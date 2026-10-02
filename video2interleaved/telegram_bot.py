"""Long-polling Telegram Bot that shares the CLI processing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from .config import load_settings
from .media import check_url
from .pipeline import process


HELP = (
    "Send a single Bilibili or YouTube video URL to process it.\n"
    "Available commands:\n"
    "/process <URL> [--images|--no-images] Choose whether to extract frames\n"
    "/status Show the task queue\n"
    "/id Show your chat ID\n"
    "/help Show help"
)


@dataclass(frozen=True)
class BotConfig:
    token: str
    allowed_chat_ids: frozenset[int]
    poll_timeout_seconds: int


def load_bot_config(path: Path) -> BotConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("telegram") or {}
    token_variable = raw.get(
        "bot_token_env",
        "TELEGRAM_BOT_TOKEN",
    )
    token = str(
        raw.get("bot_token")
        or os.environ.get(token_variable, "")
    ).strip()
    if not token:
        raise ValueError("Missing Telegram Bot Token; set telegram.bot_token in env.yaml")
    chat_ids = raw.get("allowed_chat_ids") or []
    if not isinstance(chat_ids, list):
        raise ValueError("telegram.allowed_chat_ids must be a list")
    timeout = int(raw.get("poll_timeout_seconds", 30))
    if not 1 <= timeout <= 50:
        raise ValueError("telegram.poll_timeout_seconds must be between 1 and 50")
    return BotConfig(token, frozenset(int(item) for item in chat_ids), timeout)


def parse_request(text: str, default_images: bool) -> tuple[str, bool]:
    try:
        parts = shlex.split(text.strip())
    except ValueError as exc:
        raise ValueError("Unclosed quotation mark in the message") from exc
    if parts and parts[0].split("@", 1)[0] == "/process":
        parts.pop(0)
    flags = [part for part in parts if part in {"--images", "--no-images"}]
    links = [part for part in parts if part not in {"--images", "--no-images"}]
    if len(flags) > 1 or len(links) != 1:
        raise ValueError("Usage: /process <video URL> [--images|--no-images]")
    check_url(links[0])
    return links[0], (flags[0] == "--images" if flags else default_images)


def split_message(text: str, limit: int = 3500) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def split_markdown(text: str, limit: int = 30000) -> list[str]:
    """Keep Markdown sections intact within the rich-message size limit."""
    if limit < 128:
        raise ValueError("The Markdown message limit must be at least 128 bytes")
    if not text.strip():
        return []

    sections: list[str] = []
    lines: list[str] = []
    fence = ""
    for line in text.splitlines(keepends=True):
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line)
        closes_fence = False
        if marker:
            delimiter, suffix = marker.groups()
            if not fence:
                if lines:
                    sections.append("".join(lines))
                    lines = []
                fence = delimiter
            elif (
                delimiter[0] == fence[0]
                and len(delimiter) >= len(fence)
                and not suffix.strip()
            ):
                fence = ""
                closes_fence = True
        lines.append(line)
        if not fence and (not line.strip() or closes_fence):
            sections.append("".join(lines))
            lines = []
    if lines:
        sections.append("".join(lines))

    chunks: list[str] = []
    current = ""
    for section in sections:
        if len(section.encode("utf-8")) > limit:
            if current.strip():
                chunks.append(current)
            chunks.extend(_split_long_markdown_section(section, limit))
            current = ""
        elif len((current + section).encode("utf-8")) > limit:
            chunks.append(current)
            current = section
        else:
            current += section
    if current.strip():
        chunks.append(current)
    return chunks


def _split_utf8_text(text: str, limit: int) -> list[str]:
    """Split oversized text without cutting a Unicode character."""
    chunks: list[str] = []
    while len(text.encode("utf-8")) > limit:
        prefix = text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
        cut = prefix.rfind("\n")
        if cut < len(prefix) // 2:
            cut = prefix.rfind(" ")
        if cut < len(prefix) // 2:
            cut = len(prefix)
        else:
            cut += 1
        chunks.append(text[:cut])
        text = text[cut:]
    if text:
        chunks.append(text)
    return chunks


def _split_long_markdown_section(section: str, limit: int) -> list[str]:
    """Close and reopen an oversized fenced code block in every message."""
    opening = re.match(r"( {0,3}(`{3,}|~{3,})[^\r\n]*\r?\n)", section)
    if opening is None:
        return _split_utf8_text(section, limit)

    header, delimiter = opening.groups()
    body_lines = section[len(header):].splitlines(keepends=True)
    closing_index = len(body_lines) - 1
    while closing_index >= 0 and not body_lines[closing_index].strip():
        closing_index -= 1
    closing_pattern = re.compile(r" {0,3}" + re.escape(delimiter[0]) + "{" + str(len(delimiter)) + r",}\s*")
    if closing_index >= 0 and closing_pattern.fullmatch(body_lines[closing_index]):
        body_lines = body_lines[:closing_index]
    body = "".join(body_lines)
    footer = "\n" + delimiter + "\n"
    capacity = limit - len((header + footer).encode("utf-8"))
    if capacity < 4:
        raise ValueError("The Markdown code-fence header exceeds the message limit")
    chunks: list[str] = []
    for part in _split_utf8_text(body, capacity):
        separator = "" if part.endswith("\n") else "\n"
        chunks.append(header + part + separator + delimiter + "\n")
    return chunks


class TelegramAPI:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}/"
        timeout = httpx.Timeout(
            connect=10,
            read=65,
            write=65,
            pool=10,
        )
        self.client = httpx.Client(timeout=timeout)

    def call(
        self,
        method: str,
        *,
        data: dict | None = None,
        files: dict | None = None,
    ) -> object:
        for attempt in range(4):
            if files:
                for _, item in files.items():
                    item[1].seek(0)
            try:
                response = self.client.post(
                    self.base + method,
                    data=data,
                    files=files,
                )
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Telegram {method} network error: "
                    f"{type(exc).__name__}"
                ) from exc
            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"Telegram {method} returned a non-JSON response: "
                    f"HTTP {response.status_code}"
                ) from exc
            if payload.get("ok"):
                return payload["result"]
            retry_after = (payload.get("parameters") or {}).get("retry_after")
            if attempt < 3 and (response.status_code >= 500 or retry_after):
                time.sleep(min(int(retry_after or 2 ** attempt), 30))
                continue
            description = payload.get(
                "description",
                response.status_code,
            )
            raise RuntimeError(
                f"Telegram {method} failed: {description}"
            )
        raise RuntimeError(f"Telegram {method} failed after retries")

    def send_text(self, chat_id: int, message: str) -> None:
        for part in split_message(message):
            self.call("sendMessage", data={"chat_id": chat_id, "text": part})

    def send_markdown(self, chat_id: int, message: str) -> None:
        """Send article content using Telegram's native Rich Markdown renderer."""
        for part in split_markdown(message):
            self.call(
                "sendRichMessage",
                data={
                    "chat_id": chat_id,
                    "rich_message": json.dumps({"markdown": part}, ensure_ascii=False),
                },
            )

    def send_photo(self, chat_id: int, path: Path, caption: str) -> None:
        with path.open("rb") as stream:
            self.call(
                "sendPhoto",
                data={
                    "chat_id": chat_id,
                    "caption": caption[:1000],
                },
                files={
                    "photo": (
                        path.name,
                        stream,
                        "image/jpeg",
                    )
                },
            )
        if len(caption) > 1000:
            self.send_text(chat_id, caption[1000:])

    def send_document(self, chat_id: int, path: Path) -> None:
        with path.open("rb") as stream:
            self.call(
                "sendDocument",
                data={"chat_id": chat_id},
                files={
                    "document": (
                        path.name,
                        stream,
                        "text/markdown",
                    )
                },
            )


def send_article(api: TelegramAPI, chat_id: int, output: Path) -> None:
    article = json.loads((output / "article.json").read_text(encoding="utf-8"))
    api.send_text(chat_id, f"{article['title']}\nSource: {article['source']}")
    for block in article["blocks"]:
        if block["type"] == "text":
            api.send_markdown(chat_id, block["content"])
        elif block["type"] == "image":
            image_id = block["image_id"]
            if not re.fullmatch(r"frame_\d{4}", image_id):
                raise ValueError("The article references an invalid frame ID")
            image_path = output / "frames" / f"{image_id}.jpg"
            api.send_photo(
                chat_id,
                image_path,
                block["caption"],
            )
    api.send_document(chat_id, output / "article.md")

@dataclass
class BotState:
    current_url: str | None = None


BotJob = tuple[int, str, bool]


def run_bot(
    config_path: Path,
    output_dir: Path,
    cookies_from_browser: str | None,
) -> None:
    """Start the long-polling Telegram service."""

    settings = load_settings(config_path)
    config = load_bot_config(config_path)
    _require_media_tools()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = TelegramAPI(config.token)
    identity = api.call("getMe")
    print(
        f"Telegram Bot @{identity.get('username', '?')} started",
        flush=True,
    )

    jobs: queue.Queue[BotJob] = queue.Queue(maxsize=10)
    state = BotState()
    worker = threading.Thread(
        target=_work_jobs,
        args=(
            jobs,
            state,
            api,
            config_path,
            output_dir,
            cookies_from_browser,
        ),
        daemon=True,
    )
    worker.start()
    _poll_updates(
        api,
        config,
        settings.default_images,
        jobs,
        state,
        output_dir / ".telegram_offset",
    )


def _require_media_tools() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "Install ffmpeg and ffprobe and add them to PATH"
        )


def _work_jobs(
    jobs: queue.Queue[BotJob],
    state: BotState,
    api: TelegramAPI,
    config_path: Path,
    output_dir: Path,
    cookies_from_browser: str | None,
) -> None:
    while True:
        chat_id, url, images = jobs.get()
        state.current_url = url
        try:
            args = argparse.Namespace(
                url=url,
                file=None,
                images=images,
                config=config_path,
                output_dir=output_dir,
                cookies_from_browser=cookies_from_browser,
            )
            output = process(
                args,
                lambda message: print(message, flush=True),
            )
            send_article(api, chat_id, output)
            api.send_text(chat_id, "Processing complete.")
        except Exception as exc:
            _report_job_error(api, chat_id, exc)
        finally:
            state.current_url = None
            jobs.task_done()


def _report_job_error(
    api: TelegramAPI,
    chat_id: int,
    error: Exception,
) -> None:
    print(f"Task failed: {error}", flush=True)
    try:
        api.send_text(chat_id, f"Processing failed: {error}")
    except Exception as send_error:
        print(
            f"Unable to report the error to Telegram: {send_error}",
            flush=True,
        )


def _poll_updates(
    api: TelegramAPI,
    config: BotConfig,
    default_images: bool,
    jobs: queue.Queue[BotJob],
    state: BotState,
    offset_file: Path,
) -> None:
    offset = _load_offset(offset_file)
    while True:
        try:
            updates = api.call(
                "getUpdates",
                data={
                    "offset": offset,
                    "timeout": config.poll_timeout_seconds,
                    "allowed_updates": json.dumps(["message"]),
                },
            )
            for update in updates:
                offset = update["update_id"] + 1
                _handle_update(
                    api,
                    update,
                    config,
                    default_images,
                    jobs,
                    state,
                )
            if updates:
                _save_offset(offset_file, offset)
        except KeyboardInterrupt:
            print("Bot stopped", flush=True)
            return
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            print(
                f"Telegram connection error: {exc}; retrying in 5 seconds",
                flush=True,
            )
            time.sleep(5)


def _handle_update(
    api: TelegramAPI,
    update: dict,
    config: BotConfig,
    default_images: bool,
    jobs: queue.Queue[BotJob],
    state: BotState,
) -> None:
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = message.get("text") or ""
    if chat_id is None or not text:
        return

    command = text.split(maxsplit=1)[0].split("@", 1)[0]
    if command == "/id":
        api.send_text(chat_id, f"Your chat ID: {chat_id}")
        return
    if (
        config.allowed_chat_ids
        and chat_id not in config.allowed_chat_ids
    ):
        api.send_text(chat_id, "This chat ID is not authorized to use the Bot.")
        return
    if command in {"/start", "/help"}:
        api.send_text(chat_id, HELP)
        return
    if command == "/status":
        api.send_text(
            chat_id,
            f"Current task: {state.current_url or 'none'}\n"
            f"Queued tasks: {jobs.qsize()}",
        )
        return

    _enqueue_request(
        api,
        chat_id,
        text,
        default_images,
        jobs,
        state,
    )


def _enqueue_request(
    api: TelegramAPI,
    chat_id: int,
    text: str,
    default_images: bool,
    jobs: queue.Queue[BotJob],
    state: BotState,
) -> None:
    try:
        url, images = parse_request(text, default_images)
    except ValueError as exc:
        api.send_text(chat_id, f"{exc}\n\n{HELP}")
        return

    ahead = jobs.qsize() + int(state.current_url is not None)
    try:
        jobs.put_nowait((chat_id, url, images))
    except queue.Full:
        api.send_text(chat_id, "The task queue is full; try again later.")
        return

    image_status = "enabled" if images else "disabled"
    api.send_text(
        chat_id,
        f"Queued ({ahead} tasks ahead). "
        f"Images: {image_status}",
    )


def _load_offset(offset_file: Path) -> int:
    if not offset_file.exists():
        return 0
    return int(offset_file.read_text(encoding="utf-8"))


def _save_offset(offset_file: Path, offset: int) -> None:
    temporary = offset_file.with_suffix(".tmp")
    temporary.write_text(str(offset), encoding="utf-8")
    temporary.replace(offset_file)

