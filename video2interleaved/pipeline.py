from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import AudioChunk, Settings, load_settings
from .fusion import compose_article
from .media import (
    check_url,
    download_video,
    duration_seconds,
    extract_audio,
    extract_frames,
    prefer_venv_tools,
    time_label,
)
from .model_api import transcribe


FINAL_OUTPUT_NAMES = {
    "article.json",
    "article.md",
    "audio_raw.md",
    "audio.md",
    "frame_selection.json",
    "frames",
    "metadata.json",
}


def process(
    args: argparse.Namespace,
    progress: Callable[[str], None] = print,
) -> Path:
    """Run the complete video-to-article pipeline."""

    settings = load_settings(args.config)
    _require_media_tools()

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=300,
        max_retries=2,
    )
    video, title, source, output = _prepare_source(args)
    duration = duration_seconds(video)
    use_images = (
        settings.default_images
        if args.images is None
        else args.images
    )

    progress(
        f"Processing: {title} ({time_label(duration)}), "
        f"Images: {'enabled' if use_images else 'disabled'}"
    )
    audio_chunks = extract_audio(
        video,
        output / "audio",
        duration,
        settings,
    )
    frames = _extract_optional_frames(
        video,
        output,
        duration,
        settings,
        client,
        use_images,
        progress,
    )
    raw_audio_parts = transcribe(
        client,
        audio_chunks,
        settings,
        progress,
    )
    _write_segmented_transcript(
        output / "audio_raw.md",
        "Raw segmented transcript (overlap retained)",
        audio_chunks,
        raw_audio_parts,
    )

    legacy_visual = output / "visual.md"
    if legacy_visual.exists():
        legacy_visual.unlink()

    cleaned_audio: list[str] = []
    blocks = compose_article(
        client,
        raw_audio_parts,
        frames,
        settings,
        title,
        use_images,
        progress=progress,
        chunks=audio_chunks,
        cleaned_audio=cleaned_audio,
        response_dir=output / "fusion_responses",
    )
    _write_segmented_transcript(
        output / "audio.md",
        "Audio transcript with segment boundaries reconciled",
        audio_chunks,
        cleaned_audio,
    )
    _write_article_outputs(
        output,
        title,
        source,
        blocks,
    )
    _write_metadata(
        output,
        title,
        source,
        duration,
        audio_chunks,
        frames,
        use_images,
        settings,
    )
    _cleanup_successful_output(
        output,
        settings.keep_fusion_responses,
    )
    return output


def _require_media_tools() -> None:
    prefer_venv_tools()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Install ffmpeg and ffprobe and add them to PATH")


def _prepare_source(
    args: argparse.Namespace,
) -> tuple[Path, str, str, Path]:
    if args.url:
        check_url(args.url)
        output = create_output_directory(args.output_dir, args.url)
        video, title = download_video(
            args.url,
            output,
            args.cookies_from_browser,
        )
        return video, title, args.url, output

    video = args.file.resolve()
    if not video.is_file():
        raise ValueError(f"Video file not found: {video}")

    source = str(video)
    output = create_output_directory(args.output_dir, source)
    return video, video.stem, source, output


def _extract_optional_frames(
    video: Path,
    output: Path,
    duration: float,
    settings: Settings,
    client: object,
    use_images: bool,
    progress: Callable[[str], None],
) -> list[tuple[float, Path]]:
    if not use_images:
        disabled_report = {
            "comparison_region": None,
            "region_source": "disabled",
            "similarity_threshold": settings.frame_similarity_threshold,
            "selection_mode": "disabled",
            "groups": [],
            "frames": {},
        }
        (output / "frame_selection.json").write_text(
            json.dumps(
                disabled_report,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return []
    return extract_frames(
        video,
        output / "frames",
        duration,
        settings,
        client,
        progress,
    )


def _write_segmented_transcript(
    path: Path,
    heading: str,
    chunks: list[AudioChunk],
    transcripts: list[str],
) -> None:
    sections = [
        (
            f"## {time_label(chunk.start)}–{time_label(chunk.end)}"
            f"\n\n{transcript}"
        )
        for chunk, transcript in zip(chunks, transcripts)
    ]
    content = f"# {heading}\n\n" + "\n\n".join(sections) + "\n"
    path.write_text(content, encoding="utf-8")


def _write_article_outputs(
    output: Path,
    title: str,
    source: str,
    blocks: list[dict[str, str]],
) -> None:
    final_article = render_article(title, source, blocks)

    (output / "article.md").write_text(
        final_article,
        encoding="utf-8",
    )
    article_data = {
        "title": title,
        "source": source,
        "blocks": blocks,
    }
    (output / "article.json").write_text(
        json.dumps(article_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_metadata(
    output: Path,
    title: str,
    source: str,
    duration: float,
    audio_chunks: list[AudioChunk],
    frames: list[tuple[float, Path]],
    use_images: bool,
    settings: Settings,
) -> None:
    metadata = {
        "title": title,
        "source": source,
        "duration_seconds": duration,
        "audio_segments": len(audio_chunks),
        "audio_overlap_seconds": settings.audio_overlap_seconds,
        "selected_frames": len(frames),
        "images_enabled": use_images,
        "fusion_responses_saved": settings.keep_fusion_responses,
        "models": {
            "asr": settings.asr_model,
            "fusion": settings.fusion_model,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _cleanup_successful_output(
    output: Path,
    keep_fusion_responses: bool,
) -> None:
    """Remove successful-run intermediates while retaining final artifacts."""

    retained = set(FINAL_OUTPUT_NAMES)
    if keep_fusion_responses:
        retained.add("fusion_responses")

    for child in output.iterdir():
        if child.name in retained:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def render_article(title: str, source: str, blocks: list[dict[str, str]]) -> str:
    """Render interleaved text and image blocks as Markdown."""

    rendered = [f"# {title}", f"Source: {source}"]
    for block in blocks:
        if block["type"] == "text":
            rendered.append(block["content"])
        else:
            caption = block["caption"] or block["image_id"]
            caption = (
                caption.replace("\\", "\\\\")
                .replace("]", "\\]")
                .replace("\n", " ")
            )
            rendered.append(f"![{caption}](frames/{block['image_id']}.jpg)")
    return "\n\n".join(rendered) + "\n"


def create_output_directory(root: Path, source: str) -> Path:
    """Create a collision-safe directory named with source hash and time."""

    source_id = hashlib.sha256(source.encode()).hexdigest()[:12]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    prefix = f"{source_id}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1000):
        output = root / (prefix if attempt == 0 else f"{prefix}_{attempt}")
        try:
            output.mkdir()
            return output
        except FileExistsError:
            continue
    raise RuntimeError("Unable to create a unique output directory")
