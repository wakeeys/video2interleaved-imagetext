"""Manual test: generate raw ASR transcripts using a config file and video URL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from video2interleaved.config import load_settings
from video2interleaved.media import (
    check_url, download_video, duration_seconds, extract_audio,
    prefer_venv_tools, time_label,
)
from video2interleaved.model_api import transcribe
from video2interleaved.pipeline import create_output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "env.yaml")
    parser.add_argument("--url", required=True, help="Bilibili or YouTube video URL")
    parser.add_argument(
        "--cookies-from-browser",
        choices=["chrome", "edge", "firefox", "brave"],
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
        check_url(args.url)
        prefer_venv_tools()
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("Install ffmpeg and ffprobe in PATH or the virtual environment tool directory")

        output = create_output_directory(PROJECT_ROOT / ".tmp" / "audio", args.url)
        work = output / "work"
        work.mkdir()
        # Only affect this test process; third-party temporary files also go under .tmp.
        for name in ("TEMP", "TMP", "TMPDIR"):
            os.environ[name] = str(work)
        tempfile.tempdir = str(work)
        print(f"Test directory: {output}")

        video, title = download_video(args.url, work, args.cookies_from_browser)
        duration = duration_seconds(video)
        chunks = extract_audio(video, work / "audio", duration, settings)
        with OpenAI(
            api_key=settings.api_key, base_url=settings.base_url,
            timeout=300, max_retries=2,
        ) as client:
            transcripts = transcribe(client, chunks, settings)

        if len(transcripts) != len(chunks):
            raise RuntimeError("The number of transcripts does not match the number of audio chunks")
        sections = ["# Raw audio transcript (overlap retained; no deduplication or editing)"]
        for chunk, transcript in zip(chunks, transcripts):
            sections.append(
                f"## {time_label(chunk.start)}–{time_label(chunk.end)}\n\n{transcript}"
            )
        result = output / "audio_raw.md"
        result.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
        print(f"Done: {title}; {len(chunks)} audio chunks\nRaw transcript: {result}")
        return 0
    except Exception as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else None
        print(f"Error: {detail or exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
