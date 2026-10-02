"""Manual test: extract and deduplicate frames using a config file and video URL."""

from __future__ import annotations

import argparse
import json
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
    check_url, download_video, duration_seconds, extract_frames, prefer_venv_tools,
)
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

        output = create_output_directory(PROJECT_ROOT / ".tmp" / "frames", args.url)
        work = output / "work"
        work.mkdir()
        for name in ("TEMP", "TMP", "TMPDIR"):
            os.environ[name] = str(work)
        tempfile.tempdir = str(work)
        print(f"Test directory: {output}")

        video, title = download_video(args.url, work, args.cookies_from_browser)
        duration = duration_seconds(video)
        with OpenAI(
            api_key=settings.api_key, base_url=settings.base_url,
            timeout=300, max_retries=2,
        ) as client:
            frames = extract_frames(
                video, output / "frames", duration, settings, client,
                # Stop after candidate selection; do not generate an article.
                progress=lambda message: print(message.split("; final article images", 1)[0]),
            )
        if not frames:
            raise RuntimeError("No frames extracted; check the video and sampling interval")

        report = json.loads((output / "frame_selection.json").read_text(encoding="utf-8"))
        if report["selection_mode"] == "all_candidates":
            print("All candidate frames retained: the threshold is 0 or no reliable comparison region was found; difference-based deduplication was skipped.")
        print(f"Done: {title}; retained {len(frames)} frames\nFrame directory: {output / 'frames'}")
        print(f"Selection report: {output / 'frame_selection.json'}")
        return 0
    except Exception as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else None
        print(f"Error: {detail or exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
