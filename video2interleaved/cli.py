from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .config import (
    AudioChunk,
    Settings,
    load_settings,
    parse_frame_region,
)
from .media import (
    check_url,
    choose_frame_region,
    detect_frame_region,
    download_video,
    duration_seconds,
    extract_audio,
    extract_frames,
    prefer_venv_tools,
    region_overlap,
    run_command,
    time_label,
)
from .fusion import build_fusion_prompt, compose_article
from .fusion_validation import (
    FusionFormatError,
    FusionModelReportedError,
    build_fusion_retry_prompt,
    filter_unjustified_images,
    merge_visual_additions,
    parse_blocks,
    parse_fusion_response,
)
from .model_api import response_text, transcribe
from .pipeline import create_output_directory, process, render_article


__all__ = [
    "AudioChunk",
    "FusionFormatError",
    "FusionModelReportedError",
    "Settings",
    "build_fusion_prompt",
    "build_fusion_retry_prompt",
    "check_url",
    "choose_frame_region",
    "compose_article",
    "create_output_directory",
    "detect_frame_region",
    "download_video",
    "duration_seconds",
    "extract_audio",
    "extract_frames",
    "filter_unjustified_images",
    "load_settings",
    "main",
    "merge_visual_additions",
    "parse_blocks",
    "parse_frame_region",
    "parse_fusion_response",
    "prefer_venv_tools",
    "process",
    "region_overlap",
    "render_article",
    "response_text",
    "run_command",
    "time_label",
    "transcribe",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        description="Generate study materials from Bilibili, YouTube, or a local video"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="A single Bilibili or YouTube video URL")
    source.add_argument("--file", type=Path, help="Local video path")
    source.add_argument("--bot", action="store_true", help="Run the Telegram Bot continuously")

    parser.add_argument("--config", type=Path, default=Path("env.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable frame extraction; defaults to env.yaml",
    )
    parser.add_argument(
        "--cookies-from-browser",
        choices=["chrome", "edge", "firefox", "brave"],
        help="Use browser cookies when the video requires authentication",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface and convert failures to an exit code."""

    args = build_parser().parse_args(argv)
    try:
        if args.bot:
            from .telegram_bot import run_bot

            run_bot(args.config, args.output_dir, args.cookies_from_browser)
            return 0
        output = process(args)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Done: {output / 'article.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
