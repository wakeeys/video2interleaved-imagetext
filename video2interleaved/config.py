from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    asr_model: str
    fusion_model: str
    audio_chunk_seconds: int
    audio_overlap_seconds: float
    audio_bitrate_kbps: int
    frame_interval_seconds: float
    max_frames: int
    frame_similarity_threshold: float
    vision_batch_size: int
    default_images: bool
    output_language: str
    asr_url: str = ""
    frame_region_detection: bool = True
    frame_compare_region: tuple[float, float, float, float] | None = None
    keep_fusion_responses: bool = False


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start: float
    end: float


ASR_MODELS = {"qwen3-asr-flash", "qwen-audio-3.1-asr-flash"}


def parse_frame_region(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        return None
    left, top, right, bottom = (float(item) for item in value)
    coordinates = (left, top, right, bottom)
    if not all(math.isfinite(item) and 0 <= item <= 1 for item in coordinates):
        return None
    width = right - left
    height = bottom - top
    if width < 0.25 or height < 0.25 or width * height < 0.15:
        return None
    return left, top, right, bottom


def load_settings(path: Path) -> Settings:
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}; copy env.example.yaml to get started")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    api, models, processing = (
        data.get(section) or {}
        for section in ("api", "models", "processing")
    )
    key_variable = api.get("api_key_env", "DASHSCOPE_API_KEY")
    key = api.get("api_key") or os.environ.get(key_variable, "")
    if not key:
        raise ValueError("Missing API key; set DASHSCOPE_API_KEY or api.api_key in env.yaml")
    base_url = str(api.get("base_url", "")).rstrip("/")
    if not base_url.startswith("https://"):
        raise ValueError("api.base_url must be an HTTPS URL")
    asr_model = str(models.get("asr", "qwen-audio-3.1-asr-flash"))
    if "filetrans" in asr_model.lower():
        raise ValueError(
            "FileTrans ASR models require a public audio URL and an asynchronous task API; "
            "this program uses a synchronous API with Base64-encoded local audio; use qwen-audio-3.1-asr-flash or qwen3-asr-flash"
        )
    if asr_model not in ASR_MODELS:
        raise ValueError("models.asr currently supports qwen3-asr-flash or qwen-audio-3.1-asr-flash")
    if "fusion" not in models and ("vision" in models or "text" in models):
        raise ValueError("Model settings have been unified; replace models.vision and models.text with models.fusion")
    asr_url = str(api.get("asr_url") or "")
    if asr_model == "qwen-audio-3.1-asr-flash":
        if not asr_url:
            suffix = "/compatible-mode/v1"
            if not base_url.endswith(suffix):
                raise ValueError("Set api.asr_url when using qwen-audio-3.1-asr-flash with a nonstandard base URL")
            asr_url = (
                base_url[: -len(suffix)]
                + "/api/v1/services/aigc/multimodal-generation/generation"
            )
        if not asr_url.startswith("https://"):
            raise ValueError("api.asr_url must be an HTTPS URL")
    configured_region = processing.get("frame_compare_region")
    region = (
        parse_frame_region(configured_region)
        if configured_region is not None
        else None
    )
    if configured_region is not None and region is None:
        raise ValueError("frame_compare_region must be [left, top, right, bottom] normalized to 0-1 and cover a sufficiently large region")
    settings = Settings(
        api_key=key,
        base_url=base_url,
        asr_model=asr_model,
        fusion_model=str(models.get("fusion", "qwen3.8-flash")),
        audio_chunk_seconds=int(processing.get("audio_chunk_seconds", 120)),
        audio_overlap_seconds=float(processing.get("audio_overlap_seconds", 4)),
        audio_bitrate_kbps=int(processing.get("audio_bitrate_kbps", 32)),
        frame_interval_seconds=float(processing.get("frame_interval_seconds", 15)),
        max_frames=int(processing.get("max_frames", 120)),
        frame_similarity_threshold=float(
            processing.get("frame_similarity_threshold", 3.0)
        ),
        vision_batch_size=int(processing.get("vision_batch_size", 8)),
        default_images=processing.get("default_images", True),
        output_language=str(processing.get("output_language", "zh")),
        asr_url=asr_url,
        frame_region_detection=processing.get("frame_region_detection", True),
        frame_compare_region=region,
        keep_fusion_responses=processing.get(
            "keep_fusion_responses",
            False,
        ),
    )
    chunk_duration = settings.audio_chunk_seconds + settings.audio_overlap_seconds
    if not 1 <= settings.audio_chunk_seconds <= 290 or chunk_duration > 290:
        raise ValueError("Audio chunks including overlap must not exceed 290 seconds, leaving a margin below the ASR five-minute limit")
    if (
        not 1 <= settings.audio_overlap_seconds <= 10
        or settings.audio_overlap_seconds >= settings.audio_chunk_seconds
    ):
        raise ValueError("audio_overlap_seconds must be between 1 and 10 and less than audio_chunk_seconds")
    if not 8 <= settings.audio_bitrate_kbps <= 128:
        raise ValueError("audio_bitrate_kbps must be between 8 and 128")
    if (
        settings.frame_interval_seconds <= 0
        or settings.max_frames < 1
        or settings.vision_batch_size < 1
    ):
        raise ValueError("Frame interval, maximum frame count, and vision batch size must be greater than 0")
    if not 0 <= settings.frame_similarity_threshold <= 100:
        raise ValueError("frame_similarity_threshold must be between 0 and 100")
    if not isinstance(settings.default_images, bool):
        raise ValueError("default_images must be true or false")
    if not isinstance(settings.frame_region_detection, bool):
        raise ValueError("frame_region_detection must be true or false")
    if not isinstance(settings.keep_fusion_responses, bool):
        raise ValueError("keep_fusion_responses must be true or false")
    return settings
