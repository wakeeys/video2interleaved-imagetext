from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from PIL import Image, ImageChops, ImageStat

from .config import AudioChunk, Settings, parse_frame_region
from .model_api import image_data_url, response_text


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".flv", ".avi"}
ALLOWED_HOSTS = {"youtube.com", "youtu.be", "bilibili.com", "b23.tv"}

def check_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not any(
        host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_HOSTS
    ):
        raise ValueError("Only Bilibili, b23.tv, and YouTube video URLs are accepted")


def run_command(*args: str) -> str:
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def prefer_venv_tools() -> None:
    """Make local ffmpeg and Deno available even without activating the venv."""
    if sys.prefix == sys.base_prefix:
        return
    scripts = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    if scripts.is_dir():
        current = os.environ.get("PATH", "")
        first = current.split(os.pathsep, 1)[0]
        if first.lower() != str(scripts).lower():
            os.environ["PATH"] = str(scripts) + os.pathsep + current


def duration_seconds(path: Path) -> float:
    raw = run_command(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    )
    duration = float(json.loads(raw)["format"]["duration"])
    if duration <= 0:
        raise ValueError("Invalid video duration")
    return duration


def download_video(
    url: str,
    directory: Path,
    cookies_from_browser: str | None,
) -> tuple[Path, str]:
    import yt_dlp

    opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(directory / "source.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": False,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.stem == "source"
        and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if len(candidates) != 1:
        raise RuntimeError("Unable to find a unique video file after download; check the yt-dlp output")
    return candidates[0], str(info.get("title") or info.get("id") or url)


def extract_audio(
    video: Path,
    directory: Path,
    duration: float,
    settings: Settings,
) -> list[AudioChunk]:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("audio_*.mp3"):
        old.unlink()
    chunks: list[AudioChunk] = []
    chunk_count = math.ceil(duration / settings.audio_chunk_seconds)
    if chunk_count > 1:
        tail = duration - (chunk_count - 1) * settings.audio_chunk_seconds
        merged_duration = (
            settings.audio_chunk_seconds + settings.audio_overlap_seconds + tail
        )
        if tail <= settings.audio_overlap_seconds and merged_duration <= 290:
            chunk_count -= 1
    for index in range(chunk_count):
        nominal_start = index * settings.audio_chunk_seconds
        overlap = settings.audio_overlap_seconds if index else 0.0
        start = max(0.0, nominal_start - overlap)
        end = (
            duration
            if index == chunk_count - 1
            else nominal_start + settings.audio_chunk_seconds
        )
        path = directory / f"audio_{index:04d}.mp3"
        run_command(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video),
            "-t", f"{end - start:.3f}",
            "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", "16000",
            "-c:a", "libmp3lame",
            "-b:a", f"{settings.audio_bitrate_kbps}k",
            str(path),
        )
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Audio chunk extraction failed: {path}")
        chunks.append(AudioChunk(path, start, end))
    if not chunks:
        raise RuntimeError("No audio extracted; confirm that the video has an audio track")
    if settings.asr_model == "qwen3-asr-flash":
        for chunk in chunks:
            if chunk.path.stat().st_size > 10_000_000:
                raise ValueError(f"Audio chunk exceeds the Qwen3-ASR-Flash 10 MB limit: {chunk.path}")
    return chunks


def time_label(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}"


FrameRegion = tuple[float, float, float, float]


def detect_frame_region(
    client: object,
    frame: Path,
    model: str,
) -> FrameRegion | None:
    """Let the model choose the PPT comparison region and exclude overlays."""
    instruction = (
        "请根据当前视频截图，为本地帧间比较确定 PPT（幻灯片）的有效内容区域。"
        "先判断是否需要排除摄像头、视频字幕等干扰，再决定比较区域的边界。默认使用整张截图，不做无必要的缩小。\n"
        "判断规则：\n"
        "1. PPT 占满或基本占满画面，且没有摄像头、视频字幕等需要排除的干扰时，"
        "直接返回 has_slide=true、bbox=[0,0,1,1]。纯文字、只有标题或留白较多的幻灯片都适用。\n"
        "2. 如果存在讲者摄像头、视频字幕及其背景条、水印、明显的非 PPT 背景、黑边或播放器控件，"
        "请直接判断这些部分的位置，返回尽量避开它们、同时覆盖最多有效 PPT 内容的单个连续矩形。"
        "分屏时定位 PPT 所在的区域；字幕或摄像头叠加在 PPT 边缘时，适度收紧对应边界。\n"
        "特别检查叠加在 PPT 内部的英文字幕、中文字幕和半透明背景条：它们即使位于白色幻灯片内，"
        "也必须从比较框中排除。底部有字幕时，下边界应在最上方字幕及背景条的上方，"
        "不要直接返回包含字幕的整页 PPT 外边界。\n"
        "3. 尽量保留 PPT 的正文、图表、图注、坐标标签及必要上下文。"
        "PPT 自身的留白、页脚和原生文字属于页面，不能当成外部背景或视频字幕排除。"
        "不要只圈标题、单个图表、文字密集处，也不要无依据地一律取上半页。\n"
        "4. 有干扰时，以保留足够大且能代表 PPT 内容的可见区域为准，不能只剩零散文字或空白。"
        "请在本次返回的 bbox 中完成摄像头、字幕等区域的排除；程序将直接沿用该范围，"
        "不会再进行本地字幕识别或自动收紧边界。\n"
        "5. 如果画面中没有可辨认的 PPT，例如只有人物或视频字幕，返回 has_slide=false。"
        "若遮挡严重到无法可靠判断 PPT 的存在或范围，也返回 has_slide=false，不猜测坐标。\n"
        "用途说明：bbox 只用于比较画面变化，最终保存的截图仍是原图。识别出 PPT 不代表文章必须插图；"
        "后续融合时，只有文字不足以清楚说明且原图确有帮助，才会引用图片。\n"
        "输出要求：只返回包含 has_slide 和 bbox 的合法 JSON 对象，不附带解释或 Markdown。"
        "有明确区域时，has_slide 为 true，bbox 为 [左, 上, 右, 下]；"
        "坐标均为相对整张图片宽高的 0～1 小数，满足左 < 右、上 < 下。"
        "无明确区域时，严格返回 {\"has_slide\":false,\"bbox\":null}。"
    )
    user_content = [
        {
            "type": "text",
            "text": (
                "请先判断是否存在需要排除的摄像头、字幕等干扰；"
                "无干扰的满屏 PPT 使用整图，否则给出避开干扰后的 PPT 比较区域。"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_data_url(frame)},
        },
    ]
    request_options = (
        {"extra_body": {"enable_thinking": False}}
        if model.startswith("qwen3.8")
        else {}
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        **request_options,
    )
    payload = json.loads(response_text(response))
    if not isinstance(payload, dict) or payload.get("has_slide") is not True:
        return None
    return parse_frame_region(payload.get("bbox"))


def region_overlap(first: FrameRegion, second: FrameRegion) -> float:
    """Intersection over union of two normalized comparison rectangles."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    width = max(0.0, min(first[2], second[2]) - left)
    height = max(0.0, min(first[3], second[3]) - top)
    intersection = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def choose_frame_region(
    regions: list[FrameRegion | None],
) -> int | None:
    """Choose the original box with the highest mean overlap, if a pair agrees."""
    valid = [index for index, region in enumerate(regions) if region is not None]
    if len(regions) == 1:
        return valid[0] if valid else None
    overlaps: dict[int, list[float]] = {index: [] for index in valid}
    has_agreement = False
    for position, first_index in enumerate(valid):
        for second_index in valid[position + 1:]:
            first = regions[first_index]
            second = regions[second_index]
            assert first is not None and second is not None
            overlap = region_overlap(first, second)
            overlaps[first_index].append(overlap)
            overlaps[second_index].append(overlap)
            has_agreement |= overlap >= 0.75
    if not has_agreement:
        return None
    def score(index: int) -> tuple[float, float, int]:
        region = regions[index]
        assert region is not None
        mean_overlap = sum(overlaps[index]) / len(overlaps[index])
        area = (region[2] - region[0]) * (region[3] - region[1])
        return mean_overlap, area, -index

    return max(valid, key=score)


def extract_frames(
    video: Path,
    directory: Path,
    duration: float,
    settings: Settings,
    client: object | None = None,
    progress: Callable[[str], None] = print,
) -> list[tuple[float, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("frame_*.jpg"):
        old.unlink()
    interval = max(settings.frame_interval_seconds, duration / settings.max_frames)
    run_command(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-an",
        "-vf",
        f"fps=1/{interval:.6f},scale='min(1280,iw)':-2",
        "-q:v", "3",
        str(directory / "frame_%04d.jpg"),
    )
    files = sorted(directory.glob("frame_*.jpg"))[: settings.max_frames]
    region = (
        settings.frame_compare_region
        if settings.frame_similarity_threshold > 0
        else None
    )
    region_source = "manual" if region else "none"
    sample_name = None
    region_samples: list[dict[str, object]] = []
    should_detect_region = (
        bool(files)
        and region is None
        and settings.frame_similarity_threshold > 0
        and settings.frame_region_detection
        and client is not None
    )
    if should_detect_region:
        sample_count = min(3, len(files))
        sample_indices = [
            round((index + 0.5) * len(files) / sample_count - 0.5)
            for index in range(sample_count)
        ]
        candidates: list[FrameRegion | None] = []
        for sample_index in sample_indices:
            sample = files[sample_index]
            try:
                candidate = detect_frame_region(client, sample, settings.fusion_model)
                region_samples.append(
                    {
                        "frame": sample.stem,
                        "seconds": round(sample_index * interval, 3),
                        "bbox": list(candidate) if candidate else None,
                        "status": "valid" if candidate else "not_found",
                    }
                )
            except Exception as exc:
                candidate = None
                region_samples.append(
                    {
                        "frame": sample.stem,
                        "seconds": round(sample_index * interval, 3),
                        "bbox": None,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    }
                )
            candidates.append(candidate)
        selected_index = choose_frame_region(candidates)
        if selected_index is not None:
            region = candidates[selected_index]
            sample_name = files[sample_indices[selected_index]].name
            region_source = "model"
        elif any(candidate is not None for candidate in candidates):
            region_source = "inconsistent"
        elif any(item["status"] == "error" for item in region_samples):
            region_source = "detection_error"
        else:
            region_source = "not_found"
    if region is None and files and settings.frame_similarity_threshold > 0:
        progress("No reliable content region; retaining all candidate frames")
    kept: list[tuple[float, Path]] = []
    previous: Image.Image | None = None
    previous_name: str | None = None
    decisions: dict[str, dict[str, object]] = {}
    groups: list[list[Path]] = []
    for index, path in enumerate(files):
        if region is None or settings.frame_similarity_threshold == 0:
            kept.append((index * interval, path))
            decisions[path.stem] = {"difference": None, "compared_to": None, "kept": True}
            continue
        with Image.open(path) as img:
            width, height = img.size
            left, top, right, bottom = region
            crop_box = (
                round(width * left),
                round(height * top),
                round(width * right),
                round(height * bottom),
            )
            content = img.crop(crop_box)
            small = content.convert("L").resize((256, 144))
        if previous is None:
            difference = None
        else:
            pixel_difference = ImageChops.difference(previous, small)
            difference = ImageStat.Stat(pixel_difference).mean[0] / 255 * 100
        if not groups or (
            difference is not None
            and difference >= settings.frame_similarity_threshold
        ):
            groups.append([])
        groups[-1].append(path)
        decisions[path.stem] = {
            "difference": round(difference, 4) if difference is not None else None,
            "compared_to": previous_name,
            "group": len(groups),
            "kept": False,
        }
        previous = small
        previous_name = path.stem
    group_report = []
    if groups:
        selected = {group[-1] for group in groups}
        for number, group in enumerate(groups, 1):
            representative = group[-1]
            decisions[representative.stem]["kept"] = True
            group_report.append({
                "group": number,
                "first": group[0].stem,
                "last": representative.stem,
                "selected": representative.stem,
                "count": len(group),
            })
        for index, path in enumerate(files):
            if path in selected:
                kept.append((index * interval, path))
            else:
                path.unlink()
    report = {
        "comparison_region": list(region) if region else None,
        "model_region": list(region) if region_source == "model" else None,
        "region_source": region_source,
        "region_sample": sample_name,
        "region_samples": region_samples,
        "similarity_threshold": settings.frame_similarity_threshold,
        "comparison_mode": "previous_sample",
        "selection_mode": "last_frame_per_group" if groups else "all_candidates",
        "groups": group_report,
        "frames": decisions,
    }
    report_path = directory.parent / "frame_selection.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    progress(
        f"Frame selection: sampled {len(files)}, retained {len(kept)} candidates; "
        "final article images are determined during fusion"
    )
    return kept
