from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable

import httpx

from .config import AudioChunk, Settings


DASHSCOPE_ASR_PATH = (
    "/api/v1/services/aigc/multimodal-generation/generation"
)


def response_text(response: object) -> str:
    """Extract text from an OpenAI-compatible chat completion."""

    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned no text")
    return content.strip()


def image_data_url(path: Path) -> str:
    """Encode a local JPEG as an OpenAI-compatible data URL."""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def transcribe(
    client: object,
    chunks: list[AudioChunk],
    settings: Settings,
    progress: Callable[[str], None] = print,
) -> list[str]:
    """Transcribe every audio chunk in chronological order."""

    if settings.asr_model == "qwen-audio-3.1-asr-flash":
        return _transcribe_with_dashscope(chunks, settings, progress)
    return _transcribe_with_chat_api(client, chunks, settings, progress)


def _transcribe_with_dashscope(
    chunks: list[AudioChunk],
    settings: Settings,
    progress: Callable[[str], None],
) -> list[str]:
    asr_url = settings.asr_url
    if not asr_url:
        service_root = settings.base_url.removesuffix(
            "/compatible-mode/v1"
        )
        asr_url = service_root + DASHSCOPE_ASR_PATH

    transcripts: list[str] = []
    with httpx.Client(timeout=300) as audio_client:
        for index, chunk in enumerate(chunks, start=1):
            progress(f"Transcribing audio {index}/{len(chunks)}")
            response = audio_client.post(
                asr_url,
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "X-DashScope-SSE": "disable",
                },
                json={
                    "model": settings.asr_model,
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": _audio_data_url(chunk)
                                        },
                                    }
                                ],
                            }
                        ]
                    },
                    "parameters": {"format": "mp3"},
                },
            )
            response.raise_for_status()
            transcripts.append(
                _dashscope_transcript(response.json())
            )
    return transcripts


def _dashscope_transcript(payload: object) -> str:
    output = payload.get("output") if isinstance(payload, dict) else None
    transcript = output.get("text") if isinstance(output, dict) else None
    if isinstance(transcript, str):
        return transcript.strip()

    error_code = (
        payload.get("code", "unknown")
        if isinstance(payload, dict)
        else "invalid response"
    )
    raise RuntimeError(f"The ASR model returned no transcript: {error_code}")


def _transcribe_with_chat_api(
    client: object,
    chunks: list[AudioChunk],
    settings: Settings,
    progress: Callable[[str], None],
) -> list[str]:
    transcripts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        progress(f"Transcribing audio {index}/{len(chunks)}")
        response = client.chat.completions.create(
            model=settings.asr_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": _audio_data_url(chunk)
                            },
                        }
                    ],
                }
            ],
        )
        transcripts.append(response_text(response))
    return transcripts


def _audio_data_url(chunk: AudioChunk) -> str:
    encoded = base64.b64encode(
        chunk.path.read_bytes()
    ).decode("ascii")
    return f"data:audio/mpeg;base64,{encoded}"

