from __future__ import annotations

import json
import re
from difflib import SequenceMatcher


IMAGE_NEED_KINDS = {
    "trend",
    "dense_chart",
    "complex_structure",
    "spatial_relation",
}
IMAGE_CONTENT_TYPES = {
    "trend_chart",
    "dense_chart",
    "complex_diagram",
    "spatial_map",
}
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _letters(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _uses_one_based_anchors(
    items: list,
    block_count: int,
) -> bool:
    anchors = [
        item.get("after_text_block")
        for item in items
        if isinstance(item, dict)
    ]
    return (
        bool(anchors)
        and block_count in anchors
        and all(
            type(anchor) is int and 1 <= anchor <= block_count
            for anchor in anchors
        )
    )


def validated_image_id(block: dict, allowed_images: set[str]) -> str:
    """Accept the unambiguous id alias, then enforce the current image allowlist."""
    has_conflicting_ids = (
        "image_id" in block
        and "id" in block
        and block["image_id"] != block["id"]
    )
    if has_conflicting_ids:
        raise ValueError("The image_id and id fields conflict; the referenced frame is ambiguous")
    image_id = block["image_id"] if "image_id" in block else block.get("id")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("Image item is missing a valid image_id string (id is accepted as an alias)")
    if image_id not in allowed_images:
        allowed = ", ".join(sorted(allowed_images)) or "none"
        raise ValueError(
            f"Frame ID {image_id!r} is not among the candidates supplied in this request; "
            f"Allowed IDs: {allowed}"
        )
    return image_id


def parse_blocks(raw: str, allowed_images: set[str]) -> list[dict[str, str]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("The model returned invalid JSON; try again") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("blocks"), list):
        raise ValueError("The model JSON is missing the blocks list")
    blocks: list[dict[str, str]] = []
    for block in payload["blocks"]:
        if not isinstance(block, dict):
            raise ValueError("The model returned an invalid content block")
        kind = block.get("type")
        is_text = (
            kind == "text"
            and isinstance(block.get("content"), str)
            and block["content"].strip()
        )
        if is_text:
            if MARKDOWN_IMAGE.search(block["content"]):
                raise ValueError("Text blocks cannot embed images; use an image block")
            blocks.append(
                {
                    "type": "text",
                    "content": block["content"].strip(),
                }
            )
        elif kind == "image":
            blocks.append(
                {
                    "type": "image",
                    "image_id": validated_image_id(
                        block,
                        allowed_images,
                    ),
                    "caption": str(
                        block.get("caption") or ""
                    ).strip(),
                }
            )
        else:
            raise ValueError(f"The model returned an invalid content block or unknown frame ID: {kind}")
    if not blocks or not any(block["type"] == "text" for block in blocks):
        raise ValueError("The model returned no readable text blocks")
    return blocks


def remove_empty_audio_blocks(payload: dict) -> None:
    """Drop empty model text blocks and remap visual anchors to retained blocks."""
    blocks = payload["audio_blocks"]
    empty = {
        index
        for index, block in enumerate(blocks)
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("content"), str)
        and not block["content"].strip()
    }
    if not empty:
        return
    retained_indices = [
        index
        for index in range(len(blocks))
        if index not in empty
    ]
    mapping = {
        old: new
        for new, old in enumerate(retained_indices)
    }
    items = payload["visual_items"]
    one_based = _uses_one_based_anchors(items, len(blocks))
    for item in items:
        if not isinstance(item, dict):
            continue
        anchor = item.get("after_text_block")
        if type(anchor) is not int:
            continue
        old_anchor = anchor - 1 if one_based else anchor
        if old_anchor in empty:
            raise ValueError("A visual addition references an empty audio block; the insertion position is ambiguous")
        if old_anchor in mapping:
            item["after_text_block"] = mapping[old_anchor]
    payload["audio_blocks"] = [
        block
        for index, block in enumerate(blocks)
        if index not in empty
    ]


def remove_duplicate_audio_blocks(payload: dict) -> None:
    """Collapse long identical neighboring draft blocks and retain visual anchors."""
    blocks = payload["audio_blocks"]
    kept: list[dict] = []
    mapping: dict[int, int] = {}
    for index, block in enumerate(blocks):
        content = block.get("content") if isinstance(block, dict) else None
        previous = (
            kept[-1].get("content")
            if kept and isinstance(kept[-1], dict)
            else None
        )
        is_duplicate = (
            isinstance(content, str)
            and isinstance(previous, str)
            and len(content.strip()) >= 120
            and content.strip() == previous.strip()
        )
        if is_duplicate:
            mapping[index] = len(kept) - 1
        else:
            mapping[index] = len(kept)
            kept.append(block)
    if len(kept) == len(blocks):
        return
    items = payload["visual_items"]
    one_based = _uses_one_based_anchors(items, len(blocks))
    for item in items:
        if not isinstance(item, dict):
            continue
        anchor = item.get("after_text_block")
        old = anchor - 1 if one_based and type(anchor) is int else anchor
        if type(old) is int and old in mapping:
            item["after_text_block"] = mapping[old]
    payload["audio_blocks"] = kept


def merge_visual_additions(
    audio_blocks: list[dict[str, str]],
    raw: str,
    allowed_images: set[str],
) -> list[dict[str, str]]:
    """Insert visual additions without changing or removing the audio draft."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("The visual addition response is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("The visual addition JSON is missing the items list")
    # An anchor equal to the block count is invalid in zero-based indexing but
    # unambiguously identifies a one-based response when every anchor is 1..N.
    one_based = _uses_one_based_anchors(
        payload["items"],
        len(audio_blocks),
    )
    additions: list[list[dict[str, str]]] = [[] for _ in audio_blocks]
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise ValueError("A visual addition contains an invalid content block")
        anchor = item.get("after_text_block")
        if one_based:
            anchor -= 1
        if type(anchor) is not int or not 0 <= anchor < len(audio_blocks):
            raise ValueError("A visual addition references a nonexistent audio block")
        is_text = (
            item.get("type") == "text"
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        )
        if is_text:
            content = item["content"].strip()
            if MARKDOWN_IMAGE.search(content):
                raise ValueError("Visual addition text cannot embed images")
            content = re.sub(
                r"^#{2,4}\s*(?:\u8865\u5145\u4fe1\u606f|\u753b\u9762\u8865\u5145|\u5e7b\u706f\u7247\u8865\u5145)\s*\n+",
                "",
                content,
            ).strip()
            if content:
                additions[anchor].append({"type": "text", "content": content})
        elif item.get("type") == "image":
            additions[anchor].append(
                {
                    "type": "image",
                    "image_id": validated_image_id(
                        item,
                        allowed_images,
                    ),
                    "caption": str(
                        item.get("caption") or ""
                    ).strip(),
                }
            )
        else:
            raise ValueError(
                "Invalid visual addition: type must be text or image, "
                "and text items must have nonempty content"
            )
    merged: list[dict[str, str]] = []
    for index, block in enumerate(audio_blocks):
        merged.append(block)
        merged.extend(additions[index])
    return merged


def filter_unjustified_images(
    blocks: list[dict[str, str]],
    raw: str,
) -> list[dict[str, str]]:
    """Require a concrete visual dependency before publishing a model-selected image."""
    payload = json.loads(raw)
    eligible: set[str] = set()
    for item in payload["visual_items"]:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        need = item.get("visual_need")
        if not isinstance(need, dict):
            continue
        if need.get("kind") not in IMAGE_NEED_KINDS:
            continue
        if need.get("content_type") not in IMAGE_CONTENT_TYPES:
            continue
        reason = need.get("why_text_insufficient")
        if isinstance(reason, str) and len(_letters(reason)) >= 12:
            eligible.add(item.get("image_id", item.get("id")))
    return [
        block
        for block in blocks
        if block["type"] != "image"
        or block["image_id"] in eligible
    ]


def sentence_breaks(value: str) -> list[re.Match[str]]:
    """Find sentence ends while ignoring dots inside numbers and web addresses."""
    breaks = []
    for match in re.finditer(r"[.!?。！？]+(?:[\"'”’）)]*)\s*", value):
        at = match.start()
        if value[at] == "." and at > 0 and at + 1 < len(value):
            before, after = value[at - 1], value[at + 1]
            is_internal_dot = (
                before.isalnum()
                and after.isalnum()
                and (
                    after.islower()
                    or before.isdigit() and after.isdigit()
                )
            )
            if is_internal_dot:
                continue
        breaks.append(match)
    return breaks


def defer_translated_sentence(
    payload: dict,
    revised_cleaned: str,
    source_sentence: str,
) -> None:
    """Remove the trailing translated sentence that is now held in carry."""
    raw_blocks = payload.get("audio_blocks")
    if (
        not isinstance(raw_blocks, list)
        or not raw_blocks
        or not isinstance(raw_blocks[-1], dict)
    ):
        raise ValueError("The boundary sentence was moved to carry_to_next, but no audio block can be adjusted")
    last_content = raw_blocks[-1].get("content")
    if raw_blocks[-1].get("type") != "text" or not isinstance(last_content, str):
        raise ValueError("The boundary sentence was moved to carry_to_next, but the final audio block is not text")
    translated_end = last_content.rstrip().rstrip(".!?。！？…\"'”’）)] ")
    translated_stops = sentence_breaks(translated_end)
    if translated_stops:
        translated_start = translated_stops[-1].end()
        translated_tail = translated_end[translated_start:].strip()
        source_size = len(_letters(source_sentence))
        target_size = len(_letters(translated_tail))
        trailing_punctuation = last_content.rstrip().endswith(
            ("。", "！", "？", ".", "!", "?")
        )
        # A very short, complete final target sentence is likely the preceding
        # idea: the model may already have omitted the carried sentence.
        plausible_translation_size = (
            source_size
            and 0.2 <= target_size / source_size <= 1.5
        )
        if (
            trailing_punctuation
            and source_size
            and not plausible_translation_size
        ):
            return
        revised_last = last_content[:translated_start].rstrip()
        if revised_last and not re.fullmatch(r"(?:#{1,6}\s+[^\n]+\s*)+", revised_last):
            raw_blocks[-1] = {**raw_blocks[-1], "content": revised_last}
        else:
            raw_blocks.pop()
    elif not revised_cleaned:
        raw_blocks.pop()
    else:
        source_size = len(_letters(source_sentence))
        target_size = len(_letters(translated_end))
        complete = last_content.rstrip().endswith(
            ("。", "！", "？", ".", "!", "?")
        )
        plausible_translation_size = (
            source_size
            and 0.2 <= target_size / source_size <= 1.5
        )
        if (
            not complete
            or not source_size
            or plausible_translation_size
        ):
            raise FusionFormatError(
                "The boundary sentence was moved to carry_to_next, but its translation cannot be reliably identified at the end of the audio draft; "
                "rewrite this segment's audio_blocks to cover the completed transcript and omit the deferred final sentence"
            )
        # A short complete block cannot plausibly translate the long carried
        # sentence; preserve it as already finished audio content.
        return
    if revised_cleaned and not raw_blocks:
        raise ValueError("The boundary sentence was moved to carry_to_next, but adjusting its translation left no completed audio draft")
    if not raw_blocks and payload.get("visual_items") != []:
        raise ValueError("The boundary sentence was moved to carry_to_next, but visual additions still reference the deferred audio draft")


def repair_duplicate_carry(
    payload: dict,
    cleaned: str,
    carry: str,
) -> tuple[str, str]:
    """Move the final sentence from cleaned when carry repeats its tail."""
    source_end = cleaned.rstrip().rstrip(".!?。！？\"'”’）)] ")
    carry_end = carry.rstrip().rstrip(".!?。！？\"'”’）)] ")
    if not carry_end or not source_end.casefold().endswith(carry_end.casefold()):
        raise ValueError(
            "carry_to_next duplicates cleaned_transcript, "
            "but the exact suffix cannot be located; no automatic change was made"
        )
    carry_start = len(source_end) - len(carry_end)
    previous_stops = sentence_breaks(source_end[:carry_start])
    sentence_start = previous_stops[-1].end() if previous_stops else 0
    sentence = source_end[sentence_start:]
    sentence_start += len(sentence) - len(sentence.lstrip())
    revised_cleaned = source_end[:sentence_start].rstrip()
    revised_carry = cleaned[sentence_start:].strip()
    if not _letters(revised_carry).endswith(_letters(carry)):
        raise ValueError("carry_to_next duplicates cleaned_transcript, but the full final sentence cannot be identified")
    defer_translated_sentence(payload, revised_cleaned, revised_carry)
    return revised_cleaned, revised_carry


def repair_split_carry(
    payload: dict,
    cleaned: str,
    carry: str,
    transcript: str,
) -> tuple[str, str]:
    """Join a sentence split between cleaned and carry, then defer it intact."""
    source = transcript.rstrip()
    if not source.endswith(carry):
        raise ValueError("carry_to_next cannot be aligned with the raw transcript suffix; no automatic change was made")
    carry_start = len(source) - len(carry)
    previous_stops = sentence_breaks(source[:carry_start])
    sentence_start = previous_stops[-1].end() if previous_stops else 0
    source_sentence = source[sentence_start:].strip()
    leading_piece = source[sentence_start:carry_start].strip()
    if not leading_piece or not cleaned.casefold().endswith(leading_piece.casefold()):
        raise ValueError(
            "cleaned_transcript and carry_to_next "
            "cannot be reliably joined into the final sentence; no automatic change was made"
        )
    revised_cleaned = cleaned[:-len(leading_piece)].rstrip()
    if revised_cleaned and revised_cleaned[-1] not in ".!?。！？\n":
        raise ValueError("The sentence start cannot be identified at the end of cleaned_transcript; no automatic change was made")
    defer_translated_sentence(payload, revised_cleaned, source_sentence)
    return revised_cleaned, source_sentence


class FusionFormatError(ValueError):
    """The response cannot be interpreted as a fusion JSON object."""


class FusionModelReportedError(ValueError):
    """The model returned a concrete reason it could not complete fusion."""


def validate_audio_coverage(cleaned: str, payload: dict) -> None:
    """Catch obvious wholesale omissions before trying to repair a boundary sentence."""
    source_size = len(_letters(cleaned))
    if source_size < 500:
        return
    blocks = payload["audio_blocks"]
    target_size = sum(
        len(_letters(block.get("content", "")))
        for block in blocks
        if isinstance(block, dict)
        and isinstance(block.get("content"), str)
    )
    if target_size < source_size * 0.15:
        raise FusionFormatError(
            f"The audio draft appears to omit substantial content: cleaned_transcript has about {source_size} characters, "
            f"but audio_blocks has only {target_size}; include all completed content from this segment, "
            "not just the previous carry or opening sentence, and keep the unfinished final sentence out of the draft"
        )


def build_fusion_retry_prompt(error: str, attempt: int, previous_reply: str) -> str:
    """Explain the latest failed attempt and require a top-level JSON object."""
    return (
        f"你已经尝试生成 {attempt} 次，但最近一次程序报告错误：{error}。\n"
        f"最近一次回复的开头：{previous_reply[:300]!r}\n"
        "本次请根据当前段原始转写、上段待接续句和当前段截图，重新生成完整融合结果，"
        "针对上述具体错误修正；不要只返回上次正文块或空结果。\n"
        "成功时只返回一个合法 JSON 对象，顶层以 { 开始、以 } 结束，"
        "必须包含 cleaned_transcript、carry_to_next、audio_blocks、visual_items 四个字段。"
        "明确禁止顶层返回 [] 或任何其他数组，包括只列出 text 块的数组；"
        "audio_blocks 和 visual_items 是对象内部的数组字段，仍须按原规则填写。"
        "没有截图或视觉补充时将 visual_items 设为 []，不能将整个回复设为 []。\n"
        "即使上次 JSON 语法合法，也要修正结构或内容错误。audio_blocks 必须覆盖 "
        "cleaned_transcript 中全部已完成的语义，不能只写上段 carry 接成的第一句话；"
        "原有的忠实性、去重、边界句和图片必要性要求继续适用。"
        "不加代码围栏、前言、摘要、省略号或处理说明。\n"
        "如果仍无法按要求完成，请只返回合法 JSON 对象 "
        '{"error":"具体原因"}，明确说明遇到的实际障碍；成功时不写 error 或解释。'
    )


def parse_fusion_response(
    raw: str,
    transcript: str,
    allowed_images: set[str],
    is_last: bool = False,
) -> tuple[str, str, list[dict[str, str]], list[dict[str, str]]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FusionFormatError("The fusion model returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise FusionFormatError("The fusion model did not return a JSON object")
    if isinstance(payload.get("error"), str) and payload["error"].strip():
        reason = payload["error"].strip()
        raise FusionModelReportedError(
            f"The fusion model reported that it cannot generate the result: {reason}"
        )
    cleaned = payload.get("cleaned_transcript")
    carry = payload.get("carry_to_next")
    if not isinstance(cleaned, str) or not isinstance(carry, str):
        raise FusionFormatError("The fusion response is missing cleaned_transcript or carry_to_next strings")
    has_block_lists = (
        isinstance(payload.get("audio_blocks"), list)
        and isinstance(payload.get("visual_items"), list)
    )
    if not has_block_lists:
        raise FusionFormatError("The fusion response is missing audio_blocks or visual_items arrays")
    cleaned, carry = cleaned.strip(), carry.strip()
    validate_audio_coverage(cleaned, payload)
    if is_last and carry:
        raise ValueError("The final segment must leave carry_to_next empty")
    if not cleaned and not carry:
        raise ValueError("The fusion response retained no transcript text")
    carry_key = _letters(carry)
    if carry:
        if not carry_key:
            raise ValueError("carry_to_next cannot contain only punctuation or whitespace")
        if _letters(cleaned).endswith(carry_key):
            cleaned, carry = repair_duplicate_carry(payload, cleaned, carry)
        # A complete carry is unchanged. If it is only the latter half of the
        # source's final sentence, move the matching leading half out of cleaned.
        source = transcript.rstrip()
        if source.endswith(carry):
            prefix = source[:-len(carry)].rstrip()
            if prefix and prefix[-1] not in ".!?。！？\n":
                cleaned, carry = repair_split_carry(
                    payload,
                    cleaned,
                    carry,
                    transcript,
                )
    original_key = _letters(transcript)
    revised_key = _letters(cleaned + carry)
    # Comparing individual letters with autojunk=True can discard common letters
    # in long transcripts and mistake small edits for wholesale content loss.
    similarity = SequenceMatcher(
        None,
        original_key,
        revised_key,
        autojunk=False,
    ).ratio()
    if len(original_key) >= 100 and similarity < 0.55:
        raise ValueError(
            "Boundary reconciliation changed too much and may have omitted transcript content"
            f" (character similarity {similarity:.1%}; at least 55% is required)"
        )
    remove_duplicate_audio_blocks(payload)
    remove_empty_audio_blocks(payload)
    raw_audio_blocks = payload.get("audio_blocks")
    if not cleaned and carry and raw_audio_blocks == []:
        audio_blocks: list[dict[str, str]] = []
    else:
        audio_blocks = parse_blocks(
            json.dumps({"blocks": raw_audio_blocks}),
            set(),
        )
    if not cleaned and audio_blocks:
        raise ValueError("audio_blocks must be empty when cleaned_transcript is empty")
    has_unfinished_ending = (
        not is_last
        and not carry
        and audio_blocks
        and re.search(
            r"(?:…+|\.{3,})\s*$",
            audio_blocks[-1]["content"],
        )
    )
    if has_unfinished_ending:
        raise FusionFormatError(
            "The audio draft ends with an ellipsis or unfinished sentence, but carry_to_next is empty; if the speaker was interrupted, "
            "defer the speaker's unfinished sentence to the next segment instead of leaving a fragment in the draft"
        )
    visual_items = payload.get("visual_items")
    if not allowed_images and visual_items != []:
        raise ValueError("Segments without screenshots cannot add visual content")
    if not audio_blocks:
        if visual_items != []:
            raise ValueError("visual_items must be empty when the entire segment is deferred")
        combined = []
    else:
        combined = merge_visual_additions(
            audio_blocks,
            json.dumps({"items": visual_items}),
            allowed_images,
        )
    return cleaned, carry, audio_blocks, combined
