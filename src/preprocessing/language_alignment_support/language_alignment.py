from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


_GUIDELINE_RE = re.compile(r"\s*\[RESPONSE LENGTH GUIDELINE\][\s\S]*$", re.IGNORECASE)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\{\{[^{}]{0,120}\}\}|<\/?(?:user|char|assistant|system)>", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]{1,200}>")
_CODE_RE = re.compile(r"```[\s\S]*?```|`[^`]*`")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Detection:
    language: str
    confidence: float
    margin: float
    top3: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PairClassification:
    status: str
    discard: bool
    reason: str
    broad_mismatch: bool


def clean_text(text: str | None) -> str:
    value = text or ""
    value = _GUIDELINE_RE.sub(" ", value)
    value = _IMAGE_RE.sub(" ", value)
    value = _URL_RE.sub(" ", value)
    value = _PLACEHOLDER_RE.sub(" ", value)
    value = _HTML_RE.sub(" ", value)
    value = _CODE_RE.sub(" ", value)
    return _SPACE_RE.sub(" ", value).strip()


def informative_char_count(text: str | None) -> int:
    return sum(unicodedata.category(char).startswith("L") for char in (text or ""))


def extract_pair(messages: list[dict[str, Any]]) -> tuple[str, str]:
    target_index = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and int(message.get("weight") or 0) == 1:
            target_index = index
    if target_index is None:
        raise ValueError("no assistant weight=1 target")
    user_content = None
    for message in reversed(messages[:target_index]):
        if message.get("role") == "user":
            user_content = message.get("content")
            break
    if user_content is None:
        raise ValueError("no user message before target")
    return str(user_content or ""), str(messages[target_index].get("content") or "")


def language_group(language: str | None) -> str:
    value = (language or "unknown").lower().replace("_", "-")
    if value.startswith("zh"):
        return "zh"
    if value in {"no", "nb", "nn"}:
        return "no"
    if value in {"bs", "hr", "sr"}:
        return "bcs"
    if value in {"id", "ms"}:
        return "msa"
    return value


def classify_pair(
    user: Detection,
    assistant: Detection,
    user_chars: int,
    assistant_chars: int,
    *,
    user_share_in_assistant: float = 0.0,
    assistant_share_in_user: float = 0.0,
) -> PairClassification:
    if user_chars < 20 or assistant_chars < 20:
        return PairClassification("uncertain_short", False, "insufficient_informative_text", False)
    if user.language in {"unknown", ""} or assistant.language in {"unknown", ""}:
        return PairClassification("uncertain_language", False, "language_not_detected", False)
    user_group = language_group(user.language)
    assistant_group = language_group(assistant.language)
    if user_group == assistant_group:
        status = "aligned" if user.language == assistant.language else "compatible"
        return PairClassification(status, False, "same_or_compatible_language", False)

    strict = user.confidence >= 0.80 and assistant.confidence >= 0.80 and user.margin >= 0.25 and assistant.margin >= 0.25
    broad = user.confidence >= 0.65 and assistant.confidence >= 0.65 and user.margin >= 0.15 and assistant.margin >= 0.15
    if strict:
        if user_share_in_assistant >= 0.35 or assistant_share_in_user >= 0.35:
            return PairClassification("code_switch_compatible", False, "counterpart_language_share_at_least_35pct", True)
        return PairClassification("mismatch", True, "high_confidence_language_mismatch", True)
    if broad:
        return PairClassification("likely_mismatch", False, "moderate_confidence_language_mismatch", True)
    return PairClassification("uncertain_confidence", False, "confidence_or_margin_below_threshold", False)
