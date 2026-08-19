"""Deterministic text hard-screening rules for the private training pipeline."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata


MAX_TURN_CHARS = 32_768
_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TOKEN = re.compile(r"[^\W_]+|[^\w\s]", re.UNICODE)
_REASONING_BLOCK = re.compile(
    r"<(think|analysis)(?:\s[^>]*)?>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_PROVIDER_TERMS = re.compile(
    r"(?:rate\s*limit|too many requests|bad gateway|service unavailable|"
    r"internal server error|gateway timeout|model (?:is )?overloaded|"
    r"request failed with status code|upstream (?:error|timeout)|"
    r"temporarily unavailable)",
    re.IGNORECASE,
)
_MEDIA_PLACEHOLDERS = {
    "[image]", "<image>", "[audio]", "<audio>", "[video]", "<video>",
    "[attachment]", "<attachment>", "[file]", "<file>",
}
_LEAKED_CONTROL_TOKEN = re.compile(r"<[|｜][^>\n]{1,80}[|｜]>")


@dataclass(frozen=True)
class TurnScreenResult:
    keep: bool
    reasons: tuple[str, ...]
    flags: tuple[str, ...]
    user_hash: str
    assistant_hash: str
    user_norm_chars: int
    assistant_norm_chars: int


def canonicalize_for_hash(text: str | None) -> str:
    """Return a Unicode/whitespace shadow form; original text is never mutated."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace("\r", "\n")
    cleaned: list[str] = []
    for char in value:
        if char in "\n\t":
            cleaned.append(char)
        elif unicodedata.category(char).startswith("C"):
            continue
        else:
            cleaned.append(char)
    value = "".join(cleaned)
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in value.split("\n")]
    value = "\n".join(lines).strip()
    value = _BLANK_LINES.sub("\n\n", value)
    return value.casefold()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_provider_error(raw: str, normalized: str) -> bool:
    stripped = raw.strip()
    if not stripped:
        return False
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            serialized = json.dumps(parsed, ensure_ascii=False)
            if _PROVIDER_TERMS.search(serialized) or re.search(r'"(?:code|status)"\s*:\s*5?\d{2}', serialized):
                return True
    if "<html" in normalized and _PROVIDER_TERMS.search(normalized):
        return True
    compact = " ".join(normalized.split())
    return bool(_PROVIDER_TERMS.fullmatch(compact) or re.fullmatch(r"(?:error\s*:\s*)?(?:4(?:08|29)|5\d{2})", compact))


def _line_repetition(normalized: str) -> bool:
    if len(normalized) < 200:
        return False
    units = [line.strip() for line in normalized.split("\n") if line.strip()]
    if len(units) < 4:
        return False
    counts = Counter(units)
    repeated_chars = sum(len(unit) for unit in units if counts[unit] >= 4)
    total_chars = sum(len(unit) for unit in units)
    return total_chars > 0 and repeated_chars / total_chars >= 0.70


def _token_repetition(normalized: str) -> bool:
    if len(normalized) < 200:
        return False
    tokens = _TOKEN.findall(normalized)
    if len(tokens) < 40:
        return False
    return Counter(tokens).most_common(1)[0][1] / len(tokens) >= 0.80


def _ngram_repetition(normalized: str, n: int = 4) -> bool:
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) < 500:
        return False
    grams = [compact[index:index + n] for index in range(len(compact) - n + 1)]
    return bool(grams) and len(set(grams)) / len(grams) < 0.05


def screen_turn(user_text: str | None, assistant_text: str | None) -> TurnScreenResult:
    user_raw = "" if user_text is None else str(user_text)
    assistant_raw = "" if assistant_text is None else str(assistant_text)
    user_norm = canonicalize_for_hash(user_raw)
    assistant_norm = canonicalize_for_hash(assistant_raw)
    reasons: list[str] = []
    flags: list[str] = []

    if not user_norm:
        reasons.append("empty_user")
    if not assistant_norm:
        reasons.append("empty_assistant")
    if len(user_raw) > MAX_TURN_CHARS:
        reasons.append("oversize_user")
    if len(assistant_raw) > MAX_TURN_CHARS:
        reasons.append("oversize_assistant")
    if assistant_norm in _MEDIA_PLACEHOLDERS:
        reasons.append("media_placeholder_only")
    nonspace = [char for char in assistant_norm if not char.isspace()]
    if nonspace and all(unicodedata.category(char).startswith("P") for char in nonspace):
        reasons.append("nonsemantic_assistant_only")
    if _LEAKED_CONTROL_TOKEN.search(assistant_raw):
        reasons.append("leaked_control_token")
    if _is_provider_error(assistant_raw, assistant_norm):
        reasons.append("provider_error")
    if user_norm == assistant_norm and len(assistant_norm) >= 10:
        reasons.append("assistant_equals_user")
    if _REASONING_BLOCK.search(assistant_raw):
        reasons.append("reasoning_leak")
    if _line_repetition(assistant_norm):
        reasons.append("severe_line_repetition")
    if _token_repetition(assistant_norm):
        reasons.append("severe_token_repetition")
    if _ngram_repetition(assistant_norm):
        reasons.append("severe_ngram_repetition")

    if assistant_norm and len(assistant_norm) < 20:
        flags.append("short_assistant")
    if re.search(r"\b(?:as an ai|i (?:can(?:not|'t)|won't) (?:help|assist|comply))\b", assistant_norm):
        flags.append("possible_refusal")

    return TurnScreenResult(
        keep=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        flags=tuple(dict.fromkeys(flags)),
        user_hash=_sha256(user_norm),
        assistant_hash=_sha256(assistant_norm),
        user_norm_chars=len(user_norm),
        assistant_norm_chars=len(assistant_norm),
    )
