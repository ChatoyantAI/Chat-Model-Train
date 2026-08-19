from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class CotStripResult:
    content: str
    removed: bool
    valid: bool
    removed_chars: int = 0

_LEADING_THINK = re.compile(r"^\s*<think(?:\s[^>]*)?>[\s\S]*?</think>\s*", re.IGNORECASE)

def strip_provider_cot(content: str | None) -> CotStripResult:
    text = content or ""
    match = _LEADING_THINK.match(text)
    if not match:
        return CotStripResult(text, False, bool(text.strip()), 0)
    cleaned = text[match.end():].strip()
    return CotStripResult(cleaned, True, bool(cleaned), match.end())
