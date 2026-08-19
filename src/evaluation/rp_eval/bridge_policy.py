from __future__ import annotations

import re
from typing import Any


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def bridge_contract_error(original: str, previous_user: str | None, result: Any) -> str | None:
    """Return a structural Bridge contract violation that no model verifier may override."""
    if result.action == "UNBRIDGEABLE":
        return None
    adapted = result.adapted_user_message
    if not adapted.strip():
        return "empty_adapted_user"
    if result.action == "KEEP":
        if adapted != original:
            return "keep_must_equal_original_next_user"
        if not result.original_intent_preserved or not result.user_style_preserved or result.added_new_fact:
            return "keep_semantic_flags_invalid"
    if previous_user is not None and original != previous_user and adapted == previous_user:
        return "stale_previous_user_replay"
    return None


def local_bridge_ok(original: str, result: Any) -> bool:
    if not result.original_intent_preserved or not result.user_style_preserved or result.added_new_fact:
        return False
    adapted = result.adapted_user_message
    if not adapted.strip():
        return False
    if result.action == "KEEP":
        return adapted == original
    if result.action == "MINOR_EDIT":
        ratio = len(adapted) / max(1, len(original))
        if ratio < 0.5 or ratio > 1.8:
            return False
        if sorted(NUMBER_RE.findall(adapted)) != sorted(NUMBER_RE.findall(original)):
            return False
    return True
