#!/usr/bin/env python3
"""Leak-safe RP dialogue view extraction.

The input is one SFT target's prompt snapshot.  The target assistant reply is
context for the training label, never context for clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ViewBoundaryError(ValueError):
    """The prompt snapshot cannot be bounded without risking target leakage."""


@dataclass(frozen=True)
class SafeDialogueViews:
    user_texts: list[str]
    recent_scene_text: str
    target_assistant_index: int
    current_user_index: int
    history_pair_indices: tuple[tuple[int, int], ...]
    target_boundary_source: str


def _text(value: Any) -> str:
    return str(value or "").strip()


def _role(message: dict[str, Any]) -> str:
    return str(message.get("role") or "").strip().casefold()


def _weight_is_one(message: dict[str, Any]) -> bool:
    value = message.get("weight")
    if value is None:
        return False
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def _target_boundary(messages: list[dict[str, Any]]) -> tuple[int, str]:
    explicit = [
        index
        for index, message in enumerate(messages)
        if _role(message) == "assistant"
        and _text(message.get("content"))
        and _weight_is_one(message)
    ]
    if len(explicit) > 1:
        raise ViewBoundaryError(
            f"multiple non-empty assistant(weight=1) targets: {explicit}"
        )
    if explicit:
        target_index = explicit[0]
        source = "explicit_weight_1"
    else:
        nonempty = [
            index
            for index, message in enumerate(messages)
            if _text(message.get("content"))
        ]
        if not nonempty:
            raise ViewBoundaryError("empty prompt snapshot")
        target_index = nonempty[-1]
        if _role(messages[target_index]) != "assistant":
            raise ViewBoundaryError(
                "weight is unavailable and final non-empty message is not assistant"
            )
        source = "inferred_final_assistant"

    trailing = [
        index
        for index in range(target_index + 1, len(messages))
        if _text(messages[index].get("content"))
    ]
    if trailing:
        raise ViewBoundaryError(
            f"non-empty messages follow target assistant at positions {trailing}"
        )
    return target_index, source


def extract_safe_dialogue_views(
    messages: list[dict[str, Any]] | None,
    *,
    max_history_pairs: int = 3,
) -> SafeDialogueViews:
    if max_history_pairs < 0:
        raise ValueError("max_history_pairs must be non-negative")
    normalized = list(messages or [])
    target_index, boundary_source = _target_boundary(normalized)

    user_positions = [
        index
        for index, message in enumerate(normalized[:target_index])
        if _role(message) == "user" and _text(message.get("content"))
    ]
    if not user_positions:
        raise ViewBoundaryError("target assistant has no preceding non-empty user")
    current_user_index = user_positions[-1]

    # Only user messages visible before the bounded target are used.  In
    # particular, no message after the target can enter either view.
    user_texts = [
        _text(normalized[index].get("content")) for index in user_positions
    ]

    pairs: list[tuple[int, int, str, str]] = []
    pending_user: tuple[int, str] | None = None
    for index, message in enumerate(normalized[:current_user_index]):
        role = _role(message)
        content = _text(message.get("content"))
        if not content:
            continue
        if role == "user":
            # A later user supersedes an unmatched user.  This refuses to
            # invent an assistant response for an incomplete historical turn.
            pending_user = (index, content)
        elif role == "assistant":
            if _weight_is_one(message):
                raise ViewBoundaryError(
                    f"assistant(weight=1) appears before current target at {index}"
                )
            if pending_user is not None:
                user_index, user_content = pending_user
                pairs.append((user_index, index, user_content, content))
                pending_user = None
            # Assistant before the first user is greeting; other unmatched
            # assistant messages are ignored instead of fabricating a pair.

    recent_pairs = pairs[-max_history_pairs:] if max_history_pairs else []
    blocks = [
        f"[USER]\n{user}\n[ASSISTANT]\n{assistant}"
        for _, _, user, assistant in recent_pairs
    ]
    blocks.append(
        f"[CURRENT_USER]\n{_text(normalized[current_user_index].get('content'))}"
    )
    recent_scene_text = "\n\n".join(blocks)
    if not recent_scene_text:
        raise ViewBoundaryError("recent scene unexpectedly empty")

    return SafeDialogueViews(
        user_texts=user_texts,
        recent_scene_text=recent_scene_text,
        target_assistant_index=target_index,
        current_user_index=current_user_index,
        history_pair_indices=tuple(
            (user_index, assistant_index)
            for user_index, assistant_index, _, _ in recent_pairs
        ),
        target_boundary_source=boundary_source,
    )
