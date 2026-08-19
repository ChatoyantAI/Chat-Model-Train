"""Deterministic repair for balanced RP quote/asterisk nesting.

Only pure, individually-balanced nesting is repaired. Missing delimiters,
unexpected closers, and ambiguous/crossing combinations are never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


OPEN_TO_CLOSE = {
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "«": "»",
    "‹": "›",
    "„": "“",
    "‚": "‘",
    "»": "«",
}
CLOSERS = set(OPEN_TO_CLOSE.values())
NESTING_ERRORS = {"quote_inside_asterisk", "asterisk_inside_quote"}


@dataclass(frozen=True)
class FormatRepair:
    content: str
    repaired: bool
    repairable: bool
    operations: tuple[dict[str, Any], ...]
    reason: str


def _escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _single_star_positions(text: str) -> list[int]:
    positions: list[int] = []
    index = 0
    while index < len(text):
        if text[index] != "*" or _escaped(text, index):
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] == "*":
            end += 1
        if end - index == 1:
            positions.append(index)
        index = end
    return positions


def _star_spans(text: str) -> list[tuple[int, int]] | None:
    positions = _single_star_positions(text)
    if len(positions) % 2:
        return None
    return [(positions[index], positions[index + 1]) for index in range(0, len(positions), 2)]


def _quote_spans(text: str) -> list[tuple[int, int]] | None:
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            if stack and stack[-1][0] == '"':
                _, opened = stack.pop()
                spans.append((opened, index))
            else:
                stack.append(('"', index))
            index += 1
            continue
        if char in CLOSERS and stack and stack[-1][0] == char:
            _, opened = stack.pop()
            spans.append((opened, index))
            index += 1
            continue
        if char in OPEN_TO_CLOSE:
            stack.append((OPEN_TO_CLOSE[char], index))
            index += 1
            continue
        if char in CLOSERS and char != "’":
            return None
        index += 1
    if stack:
        return None
    return sorted(spans)


def _relationship(first: tuple[int, int], second: tuple[int, int]) -> str:
    a0, a1 = first
    b0, b1 = second
    if a1 < b0 or b1 < a0:
        return "disjoint"
    if a0 < b0 and b1 < a1:
        return "contains"
    if b0 < a0 and a1 < b1:
        return "inside"
    return "crossing"


def _top_level_quotes_within(
    star: tuple[int, int], quote_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    contained = [quote for quote in quote_spans if _relationship(star, quote) == "contains"]
    return [
        quote
        for quote in contained
        if not any(
            other != quote
            and other[0] < quote[0]
            and quote[1] < other[1]
            for other in contained
        )
    ]


def _wrap_action_segment(segment: str) -> str:
    if not segment or not segment.strip():
        return segment
    left = len(segment) - len(segment.lstrip())
    right = len(segment) - len(segment.rstrip())
    prefix = segment[:left]
    suffix = segment[len(segment) - right :] if right else ""
    core_end = len(segment) - right if right else len(segment)
    core = segment[left:core_end]
    return prefix + "*" + core + "*" + suffix


def _split_star_around_quotes(
    text: str, star: tuple[int, int], quote_spans: list[tuple[int, int]]
) -> tuple[str, int]:
    start, end = star
    direct_quotes = _top_level_quotes_within(star, quote_spans)
    cursor = start + 1
    pieces: list[str] = []
    for quote_start, quote_end in direct_quotes:
        pieces.append(_wrap_action_segment(text[cursor:quote_start]))
        pieces.append(text[quote_start : quote_end + 1])
        cursor = quote_end + 1
    pieces.append(_wrap_action_segment(text[cursor:end]))
    return "".join(pieces), len(direct_quotes)


def repair_balanced_nesting(
    text: str,
    validation: dict[str, Any],
    validator: Callable[[str], dict[str, Any]],
) -> FormatRepair:
    """Repair pure balanced nesting and require strict post-validation."""

    errors = set(validation.get("format_error_type") or [])
    if not errors:
        return FormatRepair(text, False, False, (), "already_valid")
    if not errors.issubset(NESTING_ERRORS):
        return FormatRepair(text, False, False, (), "contains_non_nesting_error")
    # The three observed both-direction cases are ambiguous with a symmetric
    # single-star delimiter; do not guess the author's intended pairing.
    if errors == NESTING_ERRORS:
        return FormatRepair(text, False, False, (), "ambiguous_bidirectional_nesting")

    stars = _star_spans(text)
    quotes = _quote_spans(text)
    if stars is None or quotes is None:
        return FormatRepair(text, False, False, (), "delimiter_pairing_failed")
    for star in stars:
        for quote in quotes:
            if _relationship(star, quote) == "crossing":
                return FormatRepair(text, False, False, (), "true_crossing")

    operations: list[dict[str, Any]] = []
    repaired = text
    if errors == {"asterisk_inside_quote"}:
        removals: set[int] = set()
        for star in stars:
            containing = [quote for quote in quotes if _relationship(star, quote) == "inside"]
            if not containing:
                # A separate, valid action span may coexist with a star span
                # nested inside dialogue. Preserve the valid span verbatim.
                continue
            removals.update(star)
            operations.append(
                {
                    "operation": "remove_star_delimiters_inside_quote",
                    "start": star[0],
                    "end": star[1],
                }
            )
        if not removals:
            return FormatRepair(text, False, False, (), "nested_star_not_found")
        repaired = "".join(char for index, char in enumerate(text) if index not in removals)
    elif errors == {"quote_inside_asterisk"}:
        replacements: list[tuple[int, int, str, int]] = []
        for star in stars:
            contained = [quote for quote in quotes if _relationship(star, quote) == "contains"]
            if not contained:
                continue
            replacement, count = _split_star_around_quotes(text, star, quotes)
            replacements.append((star[0], star[1], replacement, count))
        if not replacements:
            return FormatRepair(text, False, False, (), "nested_quote_not_found")
        for start, end, replacement, count in sorted(replacements, reverse=True):
            repaired = repaired[:start] + replacement + repaired[end + 1 :]
            operations.append(
                {
                    "operation": "split_action_around_quotes",
                    "start": start,
                    "end": end,
                    "quote_count": count,
                }
            )

    post = validator(repaired)
    if not post.get("format_valid"):
        return FormatRepair(text, False, False, tuple(operations), "post_validation_failed")
    return FormatRepair(repaired, repaired != text, True, tuple(operations), "repaired")
