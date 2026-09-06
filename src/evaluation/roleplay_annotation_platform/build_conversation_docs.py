"""Render the exact runtime conversation rubric for repository readers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .conversation_filter import rubric_document

TARGET = (
    Path(__file__).resolve().parents[3]
    / "docs/evaluation/ROLEPLAY_CONVERSATION_FILTER_RUBRIC_V1.md"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting stale documentation",
    )
    args = parser.parse_args()
    expected = rubric_document() + "\n"
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != expected:
            raise SystemExit(
                "Conversation rubric documentation is stale; run this module without --check."
            )
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(expected, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
