#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from typing import Any

from lingua import LanguageDetectorBuilder


def code(language: Any) -> str:
    iso = language.iso_code_639_1
    return iso.name.lower() if iso is not None else language.name.lower()


def primary(detector: Any, text: str) -> dict[str, Any]:
    values = detector.compute_language_confidence_values(text)
    if not values:
        return {"language": "unknown", "confidence": 0.0, "margin": 0.0, "top3": []}
    top3 = [{"language": code(value.language), "confidence": float(value.value)} for value in values[:3]]
    first = float(values[0].value)
    second = float(values[1].value) if len(values) > 1 else 0.0
    return {"language": code(values[0].language), "confidence": first, "margin": first - second, "top3": top3}


def multiple(detector: Any, text: str) -> list[dict[str, Any]]:
    return [
        {
            "language": code(section.language),
            "start": int(section.start_index),
            "end": int(section.end_index),
            "word_count": int(section.word_count),
        }
        for section in detector.detect_multiple_languages_of(text)
    ]


def main() -> int:
    detector = LanguageDetectorBuilder.from_all_languages().build()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            texts = request["texts"]
            if not isinstance(texts, list) or len(texts) > 2048:
                raise ValueError("texts must be a list with at most 2048 items")
            mode = request.get("mode", "primary")
            if mode == "primary":
                response = [primary(detector, str(text or "")) for text in texts]
            elif mode == "multi":
                response = [multiple(detector, str(text or "")) for text in texts]
            else:
                raise ValueError("unknown mode")
            sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
