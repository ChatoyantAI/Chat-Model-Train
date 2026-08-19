#!/usr/bin/env python3
"""Run target user/assistant Lingua alignment on context-plan rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_ROOT = Path(__file__).resolve().parent
LANGUAGE_ROOT = Path(
    os.environ.get(
        "LUSY_LANGUAGE_ALIGNMENT_ROOT",
        str(SCRIPT_ROOT / "language_alignment_support"),
    )
)
sys.path.insert(0, str(LANGUAGE_ROOT))

from build_language_alignment import (  # noqa: E402
    DETECTOR_NAME,
    DETECTOR_VERSION,
    DetectorPool,
    detection,
    excerpt,
    section_share,
)
from language_alignment import (  # noqa: E402
    classify_pair,
    clean_text,
    informative_char_count,
)


ROOT = Path("/home/gyr/lusychat-training/pre-mds-context-v2")
INPUT = ROOT / "02_context_plan/parts"
OUTPUT = ROOT / "03_language_alignment/features"
RULE_VERSION = "language-alignment-v2-context-plan"

SCHEMA = pa.schema(
    [
        ("target_id", pa.string()),
        ("part_name", pa.string()),
        ("row_index", pa.int32()),
        ("conversation_id", pa.string()),
        ("user_id", pa.string()),
        ("character_id", pa.string()),
        ("pool", pa.string()),
        ("source_kind", pa.string()),
        ("assistant_language_existing", pa.string()),
        ("user_language", pa.string()),
        ("assistant_language", pa.string()),
        ("user_confidence", pa.float32()),
        ("assistant_confidence", pa.float32()),
        ("user_margin", pa.float32()),
        ("assistant_margin", pa.float32()),
        ("user_informative_chars", pa.int32()),
        ("assistant_informative_chars", pa.int32()),
        ("user_top3", pa.string()),
        ("assistant_top3", pa.string()),
        ("user_share_in_assistant", pa.float32()),
        ("assistant_share_in_user", pa.float32()),
        ("status", pa.string()),
        ("discard", pa.bool_()),
        ("broad_mismatch", pa.bool_()),
        ("reason", pa.string()),
        ("user_text_sha256", pa.string()),
        ("assistant_text_sha256", pa.string()),
        ("detector_name", pa.string()),
        ("detector_version", pa.string()),
        ("rule_version", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", required=True)
    parser.add_argument("--detector-workers", type=int, default=2)
    parser.add_argument("--input-root", type=Path, default=INPUT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    return parser.parse_args()


def parse_range(value: str) -> list[int]:
    if "-" not in value:
        return [int(value)]
    start, end = value.split("-", 1)
    return list(range(int(start), int(end) + 1))


def process_part(path: Path, output: Path, pool: DetectorPool) -> dict[str, Any]:
    rows = pq.read_table(
        path,
        columns=[
            "target_id",
            "conversation_id",
            "user_id",
            "character_id",
            "pool",
            "source_kind",
            "language_final",
            "target_user_content",
            "target_content",
        ],
    ).to_pylist()
    pairs = [
        (
            clean_text(str(row.get("target_user_content") or "")),
            clean_text(str(row.get("target_content") or "")),
        )
        for row in rows
    ]
    texts = [excerpt(text) for pair in pairs for text in pair]
    detected = pool.detect("primary", texts)
    users = [detection(detected[index * 2]) for index in range(len(rows))]
    assistants = [
        detection(detected[index * 2 + 1]) for index in range(len(rows))
    ]
    chars = [
        (informative_char_count(user), informative_char_count(assistant))
        for user, assistant in pairs
    ]
    preliminary = [
        classify_pair(users[index], assistants[index], *chars[index])
        for index in range(len(rows))
    ]
    strict_candidates = [
        index for index, result in enumerate(preliminary) if result.discard
    ]
    multi_texts = [
        excerpt(text) for index in strict_candidates for text in pairs[index]
    ]
    multi = pool.detect("multi", multi_texts, batch_size=128)
    shares: dict[int, tuple[float, float]] = {}
    for offset, index in enumerate(strict_candidates):
        user_sections = multi[offset * 2]
        assistant_sections = multi[offset * 2 + 1]
        shares[index] = (
            section_share(assistant_sections, users[index].language),
            section_share(user_sections, assistants[index].language),
        )

    features: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    discarded = 0
    for index, row in enumerate(rows):
        user_text, assistant_text = pairs[index]
        user_share, assistant_share = shares.get(index, (0.0, 0.0))
        result = classify_pair(
            users[index],
            assistants[index],
            *chars[index],
            user_share_in_assistant=user_share,
            assistant_share_in_user=assistant_share,
        )
        statuses[result.status] += 1
        discarded += int(result.discard)
        features.append(
            {
                "target_id": row["target_id"],
                "part_name": path.name,
                "row_index": index,
                "conversation_id": row.get("conversation_id"),
                "user_id": row.get("user_id"),
                "character_id": row.get("character_id"),
                "pool": row.get("pool"),
                "source_kind": row.get("source_kind"),
                "assistant_language_existing": row.get("language_final"),
                "user_language": users[index].language,
                "assistant_language": assistants[index].language,
                "user_confidence": users[index].confidence,
                "assistant_confidence": assistants[index].confidence,
                "user_margin": users[index].margin,
                "assistant_margin": assistants[index].margin,
                "user_informative_chars": chars[index][0],
                "assistant_informative_chars": chars[index][1],
                "user_top3": json.dumps(users[index].top3, ensure_ascii=False),
                "assistant_top3": json.dumps(
                    assistants[index].top3, ensure_ascii=False
                ),
                "user_share_in_assistant": user_share,
                "assistant_share_in_user": assistant_share,
                "status": result.status,
                "discard": result.discard,
                "broad_mismatch": result.broad_mismatch,
                "reason": result.reason,
                "user_text_sha256": hashlib.sha256(
                    user_text.encode("utf-8")
                ).hexdigest(),
                "assistant_text_sha256": hashlib.sha256(
                    assistant_text.encode("utf-8")
                ).hexdigest(),
                "detector_name": DETECTOR_NAME,
                "detector_version": DETECTOR_VERSION,
                "rule_version": RULE_VERSION,
            }
        )
    temp = output.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pylist(features, schema=SCHEMA),
        temp,
        compression="zstd",
        compression_level=6,
        row_group_size=16384,
    )
    os.replace(temp, output)
    return {
        "rows": len(rows),
        "discarded": discarded,
        "status": dict(statuses),
    }


def main() -> int:
    args = parse_args()
    buckets = parse_range(args.buckets)
    args.output_root.mkdir(parents=True, exist_ok=True)
    receipt_root = ROOT / "00_manifest/language_receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total = 0
    discarded = 0
    statuses: Counter[str] = Counter()
    pool = DetectorPool(args.detector_workers)
    try:
        for bucket in buckets:
            path = args.input_root / f"part-{bucket:03d}.parquet"
            output = args.output_root / path.name
            result = process_part(path, output, pool)
            total += result["rows"]
            discarded += result["discarded"]
            statuses.update(result["status"])
    finally:
        pool.close()
    report = {
        "buckets": args.buckets,
        "parts": len(buckets),
        "rows": total,
        "discarded": discarded,
        "status": dict(statuses),
        "detector_workers": args.detector_workers,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    receipt = receipt_root / f"range-{buckets[0]:03d}-{buckets[-1]:03d}.json"
    receipt.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
