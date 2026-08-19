#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import statistics
from collections import Counter

import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = pathlib.Path("/home/gyr/lusychat-training")
ELITE = ROOT / "judge-dialogue-selection-v1/02_tiers/elite_A.parquet"
TURNS = (
    ROOT
    / "judge-dialogue-union-50pct-v1/01_input/turn_manifest.parquet"
)
MATERIAL = (
    ROOT
    / "judge-dialogue-union-50pct-v1/01_input/dialogue_material.parquet"
)
SMOKE_SCRIPT = (
    ROOT
    / "judge-deepinfra-glm47-turn-v46-promptonly-softjson-smoke-20260728"
    / "scripts/run_v46_promptonly_softjson_smoke.py"
)


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("v46_softjson", SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load v46 smoke module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    v46 = load_smoke_module()
    elite_ids_array = pq.read_table(ELITE, columns=["dialogue_id"])[
        "dialogue_id"
    ]
    elite_ids = {str(value) for value in elite_ids_array.to_pylist()}

    turn_table = pq.read_table(
        TURNS,
        columns=[
            "dialogue_id",
            "conversation_id",
            "assistant_ordinal",
            "native_message_id",
            "content_sha256",
            "replay_mode",
            "memory_id",
            "memory_boundary_increase_id",
        ],
    )
    eligible = turn_table.filter(
        pc.is_in(turn_table["dialogue_id"], value_set=elite_ids_array)
    )
    turn_rows = eligible.to_pylist()

    material_table = pq.read_table(
        MATERIAL,
        columns=[
            "dialogue_id",
            "conversation_id",
            "replay_mode",
            "dominant_language",
            "representative_topic",
            "messages",
        ],
        filters=[("dialogue_id", "in", list(elite_ids))],
    )
    material_by_dialogue = {
        str(row["dialogue_id"]): row for row in material_table.to_pylist()
    }

    parsed_by_dialogue = {}
    greeting_counts = Counter()
    language_counts = Counter()
    topic_counts = Counter()
    ordinal_counts = Counter()
    errors = Counter()
    valid_rows = []
    keys = set()
    request_sizes = []

    for dialogue_id, material in material_by_dialogue.items():
        try:
            parsed = v46.parse_replay(material["messages"] or [])
        except ValueError:
            errors["parse_error_dialogues"] += 1
            continue
        parsed_by_dialogue[dialogue_id] = parsed
        greeting_counts[
            "present" if v46.greeting_present(parsed) else "absent"
        ] += 1

    for row in turn_rows:
        dialogue_id = str(row["dialogue_id"])
        ordinal = int(row["assistant_ordinal"])
        key = (dialogue_id, ordinal)
        if key in keys:
            errors["duplicate_dialogue_ordinal"] += 1
            continue
        keys.add(key)
        material = material_by_dialogue.get(dialogue_id)
        parsed = parsed_by_dialogue.get(dialogue_id)
        if material is None:
            errors["missing_material_rows"] += 1
            continue
        if parsed is None:
            errors["unparsed_material_rows"] += 1
            continue
        if ordinal < 1 or ordinal > len(parsed["pairs"]):
            errors["ordinal_error_rows"] += 1
            continue
        target = parsed["pairs"][ordinal - 1][1]
        if hashlib.sha256(target.encode("utf-8")).hexdigest() != str(
            row["content_sha256"]
        ):
            errors["content_sha256_mismatch_rows"] += 1
            continue
        candidate = {
            **row,
            "dialogue_id": dialogue_id,
            "conversation_id": str(row["conversation_id"]),
            "native_message_id": str(row["native_message_id"]),
            "assistant_ordinal": ordinal,
            "language": str(material["dominant_language"]),
            "topic": str(material["representative_topic"]),
            "parsed": parsed,
            "has_greeting": v46.greeting_present(parsed),
            "turn_bucket": v46.turn_bucket(max(ordinal, 5)),
        }
        valid_rows.append(candidate)
        language_counts[candidate["language"]] += 1
        topic_counts[candidate["topic"]] += 1
        ordinal_counts[
            "01_04"
            if ordinal <= 4
            else "05_08"
            if ordinal <= 8
            else "09_16"
            if ordinal <= 16
            else "17_32"
            if ordinal <= 32
            else "33_plus"
        ] += 1

    valid_rows.sort(
        key=lambda row: (row["dialogue_id"], row["assistant_ordinal"])
    )
    sample_count = min(4096, len(valid_rows))
    if sample_count:
        indexes = sorted(
            {
                round(i * (len(valid_rows) - 1) / (sample_count - 1))
                for i in range(sample_count)
            }
        )
        for index in indexes:
            row = valid_rows[index]
            variant = (
                "greeting_present"
                if row["has_greeting"]
                else "greeting_absent"
            )
            request = {
                "custom_id": (
                    f"a_{hashlib.sha256(row['dialogue_id'].encode()).hexdigest()[:16]}"
                    f"_{row['assistant_ordinal']:04d}"
                ),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": v46.request_body(row, variant),
            }
            request_sizes.append(
                len(
                    json.dumps(
                        request,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                + 1
            )

    report = {
        "elite_dialogues": len(elite_ids),
        "elite_turn_rows": len(turn_rows),
        "validated_turn_rows": len(valid_rows),
        "unique_dialogue_ordinal": len(keys),
        "material_rows": material_table.num_rows,
        "parsed_dialogues": len(parsed_by_dialogue),
        "errors": dict(errors),
        "assistant_ordinal_buckets": dict(ordinal_counts),
        "language_top20": dict(language_counts.most_common(20)),
        "topic": dict(topic_counts),
        "greeting_dialogues": dict(greeting_counts),
        "request_size_sample": {
            "rows": len(request_sizes),
            "mean_bytes": statistics.mean(request_sizes),
            "p50_bytes": percentile(request_sizes, 0.50),
            "p90_bytes": percentile(request_sizes, 0.90),
            "p95_bytes": percentile(request_sizes, 0.95),
            "p99_bytes": percentile(request_sizes, 0.99),
            "max_bytes": max(request_sizes, default=0),
            "estimated_total_bytes": (
                round(statistics.mean(request_sizes) * len(valid_rows))
                if request_sizes
                else 0
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
