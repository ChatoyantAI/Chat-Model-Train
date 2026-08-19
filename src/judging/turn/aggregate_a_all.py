#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from collections import Counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


TRAINING_ROOT = pathlib.Path(
    os.environ.get("LUSY_TRAINING_ROOT", "/home/gyr/lusychat-training")
)
FIRST_ROOT = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-v47-softjson-direct64-a50-dialogues-20260728"
)
SECOND_ROOT = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-v47-softjson-direct64-a50-second-dialogues-20260729"
)
OUTPUT_ROOT = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-v47-softjson-direct64-a-all-aggregated-20260729"
)
ARTIFACTS = OUTPUT_ROOT / "artifacts"

EXPECTED_FIRST = 151_503
EXPECTED_SECOND = 151_020
EXPECTED_TOTAL = 302_523
EXPECTED_DIALOGUES = 15_772
BATCH_SIZE = 5_000

DIMENSIONS = [
    "character_fidelity",
    "context_coherence",
    "current_turn_fit",
    "relationship_continuity",
    "scene_progression",
    "language_roleplay_quality",
]
CRITICAL_FLAGS = [
    "ai_identity_break",
    "character_identity_break",
    "context_break",
    "relationship_reset",
    "wrong_language",
    "format_collapse",
    "meta_refusal",
]
SCORE_VALUES = range(5)


def atomic_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def result_schema() -> pa.Schema:
    fields = [
        pa.field("source_half", pa.string(), nullable=False),
        pa.field("custom_id", pa.string(), nullable=False),
        pa.field("dialogue_id", pa.string(), nullable=False),
        pa.field("conversation_id", pa.string(), nullable=False),
        pa.field("assistant_ordinal", pa.int32(), nullable=False),
        pa.field("native_message_id", pa.string(), nullable=False),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("language", pa.string()),
        pa.field("topic", pa.string()),
        pa.field("has_greeting", pa.bool_()),
        pa.field("turn_bucket", pa.string()),
        pa.field("success", pa.bool_(), nullable=False),
        pa.field("strict_valid", pa.bool_(), nullable=False),
        pa.field("validation_errors", pa.list_(pa.string())),
        pa.field("status_code", pa.int16()),
        pa.field("attempt_count", pa.int16(), nullable=False),
        pa.field("had_429", pa.bool_(), nullable=False),
        pa.field("input_tokens", pa.int64(), nullable=False),
        pa.field("cached_tokens", pa.int64(), nullable=False),
        pa.field("output_tokens", pa.int64(), nullable=False),
        pa.field("estimated_cost", pa.float64(), nullable=False),
        pa.field("reused_from", pa.string()),
        pa.field("response_id", pa.string()),
        pa.field("finish_reason", pa.string()),
        pa.field("error", pa.string()),
        pa.field("raw_result_json", pa.large_string()),
    ]
    for dimension in DIMENSIONS:
        fields.extend(
            [
                pa.field(f"{dimension}_score", pa.int8()),
                pa.field(f"{dimension}_reason", pa.large_string()),
            ]
        )
    fields.extend(
        [
            pa.field("intimacy_applicable", pa.bool_()),
            pa.field("intimacy_score", pa.int8()),
            pa.field("intimacy_reason", pa.large_string()),
            pa.field("semantic_repetition_severity", pa.string()),
            pa.field("semantic_repetition_reason", pa.large_string()),
        ]
    )
    for flag in CRITICAL_FLAGS:
        fields.append(pa.field(f"critical_{flag}", pa.bool_()))
    fields.extend(
        [
            pa.field("critical_reason", pa.large_string()),
            pa.field("nsfw_level", pa.int8()),
            pa.field("nsfw_reason", pa.large_string()),
            pa.field("confidence", pa.int8()),
            pa.field("overall_reason", pa.large_string()),
        ]
    )
    return pa.schema(fields)


def flatten(source_half: str, row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    raw = row.get("raw_result")
    raw_object = raw if isinstance(raw, dict) else {}
    dimensions = raw_object.get("dimensions")
    dimensions = dimensions if isinstance(dimensions, dict) else {}
    attempts = row.get("attempts")
    attempts = attempts if isinstance(attempts, list) else []
    usage = row.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    value: dict[str, Any] = {
        "source_half": source_half,
        "custom_id": str(row.get("custom_id") or ""),
        "dialogue_id": str(metadata.get("dialogue_id") or ""),
        "conversation_id": str(metadata.get("conversation_id") or ""),
        "assistant_ordinal": int(metadata.get("assistant_ordinal") or 0),
        "native_message_id": str(metadata.get("native_message_id") or ""),
        "content_sha256": str(metadata.get("content_sha256") or ""),
        "language": text_or_none(metadata.get("language")),
        "topic": text_or_none(metadata.get("topic")),
        "has_greeting": (
            metadata.get("has_greeting")
            if isinstance(metadata.get("has_greeting"), bool)
            else None
        ),
        "turn_bucket": text_or_none(metadata.get("turn_bucket")),
        "success": bool(row.get("success")),
        "strict_valid": bool(row.get("strict_valid")),
        "validation_errors": [
            str(error) for error in (row.get("validation_errors") or [])
        ],
        "status_code": as_int(row.get("status_code")),
        "attempt_count": len(attempts),
        "had_429": any(
            attempt.get("status_code") == 429
            for attempt in attempts
            if isinstance(attempt, dict)
        ),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_tokens": int(usage.get("cached_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "estimated_cost": as_float(usage.get("estimated_cost")),
        "reused_from": text_or_none(row.get("reused_from")),
        "response_id": text_or_none(row.get("response_id")),
        "finish_reason": text_or_none(row.get("finish_reason")),
        "error": text_or_none(row.get("error")),
        "raw_result_json": (
            json.dumps(
                raw,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if raw is not None
            else None
        ),
    }
    for dimension in DIMENSIONS:
        item = dimensions.get(dimension)
        item = item if isinstance(item, dict) else {}
        value[f"{dimension}_score"] = as_int(item.get("score"))
        value[f"{dimension}_reason"] = text_or_none(item.get("reason"))
    intimacy = raw_object.get("intimacy_quality")
    intimacy = intimacy if isinstance(intimacy, dict) else {}
    repetition = raw_object.get("semantic_repetition")
    repetition = repetition if isinstance(repetition, dict) else {}
    flags = raw_object.get("critical_flags")
    flags = flags if isinstance(flags, dict) else {}
    value.update(
        {
            "intimacy_applicable": (
                intimacy.get("applicable")
                if isinstance(intimacy.get("applicable"), bool)
                else None
            ),
            "intimacy_score": as_int(intimacy.get("score")),
            "intimacy_reason": text_or_none(intimacy.get("reason")),
            "semantic_repetition_severity": text_or_none(
                repetition.get("severity")
            ),
            "semantic_repetition_reason": text_or_none(
                repetition.get("reason")
            ),
            "critical_reason": text_or_none(
                raw_object.get("critical_reason")
            ),
            "nsfw_level": as_int(raw_object.get("nsfw_level")),
            "nsfw_reason": text_or_none(raw_object.get("nsfw_reason")),
            "confidence": as_int(raw_object.get("confidence")),
            "overall_reason": text_or_none(raw_object.get("overall_reason")),
        }
    )
    for flag in CRITICAL_FLAGS:
        flag_value = flags.get(flag)
        value[f"critical_{flag}"] = (
            flag_value if isinstance(flag_value, bool) else None
        )
    return value


def new_dialogue_aggregate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dialogue_id": row["dialogue_id"],
        "conversation_id": row["conversation_id"],
        "language": row["language"],
        "topic": row["topic"],
        "total_turns": 0,
        "success_turns": 0,
        "strict_valid_turns": 0,
        "failed_turns": 0,
        "invalid_turns": 0,
        "reused_turns": 0,
        "dimension_scores": {
            dimension: Counter() for dimension in DIMENSIONS
        },
        "intimacy_applicable_turns": 0,
        "intimacy_scores": Counter(),
        "repetition": Counter(),
        "critical": Counter(),
        "nsfw": Counter(),
        "confidence": Counter(),
    }


def update_dialogue(aggregate: dict[str, Any], row: dict[str, Any]) -> None:
    aggregate["total_turns"] += 1
    aggregate["success_turns"] += int(row["success"])
    aggregate["strict_valid_turns"] += int(row["strict_valid"])
    aggregate["failed_turns"] += int(not row["success"])
    aggregate["invalid_turns"] += int(not row["strict_valid"])
    aggregate["reused_turns"] += int(row["reused_from"] is not None)
    if not row["strict_valid"]:
        return
    for dimension in DIMENSIONS:
        score = row[f"{dimension}_score"]
        if score is not None:
            aggregate["dimension_scores"][dimension][int(score)] += 1
    if row["intimacy_applicable"]:
        aggregate["intimacy_applicable_turns"] += 1
        if row["intimacy_score"] is not None:
            aggregate["intimacy_scores"][int(row["intimacy_score"])] += 1
    severity = row["semantic_repetition_severity"]
    if severity is not None:
        aggregate["repetition"][severity] += 1
    for flag in CRITICAL_FLAGS:
        if row[f"critical_{flag}"] is True:
            aggregate["critical"][flag] += 1
    if row["nsfw_level"] is not None:
        aggregate["nsfw"][int(row["nsfw_level"])] += 1
    if row["confidence"] is not None:
        aggregate["confidence"][int(row["confidence"])] += 1


def counter_mean(counter: Counter) -> float | None:
    total = sum(counter.values())
    if not total:
        return None
    return sum(float(value) * count for value, count in counter.items()) / total


def dialogue_row(aggregate: dict[str, Any]) -> dict[str, Any]:
    valid = aggregate["strict_valid_turns"]
    row: dict[str, Any] = {
        "dialogue_id": aggregate["dialogue_id"],
        "conversation_id": aggregate["conversation_id"],
        "language": aggregate["language"],
        "topic": aggregate["topic"],
        "total_turns": aggregate["total_turns"],
        "success_turns": aggregate["success_turns"],
        "strict_valid_turns": valid,
        "failed_turns": aggregate["failed_turns"],
        "invalid_turns": aggregate["invalid_turns"],
        "reused_turns": aggregate["reused_turns"],
        "strict_valid_fraction": (
            valid / aggregate["total_turns"]
            if aggregate["total_turns"]
            else 0.0
        ),
    }
    for dimension in DIMENSIONS:
        scores = aggregate["dimension_scores"][dimension]
        row[f"{dimension}_mean"] = counter_mean(scores)
        row[f"{dimension}_min"] = min(scores) if scores else None
        row[f"{dimension}_max"] = max(scores) if scores else None
        for score in SCORE_VALUES:
            row[f"{dimension}_score_{score}_count"] = scores[score]
    row["intimacy_applicable_turns"] = aggregate[
        "intimacy_applicable_turns"
    ]
    row["intimacy_mean"] = counter_mean(aggregate["intimacy_scores"])
    for score in [-1, 0, 1, 2, 3, 4]:
        row[f"intimacy_score_{score}_count"] = aggregate[
            "intimacy_scores"
        ][score]
    for severity in ["none", "minor", "major", "critical"]:
        row[f"semantic_repetition_{severity}_count"] = aggregate[
            "repetition"
        ][severity]
    for flag in CRITICAL_FLAGS:
        row[f"critical_{flag}_count"] = aggregate["critical"][flag]
    row["nsfw_mean"] = counter_mean(aggregate["nsfw"])
    for level in range(5):
        row[f"nsfw_level_{level}_count"] = aggregate["nsfw"][level]
    row["confidence_mean"] = counter_mean(aggregate["confidence"])
    for score in range(5):
        row[f"confidence_{score}_count"] = aggregate["confidence"][score]
    return row


def main() -> None:
    started_at = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    result_path = ARTIFACTS / "all_turn_results.parquet"
    result_part = result_path.with_suffix(".parquet.part")
    dialogue_path = ARTIFACTS / "dialogue_summary.parquet"
    dialogue_part = dialogue_path.with_suffix(".parquet.part")
    retry_path = ARTIFACTS / "retry_needed.jsonl"
    retry_part = retry_path.with_suffix(".jsonl.part")
    schema = result_schema()
    writer = pq.ParquetWriter(
        result_part,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    retry_output = retry_part.open("w", encoding="utf-8", newline="\n")
    batch: list[dict[str, Any]] = []
    seen_custom_ids: set[str] = set()
    seen_turn_keys: set[tuple[str, int, str]] = set()
    dialogue_ids: set[str] = set()
    dialogues: dict[str, dict[str, Any]] = {}
    counts = Counter()
    usage = Counter()
    validation_errors = Counter()
    language = Counter()
    topic = Counter()
    turn_bucket = Counter()
    greeting = Counter()
    score_distributions = {
        dimension: Counter() for dimension in DIMENSIONS
    }
    intimacy_scores = Counter()
    repetition_severity = Counter()
    critical_true = Counter()
    nsfw_levels = Counter()
    confidence_scores = Counter()

    sources = [
        (
            "first",
            FIRST_ROOT / "artifacts/direct_results.jsonl",
            EXPECTED_FIRST,
        ),
        (
            "second",
            SECOND_ROOT / "artifacts/direct_results.jsonl",
            EXPECTED_SECOND,
        ),
    ]
    try:
        for source_half, source_path, expected in sources:
            source_count = 0
            with source_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    source_row = json.loads(line)
                    row = flatten(source_half, source_row)
                    custom_id = row["custom_id"]
                    turn_key = (
                        row["dialogue_id"],
                        row["assistant_ordinal"],
                        row["content_sha256"],
                    )
                    if custom_id in seen_custom_ids:
                        raise RuntimeError(
                            f"duplicate custom_id {custom_id}"
                        )
                    if turn_key in seen_turn_keys:
                        raise RuntimeError(
                            f"duplicate turn key {turn_key}"
                        )
                    seen_custom_ids.add(custom_id)
                    seen_turn_keys.add(turn_key)
                    dialogue_ids.add(row["dialogue_id"])
                    source_count += 1
                    counts["total"] += 1
                    counts[f"source_{source_half}"] += 1
                    counts["success"] += int(row["success"])
                    counts["failed"] += int(not row["success"])
                    counts["strict_valid"] += int(row["strict_valid"])
                    counts["strict_invalid"] += int(not row["strict_valid"])
                    counts["reused"] += int(row["reused_from"] is not None)
                    counts["had_429"] += int(row["had_429"])
                    validation_errors.update(row["validation_errors"])
                    language[row["language"] or "unknown"] += 1
                    topic[row["topic"] or "unknown"] += 1
                    turn_bucket[row["turn_bucket"] or "unknown"] += 1
                    greeting[
                        "present"
                        if row["has_greeting"]
                        else "absent"
                    ] += 1
                    for key in [
                        "input_tokens",
                        "cached_tokens",
                        "output_tokens",
                        "estimated_cost",
                    ]:
                        usage[key] += row[key]
                    if row["strict_valid"]:
                        for dimension in DIMENSIONS:
                            score = row[f"{dimension}_score"]
                            if score is not None:
                                score_distributions[dimension][
                                    str(score)
                                ] += 1
                        if row["intimacy_score"] is not None:
                            intimacy_scores[str(row["intimacy_score"])] += 1
                        if row["semantic_repetition_severity"] is not None:
                            repetition_severity[
                                row["semantic_repetition_severity"]
                            ] += 1
                        for flag in CRITICAL_FLAGS:
                            if row[f"critical_{flag}"] is True:
                                critical_true[flag] += 1
                        if row["nsfw_level"] is not None:
                            nsfw_levels[str(row["nsfw_level"])] += 1
                        if row["confidence"] is not None:
                            confidence_scores[str(row["confidence"])] += 1
                    aggregate = dialogues.get(row["dialogue_id"])
                    if aggregate is None:
                        aggregate = new_dialogue_aggregate(row)
                        dialogues[row["dialogue_id"]] = aggregate
                    update_dialogue(aggregate, row)
                    if not row["strict_valid"]:
                        retry_output.write(
                            json.dumps(
                                {
                                    "source_half": source_half,
                                    "custom_id": custom_id,
                                    "metadata": source_row.get("metadata")
                                    or {},
                                    "success": row["success"],
                                    "status_code": row["status_code"],
                                    "validation_errors": row[
                                        "validation_errors"
                                    ],
                                },
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    batch.append(row)
                    if len(batch) >= BATCH_SIZE:
                        writer.write_table(
                            pa.Table.from_pylist(batch, schema=schema),
                            row_group_size=BATCH_SIZE,
                        )
                        batch.clear()
            if source_count != expected:
                raise RuntimeError(
                    f"{source_half} source has {source_count}, "
                    f"expected {expected}"
                )
        if batch:
            writer.write_table(
                pa.Table.from_pylist(batch, schema=schema),
                row_group_size=BATCH_SIZE,
            )
            batch.clear()
    finally:
        writer.close()
        retry_output.close()

    if counts["total"] != EXPECTED_TOTAL:
        raise RuntimeError(
            f"total {counts['total']}, expected {EXPECTED_TOTAL}"
        )
    if len(dialogue_ids) != EXPECTED_DIALOGUES:
        raise RuntimeError(
            f"dialogues {len(dialogue_ids)}, expected {EXPECTED_DIALOGUES}"
        )
    result_part.replace(result_path)
    retry_part.replace(retry_path)

    dialogue_rows = [
        dialogue_row(dialogues[dialogue_id])
        for dialogue_id in sorted(dialogues)
    ]
    dialogue_table = pa.Table.from_pylist(dialogue_rows)
    pq.write_table(
        dialogue_table,
        dialogue_part,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=2_000,
    )
    dialogue_part.replace(dialogue_path)

    artifacts = {}
    for path in [result_path, dialogue_path, retry_path]:
        artifacts[path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    summary = {
        "schema_version": "lusy-turn-judge-v47-a-all-aggregate-v1",
        "created_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "sources": {
            "first": str(
                FIRST_ROOT / "artifacts/direct_results.jsonl"
            ),
            "second": str(
                SECOND_ROOT / "artifacts/direct_results.jsonl"
            ),
        },
        "expected": {
            "turns": EXPECTED_TOTAL,
            "dialogues": EXPECTED_DIALOGUES,
        },
        "integrity": {
            "rows": counts["total"],
            "unique_custom_ids": len(seen_custom_ids),
            "unique_turn_keys": len(seen_turn_keys),
            "unique_dialogues": len(dialogue_ids),
            "duplicates": 0,
            "missing": 0,
        },
        "results": {
            "success": counts["success"],
            "failed": counts["failed"],
            "strict_valid": counts["strict_valid"],
            "strict_invalid": counts["strict_invalid"],
            "strict_valid_rate": (
                counts["strict_valid"] / counts["total"]
            ),
            "reused": counts["reused"],
            "had_429": counts["had_429"],
        },
        "usage": {
            "input_tokens": int(usage["input_tokens"]),
            "cached_tokens": int(usage["cached_tokens"]),
            "output_tokens": int(usage["output_tokens"]),
            "estimated_cost": float(usage["estimated_cost"]),
            "cache_fraction": (
                usage["cached_tokens"] / usage["input_tokens"]
                if usage["input_tokens"]
                else 0
            ),
        },
        "validation_error_counts": dict(validation_errors),
        "coverage": {
            "language": dict(language),
            "topic": dict(topic),
            "turn_bucket": dict(turn_bucket),
            "greeting": dict(greeting),
        },
        "score_distributions": {
            dimension: dict(counter)
            for dimension, counter in score_distributions.items()
        },
        "intimacy_score_distribution": dict(intimacy_scores),
        "semantic_repetition_distribution": dict(repetition_severity),
        "critical_flag_true_counts": dict(critical_true),
        "nsfw_level_distribution": dict(nsfw_levels),
        "confidence_distribution": dict(confidence_scores),
        "artifacts": artifacts,
    }
    atomic_json(ARTIFACTS / "SUMMARY.json", summary)
    atomic_json(
        ARTIFACTS / "SOURCE_MANIFEST.json",
        {
            "schema_version": summary["schema_version"],
            "sources": summary["sources"],
            "source_rows": {
                "first": EXPECTED_FIRST,
                "second": EXPECTED_SECOND,
            },
            "partition": (
                "15,772 Elite A dialogues sorted by the frozen stable "
                "dialogue hash, split into two non-overlapping halves"
            ),
            "judge_contract": (
                "v47 prompt-only soft JSON, GLM-4.7-Flash, "
                "response_format=json_object, max_tokens=1536"
            ),
        },
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
