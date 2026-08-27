#!/usr/bin/env python3
"""Create one row per accepted user/assistant turn from frozen v2.2 JSONL.

Only dialogue_sample_ids retained by the exact-dedup branch are included. The
frozen source and the independent-v1 artifacts are read-only inputs. Output is
one atomic Parquet shard per source bucket and can resume at bucket granularity.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from common import SCHEMA_VERSION, read_json, update_status, utc_now, write_json_atomic


EXPECTED_SOURCE_SCHEMA = "lusy-rp-dialogue-data-juicer-v2.2"

TURN_SCHEMA = pa.schema(
    [
        ("turn_schema_version", pa.string()),
        ("turn_id", pa.string()),
        ("conversation_id", pa.string()),
        ("dialogue_sample_id", pa.string()),
        ("character_id", pa.string()),
        ("conversation_bucket", pa.int16()),
        ("turn_index", pa.int32()),
        ("turn_count", pa.int32()),
        ("output_language", pa.string()),
        ("event_language", pa.string()),
        ("evaluation", pa.int8()),
        ("user_event_id", pa.int64()),
        ("assistant_event_id", pa.int64()),
        ("model_id", pa.int32()),
        ("teacher_family", pa.string()),
        ("user_text", pa.string()),
        ("assistant_text", pa.string()),
        ("user_chars", pa.int32()),
        ("assistant_chars", pa.int32()),
    ]
)


def int_or(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def retained_ids_by_bucket(path: Path) -> tuple[dict[int, set[str]], int, int]:
    table = pq.read_table(path, columns=["dialogue_sample_id", "conversation_bucket", "turn_count"])
    grouped: dict[int, set[str]] = defaultdict(set)
    expected_turns = 0
    sample_ids = table["dialogue_sample_id"].to_pylist()
    buckets = table["conversation_bucket"].to_pylist()
    turns = table["turn_count"].to_pylist()
    for sample_id, bucket, turn_count in zip(sample_ids, buckets, turns, strict=True):
        grouped[int(bucket)].add(str(sample_id))
        expected_turns += int(turn_count)
    expected_dialogues = table.num_rows
    unique_dialogues = sum(len(values) for values in grouped.values())
    if unique_dialogues != expected_dialogues:
        raise ValueError(
            f"dialogue_sample_id is not unique: rows={expected_dialogues}, unique={unique_dialogues}"
        )
    return dict(grouped), expected_dialogues, expected_turns


def extract_turn_rows(record: dict[str, Any], bucket: int) -> list[dict[str, Any]]:
    if record.get("schema_version") != EXPECTED_SOURCE_SCHEMA:
        raise ValueError(f"unexpected source schema: {record.get('schema_version')!r}")
    if int_or(record.get("conversation_bucket"), -1) != bucket:
        raise ValueError("conversation bucket mismatch")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[0].get("role") != "system":
        raise ValueError("invalid messages")

    memory = record.get("memory") or {}
    greeting_index = 2 if memory.get("mode") == "effective_memory_tail" else 1
    if len(messages) <= greeting_index or messages[greeting_index].get("role") != "assistant":
        raise ValueError("invalid greeting position")

    body = messages[greeting_index + 1 :]
    if len(body) % 2 != 0:
        raise ValueError(f"body does not contain complete pairs: messages={len(body)}")

    events = record.get("assistant_events") or []
    expected_turns = int_or(record.get("turn_count"))
    if len(body) // 2 != expected_turns or len(events) != expected_turns:
        raise ValueError(
            f"turn count mismatch: declared={expected_turns}, pairs={len(body) // 2}, events={len(events)}"
        )

    sample_id = str(record.get("dialogue_sample_id") or "")
    conversation_id = str(record.get("conversation_id") or "")
    character_id = str(record.get("character_id") or "")
    output_language = str(record.get("output_language") or "unknown")
    rows: list[dict[str, Any]] = []

    for turn_index in range(expected_turns):
        user_message = body[turn_index * 2]
        assistant_message = body[turn_index * 2 + 1]
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            raise ValueError(f"invalid role ordering at turn {turn_index}")
        event = events[turn_index] or {}
        user_text = str(user_message.get("content") or "")
        assistant_text = str(assistant_message.get("content") or "")
        if not user_text or not assistant_text:
            raise ValueError(f"empty text at turn {turn_index}")
        rows.append(
            {
                "turn_schema_version": SCHEMA_VERSION,
                "turn_id": f"{sample_id}:{turn_index:05d}",
                "conversation_id": conversation_id,
                "dialogue_sample_id": sample_id,
                "character_id": character_id,
                "conversation_bucket": bucket,
                "turn_index": turn_index,
                "turn_count": expected_turns,
                "output_language": output_language,
                "event_language": str(event.get("language") or "unknown"),
                "evaluation": int_or(event.get("evaluation")),
                "user_event_id": int_or(event.get("user_event_id")),
                "assistant_event_id": int_or(event.get("assistant_event_id")),
                "model_id": int_or(event.get("model_id")),
                "teacher_family": str(event.get("teacher_family") or "unknown"),
                "user_text": user_text,
                "assistant_text": assistant_text,
                "user_chars": len(user_text),
                "assistant_chars": len(assistant_text),
            }
        )
    return rows


def process_bucket(
    bucket: int,
    retained_ids: set[str],
    source_dir: str,
    output_dir: str,
    force: bool,
) -> dict[str, Any]:
    source = Path(source_dir) / f"bucket-{bucket:03d}.jsonl"
    destination = Path(output_dir) / f"bucket-{bucket:03d}.parquet"
    report_path = Path(output_dir) / "reports" / f"bucket-{bucket:03d}.json"
    existing = read_json(report_path, {}) or {}
    if (
        not force
        and destination.exists()
        and existing.get("status") == "completed"
        and existing.get("retained_dialogues") == len(retained_ids)
    ):
        return existing

    rows: list[dict[str, Any]] = []
    source_dialogues = 0
    matched_dialogues = 0
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                source_dialogues += 1
                record = json.loads(line)
                sample_id = str(record.get("dialogue_sample_id") or "")
                if sample_id not in retained_ids:
                    continue
                matched_dialogues += 1
                try:
                    rows.extend(extract_turn_rows(record, bucket))
                except Exception as error:
                    raise ValueError(f"{source.name}:{line_number}: {error}") from error

        if matched_dialogues != len(retained_ids):
            raise ValueError(
                f"retained/source join mismatch: expected={len(retained_ids)}, matched={matched_dialogues}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        table = pa.Table.from_pylist(rows, schema=TURN_SCHEMA)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=3,
            use_dictionary=[
                "output_language",
                "event_language",
                "teacher_family",
                "character_id",
            ],
            row_group_size=32_768,
        )
        os.replace(temporary, destination)
        report = {
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "bucket": bucket,
            "source": str(source),
            "output": str(destination),
            "source_dialogues": source_dialogues,
            "retained_dialogues": matched_dialogues,
            "turns": len(rows),
            "user_chars": sum(row["user_chars"] for row in rows),
            "assistant_chars": sum(row["assistant_chars"] for row in rows),
            "completed_at": utc_now(),
        }
        write_json_atomic(report_path, report)
        return report
    except Exception:
        report = {
            "status": "failed",
            "schema_version": SCHEMA_VERSION,
            "bucket": bucket,
            "retained_dialogues": len(retained_ids),
            "error": traceback.format_exc(),
            "failed_at": utc_now(),
        }
        write_json_atomic(report_path, report)
        raise


def parse_buckets(value: str | None) -> list[int]:
    if not value:
        return list(range(256))
    selected: set[int] = set()
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected or min(selected) < 0 or max(selected) > 255:
        raise ValueError("buckets must be between 0 and 255")
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--retained-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--buckets", help="e.g. 0 or 0-15,20; default is all 256")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    grouped, expected_dialogues, expected_turns = retained_ids_by_bucket(args.retained_parquet)
    selected = parse_buckets(args.buckets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current = read_json(args.status_path, {}) or {}
    current.update(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "stage": "derive_turns",
            "started_at": current.get("started_at") or utc_now(),
            "source_dir": str(args.source_dir),
            "retained_parquet": str(args.retained_parquet),
            "expected_dialogues": expected_dialogues,
            "expected_turns": expected_turns,
        }
    )
    update_status(args.status_path, **current)

    reports: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_bucket,
                bucket,
                grouped.get(bucket, set()),
                str(args.source_dir),
                str(args.output_dir),
                args.force,
            ): bucket
            for bucket in selected
        }
        for future in as_completed(futures):
            bucket = futures[future]
            reports[bucket] = future.result()
            completed = [report for report in reports.values() if report.get("status") == "completed"]
            update_status(
                args.status_path,
                status="running",
                stage="derive_turns",
                derive_completed_shards=len(completed),
                derive_selected_shards=len(selected),
                derive_dialogues=sum(int(report.get("retained_dialogues", 0)) for report in completed),
                derive_turns=sum(int(report.get("turns", 0)) for report in completed),
            )

    if len(selected) == 256:
        all_reports = [
            read_json(args.output_dir / "reports" / f"bucket-{bucket:03d}.json", {}) or {}
            for bucket in range(256)
        ]
        total_dialogues = sum(int(report.get("retained_dialogues", 0)) for report in all_reports)
        total_turns = sum(int(report.get("turns", 0)) for report in all_reports)
        if total_dialogues != expected_dialogues or total_turns != expected_turns:
            raise ValueError(
                f"global turn-view mismatch: dialogues={total_dialogues}/{expected_dialogues}, "
                f"turns={total_turns}/{expected_turns}"
            )
        update_status(
            args.status_path,
            status="running",
            stage="turn_view_complete",
            derive_completed_shards=256,
            derive_selected_shards=256,
            derive_dialogues=total_dialogues,
            derive_turns=total_turns,
            derive_completed_at=utc_now(),
        )


if __name__ == "__main__":
    main()

