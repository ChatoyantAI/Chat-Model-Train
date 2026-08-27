#!/usr/bin/env python3
"""Derive compact, operator-facing text views from frozen v2.2 dialogues.

The source JSONL is never modified. Each source bucket is converted to one
atomic Parquet artifact plus one small report so interrupted runs can resume at
bucket granularity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "lusy-data-juicer-independent-v1"
EXPECTED_SOURCE_SCHEMA = "lusy-rp-dialogue-data-juicer-v2.2"
ROLE_SEPARATOR = "\n\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_messages(messages: list[dict[str, Any]]) -> str:
    compact = [[message.get("role"), message.get("content")] for message in messages]
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def render_dialogue_body(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown").upper()
        content = str(message.get("content") or "")
        rendered.append(f"{role}:\n{content}")
    return ROLE_SEPARATOR.join(rendered)


def exact_repeat_stats(values: list[str]) -> tuple[int, int]:
    counts = Counter(value for value in values if value)
    duplicate_instances = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    return duplicate_instances, duplicate_groups


def transform_record(record: dict[str, Any], expected_bucket: int) -> dict[str, Any]:
    if record.get("schema_version") != EXPECTED_SOURCE_SCHEMA:
        raise ValueError(f"unexpected schema_version={record.get('schema_version')!r}")
    if record.get("conversation_bucket") != expected_bucket:
        raise ValueError(
            f"bucket mismatch: expected={expected_bucket}, got={record.get('conversation_bucket')!r}"
        )

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages is empty or not a list")
    if messages[0].get("role") != "system":
        raise ValueError("first message is not system")

    memory = record.get("memory") or {}
    greeting_index = 2 if memory.get("mode") == "effective_memory_tail" else 1
    if len(messages) <= greeting_index or messages[greeting_index].get("role") != "assistant":
        raise ValueError(f"invalid greeting at index {greeting_index}")

    body_messages = messages[greeting_index + 1 :]
    assistant_replies = [
        str(message.get("content") or "")
        for message in body_messages
        if message.get("role") == "assistant"
    ]
    user_turns = [
        str(message.get("content") or "")
        for message in body_messages
        if message.get("role") == "user"
    ]
    turn_pairs = [
        user_turns[index] + "\u0000" + assistant_replies[index]
        for index in range(min(len(user_turns), len(assistant_replies)))
    ]
    greeting_text = str(messages[greeting_index].get("content") or "")
    dialogue_body_text = render_dialogue_body(body_messages)
    assistant_body_text = ROLE_SEPARATOR.join(assistant_replies)
    assistant_repeat_instances, assistant_repeat_groups = exact_repeat_stats(assistant_replies)
    user_repeat_instances, user_repeat_groups = exact_repeat_stats(user_turns)
    pair_repeat_instances, pair_repeat_groups = exact_repeat_stats(turn_pairs)

    return {
        "view_schema_version": SCHEMA_VERSION,
        "conversation_id": str(record.get("conversation_id") or ""),
        "dialogue_sample_id": str(record.get("dialogue_sample_id") or ""),
        "character_id": str(record.get("character_id") or ""),
        "conversation_bucket": expected_bucket,
        "turn_count": int(record.get("turn_count") or 0),
        "output_language": str(record.get("output_language") or "unknown"),
        "memory_mode": str(memory.get("mode") or "unknown"),
        "full_training_sha256": sha256_text(canonical_messages(messages)),
        "dialogue_body_text": dialogue_body_text,
        "assistant_body_text": assistant_body_text,
        "greeting_text": greeting_text,
        "assistant_preview": assistant_body_text[:1200],
        "body_chars": len(dialogue_body_text),
        "assistant_body_chars": len(assistant_body_text),
        "greeting_chars": len(greeting_text),
        "fuzzy_eligible": len(dialogue_body_text) >= 32,
        "greeting_equals_later_reply": greeting_text in set(assistant_replies),
        "assistant_exact_repeat_instances": assistant_repeat_instances,
        "assistant_exact_repeat_groups": assistant_repeat_groups,
        "user_exact_repeat_instances": user_repeat_instances,
        "user_exact_repeat_groups": user_repeat_groups,
        "turn_pair_exact_repeat_instances": pair_repeat_instances,
        "turn_pair_exact_repeat_groups": pair_repeat_groups,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    languages = Counter(row["output_language"] for row in rows)
    memory_modes = Counter(row["memory_mode"] for row in rows)
    return {
        "records": len(rows),
        "turns": sum(row["turn_count"] for row in rows),
        "body_chars": sum(row["body_chars"] for row in rows),
        "assistant_body_chars": sum(row["assistant_body_chars"] for row in rows),
        "greeting_chars": sum(row["greeting_chars"] for row in rows),
        "fuzzy_eligible": sum(bool(row["fuzzy_eligible"]) for row in rows),
        "greeting_equals_later_reply": sum(bool(row["greeting_equals_later_reply"]) for row in rows),
        "rows_with_assistant_exact_repeat": sum(row["assistant_exact_repeat_instances"] > 0 for row in rows),
        "rows_with_user_exact_repeat": sum(row["user_exact_repeat_instances"] > 0 for row in rows),
        "rows_with_turn_pair_exact_repeat": sum(row["turn_pair_exact_repeat_instances"] > 0 for row in rows),
        "assistant_exact_repeat_instances": sum(row["assistant_exact_repeat_instances"] for row in rows),
        "user_exact_repeat_instances": sum(row["user_exact_repeat_instances"] for row in rows),
        "turn_pair_exact_repeat_instances": sum(row["turn_pair_exact_repeat_instances"] for row in rows),
        "languages": dict(languages),
        "memory_modes": dict(memory_modes),
    }


def process_bucket(source_path: str, output_dir: str, report_dir: str) -> dict[str, Any]:
    source = Path(source_path)
    bucket = int(source.stem.rsplit("-", 1)[-1])
    output = Path(output_dir) / f"bucket-{bucket:03d}.parquet"
    report_path = Path(report_dir) / f"bucket-{bucket:03d}.json"

    if output.exists() and report_path.exists():
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            if pq.ParquetFile(output).metadata.num_rows == existing_report["counts"]["records"]:
                existing_report["resumed"] = True
                return existing_report
        except Exception:
            pass

    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(transform_record(json.loads(line), bucket))
            except Exception as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=[
            "view_schema_version",
            "output_language",
            "memory_mode",
        ],
        write_statistics=True,
        row_group_size=512,
    )
    os.replace(temporary, output)

    report = {
        "bucket": bucket,
        "source_name": source.name,
        "output_name": output.name,
        "output_size_bytes": output.stat().st_size,
        "completed_at": utc_now(),
        "resumed": False,
        "counts": summarize_rows(rows),
    }
    write_json_atomic(report_path, report)
    return report


def merge_counts(reports: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "records",
        "turns",
        "body_chars",
        "assistant_body_chars",
        "greeting_chars",
        "fuzzy_eligible",
        "greeting_equals_later_reply",
        "rows_with_assistant_exact_repeat",
        "rows_with_user_exact_repeat",
        "rows_with_turn_pair_exact_repeat",
        "assistant_exact_repeat_instances",
        "user_exact_repeat_instances",
        "turn_pair_exact_repeat_instances",
    ]
    merged: dict[str, Any] = {key: 0 for key in scalar_keys}
    languages: Counter[str] = Counter()
    memory_modes: Counter[str] = Counter()
    for report in reports:
        counts = report["counts"]
        for key in scalar_keys:
            merged[key] += counts[key]
        languages.update(counts.get("languages", {}))
        memory_modes.update(counts.get("memory_modes", {}))
    merged["languages"] = dict(languages.most_common())
    merged["memory_modes"] = dict(memory_modes.most_common())
    return merged


def update_run_progress(
    status_path: Path,
    completed: int,
    total: int,
    reports: list[dict[str, Any]],
    failed: list[dict[str, str]],
) -> None:
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status["updated_at"] = utc_now()
    status["state"] = "failed" if failed else "running"
    status["current_stage"] = "derive_views"
    status["progress"] = {
        "buckets_completed": completed,
        "buckets_total": total,
        "records_completed": sum(report["counts"]["records"] for report in reports),
        "turns_completed": sum(report["counts"]["turns"] for report in reports),
        "failed_buckets": len(failed),
    }
    if failed:
        status["last_error"] = failed[-1]
    stages = status.setdefault("stages", {})
    stage = stages.setdefault("derive_views", {"label": "派生独立文本视图"})
    stage.update(
        {
            "status": "failed" if failed else ("completed" if completed == total else "running"),
            "input_records": None,
            "output_records": status["progress"]["records_completed"],
            "buckets_completed": completed,
            "buckets_total": total,
        }
    )
    if completed == total and not failed:
        stage["finished_at"] = utc_now()
    write_json_atomic(status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-buckets", type=int, default=256)
    args = parser.parse_args()

    source_paths = sorted(args.source_dir.glob("bucket-*.jsonl"))
    if len(source_paths) != args.expected_buckets:
        raise SystemExit(
            f"expected {args.expected_buckets} source buckets, found {len(source_paths)} in {args.source_dir}"
        )

    output_dir = args.run_root / "01_derived_profile" / "parquet"
    report_dir = args.run_root / "01_derived_profile" / "reports"
    dashboard_dir = args.run_root / "07_dashboard"
    status_path = dashboard_dir / "run_status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    update_run_progress(status_path, 0, len(source_paths), reports, failed)

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(process_bucket, str(path), str(output_dir), str(report_dir)): path
            for path in source_paths
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                reports.append(future.result())
            except Exception as exc:
                failed.append(
                    {
                        "source": source.name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
            update_run_progress(status_path, len(reports), len(source_paths), reports, failed)

    reports.sort(key=lambda report: report["bucket"])
    if failed:
        write_json_atomic(args.run_root / "01_derived_profile" / "failures.json", {"failures": failed})
        raise SystemExit(f"{len(failed)} bucket(s) failed")

    profile = {
        "profile_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_schema_version": EXPECTED_SOURCE_SCHEMA,
        "source_bucket_count": len(source_paths),
        "source_is_untouched": True,
        "fork_metadata_used": False,
        "record_unit": "one independent complete dialogue branch",
        "view_contract": {
            "full_training_sha256": "SHA-256 of canonical [role, content] messages including system and greeting",
            "dialogue_body_text": "post-greeting user/assistant turns with role labels; system, memory and greeting excluded",
            "assistant_body_text": "post-greeting assistant replies only",
            "greeting_text": "the constructed assistant greeting only",
        },
        "counts": merge_counts(reports),
        "bucket_reports": [
            {
                "bucket": report["bucket"],
                "records": report["counts"]["records"],
                "turns": report["counts"]["turns"],
                "output_size_bytes": report["output_size_bytes"],
            }
            for report in reports
        ],
    }
    write_json_atomic(args.run_root / "01_derived_profile" / "PROFILE_MANIFEST.json", profile)
    update_run_progress(status_path, len(reports), len(source_paths), reports, failed)


if __name__ == "__main__":
    main()
