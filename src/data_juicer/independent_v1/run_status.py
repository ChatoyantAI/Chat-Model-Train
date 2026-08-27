#!/usr/bin/env python3
"""Maintain the small atomic status document consumed by the dashboard."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_LABELS = {
    "derive_views": "派生独立文本视图",
    "exact": "完整 messages 精确去重",
    "minhash": "多语言 MinHash 近似去重",
    "simhash": "字符 SimHash 模板去重",
    "repetition": "assistant 单对话复读率",
    "summarize": "生成可视化汇总",
}


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


def read_status(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def parse_optional_int(value: str | None) -> int | None:
    return None if value in (None, "", "null", "None") else int(value)


def command_init(args: argparse.Namespace) -> None:
    now = utc_now()
    stages = {
        key: {
            "label": label,
            "status": "pending",
            "input_records": None,
            "output_records": None,
        }
        for key, label in STAGE_LABELS.items()
    }
    payload = {
        "run_version": "independent_v1",
        "state": "running",
        "scope": "full_v2.2_candidate_pool",
        "record_unit": "one independent dialogue; fork metadata ignored",
        "source_is_untouched": True,
        "cpu_only": True,
        "started_at": now,
        "updated_at": now,
        "current_stage": "derive_views",
        "progress": {
            "buckets_completed": 0,
            "buckets_total": args.expected_buckets,
            "records_completed": 0,
            "turns_completed": 0,
            "failed_buckets": 0,
        },
        "stages": stages,
        "parameters": {
            "exact": "Data-Juicer document_deduplicator over canonical full-messages SHA-256 key",
            "minhash": {
                "operator": "document_minhash_deduplicator",
                "view": "dialogue_body_text",
                "tokenization": "sentencepiece (XLM-R multilingual)",
                "window_size": 5,
                "num_permutations": 256,
                "jaccard_threshold": 0.85,
            },
            "simhash": {
                "operator": "document_simhash_deduplicator",
                "view": "dialogue_body_text",
                "tokenization": "character",
                "window_size": 6,
                "num_blocks": 6,
                "hamming_distance": 4,
            },
            "repetition": {
                "operators": ["word_repetition_filter", "character_repetition_filter"],
                "view": "assistant_body_text",
                "rep_len": 10,
                "mode": "score_only_no_threshold_deletion",
            },
        },
    }
    payload["stages"]["derive_views"]["status"] = "running"
    payload["stages"]["derive_views"]["started_at"] = now
    write_json_atomic(args.status_path, payload)


def command_stage(args: argparse.Namespace) -> None:
    status = read_status(args.status_path)
    now = utc_now()
    stage = status.setdefault("stages", {}).setdefault(
        args.stage,
        {"label": STAGE_LABELS.get(args.stage, args.stage)},
    )
    stage["status"] = args.stage_status
    if args.stage_status == "running" and "started_at" not in stage:
        stage["started_at"] = now
    if args.stage_status in {"completed", "failed", "skipped"}:
        stage["finished_at"] = now
    if args.input_records is not None:
        stage["input_records"] = parse_optional_int(args.input_records)
    if args.output_records is not None:
        stage["output_records"] = parse_optional_int(args.output_records)
    if args.message:
        stage["message"] = args.message
    status["current_stage"] = args.stage
    status["state"] = "failed" if args.stage_status == "failed" else "running"
    status["updated_at"] = now
    write_json_atomic(args.status_path, status)


def command_fail(args: argparse.Namespace) -> None:
    status = read_status(args.status_path)
    now = utc_now()
    status["state"] = "failed"
    status["updated_at"] = now
    status["last_error"] = {"message": args.message, "at": now}
    current = status.get("current_stage")
    if current:
        stage = status.setdefault("stages", {}).setdefault(current, {"label": current})
        stage["status"] = "failed"
        stage["finished_at"] = now
    write_json_atomic(args.status_path, status)


def command_complete(args: argparse.Namespace) -> None:
    status = read_status(args.status_path)
    now = utc_now()
    status["state"] = "completed"
    status["current_stage"] = "complete"
    status["updated_at"] = now
    status["finished_at"] = now
    write_json_atomic(args.status_path, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-path", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--expected-buckets", type=int, default=256)
    init_parser.set_defaults(function=command_init)

    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("stage")
    stage_parser.add_argument("stage_status", choices=["pending", "running", "completed", "failed", "skipped"])
    stage_parser.add_argument("--input-records")
    stage_parser.add_argument("--output-records")
    stage_parser.add_argument("--message")
    stage_parser.set_defaults(function=command_stage)

    fail_parser = subparsers.add_parser("fail")
    fail_parser.add_argument("--message", required=True)
    fail_parser.set_defaults(function=command_fail)

    complete_parser = subparsers.add_parser("complete")
    complete_parser.set_defaults(function=command_complete)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
