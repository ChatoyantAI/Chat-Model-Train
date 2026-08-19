#!/usr/bin/env python3
"""Build target-time online replay plans without materializing repeated prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path("/home/gyr/lusychat-training/pre-mds-context-v2")
MAIN_TARGET_ROOT = Path(
    "/home/gyr/lusychat-training/lusy-chat-4m-v2/02_features/target_features"
)
BUNDLE_ROOT = Path("/home/gyr/lusychat-training/lusy-chat-4m-v1/bundles")
LIKED_ROOT = ROOT / "01_target_index/liked_technical_shards"
MODEL_INFO = Path(
    "/home/gyr/lusychat-training/lusy-chat-4m-v2/01_canonical/model_info.parquet"
)
DEFAULT_OUTPUT = ROOT / "02_context_plan/parts"
SEED = "lusy-context-cap-v2-20260724"
THINK_HEAD_RE = re.compile(
    r"^\s*<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL
)


PLAN_SCHEMA = pa.schema(
    [
        ("target_id", pa.string()),
        ("target_message_id", pa.string()),
        ("target_increase_id", pa.int64()),
        ("conversation_id", pa.string()),
        ("user_id", pa.string()),
        ("character_id", pa.string()),
        ("conversation_bucket", pa.int16()),
        ("source_kind", pa.string()),
        ("pool", pa.string()),
        ("model_tier", pa.string()),
        ("core_protected", pa.bool_()),
        ("target_position_lifetime", pa.int32()),
        ("conversation_length_lifetime", pa.int32()),
        ("target_time", pa.timestamp("us", tz="UTC")),
        ("target_user_content", pa.large_string()),
        ("target_content", pa.large_string()),
        ("target_user_language_raw", pa.string()),
        ("language_raw", pa.string()),
        ("language_detected", pa.string()),
        ("language_final", pa.string()),
        ("topic", pa.string()),
        ("teacher_family", pa.string()),
        ("model", pa.string()),
        ("model_id", pa.int32()),
        ("model_context_window_k", pa.int32()),
        ("evaluation", pa.int8()),
        ("continued", pa.bool_()),
        ("guideline_mode", pa.string()),
        ("guideline_source", pa.string()),
        ("guideline_attached", pa.bool_()),
        ("guideline", pa.string()),
        ("lexical_units", pa.int32()),
        ("paid_at_target", pa.string()),
        ("quality_score", pa.float32()),
        ("format_valid", pa.bool_()),
        ("format_repaired", pa.bool_()),
        ("cot_removed", pa.bool_()),
        ("identity_break_confidence", pa.float32()),
        ("memory_effective", pa.bool_()),
        ("memory_record_state", pa.string()),
        ("replay_mode", pa.string()),
        ("memory_id", pa.int32()),
        ("memory_created_at", pa.timestamp("us", tz="UTC")),
        ("memory_boundary_increase_id", pa.int64()),
        ("visible_start_turn", pa.int32()),
        ("visible_end_turn", pa.int32()),
        ("visible_user_turns", pa.int32()),
        ("visible_history_assistant_turns", pa.int32()),
        ("memory_covered_turns", pa.int32()),
        ("memory_summary_chars", pa.int32()),
        ("memory_state_chars", pa.int32()),
        ("target_user_chars", pa.int32()),
        ("history_user_chars", pa.int64()),
        ("history_assistant_chars", pa.int64()),
        ("target_chars", pa.int32()),
        ("greeting_chars", pa.int32()),
        ("character_source_chars", pa.int32()),
        ("prompt_replay_chars_without_system", pa.int64()),
        ("selection_hash_u64", pa.uint64()),
        ("replay_invariant_pass", pa.bool_()),
        ("replay_error", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def stable_hash(target_id: str) -> int:
    digest = hashlib.sha256(f"{SEED}\0{target_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def strip_cot(value: Any) -> str:
    text = str(value or "")
    return THINK_HEAD_RE.sub("", text, count=1).strip()


def state_payload(value: Any) -> tuple[bool, int]:
    if value is None:
        return False, 0
    if isinstance(value, (dict, list)):
        return bool(value), len(json.dumps(value, ensure_ascii=False))
    text = str(value).strip()
    if not text:
        return False, 0
    try:
        parsed = json.loads(text)
    except Exception:
        return True, len(text)
    return bool(parsed), len(text)


def select_memory_as_of(
    memories: list[dict[str, Any]],
    target_increase_id: int,
    target_time: datetime | None,
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for memory in memories or []:
        boundary = memory.get("compressed_until_increase_id")
        created = memory.get("created_at")
        if boundary is None or int(boundary) >= target_increase_id:
            continue
        if target_time is not None and created is not None and created > target_time:
            continue
        eligible.append(memory)
    if not eligible:
        return None
    # Production reads latest by descending native id.
    return max(
        eligible,
        key=lambda item: (
            int(item.get("id") or -1),
            item.get("created_at")
            or datetime.min.replace(
                tzinfo=target_time.tzinfo if target_time else None
            ),
        ),
    )


def model_maps() -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = pq.read_table(MODEL_INFO).to_pylist()
    by_id = {int(row["id"]): row for row in rows if row.get("id") is not None}
    by_name = {
        str(row["name"]).casefold(): row
        for row in rows
        if str(row.get("name") or "").strip()
    }
    return by_id, by_name


def load_liked(bucket: int) -> list[dict[str, Any]]:
    shard = LIKED_ROOT / f"shard-{bucket // 16:02d}/eligible.parquet"
    if not shard.exists():
        return []
    table = pq.read_table(shard)
    table = table.filter(pc.equal(table["conversation_bucket"], bucket))
    return table.to_pylist()


def normalize_main(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "source_kind": "main_preselection",
        "model_tier": None,
        "core_protected": row.get("pool") != "neutral_flash",
        "format_repaired": False,
        "target_user_language_raw": None,
    }


def normalize_liked(row: dict[str, Any]) -> dict[str, Any]:
    tier = row.get("model_tier")
    guideline_mode = row.get("guideline_mode")
    guideline_by_mode = {
        "chat": "Reply in pure dialogue using approximately 40–50 lexical units.",
        "scene": "Write approximately 200–250 lexical units, interweaving dialogue and narration.",
        "longText": "Write an immersive long-form reply of approximately 520–650 lexical units.",
    }
    return {
        "target_id": row.get("target_id"),
        "target_message_id": row.get("target_message_id"),
        "target_increase_id": row.get("target_increase_id"),
        "conversation_id": row.get("conversation_id"),
        "user_id": row.get("user_id"),
        "character_id": row.get("character_id"),
        "conversation_bucket": row.get("conversation_bucket"),
        "source_kind": "liked_inventory",
        "pool": f"liked_{tier}",
        "model_tier": tier,
        "core_protected": tier == "strong_known",
        "target_position": row.get("target_position"),
        "conversation_length": row.get("conversation_length"),
        "target_time": row.get("target_time"),
        "target_content": row.get("target_content"),
        "target_user_language_raw": row.get("user_language_existing"),
        "language_raw": row.get("assistant_language_existing"),
        "language_detected": None,
        "language_final": row.get("assistant_language_existing"),
        "topic": row.get("topic"),
        "teacher_family": row.get("teacher_family"),
        "model": row.get("model"),
        "model_id": row.get("model_id"),
        "evaluation": row.get("evaluation"),
        "continued": row.get("continued"),
        "guideline_mode": guideline_mode,
        "guideline_source": (
            "compatibility_inferred" if guideline_mode else None
        ),
        "guideline_attached": bool(guideline_mode),
        "guideline": guideline_by_mode.get(guideline_mode),
        "lexical_units": row.get("lexical_units"),
        "paid_at_target": "unknown",
        "quality_score": 2.5,
        "format_valid": row.get("format_valid"),
        "format_repaired": bool(row.get("format_repaired")),
        "cot_removed": row.get("cot_removed"),
        "identity_break_confidence": row.get("identity_break_confidence"),
    }


def character_source_chars(conversation: dict[str, Any]) -> int:
    keys = [
        "character_name",
        "character_intro",
        "character_description",
        "character_personality",
        "character_scenario",
        "character_prompt",
        "character_qa_examples",
        "localized_name",
        "localized_intro",
        "localized_description",
        "localized_personality",
        "localized_scenario",
        "localized_prompt",
        "localized_qa_examples",
    ]
    return sum(len(str(conversation.get(key) or "")) for key in keys)


def build_plan(
    conversation: dict[str, Any],
    candidate: dict[str, Any],
    by_model_id: dict[int, dict[str, Any]],
    by_model_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_id = int(candidate["target_increase_id"])
    turns = conversation.get("turns") or []
    target_index = next(
        (
            index
            for index, turn in enumerate(turns)
            if int(turn.get("assistant_increase_id") or -1) == target_id
        ),
        None,
    )
    error = None
    if target_index is None:
        error = "target_turn_not_found"
        target_index = -1
        target_turn: dict[str, Any] = {}
    else:
        target_turn = turns[target_index]

    model_id = candidate.get("model_id")
    model_row = by_model_id.get(int(model_id)) if model_id is not None else None
    if model_row is None:
        model_row = by_model_name.get(str(candidate.get("model") or "").casefold())
    context_window_k = int((model_row or {}).get("context_window_k") or 0)
    small_context = 0 < context_window_k < 16

    target_time = candidate.get("target_time") or target_turn.get(
        "assistant_create_time"
    )
    memory = (
        select_memory_as_of(
            conversation.get("memory_summaries") or [],
            target_id,
            target_time,
        )
        if target_index >= 0 and not small_context
        else None
    )
    summary_text = str((memory or {}).get("summary_text") or "").strip()
    has_state, state_chars = state_payload((memory or {}).get("state_json"))
    memory_effective = bool(summary_text or has_state)
    boundary = (
        int(memory["compressed_until_increase_id"])
        if memory and memory.get("compressed_until_increase_id") is not None
        else None
    )

    if target_index < 0:
        selected: list[dict[str, Any]] = []
        replay_mode = "invalid"
        memory_state = "invalid"
    elif small_context:
        selected = turns[max(0, target_index - 24) : target_index + 1]
        replay_mode = "small_context_tail25"
        memory_state = "ignored_for_small_context"
        memory_effective = False
        boundary = None
        memory = None
    elif memory is None:
        selected = turns[: target_index + 1]
        replay_mode = "no_effective_memory_full"
        memory_state = "absent"
    else:
        selected = [
            turn
            for turn in turns[: target_index + 1]
            if int(turn.get("user_increase_id") or -1) > int(boundary)
        ]
        if memory_effective:
            replay_mode = "effective_memory_tail"
            memory_state = "effective"
        else:
            replay_mode = "no_effective_memory_bootstrap_tail"
            memory_state = "empty_bootstrap"

    if target_index >= 0 and (
        not selected
        or int(selected[-1].get("assistant_increase_id") or -1) != target_id
    ):
        error = error or "target_not_last_visible_turn"

    visible_start = (
        int(selected[0].get("turn_order") or selected[0].get("turn_index") or 1)
        if selected
        else None
    )
    visible_end = (
        int(
            selected[-1].get("turn_order")
            or selected[-1].get("turn_index")
            or target_index + 1
        )
        if selected
        else None
    )
    history_user_chars = sum(
        len(str(turn.get("user_content") or "")) for turn in selected
    )
    history_assistant_chars = sum(
        len(strip_cot(turn.get("assistant_content"))) for turn in selected[:-1]
    )
    target_user_content = str(target_turn.get("user_content") or "")
    target_content = str(candidate.get("target_content") or "")
    greeting = str(
        conversation.get("initial_greeting")
        or conversation.get("character_greeting")
        or ""
    )
    covered_turns = (
        sum(
            int(turn.get("assistant_increase_id") or -1) <= int(boundary)
            for turn in turns[: target_index + 1]
        )
        if boundary is not None
        else 0
    )
    memory_payload_chars = len(summary_text) + state_chars
    replay_chars = (
        history_user_chars
        + history_assistant_chars
        + len(target_content)
        + len(greeting)
        + memory_payload_chars
    )

    row = {
        "target_id": str(candidate.get("target_id") or target_id),
        "target_message_id": candidate.get("target_message_id"),
        "target_increase_id": target_id,
        "conversation_id": candidate.get("conversation_id"),
        "user_id": candidate.get("user_id"),
        "character_id": candidate.get("character_id"),
        "conversation_bucket": int(candidate.get("conversation_bucket") or 0),
        "source_kind": candidate.get("source_kind"),
        "pool": candidate.get("pool"),
        "model_tier": candidate.get("model_tier"),
        "core_protected": bool(candidate.get("core_protected")),
        "target_position_lifetime": int(
            candidate.get("target_position") or target_index + 1
        ),
        "conversation_length_lifetime": int(
            candidate.get("conversation_length") or len(turns)
        ),
        "target_time": target_time,
        "target_user_content": target_user_content,
        "target_content": target_content,
        "target_user_language_raw": candidate.get("target_user_language_raw")
        or target_turn.get("user_language"),
        "language_raw": candidate.get("language_raw")
        or target_turn.get("assistant_language"),
        "language_detected": candidate.get("language_detected"),
        "language_final": candidate.get("language_final"),
        "topic": candidate.get("topic"),
        "teacher_family": candidate.get("teacher_family"),
        "model": candidate.get("model"),
        "model_id": int(model_id) if model_id is not None else None,
        "model_context_window_k": context_window_k or None,
        "evaluation": int(candidate.get("evaluation") or 0),
        "continued": bool(candidate.get("continued")),
        "guideline_mode": candidate.get("guideline_mode"),
        "guideline_source": candidate.get("guideline_source"),
        "guideline_attached": bool(candidate.get("guideline_attached")),
        "guideline": candidate.get("guideline"),
        "lexical_units": candidate.get("lexical_units"),
        "paid_at_target": candidate.get("paid_at_target") or "unknown",
        "quality_score": float(candidate.get("quality_score") or 0.0),
        "format_valid": bool(candidate.get("format_valid")),
        "format_repaired": bool(candidate.get("format_repaired")),
        "cot_removed": bool(candidate.get("cot_removed")),
        "identity_break_confidence": float(
            candidate.get("identity_break_confidence") or 0.0
        ),
        "memory_effective": memory_effective,
        "memory_record_state": memory_state,
        "replay_mode": replay_mode,
        "memory_id": int(memory["id"]) if memory and memory.get("id") else None,
        "memory_created_at": memory.get("created_at") if memory else None,
        "memory_boundary_increase_id": boundary,
        "visible_start_turn": visible_start,
        "visible_end_turn": visible_end,
        "visible_user_turns": len(selected),
        "visible_history_assistant_turns": max(0, len(selected) - 1),
        "memory_covered_turns": covered_turns,
        "memory_summary_chars": len(summary_text),
        "memory_state_chars": state_chars,
        "target_user_chars": len(target_user_content),
        "history_user_chars": history_user_chars,
        "history_assistant_chars": history_assistant_chars,
        "target_chars": len(target_content),
        "greeting_chars": len(greeting),
        "character_source_chars": character_source_chars(conversation),
        "prompt_replay_chars_without_system": replay_chars,
        "selection_hash_u64": stable_hash(str(candidate.get("target_id") or target_id)),
        "replay_invariant_pass": error is None,
        "replay_error": error,
    }
    return row


def main() -> int:
    args = parse_args()
    started = time.time()
    bucket = args.bucket
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    main_path = MAIN_TARGET_ROOT / f"part-{bucket:03d}.parquet"
    main_rows = [
        normalize_main(row) for row in pq.read_table(main_path).to_pylist()
    ]
    liked_rows = [normalize_liked(row) for row in load_liked(bucket)]
    candidates = main_rows + liked_rows
    wanted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        wanted[str(candidate["conversation_id"])].append(candidate)

    by_model_id, by_model_name = model_maps()
    bundle_path = (
        BUNDLE_ROOT
        / f"conversation_bucket={bucket:03d}"
        / "part-000000.parquet"
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in pq.ParquetFile(bundle_path).iter_batches(batch_size=128):
        for conversation in batch.to_pylist():
            conversation_id = str(conversation["conversation_id"])
            if conversation_id not in wanted:
                continue
            for candidate in wanted[conversation_id]:
                row = build_plan(
                    conversation,
                    candidate,
                    by_model_id,
                    by_model_name,
                )
                rows.append(row)
                seen.add(row["target_id"])

    missing = [
        candidate
        for candidate in candidates
        if str(candidate["target_id"]) not in seen
    ]
    if missing:
        for candidate in missing:
            rows.append(
                {
                    "target_id": str(candidate["target_id"]),
                    "target_message_id": candidate.get("target_message_id"),
                    "target_increase_id": int(candidate["target_increase_id"]),
                    "conversation_id": candidate.get("conversation_id"),
                    "user_id": candidate.get("user_id"),
                    "character_id": candidate.get("character_id"),
                    "conversation_bucket": bucket,
                    "source_kind": candidate.get("source_kind"),
                    "pool": candidate.get("pool"),
                    "model_tier": candidate.get("model_tier"),
                    "core_protected": bool(candidate.get("core_protected")),
                    "target_content": candidate.get("target_content"),
                    "replay_invariant_pass": False,
                    "replay_error": "conversation_not_found",
                    "selection_hash_u64": stable_hash(str(candidate["target_id"])),
                }
            )

    table = pa.Table.from_pylist(rows, schema=PLAN_SCHEMA)
    output = output_root / f"part-{bucket:03d}.parquet"
    temp = output.with_suffix(".tmp")
    pq.write_table(
        table,
        temp,
        compression="zstd",
        compression_level=6,
        row_group_size=16384,
    )
    os.replace(temp, output)
    counts = Counter(row["memory_record_state"] for row in rows)
    errors = Counter(
        row["replay_error"] for row in rows if row.get("replay_error")
    )
    report = {
        "bucket": bucket,
        "main_candidates": len(main_rows),
        "liked_candidates": len(liked_rows),
        "output_rows": len(rows),
        "memory_record_state": dict(counts),
        "replay_errors": dict(errors),
        "accounting_pass": len(rows) == len(candidates),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = ROOT / f"00_manifest/context_plan_receipts/part-{bucket:03d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["accounting_pass"] and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
