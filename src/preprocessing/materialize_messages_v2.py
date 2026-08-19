#!/usr/bin/env python3
"""Materialize final OpenAI-style messages from selected replay plans."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


ROOT = Path("/home/gyr/lusychat-training/pre-mds-context-v2")
SELECTED = ROOT / "04_cap_selection/selected_parts"
BUNDLE_ROOT = Path("/home/gyr/lusychat-training/lusy-chat-4m-v1/bundles")
CANONICAL = Path("/home/gyr/lusychat-training/lusy-chat-4m-v2/01_canonical")
OUTPUT = ROOT / "05_materialized/parts"
TOKENIZER_JSON = Path(
    "/home/gyr/qwen3.5/checkpoints/Qwen/Qwen3.5-2B/tokenizer.json"
)

SCRIPT_ROOT = Path(__file__).resolve().parent
PIPELINE_SRC = Path(
    os.environ.get("LUSY_PIPELINE_SRC", str(SCRIPT_ROOT))
)
sys.path.insert(0, str(PIPELINE_SRC))
sys.path.insert(0, str(SCRIPT_ROOT))

from lusy_training.cards import (  # noqa: E402
    attach_image_context,
    build_character_system,
    choose_description,
    format_persona,
)
from lusy_training.preference import prompt_fingerprint  # noqa: E402
from lusy_training.text import strip_provider_cot  # noqa: E402
from filter_liked_context_v2 import validate_format  # noqa: E402
from format_repair import repair_balanced_nesting  # noqa: E402


MESSAGE = pa.struct(
    [
        ("role", pa.string()),
        ("content", pa.large_string()),
        ("weight", pa.int8()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buckets", required=True)
    return parser.parse_args()


def parse_range(value: str) -> list[int]:
    if "-" not in value:
        return [int(value)]
    start, end = value.split("-", 1)
    return list(range(int(start), int(end) + 1))


def parse_state(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    return {}


def memory_wrapper(memory: dict[str, Any]) -> str | None:
    summary = str(memory.get("summary_text") or "").strip()
    state = parse_state(memory.get("state_json"))
    if not summary and not state:
        return None
    return (
        "Long-term conversation memory:\n"
        f"{summary}\n\n"
        "Structured continuity state:\n"
        f"{json.dumps(state, ensure_ascii=False)}\n\n"
        "Treat this memory as canonical background context. Raw dialogue below "
        "is the newest post-summary tail."
    )


def normalize_duplicate_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def repaired_assistant(value: Any) -> str:
    content = strip_provider_cot(value).content
    initial = validate_format(content)
    repair = repair_balanced_nesting(content, initial, validate_format)
    return repair.content if repair.repaired else content


def build_system(
    conversation: dict[str, Any],
    row: dict[str, Any],
    descriptions: dict[str, list[dict[str, Any]]],
    persona_by_conversation: dict[str, dict[str, Any]],
    persona_by_id: dict[str, dict[str, Any]],
) -> tuple[str, str, dict[str, Any] | None]:
    language = row.get("language_final_v2") or row.get("language_final")
    description, language_source = choose_description(
        descriptions.get(str(row["character_id"]), []),
        language,
    )
    if not description:
        description = {
            "name": conversation.get("character_name"),
            "description": conversation.get("character_description"),
            "personality": conversation.get("character_personality"),
            "scenario": conversation.get("character_scenario"),
            "prompt": conversation.get("character_prompt"),
            "greeting": conversation.get("character_greeting"),
            "language": "canonical",
        }
    persona = persona_by_conversation.get(str(row["conversation_id"]))
    if persona is None and conversation.get("persona_id"):
        persona = persona_by_id.get(str(conversation.get("persona_id")))
    persona_text = format_persona(persona)
    user_name = str((persona or {}).get("nickname") or "User")
    character = build_character_system(description, language, user_name)
    system = "[CHARACTER]\n" + character.strip()
    if persona_text.strip():
        system += "\n\n[USER PERSONA]\n" + persona_text.strip()
    return system, language_source, persona


def select_turns(
    turns: list[dict[str, Any]],
    target_index: int,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    replay_mode = row["replay_mode"]
    if replay_mode == "small_context_tail25":
        return turns[max(0, target_index - 24) : target_index + 1]
    if replay_mode == "no_effective_memory_full":
        return turns[: target_index + 1]
    if replay_mode in {
        "no_effective_memory_bootstrap_tail",
        "effective_memory_tail",
    }:
        boundary = int(row["memory_boundary_increase_id"])
        return [
            turn
            for turn in turns[: target_index + 1]
            if int(turn.get("user_increase_id") or -1) > boundary
        ]
    raise ValueError(f"unsupported replay mode: {replay_mode}")


def serialized_for_tokenizer(messages: list[dict[str, Any]]) -> str:
    return "".join(
        f"<|{message['role']}|>\n{message['content']}\n"
        for message in messages
    )


def load_resources() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    descriptions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(
        CANONICAL / "character_descriptions.parquet"
    ).to_pylist():
        descriptions[str(row["character_id"])].append(row)
    persona_rows = pq.read_table(CANONICAL / "personas.parquet").to_pylist()
    by_conversation = {
        str(row["conversation_id"]): row
        for row in persona_rows
        if row.get("conversation_id") and not row.get("deleted_at")
    }
    by_id = {
        str(row["persona_id"]): row
        for row in persona_rows
        if row.get("persona_id") and not row.get("deleted_at")
    }
    images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(CANONICAL / "image_contexts.parquet").to_pylist():
        images[str(row["message_id"])].append(row)
    return descriptions, by_conversation, by_id, images


def materialize_row(
    conversation: dict[str, Any],
    row: dict[str, Any],
    descriptions: dict[str, list[dict[str, Any]]],
    persona_by_conversation: dict[str, dict[str, Any]],
    persona_by_id: dict[str, dict[str, Any]],
    images: dict[str, list[dict[str, Any]]],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    target_id = int(row["target_increase_id"])
    turns = conversation.get("turns") or []
    target_index = next(
        index
        for index, turn in enumerate(turns)
        if int(turn.get("assistant_increase_id") or -1) == target_id
    )
    selected = select_turns(turns, target_index, row)
    if (
        not selected
        or int(selected[-1].get("assistant_increase_id") or -1) != target_id
    ):
        raise ValueError("target is not the last selected turn")
    if len(selected) != int(row["visible_user_turns"]):
        raise ValueError("visible turn count differs from replay plan")

    system, language_source, _ = build_system(
        conversation,
        row,
        descriptions,
        persona_by_conversation,
        persona_by_id,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system, "weight": 0}
    ]

    wrapper = None
    if row["replay_mode"] == "effective_memory_tail":
        memory_id = int(row["memory_id"])
        memory = next(
            (
                item
                for item in conversation.get("memory_summaries") or []
                if int(item.get("id") or -1) == memory_id
            ),
            None,
        )
        if memory is None:
            raise ValueError("selected memory record not found")
        wrapper = memory_wrapper(memory)
        if not wrapper:
            raise ValueError("effective memory has empty wrapper")
        messages.append({"role": "system", "content": wrapper, "weight": 0})

    greeting = str(
        conversation.get("initial_greeting")
        or conversation.get("character_greeting")
        or ""
    ).strip()
    if greeting:
        messages.append(
            {"role": "assistant", "content": greeting, "weight": 0}
        )

    for index, turn in enumerate(selected):
        logical_id = str(turn.get("logical_turn_id") or "")
        user_content = attach_image_context(
            str(turn.get("user_content") or "").strip(),
            images.get(logical_id, []),
        )
        is_target = index == len(selected) - 1
        if is_target and row.get("guideline_attached") and row.get("guideline"):
            user_content += (
                "\n\n[RESPONSE LENGTH GUIDELINE]\n"
                + str(row["guideline"])
            )
        assistant_content = (
            str(row["target_content"]).strip()
            if is_target
            else repaired_assistant(turn.get("assistant_content"))
        )
        messages.append({"role": "user", "content": user_content, "weight": 0})
        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "weight": 1 if is_target else 0,
            }
        )

    target_text = normalize_duplicate_text(messages[-1]["content"])
    duplicate_match = None
    assistant_history = [
        message
        for message in messages[:-1]
        if message["role"] == "assistant"
    ]
    for index, message in enumerate(assistant_history):
        if normalize_duplicate_text(message["content"]) == target_text:
            duplicate_match = index
            break
    duplicate = duplicate_match is not None
    token_count = len(
        tokenizer.encode(serialized_for_tokenizer(messages)).ids
    )
    weights = [
        int(message.get("weight") or 0)
        for message in messages
        if message["role"] == "assistant"
    ]
    invariant = (
        messages[-1]["role"] == "assistant"
        and int(messages[-1]["weight"]) == 1
        and sum(weights) == 1
        and sum(message["role"] == "user" for message in messages)
        == int(row["visible_user_turns"])
        and ((wrapper is not None) == bool(row["memory_effective"]))
    )
    output = {
        key: value
        for key, value in row.items()
        if key not in {"target_user_content", "target_content"}
    }
    output.update(
        {
            "messages": messages,
            "assistant_loss_weights": weights,
            "prompt_fingerprint": prompt_fingerprint(messages[:-1]),
            "prompt_replay_mode": "canonical_current",
            "historical_prompt_exact": False,
            "character_language_source": language_source,
            "visible_user_turns_verified": sum(
                message["role"] == "user" for message in messages
            ),
            "qwen35_serialized_tokens": token_count,
            "exact_duplicate_target": duplicate,
            "exact_duplicate_history_assistant_index": duplicate_match,
            "memory_wrapper_present": wrapper is not None,
            "memory_wrapper_chars": len(wrapper or ""),
            "materialization_invariant_pass": invariant,
            "materialization_error": None if invariant else "invariant_failed",
        }
    )
    return output


def main() -> int:
    args = parse_args()
    buckets = parse_range(args.buckets)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    receipt_root = ROOT / "00_manifest/materialization_receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    resources = load_resources()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    started = time.time()
    totals = Counter()

    for bucket in buckets:
        selected_path = SELECTED / f"part-{bucket:03d}.parquet"
        selected_table = pq.read_table(selected_path)
        selected_rows = selected_table.to_pylist()
        wanted: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected_rows:
            wanted[str(row["conversation_id"])].append(row)
        bundle = (
            BUNDLE_ROOT
            / f"conversation_bucket={bucket:03d}"
            / "part-000000.parquet"
        )
        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for batch in pq.ParquetFile(bundle).iter_batches(batch_size=128):
            for conversation in batch.to_pylist():
                conversation_id = str(conversation["conversation_id"])
                if conversation_id not in wanted:
                    continue
                for row in wanted[conversation_id]:
                    try:
                        rows.append(
                            materialize_row(
                                conversation,
                                row,
                                *resources,
                                tokenizer,
                            )
                        )
                    except Exception as exc:
                        errors.append(
                            {
                                "target_id": row["target_id"],
                                "conversation_id": conversation_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
        if errors:
            error_path = ROOT / f"05_materialized/errors/part-{bucket:03d}.json"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(
                json.dumps(errors, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        base_fields = [
            field
            for field in selected_table.schema
            if field.name not in {"target_user_content", "target_content"}
        ]
        output_schema = pa.schema(
            base_fields
            + [
                pa.field("messages", pa.large_list(MESSAGE)),
                pa.field("assistant_loss_weights", pa.list_(pa.int8())),
                pa.field("prompt_fingerprint", pa.string()),
                pa.field("prompt_replay_mode", pa.string()),
                pa.field("historical_prompt_exact", pa.bool_()),
                pa.field("character_language_source", pa.string()),
                pa.field("visible_user_turns_verified", pa.int32()),
                pa.field("qwen35_serialized_tokens", pa.int32()),
                pa.field("exact_duplicate_target", pa.bool_()),
                pa.field(
                    "exact_duplicate_history_assistant_index",
                    pa.int32(),
                ),
                pa.field("memory_wrapper_present", pa.bool_()),
                pa.field("memory_wrapper_chars", pa.int32()),
                pa.field("materialization_invariant_pass", pa.bool_()),
                pa.field("materialization_error", pa.string()),
            ]
        )
        output = OUTPUT / f"part-{bucket:03d}.parquet"
        temp = output.with_suffix(".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=output_schema),
            temp,
            compression="zstd",
            compression_level=6,
            row_group_size=4096,
        )
        os.replace(temp, output)
        totals["selected"] += len(selected_rows)
        totals["materialized"] += len(rows)
        totals["errors"] += len(errors)
        totals["exact_duplicate"] += sum(
            bool(row["exact_duplicate_target"]) for row in rows
        )

    report = {
        "buckets": args.buckets,
        **dict(totals),
        "accounting_pass": totals["selected"]
        == totals["materialized"] + totals["errors"],
        "elapsed_seconds": round(time.time() - started, 3),
        "tokenizer": str(TOKENIZER_JSON),
    }
    receipt = (
        receipt_root
        / f"range-{buckets[0]:03d}-{buckets[-1]:03d}.json"
    )
    receipt.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["accounting_pass"] and not totals["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
