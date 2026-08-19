#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


DEFAULT_INPUT = Path(
    "/home/gyr/lusychat-training/pre-mds-context-v2/06_pre_mds/parts"
)
DEFAULT_CANONICAL = Path(
    "/home/gyr/lusychat-training/lusy-chat-4m-v2/01_canonical"
)
DEFAULT_OUTPUT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
BUILD_VERSION = "rp-mds-dialogue-builder-v1.0.0"
SEED = "lusy-rp-mds-dialogue-v1-20260724"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_language(value: Any) -> str:
    raw = str(value or "unknown").strip().casefold().replace("_", "-")
    if raw.startswith("zh"):
        return "zh"
    for language in ("en", "es", "pt", "fr", "de", "ja", "ko", "ru", "th"):
        if raw == language or raw.startswith(language + "-"):
            return language
    if not raw or raw in {"none", "null", "unknown"}:
        return "unknown"
    return raw.split("-", 1)[0]


def language_bucket(value: Any) -> str:
    normalized = normalize_language(value)
    if normalized in {"en", "es", "pt", "fr", "zh", "de", "ja"}:
        return normalized
    return "other"


def turn_bucket(value: int) -> str:
    if value <= 8:
        return "05_08"
    if value <= 16:
        return "09_16"
    if value <= 32:
        return "17_32"
    if value <= 64:
        return "33_64"
    return "65_plus"


def response_length_bucket(value: Any) -> str:
    try:
        count = int(value)
    except Exception:
        return "unknown"
    if count < 40:
        return "lt_40"
    if count <= 50:
        return "40_50"
    if count < 200:
        return "51_199"
    if count <= 250:
        return "200_250"
    if count < 520:
        return "251_519"
    if count <= 650:
        return "520_650"
    return "gt_650"


def stable_dialogue_id(
    conversation_id: str,
    replay_mode: str,
    memory_id: Any,
    boundary: Any,
) -> str:
    payload = "\x1f".join(
        [
            str(conversation_id),
            str(replay_mode),
            "" if memory_id is None else str(memory_id),
            "" if boundary is None else str(boundary),
        ]
    )
    return hashlib.sha256(f"{SEED}\0{payload}".encode("utf-8")).hexdigest()


def text(value: Any) -> str:
    return str(value or "").strip()


def extract_dialogue_views(messages: list[dict[str, Any]] | None) -> tuple[list[str], str]:
    user_texts: list[str] = []
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None

    for message in messages or []:
        role = str(message.get("role") or "").strip().casefold()
        content = text(message.get("content"))
        if not content:
            continue
        if role == "user":
            user_texts.append(content)
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None

    recent = pairs[-3:]
    recent_scene = "\n\n".join(
        f"[USER]\n{user}\n[ASSISTANT]\n{assistant}"
        for user, assistant in recent
    )
    return user_texts, recent_scene


def choose_localized(
    descriptions: dict[tuple[str, str], dict[str, Any]],
    character_id: str,
    language: str,
) -> tuple[dict[str, Any] | None, str]:
    normalized = normalize_language(language)
    exact = descriptions.get((character_id, normalized))
    if exact is not None:
        return exact, "localized_exact"
    english = descriptions.get((character_id, "en"))
    if english is not None:
        return english, "localized_en_fallback"
    candidates = [
        row for (candidate_character_id, _), row in descriptions.items()
        if candidate_character_id == character_id
    ]
    if candidates:
        return max(candidates, key=lambda row: int(row.get("increase_id") or -1)), "localized_any_fallback"
    return None, "base_character"


def role_text(
    base: dict[str, Any] | None,
    localized: dict[str, Any] | None,
    conversation_greeting: str,
) -> str:
    base = base or {}
    localized = localized or {}

    fields = [
        ("CHARACTER NAME", localized.get("name") or base.get("name")),
        ("INTRO", localized.get("intro") or base.get("intro")),
        ("DESCRIPTION", localized.get("description") or base.get("description")),
        ("PERSONALITY", localized.get("personality") or base.get("personality")),
        ("SCENARIO", localized.get("scenario") or base.get("scenario")),
        ("TAGS", base.get("tags")),
        (
            "GREETING",
            conversation_greeting
            or text(localized.get("greeting"))
            or text(base.get("greeting")),
        ),
    ]
    return "\n\n".join(
        f"[{label}]\n{text(value)}"
        for label, value in fields
        if text(value)
    )


def greeting_from_messages(messages: list[dict[str, Any]] | None) -> str:
    seen_user = False
    for message in messages or []:
        role = str(message.get("role") or "").strip().casefold()
        content = text(message.get("content"))
        if role == "user":
            seen_user = True
        elif role == "assistant" and not seen_user and content:
            return content
    return ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )


def counter_json(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> int:
    options = args()
    output = options.output_root
    sentinel = output / "00_manifest" / "DIALOGUE_BUILD_REPORT.json"
    if sentinel.exists():
        raise SystemExit(f"Refusing to overwrite completed build: {sentinel}")

    for relative in [
        "00_manifest",
        "01_dialogues",
        "02_texts",
        "08_core_bypass",
        "scripts",
    ]:
        (output / relative).mkdir(parents=True, exist_ok=True)

    input_parts = sorted(options.input_root.glob("*.parquet"))
    if len(input_parts) != 256:
        raise RuntimeError(f"Expected 256 pre-MDS parts, found {len(input_parts)}")

    selected_columns = [
        "target_id",
        "target_message_id",
        "target_increase_id",
        "conversation_id",
        "user_id",
        "character_id",
        "pool",
        "core_protected",
        "language_final_v2",
        "topic",
        "guideline_mode",
        "lexical_units",
        "replay_mode",
        "memory_id",
        "memory_boundary_increase_id",
        "visible_user_turns_verified",
        "qwen35_serialized_tokens",
        "messages",
    ]

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    target_rows: list[dict[str, Any]] = []
    input_target_count = 0

    for part_index, path in enumerate(input_parts):
        table = pq.ParquetFile(path).read(columns=selected_columns)
        for row in table.to_pylist():
            input_target_count += 1
            key = (
                str(row["conversation_id"]),
                str(row["replay_mode"]),
                row.get("memory_id"),
                row.get("memory_boundary_increase_id"),
            )
            dialogue_id = stable_dialogue_id(*key)
            current = groups.get(key)
            if current is None:
                current = {
                    "dialogue_id": dialogue_id,
                    "conversation_id": key[0],
                    "replay_mode": key[1],
                    "memory_id": key[2],
                    "memory_boundary_increase_id": key[3],
                    "user_id": str(row["user_id"]),
                    "character_id": str(row["character_id"]),
                    "contains_core": False,
                    "candidate_target_ids": [],
                    "candidate_message_ids": [],
                    "candidate_pools": Counter(),
                    "candidate_languages": Counter(),
                    "candidate_topics": Counter(),
                    "candidate_turn_buckets": Counter(),
                    "candidate_length_buckets": Counter(),
                    "candidate_guidelines": Counter(),
                    "representative": None,
                }
                groups[key] = current

            current["contains_core"] = current["contains_core"] or bool(row["core_protected"])
            current["candidate_target_ids"].append(str(row["target_id"]))
            current["candidate_message_ids"].append(str(row["target_message_id"]))
            current["candidate_pools"][str(row["pool"] or "unknown")] += 1
            current["candidate_languages"][normalize_language(row["language_final_v2"])] += 1
            current["candidate_topics"][str(row["topic"] or "unknown")] += 1
            current["candidate_turn_buckets"][
                turn_bucket(int(row["visible_user_turns_verified"]))
            ] += 1
            current["candidate_length_buckets"][
                response_length_bucket(row.get("lexical_units"))
            ] += 1
            current["candidate_guidelines"][
                str(row.get("guideline_mode") or "rp_base")
            ] += 1

            representative = current["representative"]
            if representative is None or int(row["target_increase_id"]) > int(
                representative["target_increase_id"]
            ):
                current["representative"] = row

            target_rows.append(
                {
                    "target_id": str(row["target_id"]),
                    "target_message_id": str(row["target_message_id"]),
                    "target_increase_id": int(row["target_increase_id"]),
                    "dialogue_id": dialogue_id,
                    "conversation_id": str(row["conversation_id"]),
                    "user_id": str(row["user_id"]),
                    "character_id": str(row["character_id"]),
                    "pool": str(row["pool"]),
                    "core_protected": bool(row["core_protected"]),
                    "language": normalize_language(row["language_final_v2"]),
                    "language_bucket": language_bucket(row["language_final_v2"]),
                    "topic": str(row["topic"] or "unknown"),
                    "turn_bucket": turn_bucket(int(row["visible_user_turns_verified"])),
                    "response_length_bucket": response_length_bucket(row.get("lexical_units")),
                    "guideline_bucket": str(row.get("guideline_mode") or "rp_base"),
                    "replay_mode": str(row["replay_mode"]),
                }
            )
        print(
            json.dumps(
                {
                    "event": "read_part",
                    "part": part_index + 1,
                    "parts": len(input_parts),
                    "targets": input_target_count,
                    "groups": len(groups),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    characters_table = pq.read_table(options.canonical_root / "characters.parquet")
    wanted_characters = {str(group["character_id"]) for group in groups.values()}
    characters = {
        str(row["character_id"]): row
        for row in characters_table.to_pylist()
        if str(row["character_id"]) in wanted_characters
    }

    description_table = pq.read_table(options.canonical_root / "character_descriptions.parquet")
    descriptions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in description_table.to_pylist():
        character_id = str(row["character_id"])
        if character_id not in wanted_characters:
            continue
        key = (character_id, normalize_language(row.get("language")))
        previous = descriptions.get(key)
        if previous is None or int(row.get("increase_id") or -1) > int(
            previous.get("increase_id") or -1
        ):
            descriptions[key] = row

    dialogue_rows: list[dict[str, Any]] = []
    cluster_text_rows: list[dict[str, Any]] = []
    core_bypass_rows: list[dict[str, Any]] = []
    summary_bypass_rows: list[dict[str, Any]] = []

    class_counts = Counter()
    class_target_counts = Counter()
    language_counts = Counter()
    replay_counts = Counter()
    missing_role_text = 0
    no_user_texts = 0
    no_recent_scene = 0

    for group in groups.values():
        representative = group.pop("representative")
        messages = representative.get("messages") or []
        user_texts, recent_scene = extract_dialogue_views(messages)
        dominant_language = (
            group["candidate_languages"].most_common(1)[0][0]
            if group["candidate_languages"]
            else normalize_language(representative.get("language_final_v2"))
        )
        localized, character_source = choose_localized(
            descriptions,
            group["character_id"],
            dominant_language,
        )
        exact_greeting = greeting_from_messages(messages)
        role = role_text(
            characters.get(group["character_id"]),
            localized,
            exact_greeting,
        )

        if not role:
            missing_role_text += 1
        if not user_texts:
            no_user_texts += 1
        if not recent_scene:
            no_recent_scene += 1

        has_summary = group["replay_mode"] == "effective_memory_tail"
        if has_summary:
            selection_class = "summary_bypass"
        elif group["contains_core"]:
            selection_class = "core_bypass"
        else:
            selection_class = "neutral_cluster"

        candidate_count = len(group["candidate_target_ids"])
        class_counts[selection_class] += 1
        class_target_counts[selection_class] += candidate_count
        language_counts[(selection_class, language_bucket(dominant_language))] += 1
        replay_counts[(selection_class, group["replay_mode"])] += 1

        dialogue_row = {
            "dialogue_id": group["dialogue_id"],
            "conversation_id": group["conversation_id"],
            "replay_mode": group["replay_mode"],
            "memory_id": group["memory_id"],
            "memory_boundary_increase_id": group["memory_boundary_increase_id"],
            "user_id": group["user_id"],
            "character_id": group["character_id"],
            "selection_class": selection_class,
            "contains_core": bool(group["contains_core"]),
            "has_effective_summary": has_summary,
            "representative_target_id": str(representative["target_id"]),
            "representative_target_increase_id": int(representative["target_increase_id"]),
            "candidate_target_count": candidate_count,
            "candidate_target_ids": group["candidate_target_ids"],
            "candidate_message_ids": group["candidate_message_ids"],
            "candidate_pool_counts_json": json.dumps(
                counter_json(group["candidate_pools"]), ensure_ascii=False, sort_keys=True
            ),
            "candidate_language_counts_json": json.dumps(
                counter_json(group["candidate_languages"]), ensure_ascii=False, sort_keys=True
            ),
            "candidate_topic_counts_json": json.dumps(
                counter_json(group["candidate_topics"]), ensure_ascii=False, sort_keys=True
            ),
            "candidate_turn_bucket_counts_json": json.dumps(
                counter_json(group["candidate_turn_buckets"]), ensure_ascii=False, sort_keys=True
            ),
            "candidate_length_bucket_counts_json": json.dumps(
                counter_json(group["candidate_length_buckets"]), ensure_ascii=False, sort_keys=True
            ),
            "candidate_guideline_counts_json": json.dumps(
                counter_json(group["candidate_guidelines"]), ensure_ascii=False, sort_keys=True
            ),
            "dominant_language": dominant_language,
            "language_bucket": language_bucket(dominant_language),
            "representative_topic": str(representative.get("topic") or "unknown"),
            "representative_visible_user_turns": int(
                representative["visible_user_turns_verified"]
            ),
            "representative_turn_bucket": turn_bucket(
                int(representative["visible_user_turns_verified"])
            ),
            "representative_serialized_tokens": int(
                representative["qwen35_serialized_tokens"]
            ),
            "character_language_source": character_source,
            "user_text_count": len(user_texts),
            "role_text_chars": len(role),
            "recent_scene_chars": len(recent_scene),
            "messages": messages,
        }
        dialogue_rows.append(dialogue_row)

        text_row = {
            "dialogue_id": group["dialogue_id"],
            "language_bucket": language_bucket(dominant_language),
            "user_texts": user_texts,
            "role_text": role,
            "recent_scene_text": recent_scene,
            "candidate_target_count": candidate_count,
            "replay_mode": group["replay_mode"],
            "representative_topic": str(representative.get("topic") or "unknown"),
            "representative_turn_bucket": turn_bucket(
                int(representative["visible_user_turns_verified"])
            ),
            "candidate_topic_counts_json": dialogue_row["candidate_topic_counts_json"],
            "candidate_turn_bucket_counts_json": dialogue_row[
                "candidate_turn_bucket_counts_json"
            ],
            "candidate_length_bucket_counts_json": dialogue_row[
                "candidate_length_bucket_counts_json"
            ],
            "candidate_guideline_counts_json": dialogue_row[
                "candidate_guideline_counts_json"
            ],
        }

        if selection_class == "neutral_cluster":
            cluster_text_rows.append(text_row)
        elif selection_class == "core_bypass":
            core_bypass_rows.append(dialogue_row)
        else:
            summary_bypass_rows.append(dialogue_row)

    dialogue_rows.sort(key=lambda row: row["dialogue_id"])
    cluster_text_rows.sort(key=lambda row: row["dialogue_id"])
    core_bypass_rows.sort(key=lambda row: row["dialogue_id"])
    summary_bypass_rows.sort(key=lambda row: row["dialogue_id"])
    target_rows.sort(key=lambda row: (row["dialogue_id"], row["target_increase_id"]))

    paths = {
        "dialogue_index": output / "01_dialogues" / "dialogue_index.parquet",
        "candidate_map": output / "01_dialogues" / "candidate_assistant_map.parquet",
        "cluster_texts": output / "02_texts" / "neutral_no_summary_views.parquet",
        "core_bypass": output / "08_core_bypass" / "core_dialogues.parquet",
        "summary_bypass": output / "08_core_bypass" / "summary_dialogues.parquet",
    }
    write_parquet(dialogue_rows, paths["dialogue_index"])
    write_parquet(target_rows, paths["candidate_map"])
    write_parquet(cluster_text_rows, paths["cluster_texts"])
    write_parquet(core_bypass_rows, paths["core_bypass"])
    write_parquet(summary_bypass_rows, paths["summary_bypass"])

    dialogue_ids = {row["dialogue_id"] for row in dialogue_rows}
    cluster_ids = {row["dialogue_id"] for row in cluster_text_rows}
    core_ids = {row["dialogue_id"] for row in core_bypass_rows}
    summary_ids = {row["dialogue_id"] for row in summary_bypass_rows}
    partition_union = cluster_ids | core_ids | summary_ids

    invariants = {
        "input_target_count_matches": input_target_count == 653_370,
        "dialogue_count_matches": len(dialogue_rows) == 189_323,
        "dialogue_ids_unique": len(dialogue_ids) == len(dialogue_rows),
        "candidate_map_count_matches": len(target_rows) == input_target_count,
        "partition_disjoint": not (
            (cluster_ids & core_ids)
            or (cluster_ids & summary_ids)
            or (core_ids & summary_ids)
        ),
        "partition_complete": partition_union == dialogue_ids,
        "neutral_cluster_contains_no_core": all(
            not row["contains_core"]
            for row in dialogue_rows
            if row["selection_class"] == "neutral_cluster"
        ),
        "neutral_cluster_contains_no_summary": all(
            not row["has_effective_summary"]
            for row in dialogue_rows
            if row["selection_class"] == "neutral_cluster"
        ),
        "summary_bypass_all_effective": all(
            row["has_effective_summary"] for row in summary_bypass_rows
        ),
        "all_cluster_rows_have_user_text": all(
            bool(row["user_texts"]) for row in cluster_text_rows
        ),
        "all_cluster_rows_have_role_text": all(
            bool(row["role_text"]) for row in cluster_text_rows
        ),
        "all_cluster_rows_have_recent_scene": all(
            bool(row["recent_scene_text"]) for row in cluster_text_rows
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"Dialogue build invariant failure: {invariants}")

    checksums = {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }

    report = {
        "build_version": BUILD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "input_root": str(options.input_root),
        "canonical_root": str(options.canonical_root),
        "output_root": str(output),
        "input_parts": len(input_parts),
        "input_target_count": input_target_count,
        "dialogue_count": len(dialogue_rows),
        "raw_conversation_count": len(
            {row["conversation_id"] for row in dialogue_rows}
        ),
        "selection_class_dialogues": counter_json(class_counts),
        "selection_class_targets": counter_json(class_target_counts),
        "language_by_class": {
            f"{selection_class}|{language}": count
            for (selection_class, language), count in sorted(language_counts.items())
        },
        "replay_by_class": {
            f"{selection_class}|{replay_mode}": count
            for (selection_class, replay_mode), count in sorted(replay_counts.items())
        },
        "missing_role_text": missing_role_text,
        "no_user_texts": no_user_texts,
        "no_recent_scene": no_recent_scene,
        "invariants": invariants,
        "outputs": checksums,
    }
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (sentinel.with_suffix(".sha256")).write_text(
        f"{sha256_file(sentinel)}  {sentinel.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
