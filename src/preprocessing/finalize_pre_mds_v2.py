#!/usr/bin/env python3
"""Finalize materialized SFT parts after exact-history dedupe.

The character-share cap is recomputed after dedupe because removing rows changes
the denominator.  All concentration caps are then verified on the final neutral
pool.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import statistics

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path("/home/gyr/lusychat-training/pre-mds-context-v2")
SOURCE = ROOT / "05_materialized/parts"
FINAL_ROOT = ROOT / "06_pre_mds/parts"
CHARACTER_SHARE_CAP = 0.005
USER_CAP = 60
USER_CHARACTER_CAP = 12
CONVERSATION_CAP = 6


def quantiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def q(value: float) -> float:
        index = (len(ordered) - 1) * value
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return float(ordered[lower])
        return (
            ordered[lower] * (upper - index)
            + ordered[upper] * (index - lower)
        )

    return {
        "min": ordered[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def priority(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row["continued"] else 1,
        0 if row["paid_at_target"] == "paid" else 1,
        int(row["selection_hash_u64"]),
        str(row["target_id"]),
    )


def determine_post_dedupe_character_removals(
    source_files: list[Path],
) -> tuple[set[str], list[dict[str, int]], int]:
    by_character: dict[str, list[dict[str, Any]]] = defaultdict(list)
    neutral_rows = 0
    columns = [
        "target_id",
        "character_id",
        "pool",
        "continued",
        "paid_at_target",
        "selection_hash_u64",
        "exact_duplicate_target",
    ]
    for path in source_files:
        table = pq.read_table(path, columns=columns)
        kept = table.filter(pc.invert(table["exact_duplicate_target"]))
        for row in kept.to_pylist():
            if row["pool"] != "neutral_flash":
                continue
            neutral_rows += 1
            by_character[str(row["character_id"])].append(row)

    cap = max(1, math.floor(neutral_rows * CHARACTER_SHARE_CAP))
    history: list[dict[str, int]] = []
    while True:
        selected = sum(min(len(rows), cap) for rows in by_character.values())
        next_cap = max(1, math.floor(selected * CHARACTER_SHARE_CAP))
        history.append(
            {
                "input_cap": cap,
                "selected": selected,
                "next_cap": next_cap,
            }
        )
        if next_cap >= cap:
            break
        cap = next_cap

    remove_ids: set[str] = set()
    for rows in by_character.values():
        if len(rows) <= cap:
            continue
        rows.sort(key=priority)
        remove_ids.update(str(row["target_id"]) for row in rows[cap:])
    return remove_ids, history, cap


def main() -> int:
    source_files = sorted(SOURCE.glob("part-*.parquet"))
    if len(source_files) != 256:
        raise RuntimeError(
            f"materialized parts incomplete: {len(source_files)}"
        )
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    remove_ids, cap_history, final_character_cap = (
        determine_post_dedupe_character_removals(source_files)
    )

    counts = Counter()
    pools = Counter()
    languages = Counter()
    topics = Counter()
    replay = Counter()
    memory = Counter()
    visible_turns: list[int] = []
    tokens: list[int] = []
    unique_targets: set[str] = set()
    unique_conversations: set[str] = set()
    unique_users: set[str] = set()
    unique_characters: set[str] = set()
    neutral_users = Counter()
    neutral_pairs = Counter()
    neutral_conversations = Counter()
    neutral_characters = Counter()
    sha256s: dict[str, str] = {}
    removed_rows: list[dict[str, Any]] = []

    for path in source_files:
        table = pq.read_table(path)
        counts["materialized"] += table.num_rows
        counts["materialization_invariant_failures"] += int(
            pc.sum(pc.invert(table["materialization_invariant_pass"])).as_py()
            or 0
        )
        duplicate_mask = table["exact_duplicate_target"]
        duplicates = int(pc.sum(duplicate_mask).as_py() or 0)
        counts["exact_duplicate_removed"] += duplicates
        keep_mask = pc.invert(duplicate_mask)
        if remove_ids:
            post_cap_mask = pc.is_in(
                table["target_id"],
                value_set=pa.array(
                    sorted(remove_ids),
                    type=table["target_id"].type,
                ),
            )
            keep_mask = pc.and_(keep_mask, pc.invert(post_cap_mask))
            removed = table.filter(post_cap_mask).select(
                ["target_id", "character_id", "pool"]
            )
            for row in removed.to_pylist():
                row["selection_exclusion_reason"] = (
                    "post_dedupe_character_share_cap_0.005"
                )
                removed_rows.append(row)
        kept = table.filter(keep_mask)
        output = FINAL_ROOT / path.name
        pq.write_table(
            kept,
            output,
            compression="zstd",
            compression_level=6,
            row_group_size=4096,
        )
        sha256s[path.name] = hashlib.sha256(output.read_bytes()).hexdigest()
        counts["final"] += kept.num_rows

        selected_columns = [
            "target_id",
            "conversation_id",
            "user_id",
            "character_id",
            "pool",
            "language_final_v2",
            "topic",
            "replay_mode",
            "memory_effective",
            "visible_user_turns_verified",
            "qwen35_serialized_tokens",
        ]
        for batch in kept.to_batches(max_chunksize=16384):
            for row in batch.select(selected_columns).to_pylist():
                target_id = str(row["target_id"])
                conversation_id = str(row["conversation_id"])
                user_id = str(row["user_id"])
                character_id = str(row["character_id"])
                unique_targets.add(target_id)
                unique_conversations.add(conversation_id)
                unique_users.add(user_id)
                unique_characters.add(character_id)
                pools[row["pool"]] += 1
                languages[str(row["language_final_v2"] or "unknown")] += 1
                topics[str(row["topic"] or "unknown")] += 1
                replay[row["replay_mode"]] += 1
                memory[
                    "effective_memory"
                    if row["memory_effective"]
                    else "no_effective_memory"
                ] += 1
                visible_turns.append(int(row["visible_user_turns_verified"]))
                tokens.append(int(row["qwen35_serialized_tokens"]))
                if row["pool"] == "neutral_flash":
                    neutral_users[user_id] += 1
                    neutral_pairs[(user_id, character_id)] += 1
                    neutral_conversations[conversation_id] += 1
                    neutral_characters[character_id] += 1

    counts["post_dedupe_character_cap_removed"] = len(remove_ids)
    neutral_total = sum(neutral_characters.values())
    neutral_maxima = {
        "user": max(neutral_users.values(), default=0),
        "user_character": max(neutral_pairs.values(), default=0),
        "conversation": max(neutral_conversations.values(), default=0),
        "character": max(neutral_characters.values(), default=0),
    }
    neutral_character_share = (
        neutral_maxima["character"] / neutral_total if neutral_total else 0.0
    )
    cap_invariants = (
        neutral_maxima["user"] <= USER_CAP
        and neutral_maxima["user_character"] <= USER_CHARACTER_CAP
        and neutral_maxima["conversation"] <= CONVERSATION_CAP
        and neutral_character_share <= CHARACTER_SHARE_CAP + 1e-12
    )
    if counts["materialization_invariant_failures"]:
        raise RuntimeError("materialization invariants failed")
    if counts["final"] != len(unique_targets):
        raise RuntimeError("final target ids are not unique")
    if not cap_invariants:
        raise RuntimeError(
            "final post-dedupe concentration cap invariant failed"
        )

    excluded_path = ROOT / "06_pre_mds/post_dedupe_excluded.parquet"
    excluded_schema = pa.schema(
        [
            pa.field("target_id", pa.string()),
            pa.field("character_id", pa.string()),
            pa.field("pool", pa.string()),
            pa.field("selection_exclusion_reason", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(removed_rows, schema=excluded_schema),
        excluded_path,
        compression="zstd",
        compression_level=6,
    )

    report = {
        "schema_version": "lusy-pre-mds-materialized-v2.1",
        "counts": dict(counts),
        "unique": {
            "targets": len(unique_targets),
            "conversations": len(unique_conversations),
            "users": len(unique_users),
            "characters": len(unique_characters),
        },
        "pool": dict(pools),
        "language": dict(languages.most_common()),
        "topic": dict(topics),
        "replay_mode": dict(replay),
        "memory": dict(memory),
        "visible_user_turns": quantiles(visible_turns),
        "qwen35_serialized_tokens": quantiles(tokens),
        "post_dedupe_caps": {
            "user": USER_CAP,
            "user_character": USER_CHARACTER_CAP,
            "conversation": CONVERSATION_CAP,
            "character_share": CHARACTER_SHARE_CAP,
            "final_character_absolute_cap": final_character_cap,
            "iterations": cap_history,
            "neutral_observed_maxima": neutral_maxima,
            "neutral_max_character_share": neutral_character_share,
            "invariants_pass": cap_invariants,
        },
        "parts": len(source_files),
        "all_invariants_pass": True,
        "sha256": sha256s,
    }
    manifest = ROOT / "00_manifest/PRE_MDS_REPORT.json"
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(".sha256").write_text(
        f"{digest}  {manifest.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
