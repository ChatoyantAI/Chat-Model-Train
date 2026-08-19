#!/usr/bin/env python3
"""Select all protected core and cap neutral concentration without a global budget."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/gyr/lusychat-training/pre-mds-context-v2")
PLAN_GLOB = str(ROOT / "02_context_plan/parts/part-*.parquet")
LANG_GLOB = str(ROOT / "03_language_alignment/features/part-*.parquet")
OUTPUT = ROOT / "04_cap_selection"

USER_CAP = 60
USER_CHARACTER_CAP = 12
CONVERSATION_CAP = 6
CHARACTER_SHARE_CAP = 0.005


def rank_and_head(
    frame: pl.DataFrame,
    group: list[str],
    cap: int,
) -> pl.DataFrame:
    return (
        frame.sort(
            group
            + [
                "continued",
                "paid_priority",
                "selection_hash_u64",
                "target_id",
            ],
            descending=[False] * len(group) + [True, True, False, False],
        )
        .group_by(group, maintain_order=True)
        .head(cap)
    )


def removed(
    before: pl.DataFrame,
    after: pl.DataFrame,
    reason: str,
) -> pl.DataFrame:
    return (
        before.select("target_id")
        .join(after.select("target_id"), on="target_id", how="anti")
        .with_columns(pl.lit(reason).alias("selection_exclusion_reason"))
    )


def maxima(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty():
        return {
            "user": 0,
            "user_character": 0,
            "conversation": 0,
            "character": 0,
        }
    return {
        "user": int(frame.group_by("user_id").len()["len"].max()),
        "user_character": int(
            frame.group_by(["user_id", "character_id"]).len()["len"].max()
        ),
        "conversation": int(
            frame.group_by("conversation_id").len()["len"].max()
        ),
        "character": int(
            frame.group_by("character_id").len()["len"].max()
        ),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = pl.scan_parquet(PLAN_GLOB)
    language = pl.scan_parquet(LANG_GLOB).select(
        [
            "target_id",
            pl.col("user_language").alias("user_language_detected"),
            pl.col("assistant_language").alias("assistant_language_detected"),
            "user_confidence",
            "assistant_confidence",
            "user_margin",
            "assistant_margin",
            pl.col("status").alias("language_alignment_status"),
            pl.col("discard").alias("language_discard"),
            pl.col("reason").alias("language_alignment_reason"),
        ]
    )
    minimal_columns = [
        "target_id",
        "conversation_id",
        "user_id",
        "character_id",
        "conversation_bucket",
        "source_kind",
        "pool",
        "model_tier",
        "core_protected",
        "continued",
        "paid_at_target",
        "selection_hash_u64",
        "topic",
        "replay_mode",
        "memory_effective",
        "visible_user_turns",
        "prompt_replay_chars_without_system",
        "language_final",
        "replay_invariant_pass",
        "replay_error",
    ]
    data = (
        plan.select(minimal_columns)
        .join(language, on="target_id", how="inner")
        .collect(engine="streaming")
    )
    if data.height != data["target_id"].n_unique():
        raise RuntimeError("target_id is not unique before cap selection")

    invalid = data.filter(
        (~pl.col("replay_invariant_pass")) | pl.col("language_discard")
    ).with_columns(
        pl.when(~pl.col("replay_invariant_pass"))
        .then(pl.lit("replay_invariant_failed"))
        .otherwise(pl.lit("language_mismatch"))
        .alias("selection_exclusion_reason")
    )
    valid = data.filter(
        pl.col("replay_invariant_pass") & (~pl.col("language_discard"))
    ).with_columns(
        (pl.col("paid_at_target") == "paid")
        .cast(pl.Int8)
        .alias("paid_priority")
    )
    core = valid.filter(pl.col("core_protected"))
    neutral = valid.filter(pl.col("pool") == "neutral_flash")
    reserve = valid.filter(
        (~pl.col("core_protected")) & (pl.col("pool") != "neutral_flash")
    ).with_columns(
        pl.lit("reserve_model_tier").alias("selection_exclusion_reason")
    )

    after_conversation = rank_and_head(
        neutral,
        ["conversation_id"],
        CONVERSATION_CAP,
    )
    removed_conversation = removed(
        neutral,
        after_conversation,
        "conversation_cap_6",
    )

    after_pair = rank_and_head(
        after_conversation,
        ["user_id", "character_id"],
        USER_CHARACTER_CAP,
    )
    removed_pair = removed(
        after_conversation,
        after_pair,
        "user_character_cap_12",
    )

    after_user = rank_and_head(
        after_pair,
        ["user_id"],
        USER_CAP,
    )
    removed_user = removed(
        after_pair,
        after_user,
        "user_cap_60",
    )

    character_cap_history: list[dict[str, int]] = []
    character_cap = max(
        1,
        math.floor(after_user.height * CHARACTER_SHARE_CAP),
    )
    while True:
        after_character = rank_and_head(
            after_user,
            ["character_id"],
            character_cap,
        )
        next_cap = max(
            1,
            math.floor(after_character.height * CHARACTER_SHARE_CAP),
        )
        character_cap_history.append(
            {
                "input_cap": character_cap,
                "selected": after_character.height,
                "next_cap": next_cap,
            }
        )
        if next_cap >= character_cap:
            break
        character_cap = next_cap
    removed_character = removed(
        after_user,
        after_character,
        f"character_share_cap_{CHARACTER_SHARE_CAP}",
    )

    core_ids = core.select("target_id").with_columns(
        pl.lit("protected_core").alias("selection_reason")
    )
    neutral_ids = after_character.select("target_id").with_columns(
        pl.lit("neutral_cap_pass").alias("selection_reason")
    )
    selected_ids = pl.concat([core_ids, neutral_ids])

    full_selected = (
        pl.scan_parquet(PLAN_GLOB)
        .join(language, on="target_id", how="inner")
        .join(selected_ids.lazy(), on="target_id", how="inner")
        .with_columns(
            pl.when(
                (pl.col("assistant_confidence") >= 0.65)
                & (pl.col("assistant_margin") >= 0.15)
                & (pl.col("assistant_language_detected") != "unknown")
            )
            .then(pl.col("assistant_language_detected"))
            .otherwise(pl.col("language_final"))
            .alias("language_final_v2")
        )
        .collect(engine="streaming")
    )
    selected_path = OUTPUT / "selected_targets.parquet"
    full_selected.write_parquet(
        selected_path,
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=32768,
    )

    part_root = OUTPUT / "selected_parts"
    part_root.mkdir(parents=True, exist_ok=True)
    for bucket in range(256):
        part = full_selected.filter(pl.col("conversation_bucket") == bucket)
        part.write_parquet(
            part_root / f"part-{bucket:03d}.parquet",
            compression="zstd",
            compression_level=6,
            statistics=True,
            row_group_size=16384,
        )

    exclusions = pl.concat(
        [
            invalid.select(
                ["target_id", "selection_exclusion_reason"]
            ),
            reserve.select(
                ["target_id", "selection_exclusion_reason"]
            ),
            removed_conversation,
            removed_pair,
            removed_user,
            removed_character,
        ],
        how="vertical_relaxed",
    )
    exclusions.write_parquet(
        OUTPUT / "excluded_targets.parquet",
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=32768,
    )

    neutral_maxima = maxima(after_character)
    actual_character_share = (
        neutral_maxima["character"] / after_character.height
        if after_character.height
        else 0.0
    )
    if neutral_maxima["conversation"] > CONVERSATION_CAP:
        raise RuntimeError("conversation cap invariant failed")
    if neutral_maxima["user_character"] > USER_CHARACTER_CAP:
        raise RuntimeError("user-character cap invariant failed")
    if neutral_maxima["user"] > USER_CAP:
        raise RuntimeError("user cap invariant failed")
    if actual_character_share > CHARACTER_SHARE_CAP + 1e-12:
        raise RuntimeError("character share cap invariant failed")
    if full_selected.height + exclusions.height != data.height:
        raise RuntimeError("selection accounting invariant failed")

    def counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
        return {
            str(row[column]): int(row["len"])
            for row in frame.group_by(column)
            .len()
            .sort("len", descending=True)
            .to_dicts()
        }

    report = {
        "schema_version": "lusy-diversity-cap-selection-v2",
        "global_budget_applied": False,
        "n_core_multiplier_applied": False,
        "input": data.height,
        "invalid_or_language_discard": invalid.height,
        "valid": valid.height,
        "protected_core": core.height,
        "neutral_before_caps": neutral.height,
        "reserve_model_tier": reserve.height,
        "neutral_after_conversation_cap": after_conversation.height,
        "neutral_after_user_character_cap": after_pair.height,
        "neutral_after_user_cap": after_user.height,
        "neutral_after_character_cap": after_character.height,
        "selected_total": full_selected.height,
        "excluded_total": exclusions.height,
        "removed": {
            "conversation_cap_6": removed_conversation.height,
            "user_character_cap_12": removed_pair.height,
            "user_cap_60": removed_user.height,
            "character_share_cap": removed_character.height,
        },
        "caps": {
            "user": USER_CAP,
            "user_character": USER_CHARACTER_CAP,
            "conversation": CONVERSATION_CAP,
            "character_share": CHARACTER_SHARE_CAP,
            "final_character_absolute_cap": character_cap,
        },
        "character_cap_iterations": character_cap_history,
        "neutral_observed_maxima": neutral_maxima,
        "neutral_max_character_share": actual_character_share,
        "selected_pool": counts(full_selected, "pool"),
        "selected_language": counts(full_selected, "language_final_v2"),
        "selected_topic": counts(full_selected, "topic"),
        "selected_replay_mode": counts(full_selected, "replay_mode"),
        "accounting_pass": True,
        "cap_invariants_pass": True,
        "selection_order": [
            "conversation_cap_6",
            "user_character_cap_12",
            "user_cap_60",
            "iterative_character_share_cap_0.5pct",
        ],
        "rank_order": [
            "continued_desc",
            "paid_at_target_desc",
            "sha256_seeded_target_hash_asc",
        ],
    }
    manifest = ROOT / "00_manifest/CAP_SELECTION_REPORT.json"
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (manifest.with_suffix(".sha256")).write_text(
        f"{digest}  {manifest.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
