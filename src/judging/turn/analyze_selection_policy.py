#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


ROOT = pathlib.Path("/home/gyr/lusychat-training")
SUMMARY_PATH = (
    ROOT
    / "judge-deepinfra-glm47-turn-v47-softjson-direct64-a-all-aggregated-20260729"
    / "artifacts/dialogue_summary.parquet"
)
ELITE_PATH = ROOT / "judge-dialogue-selection-v1/02_tiers/elite_A.parquet"
CHARACTER_PATH = ROOT / "lusy-chat-4m-v2/01_canonical/characters.parquet"
OUTPUT_PATH = (
    ROOT
    / "judge-deepinfra-glm47-turn-v47-softjson-direct64-a-all-aggregated-20260729"
    / "artifacts/selection_policy_analysis.json"
)

DIMENSIONS = [
    "character_fidelity",
    "context_coherence",
    "current_turn_fit",
    "relationship_continuity",
    "scene_progression",
    "language_roleplay_quality",
]
WEIGHTS = {
    "character_fidelity": 0.30,
    "context_coherence": 0.20,
    "current_turn_fit": 0.15,
    "relationship_continuity": 0.15,
    "scene_progression": 0.10,
    "language_roleplay_quality": 0.10,
}
SEVERE_FLAGS = [
    "critical_ai_identity_break_count",
    "critical_character_identity_break_count",
    "critical_context_break_count",
    "critical_relationship_reset_count",
    "critical_wrong_language_count",
    "critical_meta_refusal_count",
]


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    if isinstance(value, tuple):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def quantiles(series: pd.Series) -> dict[str, float]:
    values = series.dropna()
    points = [0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 1]
    result = values.quantile(points)
    return {str(point): round(float(result.loc[point]), 6) for point in points}


def group_summary(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for value, group in frame.groupby(column, dropna=False):
        rows.append(
            {
                "value": "unknown" if pd.isna(value) else str(value),
                "dialogues": len(group),
                "turns": int(group["total_turns"].sum()),
                "hard_eligible": int(group["hard_eligible"].sum()),
                "hard_eligible_rate": round(
                    float(group["hard_eligible"].mean()), 6
                ),
                "quality_mean": round(
                    float(group["quality_rank_score"].mean()), 6
                ),
                "quality_p25": round(
                    float(group["quality_rank_score"].quantile(0.25)), 6
                ),
                "quality_p50": round(
                    float(group["quality_rank_score"].quantile(0.50)), 6
                ),
                "quality_p75": round(
                    float(group["quality_rank_score"].quantile(0.75)), 6
                ),
                "nsfw_2plus_turn_rate": round(
                    float(group["nsfw_2plus_turns"].sum())
                    / max(1, int(group["strict_valid_turns"].sum())),
                    6,
                ),
            }
        )
    rows.sort(key=lambda row: (-row["dialogues"], row["value"]))
    return rows


def normalize_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "null", "unknown", "other"}:
        return "unknown_other"
    if any(term in text for term in ["nonbinary", "non-binary", "genderfluid"]):
        return "nonbinary"
    if any(term in text for term in ["multiple", "group", "various", "all"]):
        return "group_multiple"
    if any(term in text for term in ["female", "woman", "girl"]):
        return "female"
    if any(term in text for term in ["male", "man", "boy"]):
        return "male"
    return "unknown_other"


def main() -> None:
    dialogue = pd.read_parquet(SUMMARY_PATH)
    elite_columns = [
        "dialogue_id",
        "character_id",
        "selection_class",
        "contains_core",
        "replay_mode",
        "scoreable_turns",
        "final_quality_score",
        "core_score_mean",
        "quality_floor",
        "nsfw_max_level",
        "nsfw_prevalence",
        "mmr_score",
        "max_similarity_to_previous",
        "cluster_uid",
    ]
    elite = pd.read_parquet(ELITE_PATH, columns=elite_columns)
    character_columns = [
        "character_id",
        "gender",
        "age",
        "tags",
        "nsfw",
        "style",
        "is_origin",
        "station",
    ]
    characters = pd.read_parquet(
        CHARACTER_PATH, columns=character_columns
    ).drop_duplicates("character_id", keep="last")
    frame = dialogue.merge(elite, on="dialogue_id", how="left", validate="1:1")
    frame = frame.merge(
        characters, on="character_id", how="left", validate="m:1"
    )
    if frame["character_id"].isna().any():
        raise RuntimeError("missing character_id after Elite A join")

    frame["weighted_dimension_mean"] = sum(
        frame[f"{dimension}_mean"] * weight
        for dimension, weight in WEIGHTS.items()
    )
    frame["dimension_low_count"] = sum(
        frame[f"{dimension}_score_0_count"]
        + frame[f"{dimension}_score_1_count"]
        for dimension in DIMENSIONS
    )
    frame["dimension_zero_count"] = sum(
        frame[f"{dimension}_score_0_count"] for dimension in DIMENSIONS
    )
    frame["dimension_low_rate"] = frame["dimension_low_count"] / (
        frame["strict_valid_turns"].clip(lower=1) * len(DIMENSIONS)
    )
    frame["severe_flag_count"] = frame[SEVERE_FLAGS].sum(axis=1)
    frame["severe_flag_rate"] = frame["severe_flag_count"] / frame[
        "strict_valid_turns"
    ].clip(lower=1)
    frame["format_collapse_rate"] = frame[
        "critical_format_collapse_count"
    ] / frame["strict_valid_turns"].clip(lower=1)
    frame["major_repetition_count"] = (
        frame["semantic_repetition_major_count"]
        + frame["semantic_repetition_critical_count"]
    )
    frame["major_repetition_rate"] = frame[
        "major_repetition_count"
    ] / frame["strict_valid_turns"].clip(lower=1)
    frame["quality_rank_score"] = (
        frame["weighted_dimension_mean"]
        - 0.60 * frame["severe_flag_rate"]
        - 0.35 * frame["major_repetition_rate"]
        - 0.15 * frame["format_collapse_rate"]
    )
    frame["hard_eligible"] = (
        (frame["strict_valid_fraction"] >= 0.90)
        & (frame["dimension_zero_count"] == 0)
        & (frame["critical_ai_identity_break_count"] == 0)
        & (frame["critical_wrong_language_count"] == 0)
        & (frame["critical_meta_refusal_count"] == 0)
        & (frame["major_repetition_count"] == 0)
        & (frame["severe_flag_rate"] <= 0.05)
        & (frame["weighted_dimension_mean"] >= 2.0)
    )

    frame["nsfw_2plus_turns"] = (
        frame["nsfw_level_2_count"]
        + frame["nsfw_level_3_count"]
        + frame["nsfw_level_4_count"]
    )
    frame["nsfw_3plus_turns"] = (
        frame["nsfw_level_3_count"] + frame["nsfw_level_4_count"]
    )
    frame["nsfw_2plus_ratio"] = frame["nsfw_2plus_turns"] / frame[
        "strict_valid_turns"
    ].clip(lower=1)
    frame["nsfw_3plus_ratio"] = frame["nsfw_3plus_turns"] / frame[
        "strict_valid_turns"
    ].clip(lower=1)
    frame["nsfw_dialogue_band"] = np.select(
        [
            frame["nsfw_level_4_count"] > 0,
            frame["nsfw_3plus_ratio"] >= 0.30,
            frame["nsfw_2plus_ratio"] >= 0.30,
            frame["nsfw_2plus_turns"] > 0,
            frame["nsfw_level_1_count"] > 0,
        ],
        [
            "L4_present",
            "L3_heavy",
            "L2_heavy",
            "L2_3_mixed",
            "romance_suggestive_only",
        ],
        default="clean_L0_only",
    )
    frame["gender_group"] = frame["gender"].map(normalize_gender)

    tag_dialogue_counts = Counter()
    tag_characters: dict[str, set[str]] = {}
    for character_id, tags in (
        frame[["character_id", "tags"]]
        .drop_duplicates("character_id")
        .itertuples(index=False)
    ):
        values = [
            item.strip()
            for item in str(tags or "").split(",")
            if item.strip()
        ]
        for tag in values:
            tag_characters.setdefault(tag, set()).add(str(character_id))
    for tags in frame["tags"]:
        values = [
            item.strip()
            for item in str(tags or "").split(",")
            if item.strip()
        ]
        tag_dialogue_counts.update(set(values))
    top_tags = [
        {
            "tag": tag,
            "dialogues": count,
            "characters": len(tag_characters.get(tag, set())),
        }
        for tag, count in tag_dialogue_counts.most_common(80)
    ]

    char_counts = frame.groupby("character_id").size().sort_values(
        ascending=False
    )
    retention_scenarios = {}
    eligible = frame[frame["hard_eligible"]].sort_values(
        ["quality_rank_score", "strict_valid_fraction", "dialogue_id"],
        ascending=[False, False, True],
    )
    fallback = frame[~frame["hard_eligible"]].sort_values(
        ["quality_rank_score", "strict_valid_fraction", "dialogue_id"],
        ascending=[False, False, True],
    )
    ordered = pd.concat([eligible, fallback], ignore_index=True)
    for fraction in [0.30, 0.40, 0.50, 0.60, 0.70]:
        target = round(len(frame) * fraction)
        selected = ordered.head(target)
        retention_scenarios[str(fraction)] = {
            "target_dialogues": target,
            "turns": int(selected["total_turns"].sum()),
            "hard_eligible_selected": int(selected["hard_eligible"].sum()),
            "hard_eligible_share": round(
                float(selected["hard_eligible"].mean()), 6
            ),
            "quality_floor": round(
                float(selected["quality_rank_score"].min()), 6
            ),
            "quality_mean": round(
                float(selected["quality_rank_score"].mean()), 6
            ),
            "nsfw_2plus_turn_rate": round(
                float(selected["nsfw_2plus_turns"].sum())
                / max(1, int(selected["strict_valid_turns"].sum())),
                6,
            ),
            "languages": {
                str(key): int(value)
                for key, value in selected["language"]
                .value_counts(dropna=False)
                .items()
            },
            "nsfw_bands": {
                str(key): int(value)
                for key, value in selected["nsfw_dialogue_band"]
                .value_counts()
                .items()
            },
            "unique_characters": int(selected["character_id"].nunique()),
        }

    old_new_correlation = frame[
        ["final_quality_score", "quality_rank_score"]
    ].corr(method="spearman").iloc[0, 1]
    report = {
        "rows": len(frame),
        "turns": int(frame["total_turns"].sum()),
        "unique_characters": int(frame["character_id"].nunique()),
        "join_missing": {
            "elite": int(frame["selection_class"].isna().sum()),
            "character": int(frame["gender"].isna().sum()),
        },
        "quality": {
            "weighted_dimension_mean_quantiles": quantiles(
                frame["weighted_dimension_mean"]
            ),
            "quality_rank_score_quantiles": quantiles(
                frame["quality_rank_score"]
            ),
            "strict_valid_fraction_quantiles": quantiles(
                frame["strict_valid_fraction"]
            ),
            "dimension_low_rate_quantiles": quantiles(
                frame["dimension_low_rate"]
            ),
            "severe_flag_rate_quantiles": quantiles(
                frame["severe_flag_rate"]
            ),
            "total_turns_quantiles": quantiles(frame["total_turns"]),
            "hard_eligible": int(frame["hard_eligible"].sum()),
            "hard_eligible_rate": round(
                float(frame["hard_eligible"].mean()), 6
            ),
            "old_new_spearman": round(float(old_new_correlation), 6),
        },
        "language": group_summary(frame, "language"),
        "topic": group_summary(frame, "topic"),
        "nsfw_dialogue_band": group_summary(
            frame, "nsfw_dialogue_band"
        ),
        "gender_group": group_summary(frame, "gender_group"),
        "raw_gender_top": [
            {"value": str(key), "dialogues": int(value)}
            for key, value in frame["gender"]
            .fillna("unknown")
            .value_counts()
            .head(30)
            .items()
        ],
        "top_tags": top_tags,
        "character_concentration": {
            "unique_characters": int(len(char_counts)),
            "dialogues_per_character_quantiles": quantiles(char_counts),
            "characters_with_1_dialogue": int((char_counts == 1).sum()),
            "characters_with_10plus_dialogues": int(
                (char_counts >= 10).sum()
            ),
            "characters_with_50plus_dialogues": int(
                (char_counts >= 50).sum()
            ),
            "top_30": [
                {
                    "character_id": str(character_id),
                    "dialogues": int(count),
                }
                for character_id, count in char_counts.head(30).items()
            ],
        },
        "retention_scenarios_quality_only": retention_scenarios,
    }
    OUTPUT_PATH.write_text(
        json.dumps(native(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(native(report), ensure_ascii=False))


if __name__ == "__main__":
    main()
