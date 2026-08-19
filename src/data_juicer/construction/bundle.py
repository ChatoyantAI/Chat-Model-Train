"""Conversation-turn construction helpers."""
from __future__ import annotations

import polars as pl


def build_turns(messages: pl.DataFrame, mutations: pl.DataFrame | None = None) -> pl.DataFrame:
    ordered = messages.sort(["conversation_id", "increase_id"])
    current_user = (pl.col("role") == 2) & pl.col("is_current_version")
    current_assistant = (pl.col("role") == 1) & pl.col("is_current_version")
    trainable_user = current_user & pl.col("is_text_trainable")
    trainable_assistant = current_assistant & pl.col("is_text_trainable")
    assistant_candidate = pl.struct(
        "increase_id", "content", "content_chars", "status", "is_current_version",
        "is_text_trainable", "evaluation", "regenerated_times", "model", "model_id",
        "teacher_family", "create_time",
    ).filter(pl.col("role") == 1)
    user_version = pl.struct(
        "increase_id", "content", "content_chars", "status", "is_current_version",
        "is_text_trainable", "evaluation", "create_time",
    ).filter(pl.col("role") == 2)
    turns = (
        ordered.group_by(["conversation_id", "logical_turn_id"], maintain_order=True)
        .agg(
            pl.col("increase_id").min().alias("turn_order"),
            pl.len().alias("physical_message_rows"),
            (pl.col("role") == 2).sum().alias("user_version_count"),
            (pl.col("role") == 1).sum().alias("assistant_candidate_count"),
            current_user.sum().alias("current_user_count"),
            current_assistant.sum().alias("current_assistant_count"),
            trainable_user.sum().alias("trainable_current_user_count"),
            trainable_assistant.sum().alias("trainable_current_assistant_count"),
            pl.col("increase_id").filter(current_user).last().alias("user_increase_id"),
            pl.col("content").filter(current_user).last().alias("user_content"),
            pl.col("content_chars").filter(current_user).last().alias("user_content_chars"),
            pl.col("language").filter(current_user).last().alias("user_language"),
            pl.col("create_time").filter(current_user).last().alias("user_create_time"),
            pl.col("user_id").filter(current_user).last().alias("source_user_id"),
            pl.col("increase_id").filter(current_assistant).last().alias("assistant_increase_id"),
            pl.col("content").filter(current_assistant).last().alias("assistant_content"),
            pl.col("content_chars").filter(current_assistant).last().alias("assistant_content_chars"),
            pl.col("language").filter(current_assistant).last().alias("assistant_language"),
            pl.col("create_time").filter(current_assistant).last().alias("assistant_create_time"),
            pl.col("evaluation").filter(current_assistant).last().alias("assistant_evaluation"),
            pl.col("model").filter(current_assistant).last().alias("assistant_model"),
            pl.col("model_id").filter(current_assistant).last().alias("assistant_model_id"),
            pl.col("teacher_family").filter(current_assistant).last().alias("assistant_teacher_family"),
            assistant_candidate.alias("assistant_candidates"),
            user_version.alias("user_versions"),
        )
        .with_columns(
            (
                (pl.col("current_user_count") == 1)
                & (pl.col("current_assistant_count") == 1)
                & (pl.col("trainable_current_user_count") == 1)
                & (pl.col("trainable_current_assistant_count") == 1)
            ).alias("turn_is_trainable"),
            pl.when(pl.col("current_user_count") == 0)
            .then(pl.lit("missing_current_user"))
            .when(pl.col("current_user_count") > 1)
            .then(pl.lit("multiple_current_users"))
            .when(pl.col("current_assistant_count") == 0)
            .then(pl.lit("missing_current_assistant"))
            .when(pl.col("current_assistant_count") > 1)
            .then(pl.lit("multiple_current_assistants"))
            .when(pl.col("trainable_current_user_count") == 0)
            .then(pl.lit("non_trainable_user"))
            .when(pl.col("trainable_current_assistant_count") == 0)
            .then(pl.lit("non_trainable_assistant"))
            .otherwise(None)
            .alias("quality_reason"),
        )
    )
    if mutations is None or mutations.height == 0:
        return turns
    mutation_events = (
        mutations.sort(["conversation_id", "message_id", "created_at", "id"])
        .group_by(["conversation_id", "message_id"], maintain_order=True)
        .agg(
            pl.struct(
                "id", "event_type", "target_role", "target_increase_id",
                "source_increase_id", "replacement_increase_id", "before_content",
                "after_content", "prompt_content_snapshot", "created_at", "user_id",
            ).alias("mutation_events")
        )
    )
    return turns.join(
        mutation_events,
        left_on=["conversation_id", "logical_turn_id"],
        right_on=["conversation_id", "message_id"],
        how="left",
    )
