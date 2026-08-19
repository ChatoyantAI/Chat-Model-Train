"""Normalize explicit feedback and derive conservative preference candidates."""
from __future__ import annotations

import polars as pl


def _nonempty(name: str) -> pl.Expr:
    return pl.col(name).is_not_null() & (pl.col(name).str.strip_chars().str.len_chars() > 0)


def build_preference_candidates(mutations: pl.DataFrame, messages: pl.DataFrame) -> pl.DataFrame:
    mutations = mutations.with_columns(
        pl.col(
            "target_increase_id", "source_increase_id", "replacement_increase_id"
        ).cast(pl.Int64, strict=False),
        pl.col(
            "before_content", "after_content", "prompt_content_snapshot"
        ).cast(pl.String, strict=False),
    )
    source = messages.select(
        pl.col("increase_id").alias("source_increase_id"),
        pl.col("role").alias("rejected_role"),
        pl.col("status").alias("rejected_status"),
        pl.col("content").alias("rejected_content"),
        pl.col("evaluation").alias("rejected_evaluation"),
        pl.col("model").alias("rejected_model"),
        pl.col("teacher_family").alias("rejected_teacher_family"),
    )
    current = (
        messages.filter((pl.col("role") == 1) & (pl.col("status") == 1))
        .sort("increase_id")
        .unique(["conversation_id", "logical_turn_id"], keep="last")
        .select(
            "conversation_id",
            "logical_turn_id",
            pl.col("increase_id").alias("chosen_increase_id"),
            pl.col("content").alias("chosen_content"),
            pl.col("evaluation").alias("chosen_evaluation"),
            pl.col("model").alias("chosen_model"),
            pl.col("teacher_family").alias("chosen_teacher_family"),
        )
    )
    target = messages.select(
        pl.col("increase_id").alias("target_increase_id"),
        pl.col("role").alias("target_message_role"),
        pl.col("status").alias("target_status"),
        pl.col("evaluation").alias("target_evaluation"),
        pl.col("model").alias("target_model"),
        pl.col("teacher_family").alias("target_teacher_family"),
    )

    regen = (
        mutations.filter((pl.col("event_type") == "regenerate") & (pl.col("target_role") == 1))
        .join(source, on="source_increase_id", how="left")
        .join(
            current,
            left_on=["conversation_id", "message_id"],
            right_on=["conversation_id", "logical_turn_id"],
            how="left",
        )
        .with_columns(
            pl.concat_str([pl.lit("regenerate:"), pl.col("id").cast(pl.String)]).alias("candidate_id"),
            pl.lit("regenerate").alias("candidate_type"),
            pl.lit("mutation_snapshot").alias("prompt_source"),
            pl.col("prompt_content_snapshot").alias("prompt_snapshot"),
            pl.col("source_increase_id").alias("rejected_increase_id"),
            pl.col("replacement_increase_id").alias("event_replacement_increase_id"),
            pl.col("chosen_teacher_family").alias("teacher_family"),
        )
        .with_columns(
            pl.when(pl.col("source_increase_id").is_null())
            .then(pl.lit("missing_source_reference"))
            .when(pl.col("rejected_status").is_null())
            .then(pl.lit("source_not_found"))
            .when(pl.col("chosen_increase_id").is_null())
            .then(pl.lit("current_answer_not_found"))
            .when(pl.col("rejected_role") != 1)
            .then(pl.lit("source_not_assistant"))
            .when(pl.col("rejected_status") != 0)
            .then(pl.lit("rejected_not_deleted"))
            .when(pl.col("rejected_evaluation") == 1)
            .then(pl.lit("rejected_was_liked"))
            .when(~_nonempty("rejected_content"))
            .then(pl.lit("empty_rejected"))
            .when(~_nonempty("chosen_content"))
            .then(pl.lit("empty_chosen"))
            .when(pl.col("rejected_content").str.strip_chars() == pl.col("chosen_content").str.strip_chars())
            .then(pl.lit("identical_content"))
            .when(~_nonempty("prompt_snapshot"))
            .then(pl.lit("missing_prompt_snapshot"))
            .otherwise(None)
            .alias("ineligibility_reason")
        )
        .with_columns(
            pl.col("ineligibility_reason").is_null().alias("eligible_for_preference"),
            pl.when((pl.col("rejected_evaluation") == 2) | (pl.col("chosen_evaluation") == 1)).then(1.0).otherwise(0.55).alias("preference_strength"),
        )
    )

    edit = (
        mutations.filter((pl.col("event_type") == "edit") & (pl.col("target_role") == 1))
        .join(target, on="target_increase_id", how="left")
        .with_columns(
            pl.concat_str([pl.lit("assistant_edit:"), pl.col("id").cast(pl.String)]).alias("candidate_id"),
            pl.lit("assistant_edit").alias("candidate_type"),
            pl.lit("reconstruct_history").alias("prompt_source"),
            pl.lit(None, dtype=pl.String).alias("prompt_snapshot"),
            pl.col("before_content").alias("rejected_content"),
            pl.col("after_content").alias("chosen_content"),
            pl.lit(None, dtype=pl.Int64).alias("rejected_increase_id"),
            pl.col("target_increase_id").alias("chosen_increase_id"),
            pl.col("replacement_increase_id").alias("event_replacement_increase_id"),
            pl.col("target_evaluation").alias("chosen_evaluation"),
            pl.lit(None, dtype=pl.Int16).alias("rejected_evaluation"),
            pl.col("target_model").alias("chosen_model"),
            pl.lit(None, dtype=pl.String).alias("rejected_model"),
            pl.col("target_teacher_family").alias("chosen_teacher_family"),
            pl.lit(None, dtype=pl.String).alias("rejected_teacher_family"),
            pl.col("target_teacher_family").alias("teacher_family"),
        )
        .with_columns(
            pl.when(pl.col("target_increase_id").is_null())
            .then(pl.lit("missing_target_reference"))
            .when(pl.col("target_status").is_null())
            .then(pl.lit("target_not_found"))
            .when(pl.col("target_message_role") != 1)
            .then(pl.lit("target_not_assistant"))
            .when(pl.col("target_status") != 1)
            .then(pl.lit("edited_answer_later_rejected"))
            .when(pl.col("target_evaluation") == 2)
            .then(pl.lit("edited_answer_disliked"))
            .when(~_nonempty("rejected_content"))
            .then(pl.lit("empty_rejected"))
            .when(~_nonempty("chosen_content"))
            .then(pl.lit("empty_chosen"))
            .when(pl.col("rejected_content").str.strip_chars() == pl.col("chosen_content").str.strip_chars())
            .then(pl.lit("identical_content"))
            .otherwise(None)
            .alias("ineligibility_reason")
        )
        .with_columns(
            pl.col("ineligibility_reason").is_null().alias("eligible_for_preference"),
            pl.when(pl.col("target_evaluation") == 1).then(1.0).otherwise(0.8).alias("preference_strength"),
        )
    )

    columns = [
        "candidate_id", "candidate_type", "id", "conversation_id", "message_id", "user_id",
        "created_at", "prompt_source", "prompt_snapshot", "target_increase_id",
        "source_increase_id", "event_replacement_increase_id", "rejected_increase_id",
        "chosen_increase_id", "rejected_content", "chosen_content", "rejected_evaluation",
        "chosen_evaluation", "rejected_model", "chosen_model", "rejected_teacher_family",
        "chosen_teacher_family", "teacher_family", "eligible_for_preference",
        "ineligibility_reason", "preference_strength",
    ]
    return pl.concat([regen.select(columns), edit.select(columns)], how="vertical_relaxed")
