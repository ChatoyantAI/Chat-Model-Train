"""Build privacy-reduced, bucketed normalized message Parquet."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl


def _bucket_expr() -> pl.Expr:
    prefix = pl.col("conversation_id").str.slice(0, 2).str.to_integer(base=16, strict=False)
    fallback = (pl.col("conversation_id").hash(seed=20260720) % 256).cast(pl.UInt16)
    return prefix.fill_null(fallback).cast(pl.UInt16).alias("conversation_bucket")


def _teacher_family_expr() -> pl.Expr:
    model = pl.col("model").fill_null("").str.to_lowercase()
    family = (
        pl.when(model.str.contains("deepseek-v4-flash"))
        .then(pl.lit("deepseek_v4_flash"))
        .when(model.str.contains("deepseek-v4-pro"))
        .then(pl.lit("deepseek_v4_pro"))
        .when(model.str.contains("deepseek-v3.2") | model.str.contains("deepseek v3.2"))
        .then(pl.lit("deepseek_v3_2"))
        .when(model.str.contains("deepseek-v3-0324") | model.str.contains("deepseek-v3-0324"))
        .then(pl.lit("deepseek_v3_0324"))
        .when(model.str.contains("lunaris"))
        .then(pl.lit("lunaris"))
        .when(model.str.contains("glm-4.6"))
        .then(pl.lit("glm_4_6"))
        .when(model.str.contains("glm-5.1"))
        .then(pl.lit("glm_5_1"))
        .when(model.str.contains("claude"))
        .then(pl.lit("claude"))
        .when(model.str.contains("gemini"))
        .then(pl.lit("gemini"))
        .when(model.str.contains("owl-alpha"))
        .then(pl.lit("owl_alpha"))
        .otherwise(pl.lit("other"))
    )
    return pl.when((pl.col("role") == 1) & pl.col("model").is_not_null()).then(family).otherwise(None).alias("teacher_family")


def normalize_message_frame(frame: pl.LazyFrame) -> pl.LazyFrame:
    content_ok = pl.col("content").is_not_null() & (pl.col("content").str.strip_chars().str.len_chars() > 0)
    role_ok = pl.col("role").is_in([1, 2])
    media_ok = (pl.col("media_type").fill_null(0) == 0) & (
        pl.col("attached_image_url").fill_null("").str.len_chars() == 0
    )
    is_alive = pl.col("status") == 1
    trainable = is_alive & role_ok & content_ok & media_ok
    reason = (
        pl.when(~is_alive)
        .then(pl.lit("deleted_version"))
        .when(~role_ok)
        .then(pl.lit("unsupported_role"))
        .when(~content_ok)
        .then(pl.lit("empty_content"))
        .when(~media_ok)
        .then(pl.lit("non_text_media"))
        .otherwise(None)
        .alias("quarantine_reason")
    )
    return frame.select(
        "increase_id",
        "id",
        pl.col("id").alias("logical_turn_id"),
        "conversation_id",
        _bucket_expr(),
        "user_id",
        "role",
        pl.when(pl.col("role") == 1).then(pl.lit("assistant")).when(pl.col("role") == 2).then(pl.lit("user")).otherwise(pl.lit("unknown")).alias("role_name"),
        "content",
        pl.col("content").str.len_chars().alias("content_chars"),
        "language",
        "create_time",
        "update_time",
        "status",
        is_alive.alias("is_alive"),
        is_alive.alias("is_current_version"),
        "evaluation",
        "regenerated_times",
        "media_type",
        "media_id",
        (pl.col("attached_image_url").fill_null("").str.len_chars() > 0).alias("has_attached_image"),
        "is_locked",
        "model",
        "model_id",
        _teacher_family_expr(),
        "inner_thought",
        (pl.col("inner_thought_content").fill_null("").str.len_chars() > 0).alias("has_inner_thought_content"),
        "inner_thought_viewed",
        "inner_thought_placeholder",
        trainable.alias("is_text_trainable"),
        reason,
    )


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def normalize_messages(source: str, output: Path, receipt: Path) -> dict:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists() or receipt.exists():
        raise FileExistsError("refusing to overwrite normalized messages or receipt")
    normalized = normalize_message_frame(pl.scan_parquet(source))
    normalized.sink_parquet(
        pl.PartitionBy(
            partial,
            key="conversation_bucket",
            include_key=True,
            max_rows_per_file=5_000_000,
            approximate_bytes_per_file=1_000_000_000,
        ),
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=100_000,
        mkdir=True,
        engine="streaming",
    )
    os.replace(partial, output)
    scan = pl.scan_parquet(str(output / "**/*.parquet"))
    counts = scan.group_by("quarantine_reason").agg(pl.len().alias("rows")).sort("rows", descending=True).collect(engine="streaming")
    summary = scan.select(pl.len().alias("rows"), pl.col("conversation_id").n_unique().alias("conversations"), pl.col("logical_turn_id").n_unique().alias("logical_turn_ids"), pl.col("is_text_trainable").sum().alias("text_trainable_rows")).collect(engine="streaming").row(0, named=True)
    result = {**summary, "files": len(list(output.rglob("*.parquet"))), "quarantine_reason": counts.to_dicts(), "source": source, "output": str(output)}
    _atomic_json(receipt, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(normalize_messages(args.source, args.output, args.receipt), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
