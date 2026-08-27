#!/usr/bin/env python3
"""Encode assistant and preceding-user turns into normalized BGE-M3 vectors."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from common import SCHEMA_VERSION, read_json, update_status, utc_now, write_json_atomic
from embedding_core import encode_to_array, load_model


FIELDS = {
    "assistant": "assistant_text",
    "user": "user_text",
}


def completed_manifest(
    manifest_path: Path,
    vector_path: Path,
    *,
    rows: int,
    dimension: int,
    model: str,
    revision: str,
    max_length: int,
) -> dict[str, Any] | None:
    manifest = read_json(manifest_path, {}) or {}
    if (
        vector_path.exists()
        and manifest.get("status") == "completed"
        and manifest.get("rows") == rows
        and manifest.get("dimension") == dimension
        and manifest.get("model") == model
        and manifest.get("revision") == revision
        and manifest.get("max_length") == max_length
    ):
        return manifest
    return None


def summarize_progress(output_root: Path) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    for field in FIELDS:
        manifests = []
        for path in sorted((output_root / f"{field}_embeddings" / "manifests").glob("bucket-*.json")):
            manifest = read_json(path, {}) or {}
            if manifest.get("status") == "completed":
                manifests.append(manifest)
        progress[field] = {
            "completed_shards": len(manifests),
            "rows": sum(int(manifest.get("rows", 0)) for manifest in manifests),
            "elapsed_seconds": sum(float(manifest.get("elapsed_seconds", 0.0)) for manifest in manifests),
            "content_tokens": sum(int(manifest.get("content_tokens", 0)) for manifest in manifests),
        }
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--fields", default="assistant,user")
    args = parser.parse_args()

    selected_fields = [value.strip() for value in args.fields.split(",") if value.strip()]
    if not selected_fields or any(field not in FIELDS for field in selected_fields):
        raise ValueError(f"fields must be selected from {sorted(FIELDS)}")

    shards = sorted(args.turn_dir.glob("bucket-*.parquet"))
    if len(shards) != 256:
        raise ValueError(f"expected 256 turn shards, found {len(shards)}")
    expected_turns = int((read_json(args.status_path, {}) or {}).get("expected_turns", 0))
    tokenizer, model = load_model(args.model, args.revision, args.device)
    dimension = int(model.config.hidden_size)
    update_status(
        args.status_path,
        status="running",
        stage="embedding",
        embedding_model=args.model,
        embedding_revision=args.revision,
        embedding_dimension=dimension,
        embedding_dtype="float16",
        embedding_max_length=args.max_length,
        embedding_batch_size=args.batch_size,
        embedding_fields=selected_fields,
        embedding_started_at=utc_now(),
    )

    wall_started = time.perf_counter()
    for shard in shards:
        bucket_name = shard.stem
        parquet_file = pq.ParquetFile(shard)
        rows = parquet_file.metadata.num_rows
        for field in selected_fields:
            text_key = FIELDS[field]
            field_root = args.output_root / f"{field}_embeddings"
            vector_path = field_root / f"{bucket_name}.f16.npy"
            manifest_path = field_root / "manifests" / f"{bucket_name}.json"
            if completed_manifest(
                manifest_path,
                vector_path,
                rows=rows,
                dimension=dimension,
                model=args.model,
                revision=args.revision,
                max_length=args.max_length,
            ):
                continue

            field_root.mkdir(parents=True, exist_ok=True)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            texts = pq.read_table(shard, columns=[text_key])[text_key].to_pylist()
            temporary = vector_path.with_suffix(vector_path.suffix + ".tmp")
            if temporary.exists():
                temporary.unlink()
            output = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=np.float16,
                shape=(rows, dimension),
            )
            stats = encode_to_array(
                model,
                tokenizer,
                texts,
                output,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            del output
            os.replace(temporary, vector_path)
            manifest = {
                "status": "completed",
                "schema_version": SCHEMA_VERSION,
                "field": field,
                "text_key": text_key,
                "source_shard": str(shard),
                "vector_path": str(vector_path),
                "rows": rows,
                "dimension": dimension,
                "dtype": "float16",
                "model": args.model,
                "revision": args.revision,
                "max_length": args.max_length,
                "configured_batch_size": args.batch_size,
                "effective_batch_size": stats.batch_size,
                "content_tokens": stats.content_tokens,
                "padded_tokens": stats.padded_tokens,
                "elapsed_seconds": stats.elapsed_seconds,
                "records_per_second": stats.records_per_second,
                "content_tokens_per_second": stats.content_tokens_per_second,
                "file_bytes": vector_path.stat().st_size,
                "completed_at": utc_now(),
            }
            write_json_atomic(manifest_path, manifest)
            progress = summarize_progress(args.output_root)
            completed_rows = sum(progress[field]["rows"] for field in selected_fields)
            expected_rows = expected_turns * len(selected_fields)
            elapsed = time.perf_counter() - wall_started
            rate = completed_rows / elapsed if elapsed and completed_rows else 0.0
            update_status(
                args.status_path,
                status="running",
                stage="embedding",
                embedding_progress=progress,
                embedding_completed_rows=completed_rows,
                embedding_expected_rows=expected_rows,
                embedding_session_records_per_second=rate,
                embedding_session_eta_seconds=(expected_rows - completed_rows) / rate if rate else None,
                embedding_last_completed={"bucket": bucket_name, "field": field},
            )

    progress = summarize_progress(args.output_root)
    completed_rows = sum(progress[field]["rows"] for field in selected_fields)
    expected_rows = expected_turns * len(selected_fields)
    if completed_rows != expected_rows:
        raise ValueError(f"embedding row mismatch: {completed_rows} != {expected_rows}")
    update_status(
        args.status_path,
        status="completed",
        stage="completed",
        embedding_progress=progress,
        embedding_completed_rows=completed_rows,
        embedding_expected_rows=expected_rows,
        completed_at=utc_now(),
    )


if __name__ == "__main__":
    main()

