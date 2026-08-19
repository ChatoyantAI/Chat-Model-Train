#!/usr/bin/env python3
"""Validate and atomically assemble causal recent-scene BGE-M3 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


DIMENSION = 1024


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1"),
    )
    parser.add_argument("--num-shards", type=int, default=3)
    args = parser.parse_args()
    stage = args.stage_root.resolve()
    formal = args.formal_root.resolve()
    started = time.time()

    recent_input = stage / "02_texts/recent_texts.parquet"
    views_path = stage / "02_texts/neutral_no_summary_views.parquet"
    index_path = formal / "03_embeddings/embedding_index.parquet"
    rows = pq.ParquetFile(recent_input).metadata.num_rows
    if rows != 185_040:
        raise RuntimeError(f"unexpected recent task rows: {rows}")
    if pq.ParquetFile(views_path).metadata.num_rows != rows:
        raise RuntimeError("view/recent row count mismatch")
    if pq.ParquetFile(index_path).metadata.num_rows != rows:
        raise RuntimeError("index/recent row count mismatch")

    manifests = []
    for shard_id in range(args.num_shards):
        path = (
            stage
            / "03_embeddings/shards/recent"
            / f"shard-{shard_id:03d}-of-{args.num_shards:03d}.json"
        )
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["view"] != "recent":
            raise RuntimeError(f"wrong view in {path}")
        if int(item["dimension"]) != DIMENSION:
            raise RuntimeError(f"wrong dimension in {path}")
        if str(item["input_sha256"]) != sha256_file(recent_input):
            raise RuntimeError(f"input checksum mismatch in {path}")
        if item.get("tokenizer_truncation_side") != "left":
            raise RuntimeError(f"recent shard is not explicit left truncation: {path}")
        if int(item.get("marker_retained_count", -1)) != int(item["count"]):
            raise RuntimeError(f"CURRENT_USER marker loss in {path}")
        if int(item.get("marker_lost_count", -1)) != 0:
            raise RuntimeError(f"CURRENT_USER marker loss in {path}")
        if int(
            item.get(
                "current_block_token_ids_full_preserved_count", -1
            )
        ) != int(item["count"]):
            raise RuntimeError(f"current-user token loss in {path}")
        if int(item.get("current_block_token_ids_lost_count", -1)) != 0:
            raise RuntimeError(f"current-user token loss in {path}")
        if int(item.get("prepared_gt_4096_count", -1)) != 0:
            raise RuntimeError(f"prepared input exceeds 4096 in {path}")
        if int(item.get("current_user_over_max_seq_length", -1)) != 0:
            raise RuntimeError(f"current-user block exceeds 4096 in {path}")
        embedding_path = Path(item["embedding_path"])
        length_path = Path(item["token_length_path"])
        if sha256_file(embedding_path) != item["embedding_sha256"]:
            raise RuntimeError(f"embedding checksum mismatch in {path}")
        if sha256_file(length_path) != item["token_length_sha256"]:
            raise RuntimeError(f"length checksum mismatch in {path}")
        manifests.append(item)
    manifests.sort(key=lambda item: int(item["start"]))
    cursor = 0
    for item in manifests:
        if int(item["start"]) != cursor:
            raise RuntimeError(f"shard gap or overlap at {cursor}")
        cursor = int(item["end"])
    if cursor != rows:
        raise RuntimeError(f"shard coverage {cursor} != {rows}")

    output_dir = stage / "03_embeddings"
    embedding_path = output_dir / "recent_causal_v2.f16.npy"
    length_path = output_dir / "recent_causal_v2.token_lengths.u16.npy"
    embedding_temp = embedding_path.with_name(embedding_path.name + ".partial")
    length_temp = length_path.with_name(length_path.name + ".partial")
    for path in [embedding_temp, length_temp]:
        if path.exists():
            path.unlink()
    output = np.lib.format.open_memmap(
        embedding_temp,
        mode="w+",
        dtype=np.float16,
        shape=(rows, DIMENSION),
    )
    lengths = np.lib.format.open_memmap(
        length_temp,
        mode="w+",
        dtype=np.uint16,
        shape=(rows,),
    )
    for item in manifests:
        start = int(item["start"])
        end = int(item["end"])
        source = np.load(item["embedding_path"], mmap_mode="r")
        source_lengths = np.load(item["token_length_path"], mmap_mode="r")
        if source.shape != (end - start, DIMENSION):
            raise RuntimeError(f"bad shard shape {source.shape}")
        if source_lengths.shape != (end - start,):
            raise RuntimeError(f"bad length shape {source_lengths.shape}")
        output[start:end] = source
        lengths[start:end] = source_lengths
    output.flush()
    lengths.flush()
    del output
    del lengths
    os.replace(embedding_temp, embedding_path)
    os.replace(length_temp, length_path)

    array = np.load(embedding_path, mmap_mode="r")
    norm_min = float("inf")
    norm_max = 0.0
    nan_count = 0
    for start in range(0, rows, 4096):
        chunk = np.asarray(array[start : start + 4096], dtype=np.float32)
        nan_count += int(np.isnan(chunk).sum())
        norms = np.linalg.norm(chunk, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    if nan_count:
        raise RuntimeError(f"causal recent embedding has {nan_count} NaNs")
    if not (0.995 <= norm_min <= norm_max <= 1.005):
        raise RuntimeError(
            f"unexpected embedding norms: min={norm_min}, max={norm_max}"
        )

    build_report_path = (
        stage / "00_manifest/CAUSAL_VIEW_BUILD_REPORT.json"
    )
    reuse_report_path = (
        stage / "00_manifest/CAUSAL_REUSE_PROOF.json"
    )
    token_audit_path = (
        stage / "00_manifest/CAUSAL_RECENT_TOKEN_BUDGET_AUDIT.json"
    )
    token_audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
    if not token_audit.get("pass"):
        raise RuntimeError("causal recent token-budget audit did not pass")
    marker_retained = sum(
        int(item["marker_retained_count"]) for item in manifests
    )
    current_preserved = sum(
        int(item["current_block_token_ids_full_preserved_count"])
        for item in manifests
    )
    raw_gt_4096 = sum(
        int(item["raw_gt_4096_count"]) for item in manifests
    )
    report = {
        "schema_version": "rp-mds-causal-recent-embedding-v2",
        "target_assistant_excluded": True,
        "causal_definition": (
            "latest target prompt snapshot; latest three complete historical "
            "user-assistant(weight=0 or missing) pairs plus current user; "
            "current assistant(weight=1) and following content excluded"
        ),
        "rows": rows,
        "current_user_preserved": current_preserved == rows,
        "truncation_side": "left",
        "tokenizer_truncation_side": "left",
        "truncation_policy": manifests[0]["truncation_policy"],
        "recent_over_4096_count": raw_gt_4096,
        "current_user_over_4096_count": sum(
            int(item["current_user_over_max_seq_length"])
            for item in manifests
        ),
        "prepared_over_4096_count": sum(
            int(item["prepared_gt_4096_count"]) for item in manifests
        ),
        "current_user_marker_retained_count": marker_retained,
        "current_user_marker_survival_rate": marker_retained / rows,
        "current_user_full_preserve_count": current_preserved,
        "current_user_full_preservation_rate": current_preserved / rows,
        "current_user_content_token_min": token_audit[
            "current_user_content_token_min"
        ],
        "current_user_content_token_max": token_audit[
            "current_user_content_token_max"
        ],
        "retained_current_user_block_token_min": token_audit[
            "retained_current_user_block_token_min"
        ],
        "final_prepared_token_max": max(
            int(item["final_prepared_token_max"]) for item in manifests
        ),
        "prepared_input_ids_shard_sha256": [
            item["prepared_input_ids_sha256"] for item in manifests
        ],
        "dimension": DIMENSION,
        "dtype": "float16",
        "embedding_path": str(embedding_path),
        "embedding_sha256": sha256_file(embedding_path),
        "embedding_shape": list(array.shape),
        "token_lengths_path": str(length_path),
        "token_lengths_sha256": sha256_file(length_path),
        "views_path": str(views_path),
        "views_sha256": sha256_file(views_path),
        "recent_task_path": str(recent_input),
        "recent_task_sha256": sha256_file(recent_input),
        "embedding_index_path": str(index_path),
        "embedding_index_sha256": sha256_file(index_path),
        "build_report_path": str(build_report_path),
        "build_report_sha256": sha256_file(build_report_path),
        "reuse_report_path": str(reuse_report_path),
        "reuse_report_sha256": sha256_file(reuse_report_path),
        "token_budget_audit_path": str(token_audit_path),
        "token_budget_audit_sha256": sha256_file(token_audit_path),
        "shards": manifests,
        "nan_count": nan_count,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "elapsed_seconds": round(time.time() - started, 3),
        "pass": True,
    }
    report_path = (
        stage / "00_manifest/CAUSAL_RECENT_EMBEDDING_REPORT.json"
    )
    temp_report = report_path.with_name(report_path.name + ".partial")
    temp_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
