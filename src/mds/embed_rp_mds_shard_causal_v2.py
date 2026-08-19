#!/usr/bin/env python3
"""Encode one deterministic shard of the RP MDS three-view corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer


ROOT_DEFAULT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
VIEW_CONFIG = {
    "user": {
        "input": "02_texts/user_unique_texts.parquet",
        "max_seq_length": 1024,
        "batch_size": 96,
        "chunk_size": 8192,
    },
    "role": {
        "input": "02_texts/role_unique_texts.parquet",
        "max_seq_length": 8192,
        "batch_size": 1,
        "chunk_size": 256,
    },
    "recent": {
        "input": "02_texts/recent_texts.parquet",
        "max_seq_length": 4096,
        "batch_size": 2,
        "chunk_size": 512,
    },
}


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def shard_range(total: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"invalid shard {shard_id}/{num_shards}")
    base, remainder = divmod(total, num_shards)
    start = shard_id * base + min(shard_id, remainder)
    size = base + (1 if shard_id < remainder else 0)
    return start, start + size


def read_parquet_range(path: Path, start: int, end: int) -> pa.Table:
    parquet_file = pq.ParquetFile(path)
    pieces: list[pa.Table] = []
    offset = 0
    for row_group in range(parquet_file.num_row_groups):
        count = parquet_file.metadata.row_group(row_group).num_rows
        group_end = offset + count
        if group_end <= start:
            offset = group_end
            continue
        if offset >= end:
            break
        table = parquet_file.read_row_group(row_group, columns=["text_index", "text"])
        local_start = max(start - offset, 0)
        local_end = min(end - offset, count)
        pieces.append(table.slice(local_start, local_end - local_start))
        offset = group_end
    if not pieces:
        return pa.table({"text_index": pa.array([], type=pa.int32()), "text": pa.array([], type=pa.string())})
    return pa.concat_tables(pieces)


def prepare_texts(
    tokenizer: Any,
    raw_texts: list[str],
    max_seq_length: int,
) -> tuple[list[str], np.ndarray, int]:
    """Tokenize once for lengths and preserve both ends when truncation is needed."""
    encoded = tokenizer(
        raw_texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids_list = encoded["input_ids"]
    lengths = np.fromiter((len(ids) + 2 for ids in ids_list), dtype=np.uint32, count=len(ids_list))
    # Reserve enough space for special tokens plus the explicit middle marker.
    content_budget = max_seq_length - 16
    head_budget = content_budget // 2
    tail_budget = content_budget - head_budget
    prepared = list(raw_texts)
    truncated = 0
    for index, ids in enumerate(ids_list):
        if len(ids) <= content_budget:
            continue
        head = tokenizer.decode(
            ids[:head_budget],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        tail = tokenizer.decode(
            ids[-tail_budget:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prepared[index] = f"{head}\n[TRUNCATED_MIDDLE]\n{tail}"
        truncated += 1
    return prepared, lengths, truncated


CURRENT_USER_MARKER = "[CURRENT_USER]\n"
TRUNCATED_HISTORY_MARKER = "[TRUNCATED_HISTORY]\n"


def prepare_recent_texts(
    tokenizer: Any,
    raw_texts: list[str],
    max_seq_length: int,
) -> tuple[list[str], np.ndarray, int, dict[str, int | float]]:
    """Keep the complete current-user block and trim only historical context.

    Every returned string is re-tokenized with special tokens and proven to be
    within max_seq_length, so downstream tokenizer truncation is never relied
    on for correctness.
    """
    raw_encoded = tokenizer(
        raw_texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    raw_lengths = np.fromiter(
        (len(ids) for ids in raw_encoded["input_ids"]),
        dtype=np.uint32,
        count=len(raw_texts),
    )
    prepared: list[str] = []
    truncated = 0
    current_token_min = None
    current_token_max = 0
    retained_current_token_min = None
    final_token_max = 0
    original_over_max = 0
    current_over_max = 0
    marker_survive = 0
    full_current_preserved = 0

    for raw_text, raw_ids in zip(
        raw_texts, raw_encoded["input_ids"], strict=True
    ):
        marker_position = raw_text.rfind(CURRENT_USER_MARKER)
        if marker_position < 0:
            raise RuntimeError("recent text is missing [CURRENT_USER] marker")
        current_block = raw_text[marker_position:]
        history = raw_text[:marker_position].rstrip()
        current_ids = tokenizer(
            current_block,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        current_with_special = tokenizer(
            current_block,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        current_count = len(current_ids)
        current_token_min = (
            current_count
            if current_token_min is None
            else min(current_token_min, current_count)
        )
        current_token_max = max(current_token_max, current_count)
        if len(current_with_special) > max_seq_length:
            current_over_max += 1
            raise RuntimeError(
                "current-user block alone exceeds recent max sequence length: "
                f"{len(current_with_special)} > {max_seq_length}"
            )

        if len(raw_ids) <= max_seq_length:
            candidate = raw_text
        else:
            original_over_max += 1
            truncated += 1
            history_ids = tokenizer(
                history,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            def candidate_for(keep_history_tokens: int) -> str:
                if keep_history_tokens <= 0:
                    return f"{TRUNCATED_HISTORY_MARKER}{current_block}"
                tail = tokenizer.decode(
                    history_ids[-keep_history_tokens:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                return (
                    f"{TRUNCATED_HISTORY_MARKER}{tail}\n\n{current_block}"
                )

            low, high = 0, len(history_ids)
            candidate = candidate_for(0)
            base_length = len(
                tokenizer(
                    candidate,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
            )
            if base_length > max_seq_length:
                raise RuntimeError(
                    "current-user block plus marker exceeds max sequence length"
                )
            while low <= high:
                middle = (low + high) // 2
                probe = candidate_for(middle)
                probe_length = len(
                    tokenizer(
                        probe,
                        add_special_tokens=True,
                        truncation=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )["input_ids"]
                )
                if probe_length <= max_seq_length:
                    candidate = probe
                    low = middle + 1
                else:
                    high = middle - 1

        final_ids = tokenizer(
            candidate,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        if len(final_ids) > max_seq_length:
            raise RuntimeError(
                f"prepared recent input exceeds max length: {len(final_ids)}"
            )
        if CURRENT_USER_MARKER not in candidate:
            raise RuntimeError("prepared recent input lost CURRENT_USER marker")
        if not candidate.endswith(current_block):
            raise RuntimeError("prepared recent input changed current-user block")
        marker_survive += 1
        full_current_preserved += 1
        retained_current_token_min = (
            current_count
            if retained_current_token_min is None
            else min(retained_current_token_min, current_count)
        )
        final_token_max = max(final_token_max, len(final_ids))
        prepared.append(candidate)

    stats: dict[str, int | float] = {
        "original_over_max_seq_length": original_over_max,
        "current_user_over_max_seq_length": current_over_max,
        "current_user_marker_survive_count": marker_survive,
        "current_user_full_preserve_count": full_current_preserved,
        "current_user_token_min": int(current_token_min or 0),
        "current_user_token_max": int(current_token_max),
        "retained_current_user_token_min": int(
            retained_current_token_min or 0
        ),
        "final_prepared_token_max": final_token_max,
    }
    return prepared, raw_lengths, truncated, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--views", default="user,role,recent")
    parser.add_argument("--user-batch-size", type=int, default=VIEW_CONFIG["user"]["batch_size"])
    parser.add_argument("--role-batch-size", type=int, default=VIEW_CONFIG["role"]["batch_size"])
    parser.add_argument("--recent-batch-size", type=int, default=VIEW_CONFIG["recent"]["batch_size"])
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    model_path = args.model_path.resolve()
    output_root = root / "03_embeddings" / "shards"
    output_root.mkdir(parents=True, exist_ok=True)
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    invalid = sorted(set(views) - set(VIEW_CONFIG))
    if invalid:
        raise ValueError(f"unknown views: {invalid}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device.startswith("cuda"):
        if not (0.0 < args.cuda_memory_fraction <= 1.0):
            raise ValueError("CUDA memory fraction must be within (0,1]")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    load_started = time.time()
    model = SentenceTransformer(
        str(model_path),
        device=args.device,
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.float16},
    )
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension != 1024:
        raise RuntimeError(f"expected BGE-M3 dimension 1024, got {dimension}")
    model.eval()
    load_seconds = time.time() - load_started

    batch_overrides = {
        "user": args.user_batch_size,
        "role": args.role_batch_size,
        "recent": args.recent_batch_size,
    }
    run_manifest: dict[str, Any] = {
        "schema_version": "rp-mds-bge-m3-shard-v1",
        "model_path": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "device": args.device,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dimension": dimension,
        "dtype": "float16",
        "load_seconds": round(load_seconds, 3),
        "views": {},
    }

    for view in views:
        config = VIEW_CONFIG[view]
        input_path = root / str(config["input"])
        parquet_file = pq.ParquetFile(input_path)
        total = parquet_file.metadata.num_rows
        start, end = shard_range(total, args.shard_id, args.num_shards)
        count = end - start
        view_dir = output_root / view
        view_dir.mkdir(parents=True, exist_ok=True)
        stem = f"shard-{args.shard_id:03d}-of-{args.num_shards:03d}"
        embedding_path = view_dir / f"{stem}.embeddings.f16.npy"
        length_path = view_dir / f"{stem}.token_lengths.u16.npy"
        manifest_path = view_dir / f"{stem}.json"

        if embedding_path.exists() and manifest_path.exists() and not args.force:
            print(f"[{view}] already complete: {embedding_path}", flush=True)
            run_manifest["views"][view] = json.loads(manifest_path.read_text(encoding="utf-8"))
            continue

        print(f"[{view}] reading [{start}, {end}) from {input_path}", flush=True)
        table = read_parquet_range(input_path, start, end)
        indices = np.asarray(table["text_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        texts = [str(value or "").strip() for value in table["text"].to_pylist()]
        expected_indices = np.arange(start, end, dtype=np.int64)
        if len(texts) != count or not np.array_equal(indices, expected_indices):
            raise RuntimeError(f"{view} shard input is not contiguous and deterministic")
        if any(not text for text in texts):
            raise RuntimeError(f"{view} shard contains empty text")

        max_seq_length = int(config["max_seq_length"])
        batch_size = int(batch_overrides[view])
        chunk_size = int(config["chunk_size"])
        model.max_seq_length = max_seq_length

        embedding_temp = embedding_path.with_name(embedding_path.name + ".partial")
        length_temp = length_path.with_name(length_path.name + ".partial")
        for path in [embedding_temp, length_temp]:
            if path.exists():
                path.unlink()
        output = np.lib.format.open_memmap(
            embedding_temp,
            mode="w+",
            dtype=np.float16,
            shape=(count, dimension),
        )
        token_lengths = np.lib.format.open_memmap(
            length_temp,
            mode="w+",
            dtype=np.uint16,
            shape=(count,),
        )

        started = time.time()
        truncated_total = 0
        recent_original_over_max = 0
        recent_current_over_max = 0
        recent_marker_survive = 0
        recent_full_current_preserve = 0
        recent_current_token_min: int | None = None
        recent_current_token_max = 0
        recent_retained_current_token_min: int | None = None
        recent_final_token_max = 0
        recent_prepared_gt_max = 0
        recent_marker_lost = 0
        recent_current_token_ids_full_preserved = 0
        recent_current_token_ids_lost = 0
        prepared_input_ids_digest = hashlib.sha256()
        norm_min = float("inf")
        norm_max = 0.0
        nan_count = 0
        for local_start in range(0, count, chunk_size):
            local_end = min(local_start + chunk_size, count)
            if view == "recent":
                model.tokenizer.truncation_side = "left"
                (
                    prepared,
                    original_lengths,
                    truncated,
                    recent_stats,
                ) = prepare_recent_texts(
                    model.tokenizer,
                    texts[local_start:local_end],
                    max_seq_length=max_seq_length,
                )
                recent_original_over_max += int(
                    recent_stats["original_over_max_seq_length"]
                )
                recent_current_over_max += int(
                    recent_stats["current_user_over_max_seq_length"]
                )
                recent_marker_survive += int(
                    recent_stats["current_user_marker_survive_count"]
                )
                recent_full_current_preserve += int(
                    recent_stats["current_user_full_preserve_count"]
                )
                chunk_current_min = int(
                    recent_stats["current_user_token_min"]
                )
                recent_current_token_min = (
                    chunk_current_min
                    if recent_current_token_min is None
                    else min(recent_current_token_min, chunk_current_min)
                )
                recent_current_token_max = max(
                    recent_current_token_max,
                    int(recent_stats["current_user_token_max"]),
                )
                chunk_retained_min = int(
                    recent_stats["retained_current_user_token_min"]
                )
                recent_retained_current_token_min = (
                    chunk_retained_min
                    if recent_retained_current_token_min is None
                    else min(
                        recent_retained_current_token_min,
                        chunk_retained_min,
                    )
                )
                recent_final_token_max = max(
                    recent_final_token_max,
                    int(recent_stats["final_prepared_token_max"]),
                )
                actual_inputs = model.tokenizer(
                    prepared,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                for local_offset, (raw_text, candidate, input_ids) in enumerate(
                    zip(
                        texts[local_start:local_end],
                        prepared,
                        actual_inputs,
                        strict=True,
                    )
                ):
                    row_index = start + local_start + local_offset
                    prepared_input_ids_digest.update(
                        struct.pack(">QQ", row_index, len(input_ids))
                    )
                    for token_id in input_ids:
                        prepared_input_ids_digest.update(
                            struct.pack(">I", int(token_id))
                        )
                    if len(input_ids) > max_seq_length:
                        recent_prepared_gt_max += 1
                    marker_position = raw_text.rfind(CURRENT_USER_MARKER)
                    current_block = raw_text[marker_position:]
                    marker_retained = CURRENT_USER_MARKER in candidate
                    full_current_retained = candidate.endswith(current_block)
                    if marker_retained:
                        pass
                    else:
                        recent_marker_lost += 1
                    if full_current_retained:
                        recent_current_token_ids_full_preserved += 1
                    else:
                        recent_current_token_ids_lost += 1
            else:
                prepared, original_lengths, truncated = prepare_texts(
                    model.tokenizer,
                    texts[local_start:local_end],
                    max_seq_length=max_seq_length,
                )
            encoded_embeddings = model.encode(
                prepared,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="float32",
            )
            encoded_embeddings = np.asarray(encoded_embeddings, dtype=np.float32)
            if encoded_embeddings.shape != (local_end - local_start, dimension):
                raise RuntimeError(f"unexpected embedding shape {encoded_embeddings.shape}")
            norms = np.linalg.norm(encoded_embeddings, axis=1)
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
            nan_count += int(np.isnan(encoded_embeddings).sum())
            output[local_start:local_end] = encoded_embeddings.astype(np.float16)
            token_lengths[local_start:local_end] = np.minimum(original_lengths, 65535).astype(np.uint16)
            truncated_total += truncated
            output.flush()
            token_lengths.flush()
            elapsed = time.time() - started
            rate = local_end / max(elapsed, 1e-9)
            remaining = (count - local_end) / max(rate, 1e-9)
            print(
                f"[{view}] {local_end:,}/{count:,} rate={rate:.1f}/s "
                f"eta={remaining / 60:.1f}m truncated={truncated_total:,}",
                flush=True,
            )

        del output
        del token_lengths
        if nan_count:
            raise RuntimeError(f"{view} produced {nan_count} NaN values")
        os.replace(embedding_temp, embedding_path)
        os.replace(length_temp, length_path)

        view_manifest = {
            "view": view,
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "start": start,
            "end": end,
            "count": count,
            "dimension": dimension,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "head_tail_truncated": truncated_total,
            "head_tail_truncated_fraction": truncated_total / max(count, 1),
            "pre_cast_norm_min": norm_min,
            "pre_cast_norm_max": norm_max,
            "nan_count": nan_count,
            "elapsed_seconds": round(time.time() - started, 3),
            "embedding_path": str(embedding_path),
            "embedding_bytes": embedding_path.stat().st_size,
            "embedding_sha256": sha256_file(embedding_path),
            "token_length_path": str(length_path),
            "token_length_bytes": length_path.stat().st_size,
            "token_length_sha256": sha256_file(length_path),
        }
        if view == "recent":
            if recent_marker_survive != count:
                raise RuntimeError(
                    f"CURRENT_USER marker survived {recent_marker_survive}/{count}"
                )
            if recent_full_current_preserve != count:
                raise RuntimeError(
                    "current-user block was not fully preserved for every row"
                )
            view_manifest.update(
                {
                    "truncation_policy": (
                        "marker-aware-left-history-trim; complete "
                        "CURRENT_USER block retained; final prepared tokens "
                        "validated <= max_seq_length"
                    ),
                    "truncation_side": model.tokenizer.truncation_side,
                    "tokenizer_truncation_side": (
                        model.tokenizer.truncation_side
                    ),
                    "original_over_max_seq_length": recent_original_over_max,
                    "current_user_over_max_seq_length": recent_current_over_max,
                    "current_user_marker_survive_count": recent_marker_survive,
                    "current_user_marker_retained_count": (
                        recent_marker_survive
                    ),
                    "current_user_marker_survival_rate": (
                        recent_marker_survive / max(count, 1)
                    ),
                    "current_user_full_preserve_count": (
                        recent_full_current_preserve
                    ),
                    "current_user_full_preservation_rate": (
                        recent_full_current_preserve / max(count, 1)
                    ),
                    "current_user_preserved": (
                        recent_full_current_preserve == count
                    ),
                    "current_user_token_min": int(
                        recent_current_token_min or 0
                    ),
                    "current_user_token_max": recent_current_token_max,
                    "retained_current_user_token_min": int(
                        recent_retained_current_token_min or 0
                    ),
                    "current_user_token_retention_min_fraction": 1.0,
                    "final_prepared_token_max": recent_final_token_max,
                    "prepared_gt_4096_count": recent_prepared_gt_max,
                    "raw_gt_4096_count": recent_original_over_max,
                    "current_block_token_ids_full_preserved_count": (
                        recent_current_token_ids_full_preserved
                    ),
                    "current_block_token_ids_lost_count": (
                        recent_current_token_ids_lost
                    ),
                    "marker_retained_count": recent_marker_survive,
                    "marker_lost_count": recent_marker_lost,
                    "prepared_input_ids_sha256": (
                        prepared_input_ids_digest.hexdigest()
                    ),
                    "prepared_input_ids_sha256_v1": (
                        prepared_input_ids_digest.hexdigest()
                    ),
                    "prepared_input_ids_digest_format": (
                        "text_index ascending; update struct.pack('>QQ', "
                        "text_index, len(input_ids)); then for each token_id "
                        "update struct.pack('>I', token_id); "
                        "add_special_tokens=true,truncation=true,"
                        "max_length=4096,padding=false"
                    ),
                }
            )
        manifest_temp = manifest_path.with_name(manifest_path.name + ".partial")
        manifest_temp.write_text(
            json.dumps(view_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_temp, manifest_path)
        run_manifest["views"][view] = view_manifest
        print(f"[{view}] complete in {view_manifest['elapsed_seconds']}s", flush=True)

    run_manifest_path = output_root / f"worker-{args.shard_id:03d}-of-{args.num_shards:03d}.json"
    run_manifest_temp = run_manifest_path.with_name(run_manifest_path.name + ".partial")
    run_manifest_temp.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(run_manifest_temp, run_manifest_path)
    print(json.dumps(run_manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
