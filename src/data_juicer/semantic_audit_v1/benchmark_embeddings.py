#!/usr/bin/env python3
"""Benchmark BGE-M3 on a real turn shard and calculate a full-corpus ETA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from common import read_json, update_status, utc_now, write_json_atomic
from embedding_core import encode_to_array, load_model


def evenly_spaced(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    indexes = np.linspace(0, len(values) - 1, num=limit, dtype=np.int64)
    return [values[int(index)] for index in indexes]


def benchmark_field(
    name: str,
    texts: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict:
    warmup = texts[: min(len(texts), max(batch_size * 2, 32))]
    warmup_output = np.empty((len(warmup), int(model.config.hidden_size)), dtype=np.float16)
    encode_to_array(
        model,
        tokenizer,
        warmup,
        warmup_output,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    output = np.empty((len(texts), int(model.config.hidden_size)), dtype=np.float16)
    stats = encode_to_array(
        model,
        tokenizer,
        texts,
        output,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return {
        "field": name,
        "records": stats.records,
        "elapsed_seconds": stats.elapsed_seconds,
        "records_per_second": stats.records_per_second,
        "content_tokens": stats.content_tokens,
        "content_tokens_per_second": stats.content_tokens_per_second,
        "padding_efficiency": stats.content_tokens / stats.padded_tokens if stats.padded_tokens else 0.0,
        "effective_batch_size": stats.batch_size,
        "max_length": stats.max_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample-size", type=int, default=2048)
    parser.add_argument("--expected-turns", type=int)
    args = parser.parse_args()

    expected_turns = args.expected_turns or int((read_json(args.status_path, {}) or {}).get("expected_turns", 0))
    if expected_turns <= 0:
        raise ValueError("expected turns is missing")

    update_status(args.status_path, status="running", stage="benchmark_embedding")
    table = pq.read_table(args.turn_shard, columns=["assistant_text", "user_text"])
    assistant_texts = evenly_spaced(table["assistant_text"].to_pylist(), args.sample_size)
    user_texts = evenly_spaced(table["user_text"].to_pylist(), args.sample_size)

    tokenizer, model = load_model(args.model, args.revision, args.device)
    assistant = benchmark_field(
        "assistant_text",
        assistant_texts,
        model,
        tokenizer,
        args.device,
        args.batch_size,
        args.max_length,
    )
    user = benchmark_field(
        "user_text",
        user_texts,
        model,
        tokenizer,
        args.device,
        args.batch_size,
        args.max_length,
    )
    assistant_seconds = expected_turns / assistant["records_per_second"]
    user_seconds = expected_turns / user["records_per_second"]
    total_seconds = assistant_seconds + user_seconds
    dimension = int(model.config.hidden_size)
    bytes_per_field = expected_turns * dimension * 2
    payload = {
        "status": "completed",
        "completed_at": utc_now(),
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "dimension": dimension,
        "dtype": "float16",
        "max_length": args.max_length,
        "configured_batch_size": args.batch_size,
        "sample_size_per_field": args.sample_size,
        "expected_turns_per_field": expected_turns,
        "assistant": assistant,
        "user": user,
        "estimated_assistant_seconds": assistant_seconds,
        "estimated_user_seconds": user_seconds,
        "estimated_total_seconds": total_seconds,
        "estimated_total_seconds_with_25pct_margin": total_seconds * 1.25,
        "raw_vector_bytes_per_field": bytes_per_field,
        "raw_vector_bytes_total": bytes_per_field * 2,
        "raw_vector_gib_total": bytes_per_field * 2 / math.pow(1024, 3),
    }
    write_json_atomic(args.output, payload)
    update_status(
        args.status_path,
        status="running",
        stage="benchmark_complete",
        benchmark=payload,
        benchmark_completed_at=utc_now(),
    )


if __name__ == "__main__":
    main()

