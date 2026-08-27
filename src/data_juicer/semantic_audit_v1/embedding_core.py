#!/usr/bin/env python3
"""Dense BGE-M3 embedding primitives shared by benchmark and full encoding."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


@dataclass
class EncodeStats:
    records: int
    padded_tokens: int
    content_tokens: int
    elapsed_seconds: float
    batch_size: int
    max_length: int

    @property
    def records_per_second(self) -> float:
        return self.records / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def content_tokens_per_second(self) -> float:
        return self.content_tokens / self.elapsed_seconds if self.elapsed_seconds else 0.0


def load_model(model_name: str, revision: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    return tokenizer, model


def _run_batch(model, tokenizer, texts: Sequence[str], device: str, max_length: int):
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    content_tokens = int(encoded["attention_mask"].sum().item())
    padded_tokens = int(encoded["attention_mask"].numel())
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
        dense = functional.normalize(output.last_hidden_state[:, 0].float(), p=2, dim=1)
    return dense.cpu().numpy().astype(np.float16, copy=False), content_tokens, padded_tokens


def encode_to_array(
    model,
    tokenizer,
    texts: Sequence[str],
    destination: np.ndarray,
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> EncodeStats:
    if len(texts) != len(destination):
        raise ValueError(f"text/vector row mismatch: {len(texts)} != {len(destination)}")
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    total_content_tokens = 0
    total_padded_tokens = 0
    started = time.perf_counter()

    offset = 0
    active_batch_size = batch_size
    while offset < len(order):
        current_indexes = order[offset : offset + active_batch_size]
        batch_texts = [texts[index] for index in current_indexes]
        try:
            vectors, content_tokens, padded_tokens = _run_batch(
                model, tokenizer, batch_texts, device, max_length
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if active_batch_size <= 1:
                raise
            active_batch_size = max(1, active_batch_size // 2)
            continue
        destination[np.asarray(current_indexes, dtype=np.int64)] = vectors
        total_content_tokens += content_tokens
        total_padded_tokens += padded_tokens
        offset += len(current_indexes)

    if hasattr(destination, "flush"):
        destination.flush()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return EncodeStats(
        records=len(texts),
        padded_tokens=total_padded_tokens,
        content_tokens=total_content_tokens,
        elapsed_seconds=time.perf_counter() - started,
        batch_size=active_batch_size,
        max_length=max_length,
    )

