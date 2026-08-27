#!/usr/bin/env python3
"""Build compact full-run artifacts for the Streamlit audit page."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_rows(path: Path) -> int | None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def count_fuzzy_eligible(path: Path, minimum_chars: int = 32) -> int | None:
    if parquet_rows(path) is None:
        return None
    count = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=65_536, columns=["body_chars"]):
        count += sum(value is not None and value >= minimum_chars for value in batch.column(0).to_pylist())
    return count


def find_trace(directory: Path, operator: str) -> Path | None:
    matches = list(directory.rglob(f"duplicate-{operator}.jsonl")) if directory.exists() else []
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    body = str(sample.get("dialogue_body_text") or "")
    return {
        "conversation_id": sample.get("conversation_id"),
        "dialogue_sample_id": sample.get("dialogue_sample_id"),
        "character_id": sample.get("character_id"),
        "conversation_bucket": sample.get("conversation_bucket"),
        "turn_count": sample.get("turn_count"),
        "output_language": sample.get("output_language"),
        "full_training_sha256": sample.get("full_training_sha256"),
        "dialogue_preview": body[:1200],
    }


def compact_duplicate_traces(run_root: Path, dashboard_dir: Path) -> dict[str, int]:
    specifications = {
        "exact": (run_root / "02_exact_dedup", "document_deduplicator"),
        "minhash": (run_root / "03_minhash_branch", "document_minhash_deduplicator"),
        "simhash": (run_root / "04_simhash_branch", "document_simhash_deduplicator"),
    }
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for key, (directory, operator) in specifications.items():
        trace_path = find_trace(directory, operator)
        count = 0
        if trace_path:
            with trace_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    records.append(
                        {
                            "operator": key,
                            "dup1": compact_sample(raw["dup1"]),
                            "dup2": compact_sample(raw["dup2"]),
                        }
                    )
                    count += 1
        counts[key] = count
    write_jsonl_atomic(dashboard_dir / "duplicate_pairs.jsonl", records)
    return counts


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] * (high - position) + sorted_values[high] * (position - low)


def describe(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "p50": quantile(ordered, 0.50),
        "p90": quantile(ordered, 0.90),
        "p95": quantile(ordered, 0.95),
        "p99": quantile(ordered, 0.99),
        "max": ordered[-1] if ordered else None,
        "over_0_10": sum(value > 0.10 for value in ordered),
        "over_0_20": sum(value > 0.20 for value in ordered),
        "over_0_30": sum(value > 0.30 for value in ordered),
        "over_0_50": sum(value > 0.50 for value in ordered),
    }


def keep_top(heap: list[tuple[float, int]], ratio: float, index: int, limit: int = 100) -> None:
    item = (ratio, index)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def summarize_repetition(run_root: Path, dashboard_dir: Path) -> dict[str, Any] | None:
    parquet_path = run_root / "05_repetition_profile" / "scored.parquet"
    stats_path = run_root / "05_repetition_profile" / "scored_stats.jsonl"
    if parquet_rows(parquet_path) is None or not stats_path.exists():
        return None

    word_values: list[float] = []
    char_values: list[float] = []
    top_word: list[tuple[float, int]] = []
    top_char: list[tuple[float, int]] = []

    with stats_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            stats = (json.loads(line).get("__dj__stats__") or {})
            word_ratio = float(stats.get("word_rep_ratio", 0.0))
            char_ratio = float(stats.get("char_rep_ratio", 0.0))
            word_values.append(word_ratio)
            char_values.append(char_ratio)
            keep_top(top_word, word_ratio, index)
            keep_top(top_char, char_ratio, index)

    selected_indexes = {index for _, index in top_word} | {index for _, index in top_char}
    ratios_by_index = {
        index: (word_values[index], char_values[index])
        for index in selected_indexes
    }

    selected_records: list[dict[str, Any]] = []
    global_index = 0
    columns = [
        "conversation_id",
        "dialogue_sample_id",
        "character_id",
        "conversation_bucket",
        "turn_count",
        "output_language",
        "assistant_body_chars",
        "assistant_preview",
    ]
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=65_536, columns=columns):
        for local_index, record in enumerate(batch.to_pylist()):
            index = global_index + local_index
            if index not in selected_indexes:
                continue
            word_ratio, char_ratio = ratios_by_index[index]
            record.update(
                {
                    "word_rep_ratio": word_ratio,
                    "char_rep_ratio": char_ratio,
                }
            )
            selected_records.append(record)
        global_index += batch.num_rows

    selected_records.sort(
        key=lambda record: max(record["word_rep_ratio"], record["char_rep_ratio"]),
        reverse=True,
    )
    write_jsonl_atomic(dashboard_dir / "repetition_top_records.jsonl", selected_records)
    return {
        "records": len(word_values),
        "word": describe(word_values),
        "character": describe(char_values),
        "top_records": len(selected_records),
    }


def operator_summary(input_count: int | None, output_count: int | None) -> dict[str, Any]:
    removed = input_count - output_count if input_count is not None and output_count is not None else None
    return {
        "input": input_count,
        "retained": output_count,
        "removed": removed,
        "removed_ratio": (removed / input_count) if removed is not None and input_count else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    dashboard_dir = args.run_root / "07_dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    profile = read_json(args.run_root / "01_derived_profile" / "PROFILE_MANIFEST.json")
    status = read_json(dashboard_dir / "run_status.json") or {}
    source_count = profile["counts"]["records"] if profile else None

    exact_path = args.run_root / "02_exact_dedup" / "retained.parquet"
    minhash_path = args.run_root / "03_minhash_branch" / "retained.parquet"
    simhash_path = args.run_root / "04_simhash_branch" / "retained.parquet"
    repetition_path = args.run_root / "05_repetition_profile" / "scored.parquet"
    exact_count = parquet_rows(exact_path)
    eligible_after_exact = count_fuzzy_eligible(exact_path) if exact_count is not None else None
    minhash_count = parquet_rows(minhash_path)
    simhash_count = parquet_rows(simhash_path)
    repetition_count = parquet_rows(repetition_path)

    trace_counts = compact_duplicate_traces(args.run_root, dashboard_dir)
    repetition = summarize_repetition(args.run_root, dashboard_dir)
    operators = {
        "exact": {
            **operator_summary(source_count, exact_count),
            "operator": "document_deduplicator",
            "view": "full_training_sha256",
            "trace_pairs": trace_counts["exact"],
        },
        "minhash": {
            **operator_summary(eligible_after_exact, minhash_count),
            "operator": "document_minhash_deduplicator",
            "view": "dialogue_body_text",
            "short_passthrough": (exact_count - eligible_after_exact)
            if exact_count is not None and eligible_after_exact is not None
            else None,
            "trace_pairs": trace_counts["minhash"],
        },
        "simhash": {
            **operator_summary(eligible_after_exact, simhash_count),
            "operator": "document_simhash_deduplicator",
            "view": "dialogue_body_text",
            "short_passthrough": (exact_count - eligible_after_exact)
            if exact_count is not None and eligible_after_exact is not None
            else None,
            "trace_pairs": trace_counts["simhash"],
        },
        "repetition": {
            "input": exact_count,
            "scored": repetition_count,
            "operator": "word_repetition_filter + character_repetition_filter",
            "view": "assistant_body_text",
            "threshold_deletion": False,
        },
    }
    summary = {
        "summary_version": "lusy-data-juicer-independent-v1-dashboard",
        "generated_at": utc_now(),
        "data_juicer_version": "v1.5.5",
        "scope": "full_v2.2_candidate_pool",
        "record_unit": "one independent complete dialogue; fork metadata ignored",
        "source_is_untouched": True,
        "cpu_only": True,
        "status": status.get("state"),
        "profile": profile,
        "operators": operators,
        "repetition": repetition,
        "important_interpretation": [
            "MinHash and SimHash are independent candidate branches after exact dedup; they are not applied sequentially to each other.",
            "Fuzzy branches exclude body texts shorter than 32 characters from similarity hashing and report them as passthrough.",
            "The repetition operators only score in this run; max_ratio=1.0 means no sample is deleted by repetition threshold.",
            "All dedup operators keep the first item selected by Data-Juicer; outputs are audit branches, not the final quality-aware keep list.",
        ],
    }
    write_json_atomic(dashboard_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
