#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import time

import pyarrow.parquet as pq
import zstandard as zstd


ROOT = pathlib.Path("/home/gyr/lusychat-training/general-sft-anchor-v1")
BUILD_SCRIPT = ROOT / "scripts" / "build_general_sft_anchor_v1.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("general_builder", BUILD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    builder = load_builder()
    build_report_path = ROOT / "00_manifest" / "BUILD_REPORT.json"
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    parquet_paths = sorted((ROOT / "03_selected" / "parquet").glob("part-*.parquet"))
    if len(parquet_paths) != 10:
        raise RuntimeError(f"expected 10 parquet shards, found {len(parquet_paths)}")

    table = pq.read_table(parquet_paths)
    rows = table.to_pylist()
    errors: list[str] = []

    if len(rows) != 100_000:
        errors.append(f"parquet_rows={len(rows)}")

    source_counts = collections.Counter(row["source_key"] for row in rows)
    language_counts = collections.Counter(row["language_final"] for row in rows)
    task_counts = collections.Counter(row["task_type"] for row in rows)
    config_counts = collections.Counter(
        row["source_config"] for row in rows if row["source_key"] == "smoltalk"
    )

    if dict(source_counts) != dict(builder.SOURCE_TARGETS):
        errors.append(f"source_counts={dict(source_counts)}")
    if dict(language_counts) != builder.LANGUAGE_TARGETS:
        errors.append(f"language_counts={dict(language_counts)}")
    if dict(config_counts) != builder.SMOLTALK_CONFIG_TARGETS:
        errors.append(f"smoltalk_counts={dict(config_counts)}")

    unique_checks = {
        "sample_id": len({row["sample_id"] for row in rows}),
        "prompt_fingerprint": len({row["prompt_fingerprint"] for row in rows}),
        "target_fingerprint": len({row["target_fingerprint"] for row in rows}),
    }
    for name, count in unique_checks.items():
        if count != 100_000:
            errors.append(f"unique_{name}={count}")

    invalid_loss_mask = 0
    invalid_terminal = 0
    think_tag_hits = 0
    ai_identity_hits = 0
    repeated_history_target = 0
    cot_removed = 0
    for row in rows:
        messages = row["messages"]
        weights = [message["weight"] for message in messages]
        if not weights or weights[-1] != 1 or any(weight != 0 for weight in weights[:-1]):
            invalid_loss_mask += 1
        if not messages or messages[-1]["role"] != "assistant":
            invalid_terminal += 1
        target = row["target_text"]
        if builder.THINK_TAG_RE.search(target):
            think_tag_hits += 1
        if builder.has_ai_identity_break(target):
            ai_identity_hits += 1
        normalized_target = builder.normalize_for_dedupe(target)
        if any(
            message["role"] == "assistant"
            and builder.normalize_for_dedupe(message["content"]) == normalized_target
            for message in messages[:-1]
        ):
            repeated_history_target += 1
        cot_removed += int(bool(row["cot_removed"]))

    if invalid_loss_mask:
        errors.append(f"invalid_loss_mask={invalid_loss_mask}")
    if invalid_terminal:
        errors.append(f"invalid_terminal={invalid_terminal}")
    if think_tag_hits:
        errors.append(f"think_tag_hits={think_tag_hits}")
    if ai_identity_hits:
        errors.append(f"ai_identity_hits={ai_identity_hits}")
    if repeated_history_target:
        errors.append(f"repeated_history_target={repeated_history_target}")

    jsonl_path = ROOT / "03_selected" / "general_sft_100k.openai.jsonl.zst"
    jsonl_rows = 0
    jsonl_bad = 0
    decompressor = zstd.ZstdDecompressor()
    with jsonl_path.open("rb") as raw:
        with decompressor.stream_reader(raw) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text:
                for line in text:
                    jsonl_rows += 1
                    payload = json.loads(line)
                    if (
                        not payload.get("id")
                        or not isinstance(payload.get("messages"), list)
                        or payload["messages"][-1].get("role") != "assistant"
                    ):
                        jsonl_bad += 1
    if jsonl_rows != 100_000:
        errors.append(f"jsonl_rows={jsonl_rows}")
    if jsonl_bad:
        errors.append(f"jsonl_bad={jsonl_bad}")

    hashes: dict[str, str] = {}
    hash_mismatches: dict[str, dict[str, str]] = {}
    expected_hashes = build_report["output"]["sha256"]
    for relative_path, expected in expected_hashes.items():
        path = ROOT / relative_path
        actual = sha256_file(path)
        hashes[relative_path] = actual
        if actual != expected:
            hash_mismatches[relative_path] = {"expected": expected, "actual": actual}
    if hash_mismatches:
        errors.append(f"hash_mismatches={len(hash_mismatches)}")

    review_path = ROOT / "05_review" / "review_samples_500.jsonl"
    review_rows = sum(1 for _ in review_path.open("r", encoding="utf-8"))
    if review_rows != 500:
        errors.append(f"review_rows={review_rows}")

    report = {
        "schema_version": builder.SCHEMA_VERSION,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "valid": not errors,
        "errors": errors,
        "parquet_rows": len(rows),
        "parquet_shards": len(parquet_paths),
        "jsonl_rows": jsonl_rows,
        "jsonl_bad": jsonl_bad,
        "review_rows": review_rows,
        "source_distribution": dict(source_counts),
        "language_distribution": dict(language_counts),
        "task_distribution": dict(task_counts),
        "smoltalk_config_distribution": dict(config_counts),
        "unique_checks": unique_checks,
        "invalid_loss_mask": invalid_loss_mask,
        "invalid_terminal": invalid_terminal,
        "think_tag_hits": think_tag_hits,
        "ai_identity_hits": ai_identity_hits,
        "repeated_history_target": repeated_history_target,
        "cot_removed": cot_removed,
        "sha256": hashes,
        "hash_mismatches": hash_mismatches,
        "database_accessed": False,
        "production_api_accessed": False,
        "repository_modified": False,
    }

    verification_path = ROOT / "00_manifest" / "FINAL_VERIFICATION.json"
    verification_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums_path = ROOT / "00_manifest" / "SHA256SUMS.json"
    checksums_path.write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
