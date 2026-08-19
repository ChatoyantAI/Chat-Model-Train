#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from language_alignment import Detection, classify_pair, clean_text, extract_pair, informative_char_count, language_group

RULE_VERSION = "language-alignment-v1.1-guideline-stripped"
DETECTOR_NAME = "lingua-language-detector"
DETECTOR_VERSION = "2.2.0"
DEFAULT_PARTS = Path("/home/gyr/lusychat-training/lusy-chat-4m-v2/03_sft/_parts/all")
DEFAULT_ROOT = Path("/home/gyr/lusychat-training/lusy-chat-4m-v2")
WORKER = Path(__file__).with_name("detector_worker.py")
VENDOR = Path(__file__).with_name("vendor")
PYTHON_313 = Path("/opt/miniconda3/bin/python")

FEATURE_SCHEMA = pa.schema([
    ("target_id", pa.string()), ("part_name", pa.string()), ("row_index", pa.int32()),
    ("conversation_id", pa.string()), ("user_id", pa.string()), ("character_id", pa.string()),
    ("pool", pa.string()), ("teacher_family", pa.string()), ("model", pa.string()), ("topic", pa.string()),
    ("assistant_language_existing", pa.string()),
    ("user_language", pa.string()), ("assistant_language", pa.string()),
    ("user_confidence", pa.float32()), ("assistant_confidence", pa.float32()),
    ("user_margin", pa.float32()), ("assistant_margin", pa.float32()),
    ("user_informative_chars", pa.int32()), ("assistant_informative_chars", pa.int32()),
    ("user_top3", pa.string()), ("assistant_top3", pa.string()),
    ("user_share_in_assistant", pa.float32()), ("assistant_share_in_user", pa.float32()),
    ("status", pa.string()), ("discard", pa.bool_()), ("broad_mismatch", pa.bool_()), ("reason", pa.string()),
    ("user_text_sha256", pa.string()), ("assistant_text_sha256", pa.string()),
    ("detector_name", pa.string()), ("detector_version", pa.string()), ("rule_version", pa.string()),
])


class DetectorProcess:
    def __init__(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(VENDOR)
        self.process = subprocess.Popen(
            [str(PYTHON_313), str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env,
        )

    def request(self, mode: str, texts: list[str]) -> list[Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("detector pipe unavailable")
        self.process.stdin.write(json.dumps({"mode": mode, "texts": texts}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"detector exited unexpectedly: {error}")
        result = json.loads(line)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        status = self.process.wait(timeout=30)
        if status:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"detector exited {status}: {error}")


class DetectorPool:
    def __init__(self, count: int) -> None:
        self.workers = [DetectorProcess() for _ in range(count)]

    def detect(self, mode: str, texts: list[str], batch_size: int = 256) -> list[Any]:
        if not texts:
            return []
        chunks = [(index, texts[index:index + batch_size]) for index in range(0, len(texts), batch_size)]
        assignments: list[list[tuple[int, list[str]]]] = [[] for _ in self.workers]
        for number, chunk in enumerate(chunks):
            assignments[number % len(self.workers)].append(chunk)

        def run(worker: DetectorProcess, jobs: list[tuple[int, list[str]]]) -> list[tuple[int, list[Any]]]:
            return [(index, worker.request(mode, batch)) for index, batch in jobs]

        completed: list[tuple[int, list[Any]]] = []
        with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
            futures = [executor.submit(run, worker, jobs) for worker, jobs in zip(self.workers, assignments) if jobs]
            for future in futures:
                completed.extend(future.result())
        ordered: list[Any] = []
        for _, batch in sorted(completed, key=lambda item: item[0]):
            ordered.extend(batch)
        return ordered

    def close(self) -> None:
        for worker in self.workers:
            worker.close()


def excerpt(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:3000] + " " + text[-1000:]


def detection(value: dict[str, Any]) -> Detection:
    top3 = tuple((item["language"], float(item["confidence"])) for item in value.get("top3", []))
    return Detection(value.get("language", "unknown"), float(value.get("confidence", 0)), float(value.get("margin", 0)), top3)


def section_share(sections: list[dict[str, Any]], wanted_language: str) -> float:
    total = sum(max(0, int(section.get("word_count", 0))) for section in sections)
    if total <= 0:
        return 0.0
    wanted = language_group(wanted_language)
    matched = sum(int(section.get("word_count", 0)) for section in sections if language_group(section.get("language")) == wanted)
    return matched / total


def process_part(path: Path, output: Path, pool: DetectorPool) -> tuple[int, Counter[str]]:
    columns = ["target_id", "conversation_id", "user_id", "character_id", "pool", "teacher_family", "model", "topic", "language_final", "messages"]
    rows = pq.ParquetFile(path).read(columns=columns).to_pylist()
    pairs: list[tuple[str, str]] = []
    for row in rows:
        user, assistant = extract_pair(row["messages"])
        pairs.append((clean_text(user), clean_text(assistant)))
    texts = [excerpt(text) for pair in pairs for text in pair]
    detected = pool.detect("primary", texts)
    user_detected = [detection(detected[index * 2]) for index in range(len(rows))]
    assistant_detected = [detection(detected[index * 2 + 1]) for index in range(len(rows))]
    char_counts = [(informative_char_count(user), informative_char_count(assistant)) for user, assistant in pairs]
    preliminary = [classify_pair(user_detected[i], assistant_detected[i], *char_counts[i]) for i in range(len(rows))]
    candidates = [index for index, item in enumerate(preliminary) if item.discard]
    multi_texts = [excerpt(text) for index in candidates for text in pairs[index]]
    multi = pool.detect("multi", multi_texts, batch_size=128)
    shares: dict[int, tuple[float, float]] = {}
    for offset, index in enumerate(candidates):
        user_sections = multi[offset * 2]
        assistant_sections = multi[offset * 2 + 1]
        shares[index] = (
            section_share(assistant_sections, user_detected[index].language),
            section_share(user_sections, assistant_detected[index].language),
        )

    features: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    for index, row in enumerate(rows):
        user, assistant = pairs[index]
        user_share, assistant_share = shares.get(index, (0.0, 0.0))
        result = classify_pair(
            user_detected[index], assistant_detected[index], *char_counts[index],
            user_share_in_assistant=user_share, assistant_share_in_user=assistant_share,
        )
        statuses[result.status] += 1
        features.append({
            "target_id": row["target_id"], "part_name": path.name, "row_index": index,
            "conversation_id": row.get("conversation_id"), "user_id": row.get("user_id"), "character_id": row.get("character_id"),
            "pool": row.get("pool"), "teacher_family": row.get("teacher_family"), "model": row.get("model"), "topic": row.get("topic"),
            "assistant_language_existing": row.get("language_final"),
            "user_language": user_detected[index].language, "assistant_language": assistant_detected[index].language,
            "user_confidence": user_detected[index].confidence, "assistant_confidence": assistant_detected[index].confidence,
            "user_margin": user_detected[index].margin, "assistant_margin": assistant_detected[index].margin,
            "user_informative_chars": char_counts[index][0], "assistant_informative_chars": char_counts[index][1],
            "user_top3": json.dumps(user_detected[index].top3, ensure_ascii=False),
            "assistant_top3": json.dumps(assistant_detected[index].top3, ensure_ascii=False),
            "user_share_in_assistant": user_share, "assistant_share_in_user": assistant_share,
            "status": result.status, "discard": result.discard, "broad_mismatch": result.broad_mismatch, "reason": result.reason,
            "user_text_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
            "assistant_text_sha256": hashlib.sha256(assistant.encode("utf-8")).hexdigest(),
            "detector_name": DETECTOR_NAME, "detector_version": DETECTOR_VERSION, "rule_version": RULE_VERSION,
        })
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(features, schema=FEATURE_SCHEMA), temporary, compression="zstd", compression_level=6)
    os.replace(temporary, output)
    return len(rows), statuses


def aggregate(feature_dir: Path, root: Path, parts_dir: Path, elapsed: float) -> dict[str, Any]:
    status = Counter(); pools = Counter(); pairs = Counter(); user_languages = Counter(); assistant_languages = Counter(); existing_disagreement = Counter()
    excluded: list[dict[str, Any]] = []
    total = broad = 0
    for path in sorted(feature_dir.glob("part-*.parquet")):
        rows = pq.ParquetFile(path).read().to_pylist()
        total += len(rows)
        for row in rows:
            status[row["status"]] += 1
            user_languages[row["user_language"]] += 1
            assistant_languages[row["assistant_language"]] += 1
            pairs[(row["user_language"], row["assistant_language"])] += 1
            if row["broad_mismatch"]:
                broad += 1
            if language_group(row["assistant_language_existing"]) != language_group(row["assistant_language"]):
                existing_disagreement[(row["assistant_language_existing"], row["assistant_language"])] += 1
            if row["discard"]:
                pools[row["pool"]] += 1
                excluded.append({k: row[k] for k in ["target_id", "conversation_id", "user_id", "character_id", "pool", "user_language", "assistant_language", "user_confidence", "assistant_confidence", "status", "reason", "part_name", "row_index"]})
    excluded_path = root / "02_features" / "language_mismatch_excluded_v1_1.parquet"
    pq.write_table(pa.Table.from_pylist(excluded), excluded_path, compression="zstd")
    report = {
        "schema_version": 1, "rule_version": RULE_VERSION, "detector": {"name": DETECTOR_NAME, "version": DETECTOR_VERSION},
        "source": str(parts_dir), "feature_dir": str(feature_dir), "total": total,
        "strict_discard": len(excluded), "remaining": total - len(excluded), "strict_discard_rate": len(excluded) / total if total else 0,
        "broad_mismatch_before_strict_safeguards": broad, "status": dict(status), "discard_by_pool": dict(pools),
        "user_languages": dict(user_languages.most_common()), "assistant_languages": dict(assistant_languages.most_common()),
        "top_language_pairs": [{"user": pair[0], "assistant": pair[1], "count": count} for pair, count in pairs.most_common(100)],
        "assistant_existing_vs_lingua_disagreement": [{"existing": pair[0], "detected": pair[1], "count": count} for pair, count in existing_disagreement.most_common(100)],
        "thresholds": {"minimum_informative_letters_each_side": 20, "strict_confidence": 0.80, "strict_margin": 0.25, "broad_confidence": 0.65, "broad_margin": 0.15, "code_switch_counterpart_share_keep": 0.35},
        "compatible_groups": {"zh": ["zh-CN", "zh-TW", "zh"], "no": ["no", "nb", "nn"], "bcs": ["bs", "hr", "sr"], "msa": ["id", "ms"]},
        "elapsed_seconds": elapsed, "excluded_path": str(excluded_path),
    }
    (root / "00_manifest" / "LANGUAGE_ALIGNMENT_V1_1_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_review_sample(feature_dir: Path, parts_dir: Path, root: Path, limit: int = 200) -> None:
    ranked: list[tuple[str, str]] = []
    for path in feature_dir.glob("part-*.parquet"):
        for row in pq.ParquetFile(path).read(columns=["target_id", "discard"]).to_pylist():
            if row["discard"]:
                ranked.append((hashlib.sha256(("lusy-language-v1:" + row["target_id"]).encode()).hexdigest(), row["target_id"]))
    wanted = {target_id for _, target_id in sorted(ranked)[:limit]}
    samples = []
    for path in sorted(parts_dir.glob("part-*.parquet")):
        feature_rows = {row["target_id"]: row for row in pq.ParquetFile(feature_dir / path.name).read().to_pylist() if row["target_id"] in wanted}
        if not feature_rows:
            continue
        for row in pq.ParquetFile(path).read(columns=["target_id", "conversation_id", "messages"]).to_pylist():
            if row["target_id"] not in feature_rows:
                continue
            user, assistant = extract_pair(row["messages"])
            feature = feature_rows[row["target_id"]]
            samples.append({"target_id": row["target_id"], "conversation_id": row["conversation_id"], "user_language": feature["user_language"], "assistant_language": feature["assistant_language"], "user_confidence": feature["user_confidence"], "assistant_confidence": feature["assistant_confidence"], "user_excerpt": user[:800], "assistant_excerpt": assistant[:1200]})
    output = root / "06_review" / "language_alignment_v1_1_samples.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    feature_dir = arguments.root / "02_features" / "language_alignment_v1_1"
    feature_dir.mkdir(parents=True, exist_ok=True)
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ["language_alignment.py", "detector_worker.py", "build_language_alignment.py"]}
    manifest = {"rule_version": RULE_VERSION, "detector_name": DETECTOR_NAME, "detector_version": DETECTOR_VERSION, "source": str(arguments.parts), "workers": arguments.workers, "code_sha256": code_hashes}
    manifest_path = feature_dir / "_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("existing feature manifest differs")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic(); total = 0; pool = DetectorPool(arguments.workers)
    try:
        parts = sorted(arguments.parts.glob("part-*.parquet"))
        for number, path in enumerate(parts, start=1):
            output = feature_dir / path.name
            if output.exists():
                rows = pq.ParquetFile(output).metadata.num_rows; total += rows
            else:
                rows, _ = process_part(path, output, pool); total += rows
            if number == 1 or number % 8 == 0 or number == len(parts):
                print(json.dumps({"parts_done": number, "parts_total": len(parts), "rows_done": total, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        pool.close()
    elapsed = time.monotonic() - started
    report = aggregate(feature_dir, arguments.root, arguments.parts, elapsed)
    write_review_sample(feature_dir, arguments.parts, arguments.root)
    print(json.dumps({"complete": True, "total": report["total"], "strict_discard": report["strict_discard"], "remaining": report["remaining"], "report": str(arguments.root / "00_manifest" / "LANGUAGE_ALIGNMENT_V1_1_REPORT.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
