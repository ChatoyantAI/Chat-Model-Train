from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "comparison_id",
    "model",
    "model_id",
    "model_name",
    "pair_group_id",
    "human_selected",
    "left",
    "right",
    "winner",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_forbidden_keys(value: Any, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                found.add(str(key))
            _find_forbidden_keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _find_forbidden_keys(child, found)


def _validate_record(record: dict[str, Any], line_number: int) -> None:
    required = {"evaluation_id", "input_messages", "output_message"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"line {line_number}: missing keys {sorted(missing)}")
    if not isinstance(record["input_messages"], list):
        raise ValueError(f"line {line_number}: input_messages must be a list")
    output = record["output_message"]
    if not isinstance(output, dict) or output.get("role") != "assistant":
        raise ValueError(f"line {line_number}: output_message must be assistant")
    forbidden: set[str] = set()
    _find_forbidden_keys(record, forbidden)
    if forbidden:
        raise ValueError(f"line {line_number}: blind-data violation {sorted(forbidden)}")


def build_samples_db(source: Path, output: Path, overwrite: bool = False) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()

    source_hash = _sha256(source)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE samples (
                seq INTEGER PRIMARY KEY,
                evaluation_id TEXT NOT NULL UNIQUE,
                trigger_type TEXT NOT NULL,
                payload_zlib BLOB NOT NULL
            );
            """
        )
        total = 0
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for total, line in enumerate(handle, start=1):
                record = json.loads(line)
                _validate_record(record, total)
                canonical = json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                connection.execute(
                    "INSERT INTO samples(seq, evaluation_id, trigger_type, payload_zlib) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        total,
                        record["evaluation_id"],
                        str(record.get("trigger_type", "unknown")),
                        sqlite3.Binary(zlib.compress(canonical, level=6)),
                    ),
                )
                if total % 250 == 0:
                    connection.commit()

        metadata = {
            "schema_version": "1",
            "source_file": source.name,
            "source_sha256": source_hash,
            "sample_count": str(total),
            "built_at": _utc_now(),
            "payload_encoding": "utf-8 JSON compressed with zlib",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.execute("CREATE INDEX idx_samples_evaluation_id ON samples(evaluation_id)")
        connection.commit()
        connection.execute("VACUUM")
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()

    os.replace(temporary, output)
    return {
        "output": str(output),
        "sample_count": total,
        "source_sha256": source_hash,
        "output_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build random-access blind sample DB")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build_samples_db(args.input, args.output, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
