"""Streaming I/O helpers for frozen roleplay evaluation data."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


def iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        handle_context = opener(path, "rt", encoding="utf-8")
    else:
        handle_context = opener(path, "r", encoding="utf-8")
    with handle_context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc


def iter_grouped_json_lines(
    path: Path,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    current_id: str | None = None
    records: list[dict[str, Any]] = []
    for record in iter_json_lines(path):
        comparison_id = str(record["comparison_id"])
        if current_id is not None and comparison_id != current_id:
            yield current_id, records
            records = []
        current_id = comparison_id
        records.append(record)
    if current_id is not None:
        yield current_id, records


class GroupReader:
    """Consume one comparison group at a time from a comparison-sorted JSONL."""

    def __init__(self, path: Path):
        self.path = path
        self._groups = iter(iter_grouped_json_lines(path))
        self._current = next(self._groups, None)

    def take(self, comparison_id: str) -> list[dict[str, Any]]:
        while self._current is not None and self._current[0] < comparison_id:
            raise ValueError(
                f"orphaned comparison group {self._current[0]} in {self.path} "
                f"before {comparison_id}"
            )
        if self._current is None or self._current[0] != comparison_id:
            return []
        records = self._current[1]
        self._current = next(self._groups, None)
        return records

    def assert_finished(self) -> None:
        if self._current is not None:
            raise ValueError(
                f"unconsumed comparison group {self._current[0]} in {self.path}"
            )


def load_runtime_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    runtime: dict[str, dict[str, Any]] = {}
    for record in iter_json_lines(path):
        comparison_id = str(record["comparison_id"])
        if comparison_id in runtime:
            raise ValueError(f"duplicate runtime snapshot for {comparison_id}")
        runtime[comparison_id] = record
    return runtime


def find_json_record(
    path: Path,
    comparison_id: str,
) -> dict[str, Any] | None:
    for record in iter_json_lines(path):
        current_id = str(record["comparison_id"])
        if current_id == comparison_id:
            return record
        if current_id > comparison_id:
            return None
    return None


def find_json_group(path: Path, comparison_id: str) -> list[dict[str, Any]]:
    for current_id, records in iter_grouped_json_lines(path):
        if current_id == comparison_id:
            return records
        if current_id > comparison_id:
            return []
    return []


def file_metadata(path: Path, records: int) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "filename": path.name,
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
