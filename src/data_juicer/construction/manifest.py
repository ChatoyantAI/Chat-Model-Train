"""Locked, atomic dataset manifest with idempotent shard recording."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
from typing import Any, Iterator

import yaml


class ManifestStore:
    def __init__(self, path: Path | str, dataset_id: str) -> None:
        self.path = Path(path)
        self.dataset_id = dataset_id
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def _empty(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "schema_version": 1, "shards": []}

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        with self.path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or self._empty()
        if data.get("dataset_id") != self.dataset_id:
            raise ValueError("manifest dataset_id mismatch")
        return data

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _atomic_write(self, data: dict[str, Any]) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        payload = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def record_shard(self, receipt: dict[str, Any]) -> None:
        required = {"shard_id", "table", "rows", "first_id", "last_id", "parquet_sha256"}
        missing = required.difference(receipt)
        if missing:
            raise ValueError(f"receipt missing fields: {sorted(missing)}")
        with self._locked():
            data = self.load()
            for existing in data["shards"]:
                if existing["shard_id"] == receipt["shard_id"]:
                    if existing == receipt:
                        return
                    raise ValueError(f"changed receipt for shard {receipt['shard_id']}")
            if receipt["table"] == "messages" and receipt["rows"]:
                for existing in data["shards"]:
                    if existing["table"] != "messages" or not existing["rows"]:
                        continue
                    disjoint = receipt["last_id"] < existing["first_id"] or receipt["first_id"] > existing["last_id"]
                    if not disjoint:
                        raise ValueError(f"message shard {receipt['shard_id']} overlaps {existing['shard_id']}")
            data["shards"].append(receipt)
            data["shards"].sort(key=lambda item: (item["table"], item["first_id"] or -1, item["shard_id"]))
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._atomic_write(data)
