"""Shared concurrency and accounting for one evaluation supervisor."""
from contextlib import contextmanager
from pathlib import Path
import threading
import time

from .benchmark_api import object_hash
from .storage import AsyncCheckpoints as DerivedStore, ProcessLedger


class EndpointLimiter:
    """One shared limit per endpoint kind for all selected benchmarks."""

    def __init__(self, limits):
        self._semaphores = {kind: threading.BoundedSemaphore(int(value))
                            for kind, value in limits.items()}
        self._active = {kind: 0 for kind in limits}
        self._lock = threading.Lock()

    @contextmanager
    def slot(self, kind):
        semaphore = self._semaphores[kind]
        semaphore.acquire()
        with self._lock:
            self._active[kind] += 1
        try:
            yield
        finally:
            with self._lock:
                self._active[kind] -= 1
            semaphore.release()

    def snapshot(self):
        with self._lock:
            return dict(self._active)


class UsageLedger:
    """Non-blocking, process-isolated accounting for requests dispatched here."""

    def __init__(self, directory, config):
        options = config.get("process_accounting", {})
        self.ledger = ProcessLedger(Path(directory) / "usage-ledger",
                                     limit=int(options.get("limit", 512)),
                                     batch_size=int(options.get("batch_size", 64)),
                                     batch_wait=float(options.get("batch_wait", .25)))

    def append(self, role, payload, status, usage=None, error=None):
        usage = usage if isinstance(usage, dict) else {}
        self.ledger.append({"at": time.time(), "role": role.get("name", "unknown"),
                            "model": role.get("model"), "request_hash": object_hash(payload),
                            "status": status, "usage": usage,
                            "cost": usage.get("cost", usage.get("estimated_cost")),
                            "error": error})

    def flush(self):
        self.ledger.flush()

    def snapshot(self):
        return self.ledger.snapshot()

    def close(self):
        self.ledger.close()
