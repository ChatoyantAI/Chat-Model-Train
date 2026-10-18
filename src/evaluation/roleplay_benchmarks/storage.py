"""Storage primitives shared by current runs and historical recovery tools.

atomic_* durably records paid request evidence before delivery. AsyncCheckpoints
publishes reconstructible progress; ProcessLedger batches derived accounting.
The bounded queues and ACK/flush behavior are unchanged by the module move.
"""
import errno
import json
import multiprocessing
from pathlib import Path
import queue
from queue import Queue
import os
import threading
import time
import uuid


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Retry the SAME bytes only, never the paid upstream request. Replacing a
    # complete file is idempotent even when a prior rename/fsync was uncertain.
    transient = {errno.EIO, errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT, errno.ESTALE, errno.EBUSY}
    for attempt in range(3):
        try:
            temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(path)
            if os.name != "nt":
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return
        except OSError as exc:
            if exc.errno not in transient or attempt == 2:
                raise
            time.sleep(.2 * (attempt + 1))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False) + "\n")


class AsyncCheckpoints:
    def __init__(self, limit=256, workers=4, writer=None):
        self.limit = limit
        self.writer = writer or self.publish
        self.pending = {}
        self.condition = threading.Condition()
        self.queue = Queue()
        self.error = None
        self.closed = False
        self.threads = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]
        for thread in self.threads:
            thread.start()

    @staticmethod
    def publish(path, text):
        # Same lightweight atomic-replace strategy as the original runner.
        # Paid request/response journals are persisted separately before reply.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)

    def check(self):
        if self.error is not None:
            raise OSError('Background checkpoint publication failed') from self.error

    def write(self, path, value):
        path = Path(path)
        text = json.dumps(value, ensure_ascii=False) + '\n'
        with self.condition:
            while path not in self.pending and len(self.pending) >= self.limit:
                self.check()
                self.condition.wait()
            self.check()
            if self.closed:
                raise RuntimeError('Checkpoint writer is closed')
            fresh = path not in self.pending
            self.pending[path] = text
            if fresh:
                self.queue.put(path)

    def read(self, path):
        with self.condition:
            self.check()
            text = self.pending.get(Path(path))
        return json.loads(text if text is not None else Path(path).read_text(encoding='utf-8'))

    def flush(self, prefix=None, name_prefix=None):
        def matches(path):
            return ((prefix is None or path == Path(prefix) or Path(prefix) in path.parents)
                    and (name_prefix is None or path.name.startswith(name_prefix)))
        with self.condition:
            while any(matches(path) for path in self.pending):
                self.check()
                self.condition.wait()
            self.check()

    def close(self):
        with self.condition:
            self.closed = True
        try:
            self.flush()
        finally:
            for _ in self.threads:
                self.queue.put(None)
            for thread in self.threads:
                thread.join()

    def _work(self):
        while True:
            path = self.queue.get()
            if path is None:
                return
            with self.condition:
                text = self.pending[path]
            try:
                self.writer(path, text)
            except Exception as exc:
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()
                return
            with self.condition:
                if self.pending[path] == text:
                    del self.pending[path]
                else:
                    self.queue.put(path)
                self.condition.notify_all()


class LedgerFailure(OSError):
    pass


def _writer(directory, incoming, acknowledgments, batch_size, batch_wait, publisher):
    if publisher is None:
        publisher = atomic_text
    while True:
        first = incoming.get()
        if first is None:
            return
        batch = [first]
        deadline = time.monotonic() + batch_wait
        while len(batch) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = incoming.get(timeout=remaining)
            except queue.Empty:
                break
            # close() only sends this sentinel after all records are acknowledged.
            if item is None:
                raise RuntimeError('Ledger closed before flush')
            batch.append(item)
        path = Path(directory) / ('batch-' + uuid.uuid4().hex + '.jsonl')
        try:
            publisher(path, ''.join(text for _, text in batch))
        except Exception as exc:
            acknowledgments.put(([key for key, _ in batch], None,
                                 {'type': type(exc).__name__, 'errno': getattr(exc, 'errno', None)}))
            return
        acknowledgments.put(([key for key, _ in batch], str(path), None))


class ProcessLedger:
    def __init__(self, directory, limit=512, batch_size=32, batch_wait=0.05,
                 publisher=None, context=None):
        if limit < 1 or batch_size < 1 or batch_size > limit or batch_wait < 0:
            raise ValueError('Invalid ledger queue configuration')
        self.limit = limit
        self.condition = threading.Condition()
        self.pending = set()
        self.error = None
        self.closed = False
        self.last_segment = None
        self.acknowledged = self.batches = 0
        self.stop_reader = threading.Event()
        context = context or multiprocessing.get_context('spawn')
        self.incoming = context.Queue(maxsize=limit)
        self.acknowledgments = context.Queue()
        self.process = context.Process(target=_writer, name='evaluation-ledger-writer',
            args=(str(directory), self.incoming, self.acknowledgments,
                  batch_size, batch_wait, publisher), daemon=True)
        self.process.start()
        self.reader = threading.Thread(target=self._read, name='ledger-acknowledgments', daemon=True)
        self.reader.start()

    def check(self):
        if self.error is not None:
            raise LedgerFailure('Ledger writer failed: ' + self.error)
        if not self.process.is_alive() and not self.closed:
            raise LedgerFailure('Ledger writer exited unexpectedly')

    def append(self, entry):
        # Snapshot before returning: later caller mutations cannot alter the record.
        text = json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n'
        key = uuid.uuid4().hex
        with self.condition:
            self.check()
            if self.closed:
                raise LedgerFailure('Ledger writer is closed')
            if len(self.pending) >= self.limit:
                raise LedgerFailure('Ledger queue is full; refusing further dispatch')
            self.pending.add(key)
            try:
                self.incoming.put_nowait((key, text))
            except queue.Full as exc:
                self.pending.remove(key)
                raise LedgerFailure('Ledger IPC queue is full') from exc

    def _read(self):
        while not self.stop_reader.is_set():
            try:
                keys, path, error = self.acknowledgments.get(timeout=0.2)
            except queue.Empty:
                if not self.process.is_alive() and not self.closed:
                    with self.condition:
                        self.error = 'writer_process_exited'
                        self.condition.notify_all()
                    return
                continue
            with self.condition:
                if error:
                    self.error = error['type'] + ':' + str(error['errno'])
                else:
                    self.pending.difference_update(keys)
                    self.acknowledged += len(keys)
                    self.batches += 1
                    self.last_segment = path
                self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {'pid': self.process.pid, 'pending_records': len(self.pending),
                    'acknowledged_records': self.acknowledged, 'durable_batches': self.batches,
                    'last_segment': self.last_segment, 'error': self.error}

    def flush(self, timeout=600):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.pending:
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LedgerFailure('Timed out flushing ledger')
                self.condition.wait(min(remaining, 0.2))
            self.check()

    def close(self):
        try:
            self.flush()
            self.closed = True
            self.incoming.put(None)
            self.process.join(10)
        finally:
            self.closed = True
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
            self.stop_reader.set()
            self.reader.join(1)
            self.incoming.cancel_join_thread()
            self.incoming.close()
            self.acknowledgments.close()
