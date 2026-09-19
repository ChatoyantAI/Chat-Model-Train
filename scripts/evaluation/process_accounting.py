"""Independent, bounded, batched persistence of reconstructible usage ledgers.

Request intents and complete responses still use the existing durable journal
before delivery. Only the derived ledger moves off the broker's global lock.
Every batch is an immutable JSONL segment, compatible with existing recovery.
"""
import json
import multiprocessing
from pathlib import Path
import queue
import threading
import time
import uuid


class LedgerFailure(OSError):
    pass


def _writer(directory, incoming, acknowledgments, batch_size, batch_wait, publisher):
    if publisher is None:
        from recovery_queue_guard import atomic_text
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


def install(full, options):
    """Wrap the already-installed transport without changing its cache identity."""
    previous = full.Broker

    class ProcessAccountingBroker(previous):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.accounting = ProcessLedger(self.journal / 'ledgers', **options)

        def persist_entry(self, entry):
            # The transport calls this under its accounting lock. This method
            # does no filesystem I/O, no flush, and never waits for queue space.
            self.accounting.append(entry)

        def _check_accounting(self):
            try:
                self.accounting.check()
            except LedgerFailure as exc:
                with self.lock:
                    self.halt = 'accounting_writer_failure'
                self.storage_failure(exc, 'asynchronous_accounting')

        def send(self, payload):
            self._check_accounting()
            result = super().send(payload)
            if result[0] != 200:
                # Unlike successes, HTTP failures have no full response journal
                # to rebuild their error metadata from. Confirm their ledger
                # before delivery, after the transport released its global lock.
                try:
                    self.accounting.flush()
                except LedgerFailure:
                    self._check_accounting()
            return result

        def snapshot(self):
            self._check_accounting()
            value = super().snapshot()
            value['accounting_writer'] = self.accounting.snapshot()
            value['current_ledger'] = self.accounting.last_segment
            return value

        def flush_accounting(self):
            self.accounting.flush()

        def close_accounting(self):
            self.accounting.close()

    full.Broker = ProcessAccountingBroker
