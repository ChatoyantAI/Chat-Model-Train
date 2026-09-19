"""Bounded asynchronous publication of reconstructible benchmark checkpoints."""
import json
from pathlib import Path
from queue import Queue
import threading
import uuid


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
