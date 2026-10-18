import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from async_checkpoints import AsyncCheckpoints


class AsyncCheckpointTests(unittest.TestCase):
    def test_nonblocking_publication_coalesces_and_reads_latest(self):
        entered, release = threading.Event(), threading.Event()
        writes = []
        def slow(path, text):
            entered.set()
            release.wait(3)
            writes.append(json.loads(text))
            AsyncCheckpoints.publish(path, text)
        with tempfile.TemporaryDirectory() as tmp:
            writer = AsyncCheckpoints(workers=1, writer=slow)
            try:
                path = Path(tmp) / 'progress.json'
                writer.write(path, {'done': 1})
                self.assertTrue(entered.wait(1))
                for n in range(2, 101):
                    writer.write(path, {'done': n})
                self.assertEqual(writer.read(path), {'done': 100})
                self.assertFalse(path.exists())
                self.assertEqual(len(writer.pending), 1)
            finally:
                release.set()
                writer.close()
            self.assertEqual(json.loads(path.read_text()), {'done': 100})
            self.assertLessEqual(len(writes), 2)

    def test_bounded_backpressure_preserves_every_distinct_attempt(self):
        release, finished = threading.Event(), threading.Event()
        def slow(path, text):
            release.wait(3)
            AsyncCheckpoints.publish(path, text)
        with tempfile.TemporaryDirectory() as tmp:
            writer = AsyncCheckpoints(limit=2, workers=1, writer=slow)
            paths = [Path(tmp) / f'attempt-{i}.json' for i in range(3)]
            writer.write(paths[0], {'n': 0})
            writer.write(paths[1], {'n': 1})
            def put():
                writer.write(paths[2], {'n': 2})
                finished.set()
            thread = threading.Thread(target=put)
            thread.start()
            self.assertFalse(finished.wait(.05))
            release.set()
            thread.join(3)
            writer.close()
            self.assertTrue(finished.is_set())
            self.assertEqual([json.loads(p.read_text())['n'] for p in paths], [0, 1, 2])

    def test_background_error_surfaces_on_flush_and_new_writes(self):
        def fail(*args):
            raise OSError('test write failure')
        writer = AsyncCheckpoints(workers=1, writer=fail)
        writer.write(Path('unused.json'), {})
        with self.assertRaises(OSError):
            writer.flush()
        with self.assertRaises(OSError):
            writer.write(Path('another.json'), {})
        with self.assertRaises(OSError):
            writer.close()


if __name__ == '__main__':
    unittest.main()
