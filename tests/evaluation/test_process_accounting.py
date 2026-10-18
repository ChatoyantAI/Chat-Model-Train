"""No paid/network calls: durability, failure, recovery, and lock isolation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/evaluation'))
from async_checkpoints import AsyncCheckpoints
from process_accounting import ProcessLedger, LedgerFailure, install as install_accounting
from recovery_queue_guard import atomic_text, install


class PausedPublisher:
    def __init__(self, entered, release):
        self.entered, self.release = entered, release

    def __call__(self, path, text):
        self.entered.set()
        if not self.release.wait(15):
            raise OSError('Test writer pause timed out')
        atomic_text(path, text)


def fail_publish(path, text):
    raise OSError(5, 'offline injected I/O failure')


class BrokerContract:
    def __init__(self, out, key, remote_url, local_url, remote_slots=4, local_slots=2, budget=100):
        self.out, self.key, self.remote_url, self.local_url = out, key, remote_url, local_url
        self.budget, self.lock, self.halt = budget, threading.Lock(), None
        self.active, self.completed, self.errors = Counter(), Counter(), Counter()
        self.cost = self.unknown_reserve = 0
        self.ledger = out / 'broker-requests.jsonl'

    def snapshot(self):
        with self.lock:
            return {'active': dict(self.active), 'halt': self.halt}


class AccountingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = multiprocessing.get_context('spawn')

    def close(self, ledger):
        try:
            ledger.close()
        except LedgerFailure:
            pass

    def make_broker(self, **options):
        full = SimpleNamespace(Broker=BrokerContract, ACTORS={'gemma': 'gemma'},
                               JUDGE='zai-org/GLM-5.3-Flash', model_guard=lambda _: None)
        checkpoints = AsyncCheckpoints()
        self.addCleanup(checkpoints.close)
        install(full, checkpoints=checkpoints, budget_guard=False)
        install_accounting(full, options)
        broker = full.Broker(self.root, 'offline-key', 'http://offline', 'http://offline', 4, 2)
        self.addCleanup(self.close, broker.accounting)
        return broker

    def opener(self):
        def reply(*args, **kwargs):
            return io.StringIO(json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'content': '2'}}], 'usage': {'cost': 0.002}}))
        return patch('recovery_queue_guard.request.build_opener',
                     return_value=SimpleNamespace(open=reply))

    @staticmethod
    def payload(i):
        return {'model': 'zai-org/GLM-5.3-Flash', 'messages': [{'role': 'user', 'content': str(i)}]}

    def test_slow_disk_does_not_hold_dispatch_lock_and_responses_are_durable(self):
        entered, release = self.context.Event(), self.context.Event()
        broker = self.make_broker(publisher=PausedPublisher(entered, release))
        try:
            with self.opener():
                self.assertEqual(broker.send(self.payload(1))[0], 200)
                self.assertTrue(entered.wait(10))
                started = time.monotonic()
                self.assertEqual(broker.send(self.payload(2))[0], 200)
                broker.snapshot()
                self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(len(list((broker.journal / 'operations-v2').glob('*/*.response.json'))), 2)
            self.assertAlmostEqual(broker.cost, .004)
            self.assertEqual(broker.accounting.snapshot()['pending_records'], 2)
        finally:
            release.set()
        broker.flush_accounting()
        rows = [json.loads(line) for f in (broker.journal / 'ledgers').glob('*.jsonl')
                for line in f.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(sum(row['cost'] for row in rows), .004)

    def test_batches_are_immutable_and_flush_keeps_every_record(self):
        ledger_dir = self.root / 'ledgers'
        ledger_dir.mkdir()
        old = ledger_dir / 'old.jsonl'
        old.write_text('{"historical":true}\n')
        ledger = ProcessLedger(ledger_dir, batch_size=32, batch_wait=.1)
        self.addCleanup(self.close, ledger)
        for i in range(80):
            ledger.append({'operation_id': str(i), 'cost': .01})
        ledger.flush()
        files = list(ledger_dir.glob('batch-*.jsonl'))
        rows = [json.loads(line) for f in files for line in f.read_text().splitlines()]
        self.assertEqual(len(rows), 80)
        self.assertEqual({row['operation_id'] for row in rows}, {str(i) for i in range(80)})
        self.assertLess(len(files), 80)
        self.assertEqual(old.read_text(), '{"historical":true}\n')
        self.assertEqual(ledger.snapshot()['pending_records'], 0)

    def test_full_queue_fails_immediately_instead_of_blocking_caller(self):
        entered, release = self.context.Event(), self.context.Event()
        ledger = ProcessLedger(self.root / 'ledgers', limit=1, batch_size=1,
                               publisher=PausedPublisher(entered, release))
        self.addCleanup(self.close, ledger)
        try:
            ledger.append({'operation_id': 'first'})
            self.assertTrue(entered.wait(10))
            started = time.monotonic()
            with self.assertRaises(LedgerFailure):
                ledger.append({'operation_id': 'second'})
            self.assertLess(time.monotonic() - started, .1)
        finally:
            release.set()

    def test_writer_failure_stops_new_upstream_calls(self):
        broker = self.make_broker(publisher=fail_publish)
        with self.opener():
            self.assertEqual(broker.send(self.payload(1))[0], 200)
        with self.assertRaises(LedgerFailure):
            broker.flush_accounting()
        with patch('recovery_queue_guard.request.build_opener',
                   side_effect=AssertionError('Must not dispatch after storage failure')):
            self.assertEqual(broker.send(self.payload(2))[0], 503)
        self.assertEqual(broker.halt, 'accounting_writer_failure')

    def test_failed_request_waits_for_error_evidence_without_holding_global_lock(self):
        entered, release = self.context.Event(), self.context.Event()
        broker = self.make_broker(publisher=PausedPublisher(entered, release))
        def fail(*args, **kwargs):
            raise HTTPError('http://offline', 502, 'offline error', {}, io.BytesIO(b'fixture error'))
        with patch('recovery_queue_guard.request.build_opener',
                   return_value=SimpleNamespace(open=fail)), ThreadPoolExecutor(1) as pool:
            future = pool.submit(broker.send, self.payload(1))
            try:
                self.assertTrue(entered.wait(10))
                self.assertFalse(future.done())
                start = time.monotonic()
                broker.snapshot()
                self.assertLess(time.monotonic() - start, .1)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=10)[0], 502)
        rows = [json.loads(line) for f in (broker.journal / 'ledgers').glob('*.jsonl')
                for line in f.read_text().splitlines()]
        self.assertEqual(rows[0]['error_type'], 'upstream_http')



if __name__ == '__main__':
    unittest.main()
