"""Offline regression tests: these must never contact a model provider."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib import error, request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from recovery_queue_guard import (FairGate, NotAdmitted, atomic_json, digest, install,
    install_journal_replay, install_coser_journal_replay, normalize_payload, seed_coser_cache, valid_journal_response,
    call_scope, transport_key, install_checkpoint_io, install_existing_rmt_judging, journal_path,
    load_legacy_response_names)
from async_checkpoints import AsyncCheckpoints
from src.evaluation.roleplay_benchmarks import benchmark_api as api


class BrokerContract:
    """The frozen constructor/snapshot contract, without migration dependencies."""
    def __init__(self, out, key, remote_url, local_url, remote_slots=6, local_slots=16, budget=100):
        self.out, self.key, self.remote_url, self.local_url = out, key, remote_url, local_url
        self.budget, self.lock, self.halt = budget, threading.Lock(), None
        self.active, self.completed, self.errors = Counter(), Counter(), Counter()
        self.ledger = out / "broker-requests.jsonl"
        rows = [json.loads(line) for line in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        self.cost = sum(x.get("cost", 0) for x in rows)
        self.unknown_reserve = sum(x.get("unknown_reserve", 0) for x in rows)

    def snapshot(self):
        return {"active": dict(self.active), "halt": self.halt}


def response(text="2", finish="stop"):
    return {"id": "offline-response", "model": "judge", "choices": [
        {"finish_reason": finish, "message": {"content": text}}],
        "usage": {"cost": .002, "prompt_tokens": 100, "completion_tokens": 1}}


def saved_legacy_index(journal, names):
    """Historical index fixture; the retired migration CLI is not a test dependency."""
    directory = Path(journal) / 'responses'
    path = Path(journal) / 'legacy-response-index-v1.json'
    atomic_json(path, {'format': 1, 'directory': str(directory.resolve()),
                       'mtime_ns': directory.stat().st_mtime_ns,
                       'names': names, 'names_sha256': digest(names)})
    return path


class TransportTests(unittest.TestCase):
    def test_more_producers_do_not_exceed_shared_upstream_limit(self):
        full = SimpleNamespace(Broker=BrokerContract, ACTORS={'gemma': 'gemma'}, JUDGE='judge', model_guard=lambda url: None)
        install(full, budget_guard=False)
        broker = full.Broker(self.root / 'shared-limit', 'fixture', 'http://offline', 'http://local', 2, 1, 100)
        lock = threading.Lock()
        active = peak = 0
        def upstream(req, timeout):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.05)
            with lock:
                active -= 1
            return io.StringIO(json.dumps(response()))
        with self.opener(upstream), ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda i: broker.send({**self.payload,
                'messages': [{'role': 'user', 'content': str(i)}]})[0], range(8)))
        self.assertEqual(results, [200] * 8)
        self.assertEqual(peak, 2)
        self.assertAlmostEqual(broker.cost, .016)

    def test_legacy_index_replays_success_without_negative_legacy_stats(self):
        key = transport_key(self.payload)
        directory = self.broker.journal / 'responses'
        atomic_json(directory / (key + '.json'), {'payload': normalize_payload(self.payload), 'response': response()})
        saved_legacy_index(self.broker.journal, [key + '.json'])
        names = load_legacy_response_names(self.broker.journal)
        exists = Path.exists
        def no_legacy_stat(path):
            if path.parent == directory:
                self.fail('Legacy file existence must be checked in memory')
            return exists(path)
        with patch.object(Path, 'exists', no_legacy_stat):
            self.assertEqual(valid_journal_response(self.broker.journal, self.payload, int, legacy_names=names)[2], 2)
            self.assertIsNone(valid_journal_response(self.broker.journal, {**self.payload, 'messages': []}, int, legacy_names=names))
        atomic_json(directory / ('f' * 64 + '.json'), {})
        previous = directory.stat().st_mtime_ns
        os.utime(directory, ns=(previous + 1000000000, previous + 1000000000))
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            load_legacy_response_names(self.broker.journal)

    def test_legacy_index_detects_missing_names_from_corruption(self):
        directory = self.broker.journal / 'responses'
        atomic_json(directory / ('a' * 64 + '.json'), {})
        index = saved_legacy_index(self.broker.journal, ['a' * 64 + '.json'])
        value = json.loads(index.read_text())
        value['names'] = []
        atomic_json(index, value)
        with self.assertRaisesRegex(RuntimeError, 'invalid'):
            load_legacy_response_names(self.broker.journal)

    def test_disabled_budget_keeps_cost_accounting_and_concurrency_limit(self):
        full = SimpleNamespace(Broker=BrokerContract, ACTORS={'gemma': 'gemma'}, JUDGE='judge', model_guard=lambda url: None)
        install(full, budget_guard=False)
        broker = full.Broker(self.root / 'uncapped', 'fixture', 'http://offline', 'http://local', 2, 1, 1)
        broker.cost, broker.unknown_reserve = 10, 20
        with self.opener(lambda req, timeout: io.StringIO(json.dumps(response()))):
            self.assertEqual(broker.send(self.payload)[0], 200)
        self.assertAlmostEqual(broker.cost, 10.002)
        self.assertEqual(broker.unknown_reserve, 20)
        self.assertEqual(broker.gates[False].capacity, 2)
        self.assertIsNone(broker.halt)
        self.assertFalse(broker.snapshot()['budget_guard_enabled'])
        self.assertIsNone(broker.snapshot()['budget_usd'])
        self.assertEqual(json.loads(broker.ledger.read_text())['cost'], .002)

    def test_interrupted_unknown_charge_is_not_redispatched(self):
        key = transport_key(self.payload)
        self.broker.quarantined.add(key)
        with self.opener(lambda *a, **k: self.fail('uncertain request was resent')):
            status, body, headers = self.broker.send(self.payload)
        self.assertEqual(status, 409)
        self.assertEqual(headers['X-Eval-Quarantined'], '1')
        self.assertFalse(self.broker.ledger.exists())

    def test_sharded_durable_response_and_drain_admission(self):
        writer = AsyncCheckpoints()
        self.addCleanup(writer.close)
        full = SimpleNamespace(Broker=BrokerContract, ACTORS={'gemma': 'gemma'}, JUDGE='judge', model_guard=lambda url: None)
        install(full, checkpoints=writer)
        broker = full.Broker(self.root / 'sharded', 'fixture', 'http://offline', 'http://local', 2, 2, 100)
        with self.opener(lambda req, timeout: io.StringIO(json.dumps(response()))):
            self.assertEqual(broker.send(self.payload)[0], 200)
        self.assertEqual(len(list((broker.journal / 'operations-v2').glob('*/*.response.json'))), 1)
        self.assertEqual(valid_journal_response(broker.journal, self.payload, int)[2], 2)
        self.assertAlmostEqual(broker.cost, .002)
        server = full.serve_broker(broker, 0)
        try:
            url = f'http://127.0.0.1:{server.server_port}/control/drain'
            with request.urlopen(request.Request(url, data=b''), timeout=2) as reply:
                self.assertEqual(json.load(reply)['halt'], 'maintenance_pause')
            with self.opener(lambda *a, **k: self.fail('drain sent an upstream request')):
                self.assertEqual(broker.send(self.payload)[0], 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_existing_rmt_only_judges_once_and_preserves_other_suites(self):
        from unittest.mock import Mock
        full = SimpleNamespace(run_public=Mock(return_value="other-suite"))
        previous = full.run_public
        community = SimpleNamespace(run=Mock(return_value={"coverage": {"complete": False}}))
        broker = SimpleNamespace(halt=None)
        install_existing_rmt_judging(full, community)
        result = full.run_public(self.root, "gemma", "rmtbench-claude45-long-v1", broker)
        self.assertFalse(result["coverage"]["complete"])
        community.run.assert_called_once_with(self.root / "gemma/rmtbench-claude45-long-v1", "judge", True, 300000)
        previous.assert_not_called()
        self.assertEqual(full.run_public(self.root, "gemma", "minimax", broker), "other-suite")
        broker.halt = "budget_guard"
        self.assertIsNone(full.run_public(self.root, "gemma", "rmtbench-claude45-long-v1", broker))
        self.assertEqual(community.run.call_count, 1)

    def test_glm_restores_frozen_payload_but_gemma_keeps_thinking_disabled(self):
        original = {"model": "zai-org/GLM-5.3-Flash", "messages": [{"role": "user", "content": "unchanged"}], "max_tokens": 8192}
        migrated = {**original, "chat_template_kwargs": {"enable_thinking": False}}
        self.assertEqual(normalize_payload(original), original)
        self.assertEqual(normalize_payload(migrated), original)
        self.assertIn("chat_template_kwargs", migrated)
        gemma = {**original, "model": "gemma4-base-it"}
        self.assertEqual(normalize_payload(gemma)["chat_template_kwargs"], {"enable_thinking": False})

    def test_glm_legacy_journal_is_reused_only_in_original_scope(self):
        payload = {"model": "zai-org/GLM-5.3-Flash", "messages": []}
        legacy = {**payload, "chat_template_kwargs": {"enable_thinking": False}}
        key = digest({"scope": "case-a", "payload": legacy})
        path = self.root / "responses" / (key + ".json")
        atomic_json(path, {"payload": legacy, "response": response()})
        before = path.read_bytes()
        found = valid_journal_response(self.root, payload, int, "case-a")
        self.assertEqual(found[2], 2)
        self.assertEqual(path.read_bytes(), before)
        self.assertIsNone(valid_journal_response(self.root, payload, int, "case-b"))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.full = SimpleNamespace(Broker=BrokerContract, ACTORS={"gemma": "gemma"}, JUDGE="judge", model_guard=lambda url: None)
        install(self.full, upstream_timeout=.15, queue_timeout=.1)
        self.broker = self.full.Broker(self.root, "not-a-real-key", "http://127.0.0.1:1/v1", "http://127.0.0.1:1/v1", 1, 1, 100)
        self.payload = {"model": "judge", "messages": [{"role": "user", "content": "原始 query"}], "stream": False, "temperature": 0, "max_tokens": 8192}

    def opener(self, callback):
        return patch("recovery_queue_guard.request.build_opener", return_value=SimpleNamespace(open=callback))

    def test_preserves_payload_and_journals_before_delivery(self):
        original = copy.deepcopy(self.payload)
        seen = []
        def upstream(req, timeout):
            seen.append(json.loads(req.data))
            return io.StringIO(json.dumps(response()))
        with self.opener(upstream):
            status, body, _ = self.broker.send(self.payload)
        self.assertEqual(status, 200)
        self.assertEqual(self.payload, original)
        self.assertEqual(seen, [normalize_payload(original)])
        self.assertEqual({k: v for k, v in seen[0].items() if k != "chat_template_kwargs"}, original)
        self.assertIsNotNone(valid_journal_response(self.broker.journal, original, int))
        self.assertEqual(self.broker.cost, .002)
        self.assertEqual(len(list((self.broker.journal / "operations").glob("*.response.json"))), 1)

    def test_identical_concurrent_requests_bill_once(self):
        entered, release = threading.Event(), threading.Event()
        def upstream(req, timeout):
            entered.set()
            self.assertTrue(release.wait(2))
            return io.StringIO(json.dumps(response()))
        with self.opener(upstream), ThreadPoolExecutor(2) as pool:
            a = pool.submit(self.broker.send, self.payload)
            self.assertTrue(entered.wait(1))
            b = pool.submit(self.broker.send, self.payload)
            deadline = time.monotonic() + 1
            while self.broker.deduplicated != 1 and time.monotonic() < deadline:
                time.sleep(.005)
            release.set()
            self.assertEqual(a.result(), b.result())
        self.assertEqual(self.broker.deduplicated, 1)
        self.assertEqual(len(self.broker.ledger.read_text().splitlines()), 1)
        self.assertEqual(self.broker.cost, .002)

    def test_identical_prompts_in_different_samples_remain_independent(self):
        barrier = threading.Barrier(2)
        # Give the independent samples capacity to run at the same time.
        self.broker.gates[False].capacity = 2
        def upstream(*a, **k):
            barrier.wait(timeout=2)
            return io.StringIO(json.dumps(response()))
        def send(scope):
            self.broker.context.scope = scope
            return self.broker.send(self.payload)
        with self.opener(upstream), ThreadPoolExecutor(2) as pool:
            results = list(pool.map(send, ["a" * 64, "b" * 64]))
        self.assertEqual([r[0] for r in results], [200, 200])
        self.assertEqual(self.broker.deduplicated, 0)
        self.assertEqual(len(self.broker.ledger.read_text().splitlines()), 2)

    def test_disconnected_request_never_dispatches_or_bills(self):
        self.broker.context.cancelled = lambda: True
        with self.opener(lambda *a, **k: self.fail("abandoned request dispatched")):
            self.assertEqual(self.broker.send(self.payload)[0], 429)
        self.assertFalse(self.broker.ledger.exists())
        self.assertEqual(self.broker.unknown_reserve, 0)

    def test_queue_deadline_is_bounded(self):
        with self.broker.gates[False].enter(time.monotonic() + 1):
            with self.opener(lambda *a, **k: self.fail("queued request dispatched")):
                result = self.broker.send(self.payload)
        self.assertEqual(result[0], 429)
        self.assertEqual(result[1]["error"], "queue_deadline_exceeded")
        self.assertFalse(self.broker.ledger.exists())

    def test_upstream_timeout_is_distinct_from_http_502(self):
        def timeout(*a, **k):
            raise TimeoutError()
        with self.opener(timeout):
            self.assertEqual(self.broker.send(self.payload)[0], 504)
        def bad_gateway(*a, **k):
            raise error.HTTPError("http://offline", 502, "bad", {}, io.BytesIO(b"upstream down"))
        with self.opener(bad_gateway):
            self.assertEqual(self.broker.send(self.payload)[0], 502)
        rows = [json.loads(x) for x in self.broker.ledger.read_text().splitlines()]
        self.assertEqual([x["error_type"] for x in rows], ["upstream_timeout", "upstream_http"])
        self.assertEqual(self.broker.unknown_reserve, .2)

    def test_budget_preserves_existing_spend_and_reserve(self):
        self.broker.cost, self.broker.unknown_reserve = 50, 49.95
        with self.opener(lambda *a, **k: self.fail("budget exceeded")):
            self.assertEqual(self.broker.send(self.payload)[0], 402)
        self.assertEqual(self.broker.cost, 50)
        self.assertEqual(self.broker.unknown_reserve, 49.95)

    def test_restart_sums_segments_without_rewriting_original_ledger(self):
        old = self.root / "broker-requests.jsonl"
        old.write_text(json.dumps({"cost": 2, "unknown_reserve": 3}) + "\n")
        before = old.read_bytes()
        with self.opener(lambda *a, **k: io.StringIO(json.dumps(response()))):
            self.assertEqual(self.broker.send(self.payload)[0], 200)
        restarted = self.full.Broker(self.root, "key", "remote", "local")
        self.assertAlmostEqual(restarted.cost, 2.002)
        self.assertEqual(restarted.unknown_reserve, 3)
        self.assertEqual(old.read_bytes(), before)

    def test_ambiguous_ledger_sync_does_not_duplicate_charge_or_api_call(self):
        fsync = os.fsync
        failed = []
        calls = []
        def reply(*a, **k):
            calls.append(1)
            return io.StringIO(json.dumps(response()))
        def ambiguous(descriptor):
            result = fsync(descriptor)
            if self.broker.ledger.exists() and not failed:
                failed.append(True)
                raise OSError(errno.EIO, "ambiguous sync", str(self.broker.ledger))
            return result
        with patch("recovery_queue_guard.os.fsync", ambiguous), patch("recovery_queue_guard.time.sleep"), self.opener(reply):
            self.assertEqual(self.broker.send(self.payload)[0], 200)
        self.assertEqual(len(calls), 1)
        self.assertEqual(failed, [True])
        rows = self.broker.ledger.read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.broker.cost, .002)

    def test_ledger_failure_reports_errno_and_halt_header(self):
        with patch.object(self.broker, "persist_entry", side_effect=OSError(errno.ENOSPC, "full", "/ledger")), self.opener(lambda *a, **k: io.StringIO(json.dumps(response()))):
            status, _, headers = self.broker.send(self.payload)
        self.assertEqual(status, 503)
        self.assertIn("X-Eval-Halted", headers)
        self.assertEqual(self.broker.snapshot()["status"], "halted")
        self.assertEqual(self.broker.last_failure["errno"], errno.ENOSPC)
        self.assertEqual(self.broker.last_failure["path"], "/ledger")

    def test_null_cost_uses_actual_estimated_cost_without_false_reserve(self):
        body = response()
        body["usage"].update(cost=None, estimated_cost=.003)
        with self.opener(lambda *a, **k: io.StringIO(json.dumps(body))):
            self.assertEqual(self.broker.send(self.payload)[0], 200)
        self.assertEqual(self.broker.cost, .003)
        self.assertEqual(self.broker.unknown_reserve, 0)

    def test_persistence_failure_before_dispatch_is_not_charged(self):
        with patch("recovery_queue_guard.atomic_json", side_effect=OSError("disk full")), self.opener(lambda *a, **k: self.fail("disk failed before dispatch")):
            self.assertEqual(self.broker.send(self.payload)[0], 503)
        self.assertTrue(self.broker.halt)
        self.assertEqual(self.broker.unknown_reserve, 0)

    def test_persistence_failure_after_response_halts_and_retains_cost(self):
        def save(path, value):
            if path.name.endswith(".response.json"):
                raise OSError("disk full")
            atomic_json(path, value)
        with patch("recovery_queue_guard.atomic_json", side_effect=save), self.opener(lambda *a, **k: io.StringIO(json.dumps(response()))):
            self.assertEqual(self.broker.send(self.payload)[0], 503)
        self.assertTrue(self.broker.halt)
        self.assertEqual(self.broker.cost, .002)

    def test_reasoning_is_rejected_but_response_and_cost_retained(self):
        body = response()
        body["choices"][0]["message"]["reasoning_content"] = "unexpected"
        with self.opener(lambda *a, **k: io.StringIO(json.dumps(body))):
            self.assertEqual(self.broker.send(self.payload)[0], 422)
        self.assertEqual(self.broker.halt, "thinking_not_disabled")
        self.assertEqual(self.broker.cost, .002)
        self.assertIsNone(valid_journal_response(self.broker.journal, self.payload, int))

    def test_historical_glm_reasoning_is_retained_and_replayable(self):
        self.full.JUDGE = "zai-org/GLM-5.3-Flash"
        payload = {**self.payload, "model": self.full.JUDGE}
        body = response()
        body["choices"][0]["message"]["reasoning_content"] = "provider reasoning"
        body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 38}
        with self.opener(lambda *a, **k: io.StringIO(json.dumps(body))):
            status, actual, _ = self.broker.send(payload)
        self.assertEqual(status, 200)
        self.assertEqual(actual, body)
        self.assertIsNone(self.broker.halt)
        self.assertEqual(valid_journal_response(self.broker.journal, payload, int)[1], body)

    def test_http_round_trip_and_disconnected_socket(self):
        server = self.full.serve_broker(self.broker, 0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = "http://127.0.0.1:" + str(server.server_port) + "/v1/chat/completions"
        with patch.object(self.broker, "dispatch", return_value=(200, response(), {})) as dispatch:
            req = request.Request(url, data=json.dumps(self.payload).encode())
            with request.urlopen(req, timeout=2) as reply:
                self.assertEqual(json.load(reply), response())
            self.assertEqual(dispatch.call_count, 1)
        with self.broker.gates[False].enter(time.monotonic() + 1):
            with patch.object(self.broker, "dispatch", side_effect=AssertionError("ghost dispatch")):
                conn = socket.create_connection(("127.0.0.1", server.server_port))
                data = json.dumps(self.payload).encode()
                conn.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
                deadline = time.monotonic() + 1
                while self.broker.rejected == 0 and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertEqual(self.broker.rejected, 1)


class ReplayTests(unittest.TestCase):
    def test_quarantine_skips_one_request_without_stopping_case(self):
        client = api.Client(self.config)
        exc = error.HTTPError('http://127.0.0.1/offline', 409, 'quarantined',
                              {'X-Eval-Quarantined': '1'}, io.BytesIO(b'{}'))
        opener = SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(exc))
        with patch.object(api, 'endpoint', return_value=('http://127.0.0.1/offline', 'unused')), patch('recovery_queue_guard.request.build_opener', return_value=opener):
            with self.assertRaises(api.CallFailed) as caught:
                client.call(self.root / 'requests', 'judge', self.messages, int)
        self.assertNotIsInstance(caught.exception, api.RequestLimit)
        self.assertEqual(client.count, 1)

    def test_durable_shard_replays_while_derived_checkpoint_is_still_pending(self):
        release = threading.Event()
        def delayed(path, text):
            release.wait(3)
            AsyncCheckpoints.publish(path, text)
        writer = AsyncCheckpoints(workers=1, writer=delayed)
        key = transport_key(self.payload, self.scope)
        atomic_json(journal_path(self.journal, 'responses', key + '.json', True),
                    {'payload': normalize_payload(self.payload), 'response': response()})
        try:
            with patch.object(api, 'write_json', writer.write):
                client = api.Client(self.config, sender=lambda *a: self.fail('duplicate paid call'))
                for _ in range(2):
                    self.assertEqual(client.call(self.root / 'requests', 'judge', self.messages, int), ('2', 2))
                self.assertEqual(client.count, 0)
                self.assertFalse((self.root / 'requests').exists())
        finally:
            release.set()
            writer.close()
        self.assertEqual(list((self.root / 'requests').glob('*.attempt-*.json')), [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.journal = self.root / "journal"
        self.config = {"roles": {"judge": {"model": "judge", "revision": "frozen", "sampling": {"temperature": 0, "max_tokens": 8192}}}, "max_attempts": 2, "max_requests_per_run": 20, "timeout_seconds": 1800}
        self.messages = [{"role": "user", "content": "same original prompt"}]
        self.payload = {"model": "judge", "messages": self.messages, "stream": False, **self.config["roles"]["judge"]["sampling"]}
        self.scope = call_scope(self.root / "requests", api.object_hash({"role": "judge", "revision": "frozen", "payload": self.payload}))
        old_call, old_init = api.Client.call, api.Client.__init__
        self.addCleanup(setattr, api.Client, "call", old_call)
        self.addCleanup(setattr, api.Client, "__init__", old_init)
        install_journal_replay(api, self.journal)

    def save(self, body, scope=None):
        atomic_json(self.journal / "responses" / (transport_key(self.payload, scope or self.scope) + ".json"), {"payload": normalize_payload(self.payload), "response": body})

    def test_replays_valid_response_with_original_hash_and_no_api_attempt(self):
        self.save(response())
        client = api.Client(self.config, sender=lambda *a: self.fail("duplicate paid call"))
        self.assertEqual(client.call(self.root / "requests", "judge", self.messages, int), ("2", 2))
        fingerprint = api.object_hash({"role": "judge", "revision": "frozen", "payload": self.payload})
        record = json.loads((self.root / "requests" / (fingerprint + ".json")).read_text())
        self.assertEqual(record["request"], self.payload)
        self.assertEqual(record["request_hash"], fingerprint)
        self.assertEqual(client.count, 0)
        self.assertEqual(list((self.root / "requests").glob("*.attempt-*.json")), [])

    def test_invalid_or_truncated_response_is_not_accepted_as_score(self):
        for body in (response("not a score"), response("2", "length")):
            self.save(body)
            self.assertIsNone(valid_journal_response(self.journal, self.payload, int, self.scope))

    def test_different_case_does_not_reuse_identical_prompt(self):
        self.save(response("2"))
        calls = []
        def sender(*args):
            calls.append(args)
            return response("3")
        client = api.Client(self.config, sender=sender)
        self.assertEqual(client.call(self.root / "different-case/requests", "judge", self.messages, int), ("3", 3))
        self.assertEqual(len(calls), 1)

    def test_internal_retry_reuses_response_after_failed_delivery(self):
        calls = []
        def sender(*args):
            calls.append(args)
            self.save(response())
            raise ConnectionResetError("delivery lost after billing")
        client = api.Client(self.config, sender=sender, sleeper=lambda s: None)
        self.assertEqual(client.call(self.root / "requests", "judge", self.messages, int), ("2", 2))
        self.assertEqual(len(calls), 1)
        rows = [json.loads(p.read_text()) for p in sorted((self.root / "requests").glob("*.attempt-*.json"))]
        self.assertEqual([r["status"] for r in rows], ["failed", "completed"])

    def test_halted_broker_stops_case_without_retrying_other_jobs(self):
        client = api.Client(self.config)
        exc = error.HTTPError("http://127.0.0.1/offline", 503, "halted",
                              {"X-Eval-Halted": "transport_io_failure"}, io.BytesIO(b"{}"))
        opener = SimpleNamespace(open=lambda *a, **k: (_ for _ in ()).throw(exc))
        with patch.object(api, "endpoint", return_value=("http://127.0.0.1/offline", "unused")), patch("recovery_queue_guard.request.build_opener", return_value=opener):
            with self.assertRaises(api.RequestLimit):
                client.call(self.root / "requests", "judge", self.messages, int)
        self.assertEqual(client.count, 1)

    def test_existing_success_wins_and_failure_evidence_is_unchanged(self):
        fingerprint = api.object_hash({"role": "judge", "revision": "frozen", "payload": self.payload})
        directory = self.root / "requests"
        atomic_json(directory / (fingerprint + ".json"), {"request_hash": fingerprint, "request": self.payload, "content": "3"})
        atomic_json(directory / (fingerprint + ".attempt-00001.json"), {"status": "failed", "error": "TimeoutError"})
        before = {p: p.read_bytes() for p in directory.iterdir()}
        self.save(response("2"))
        client = api.Client(self.config, sender=lambda *a: self.fail("successful cache re-requested"))
        self.assertEqual(client.call(directory, "judge", self.messages, int), ("3", 3))
        self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir()})

    def test_coser_cache_reuses_first_success_same_case_without_overwriting(self):
        for attempt, case, key, content in [(1, 7, "a", "first"), (2, 7, "a", "later"), (2, 7, "b", "existing"), (1, 8, "c", "different case")]:
            atomic_json(self.root / f"attempt-{attempt}/case-{case:04d}/request-cache/{key}.json", {"content": content})
        target = self.root / "attempt-3/case-0007/request-cache"
        atomic_json(target / "b.json", {"content": "target already present"})
        before = {p: p.read_bytes() for p in self.root.glob("attempt-[12]/**/*.json")}
        self.assertEqual(seed_coser_cache(self.root, 7, 3), 1)
        self.assertEqual(json.loads((target / "a.json").read_text())["content"], "first")
        self.assertEqual(json.loads((target / "b.json").read_text())["content"], "target already present")
        self.assertFalse((target / "c.json").exists())
        self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))

    def test_coser_reuses_journal_without_sending_http(self):
        # Use a tiny requests-compatible fixture, so this test stays stdlib-only.
        class Response:
            def __init__(self):
                self.headers = {}
        class Session:
            def post(self, *a, **k):
                raise AssertionError("duplicate paid CoSER request")
        module = SimpleNamespace(Session=Session, Response=Response)
        namespace = self.root / "coser/case-0007"
        scope = call_scope(namespace, hashlib.sha256(json.dumps(self.payload, sort_keys=True).encode()).hexdigest())
        self.save(response(), scope)
        install_coser_journal_replay(module, self.journal, namespace)
        result = Session().post("http://offline", json=self.payload)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(json.loads(result._content), response())
        self.assertEqual(result.headers["X-Recovery-Cache"], "response-journal")


class FairnessTests(unittest.TestCase):
    def test_fifo_prevents_fast_reacquiring_worker_starving_older_work(self):
        gate, order = FairGate(1), []
        def task(i):
            with gate.enter(time.monotonic() + 2):
                order.append(i)
        with ThreadPoolExecutor(3) as pool:
            with gate.enter(time.monotonic() + 2):
                futures = []
                for i in range(3):
                    futures.append(pool.submit(task, i))
                    limit = time.monotonic() + 1
                    while len(gate.waiters) < i + 1 and time.monotonic() < limit:
                        time.sleep(.005)
            for future in futures:
                future.result()
        self.assertEqual(order, [0, 1, 2])
        self.assertEqual(gate.active, 0)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trajectory.json"
        self.generated = []
        self.community = SimpleNamespace(trajectory=lambda *a: None,
            generate_case=lambda *a: self.generated.append(a))
        self.full, self.api = SimpleNamespace(), SimpleNamespace()
        install_checkpoint_io(self.full, self.community, self.api)

    def test_complete_valid_trajectory_skips_generation_without_rewriting(self):
        self.community.trajectory = lambda *a: [{"role": "user", "content": "original"}]
        atomic_json(self.path, {"complete": True})
        before = self.path.read_bytes()
        self.community.generate_case({}, {}, {}, self.path.parent, None)
        self.assertEqual(self.generated, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_trajectory_still_uses_original_generator(self):
        self.community.generate_case({}, {}, {}, self.path.parent, None)
        self.assertEqual(len(self.generated), 1)

    def test_cached_prefix_does_not_erase_longer_checkpoint(self):
        messages = [{"role": "user", "content": "original"}, {"role": "assistant", "content": "kept"}]
        atomic_json(self.path, {"complete": False, "messages": messages})
        before = self.path.read_bytes()
        self.community.write_json(self.path, {"complete": False, "messages": messages[:1]})
        self.assertEqual(self.path.read_bytes(), before)
        self.community.write_json(self.path, {"complete": True, "messages": messages})
        self.assertTrue(json.loads(self.path.read_text())["complete"])

    def test_changed_prefix_is_not_silently_ignored(self):
        atomic_json(self.path, {"complete": False, "messages": [{"content": "old"}]})
        self.community.write_json(self.path, {"complete": False, "messages": [{"content": "different"}]})
        self.assertEqual(json.loads(self.path.read_text())["messages"], [{"content": "different"}])

    def test_sync_error_prevents_publishing_checkpoint(self):
        with patch("recovery_queue_guard.os.fsync", side_effect=OSError("remote flush failed")):
            with self.assertRaises(OSError):
                atomic_json(self.path, {"complete": True})
        self.assertFalse(self.path.exists())

    def test_broker_halt_updates_supervisor_and_in_memory_status(self):
        state = {"status": "running"}
        self.full.write(self.path.parent / "supervisor.json", state)
        self.full.write(self.path.parent / "broker-status.json", {"halt": "transport_io_failure"})
        self.assertEqual(state["status"], "halted")
        saved = json.loads((self.path.parent / "supervisor.json").read_text())
        self.assertEqual(saved["status"], "halted")
        self.assertEqual(saved["halt"], "transport_io_failure")


if __name__ == "__main__":
    unittest.main()
