from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.evaluation.roleplay_benchmarks.unified import EndpointLimiter
from src.evaluation.roleplay_benchmarks.endpoint_transport import JournalSender, endpoint_ownership, usage_report
from src.evaluation.roleplay_benchmarks import benchmark_api as api


class Ledger:
    def __init__(self):
        self.ledger, self.rows = self, []
    def check(self):
        pass
    def append(self, row):
        self.rows.append(row)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, {'TEST_URL': 'http://127.0.0.1:1/v1', 'TEST_KEY': 'fixture'})
        self.env.start(); self.addCleanup(self.env.stop)
        self.role = {'model': 'fixture', 'revision': 'fixture-r1', 'base_url_env': 'TEST_URL',
                     'api_key_env': 'TEST_KEY', 'kind': 'remote', 'sampling': {}}
        self.config = {'actor': self.role, 'judge': self.role, 'thinking': {'remote': {}},
                       'max_requests_per_run': 300, 'timeout_seconds': 10}
        self.ledger = Ledger()
        self.sender = JournalSender(self.config, EndpointLimiter({'remote': 96}), self.ledger, self.tmp.name)
        self.payload = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'unaltered'}], 'stream': False}

    def body(self, content='ok'):
        return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}], 'usage': {'cost': .01}}

    def call(self, scope='sample-1', parser=lambda x: x):
        return self.sender.call(self.role, self.payload, 10, scope, parser)

    def test_durable_response_survives_client_checkpoint_loss(self):
        with patch.object(api, 'transport', return_value=self.body()) as http:
            self.call()
            self.assertEqual(len(list(Path(self.tmp.name).rglob('*.outcome.json'))), 1)
            fresh = JournalSender(self.config, EndpointLimiter({'remote': 96}), self.ledger, self.tmp.name)
            fresh.call(self.role, self.payload, 10, 'sample-1')
            self.assertEqual(http.call_count, 1)
            self.assertEqual(fresh.calls, 0)
        self.assertEqual(len(self.ledger.rows), 1)
        summary = usage_report(self.tmp.name)
        self.assertEqual(summary['requests'], 1)
        self.assertEqual(summary['models']['fixture']['provider_cost_usd'], .01)

    def test_extended_actor_reuses_success_and_retries_truncation_under_new_wire_key(self):
        self.role.update(kind='local', sampling={'max_tokens': 8192})
        judge = {**self.role, 'model': 'judge', 'kind': 'remote'}
        self.config.update(actor=self.role, judge=judge, protocol='her-coser-format-v1', actor_context_window=32768)
        self.sender = JournalSender(self.config, EndpointLimiter({'local': 64, 'remote': 48}), self.ledger, self.tmp.name)
        self.payload['max_tokens'] = 8192
        truncated = self.body('partial'); truncated['choices'][0]['finish_reason'] = 'length'
        with patch.object(api, 'transport', side_effect=[self.body('old success'), truncated]):
            self.call('success'); self.call('truncated')
        self.role['sampling']['max_tokens'] = 16384
        with patch.object(self.sender, 'extended_actor_payload', return_value={**self.payload, 'max_tokens': 16384}) as expand, \
                patch.object(api, 'transport', return_value=self.body('new complete')) as http:
            self.assertEqual(self.call('success')['choices'][0]['message']['content'], 'old success')
            expand.assert_not_called()
            self.call('truncated'); self.call('truncated')
            self.assertEqual(http.call_count, 1)
            self.assertEqual(http.call_args.args[1]['max_tokens'], 16384)
        outcomes = [json.loads(p.read_text()) for p in Path(self.tmp.name).rglob('*.outcome.json')]
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(sum(o['response']['choices'][0]['finish_reason'] == 'length' for o in outcomes), 1)

    def test_extended_actor_counts_exact_template_and_preserves_input(self):
        self.role.update(kind='local', sampling={'max_tokens': 16384})
        self.config.update(protocol='her-coser-format-v1', actor_context_window=32768)
        payload = {**self.payload, 'max_tokens': 8192, 'chat_template_kwargs': {'enable_thinking': False}}
        class Reply(io.BytesIO):
            pass
        with patch('src.evaluation.roleplay_benchmarks.endpoint_transport.build_opener') as opener:
            opener.return_value.open.return_value = Reply(b'{"count":20000}')
            result = self.sender.extended_actor_payload(self.role, payload, 1200)
            sent = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(sent['messages'], payload['messages'])
            self.assertEqual(sent['chat_template_kwargs'], {'enable_thinking': False})
            self.assertTrue(sent['add_generation_prompt'])
            self.assertEqual(result['max_tokens'], 12767)
            self.assertEqual(result['messages'], payload['messages'])
            self.assertEqual(payload['max_tokens'], 8192)
            opener.return_value.open.return_value = Reply(b'{"count":32768}')
            with self.assertRaisesRegex(ValueError, 'input exceeds'):
                self.sender.extended_actor_payload(self.role, payload, 1200)

    def test_ambiguous_response_quarantined_and_other_sample_runs(self):
        with patch.object(api, 'transport', side_effect=TimeoutError):
            with self.assertRaises(api.CallFailed): self.call()
        with patch.object(api, 'transport', return_value=self.body()) as http:
            with self.assertRaises(api.CallFailed): self.call()
            self.call('sample-2')
            self.assertEqual(http.call_count, 1)

    def test_parse_invalid_response_retried_without_overwriting_evidence(self):
        with patch.object(api, 'transport', side_effect=[self.body('invalid'), self.body('3')]) as http:
            self.call(parser=int)
            self.call(parser=int)
            self.call(parser=int)
            self.assertEqual(http.call_count, 2)
        self.assertEqual(len(list(Path(self.tmp.name).rglob('*.outcome.json'))), 2)

    def test_http_502_retries_keep_each_attempt(self):
        error = HTTPError('fixture', 502, 'fixture', {}, io.BytesIO(b'upstream-failure'))
        with patch.object(api, 'transport', side_effect=[error, self.body()]) as http:
            with self.assertRaises(HTTPError): self.call()
            self.call()
            self.assertEqual(http.call_count, 2)
        self.assertEqual([r['status'] for r in self.ledger.rows], [502, 200])
        summary = usage_report(self.tmp.name)
        self.assertEqual(summary['retries'], 1)
        self.assertEqual(summary['models']['fixture']['cost_unknown'], 1)

    def test_failed_writer_stops_before_next_charge(self):
        with patch.object(self.ledger, 'check', side_effect=OSError('disk failed')):
            with patch.object(api, 'transport', side_effect=AssertionError('No dispatch')):
                with self.assertRaises(api.RequestLimit): self.call()

    def test_ledger_failure_after_response_still_replays_without_new_charge(self):
        with patch.object(self.ledger, 'append', side_effect=OSError('writer died')), \
             patch.object(api, 'transport', return_value=self.body()) as http:
            with self.assertRaises(api.RequestLimit): self.call()
            fresh = JournalSender(self.config, EndpointLimiter({'remote': 96}), Ledger(), self.tmp.name)
            fresh.call(self.role, self.payload, 10, 'sample-1')
            self.assertEqual(http.call_count, 1)
        report = usage_report(self.tmp.name)
        self.assertEqual(report['requests'], 1)
        self.assertEqual(report['models']['fixture']['provider_cost_usd'], .01)

    def test_second_supervisor_cannot_multiply_endpoint_limit(self):
        with endpoint_ownership([self.role]):
            with self.assertRaises(RuntimeError):
                with endpoint_ownership([self.role]): pass
        with endpoint_ownership([self.role]): pass

    def test_same_sample_concurrent_call_is_sent_once(self):
        with patch.object(api, 'transport', side_effect=lambda *a: (time.sleep(.05), self.body())[1]) as http:
            with ThreadPoolExecutor(16) as pool:
                list(pool.map(lambda _: self.call(), range(16)))
            self.assertEqual(http.call_count, 1)

    def test_real_http_96_parallel_and_zero_call_replay(self):
        reached = threading.Event()
        counter = {'active': 0, 'peak': 0, 'calls': 0}
        guard = threading.Lock()
        body = self.body()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                with guard:
                    counter['active'] += 1; counter['calls'] += 1
                    counter['peak'] = max(counter['peak'], counter['active'])
                    if counter['active'] == 96: reached.set()
                reached.wait(8)
                raw = json.dumps(body).encode()
                self.send_response(200); self.send_header('Content-Length',str(len(raw))); self.end_headers()
                self.wfile.write(raw)
                with guard: counter['active'] -= 1
        class Server(ThreadingHTTPServer):
            request_queue_size = 256
            daemon_threads = True
        server = Server(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            os.environ['TEST_URL'] = f'http://127.0.0.1:{server.server_port}/v1'
            with ThreadPoolExecutor(128) as pool:
                list(pool.map(lambda i: self.call(str(i)), range(128)))
                list(pool.map(lambda i: self.call(str(i)), range(128)))
            self.assertEqual(counter['calls'],128)
            self.assertEqual(counter['peak'],96)
            self.assertEqual(self.sender.calls,128)
            self.assertEqual(self.sender.replays,128)
        finally:
            reached.set(); server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__': unittest.main()
