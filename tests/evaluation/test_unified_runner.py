import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks import unified
from src.evaluation.roleplay_benchmarks.benchmark_api import Client
from src.evaluation.roleplay_benchmarks.sources import ROOT, read_json


def response(text="next message"):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
                       "estimated_cost": 0.01}}


class UnifiedRunnerTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "outputs").mkdir(exist_ok=True)

    def config(self):
        return {
            "schema_version": 1,
            "protocol_label": "test-unified",
            "benchmarks": ["minimax"],
            "actor": {"model": "actor", "revision": "a1", "base_url_env": "TEST_URL",
                       "api_key_env": "TEST_KEY", "kind": "local", "sampling": {"temperature": .7}},
            "judge": {"model": "judge", "revision": "j1", "base_url_env": "TEST_URL",
                       "api_key_env": "TEST_KEY", "kind": "remote", "sampling": {"temperature": 0}},
            "simulator": {"model": "actor", "revision": "a1", "base_url_env": "TEST_URL",
                           "api_key_env": "TEST_KEY", "kind": "local", "sampling": {"temperature": .7}},
            "thinking": {"local": {"chat_template_kwargs": {"enable_thinking": False}}, "remote": {}},
            "concurrency": {"endpoints": {"local": 2, "remote": 2}, "benchmarks": {"minimax": 2}},
            "max_attempts": 2, "timeout_seconds": 5, "max_requests_per_run": 100,
            "benchmark_options": {"minimax": {"limit_cases": 1, "generated_messages": 4, "chunk_messages": 2}},
            "storage": {"checkpoint_limit": 32, "checkpoint_workers": 2},
            "process_accounting": {"limit": 32, "batch_size": 4, "batch_wait": .01},
        }

    def test_validation_does_not_allow_credentials_in_config(self):
        value = self.config()
        value["actor"]["api_key_env"] = "secret-token"
        with self.assertRaises(ValueError):
            unified.validate(value)

    def test_extended_coser_actor_requires_explicit_local_context_window(self):
        value = self.config()
        value.update(protocol='her-coser-format-v1', benchmarks=['coser'], benchmark_options={'coser': {}})
        value['concurrency']['benchmarks'] = {'coser': 128}
        value['actor']['sampling']['max_tokens'] = 16384
        with self.assertRaisesRegex(ValueError, 'actor_context_window'):
            unified.validate(value)
        value['actor_context_window'] = 32768
        unified.validate(value)
        value['judge']['sampling']['max_tokens'] = 16384
        with self.assertRaises(ValueError):
            unified.validate(value)
        value['judge']['sampling']['max_tokens'] = 8192
        value['actor']['kind'] = 'remote'
        with self.assertRaises(ValueError):
            unified.validate(value)

    def test_endpoint_limiter_is_shared_across_producers(self):
        limiter = unified.EndpointLimiter({"local": 2})
        active = peak = 0
        lock = threading.Lock()

        def one():
            nonlocal active, peak
            with limiter.slot("local"):
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.02)
                with lock:
                    active -= 1

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: one(), range(8)))
        self.assertLessEqual(peak, 2)
        self.assertEqual(limiter.snapshot(), {"local": 0})

    def test_completed_attempt_is_promoted_without_a_second_paid_call(self):
        value = self.config()
        role = {key: value["actor"][key] for key in ("model", "revision", "base_url_env", "api_key_env", "sampling")}
        cfg = {"roles": {"actor": role}, "max_attempts": 2, "max_requests_per_run": 10, "timeout_seconds": 5}
        with tempfile.TemporaryDirectory() as temp:
            first = Client({**cfg, "roles": {"actor": role}}, sender=lambda *args: response("cached"), sleeper=lambda _: None)
            first.call(temp, "actor", [], lambda text: text)
            cache = next(p for p in Path(temp).glob("*.json") if ".attempt-" not in p.name)
            cache.unlink()
            calls = []
            second = Client(cfg, sender=lambda *args: calls.append(1) or response("new"), sleeper=lambda _: None)
            self.assertEqual(second.call(temp, "actor", [], lambda text: text)[0], "cached")
            self.assertEqual(calls, [])

    def test_public_run_resumes_from_sync_request_cache_and_async_derived_files(self):
        value = self.config()
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temp:
            config_path = Path(temp) / "config.json"
            config_path.write_text(json.dumps(value), encoding="utf-8")
            run_dir = Path(temp) / "unified"
            unified.prepare(config_path, run_dir)
            calls = []

            def transport(role, payload, timeout):
                calls.append(payload)
                if "Evaluate existing" in payload["messages"][0].get("content", ""):
                    dims = ("basics", "logic", "knowledge", "diversity", "content_logic", "interaction")
                    return response(json.dumps({dim: {"score": 80, "reason": "fixture"} for dim in dims}))
                return response()

            with patch.dict(os.environ, {"TEST_URL": "http://fixture", "TEST_KEY": "test-key"}), \
                 patch.object(unified.benchmark_api, "transport", side_effect=transport):
                first = unified.run(run_dir, execute=True)
                count = len(calls)
                second = unified.run(run_dir, execute=True)
                with unified.endpoint_ownership([value['actor'], value['judge']]):
                    with self.assertRaisesRegex(RuntimeError, 'Another evaluation owns'):
                        unified.run(run_dir, execute=True)
                    independent = unified.run(run_dir, execute=True, independent_endpoints=True)
                    self.assertEqual(independent['requests'], 0)
                    self.assertEqual(read_json(run_dir / 'manifest.json')['endpoint_ownership_mode'], 'independent')
                    (run_dir / '.run.lock').write_text('another supervisor')
                    try:
                        with self.assertRaisesRegex(ValueError, 'Run is locked'):
                            unified.run(run_dir, execute=True, independent_endpoints=True)
                    finally:
                        (run_dir / '.run.lock').unlink()
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["requests"], 0)
            self.assertEqual(len(calls), count)
            self.assertTrue((run_dir / "progress.json").exists())
            self.assertTrue(list((run_dir / "usage-ledger").glob("batch-*.jsonl")))

    def test_prepare_accepts_external_local_scratch_directory(self):
        value = self.config()
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            config_path.write_text(json.dumps(value), encoding="utf-8")
            run_dir = Path(temp) / "ram-or-local-ssd-run"
            manifest = unified.prepare(config_path, run_dir)
            self.assertEqual(manifest["status"], "prepared")
            self.assertTrue((run_dir / "minimax" / "manifest.json").exists())

    def test_reviewed_public_summary_and_report_agree_and_resume_without_calls(self):
        from src.evaluation.roleplay_benchmarks.reports import describe_run
        value = self.config()
        value['assessment'] = {'version': '20261018', 'coser_role_thoughts': 'include'}
        # A one-session smoke explicitly uses one repeat; formal defaults stay 3.
        value['benchmark_options']['minimax']['repeats'] = 1
        calls = []
        def transport(role, payload, timeout):
            calls.append(payload)
            if 'Evaluate existing' in payload['messages'][0].get('content', ''):
                from src.evaluation.roleplay_benchmarks.community_protocols import MINI_DIMS
                return response(json.dumps({d: {'score': 80, 'reason': 'fixture'} for d in MINI_DIMS}))
            return response()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_URL': 'http://fixture', 'TEST_KEY': 'test'}):
            directory = Path(tmp)/'run'
            unified.prepare(value, directory)
            with patch.object(unified.benchmark_api, 'transport', side_effect=transport):
                result = unified.run(directory, execute=True)
                count = len(calls)
                repeated = unified.run(directory, execute=True)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(repeated['requests'], 0)
            self.assertEqual(len(calls), count)
            summary = read_json(directory/'minimax/summary.json')
            self.assertEqual(summary['scores']['en']['complete_seeds'] + summary['scores']['zh']['complete_seeds'], 1)
            self.assertTrue(summary['score_coverage']['complete'])
            reported = describe_run(directory/'minimax')
            self.assertEqual(reported['native_scores'], summary['scores'])
            self.assertTrue(reported['eligible_for_comparison'])

    def test_character_uses_api_judge_and_preserves_frozen_sampling(self):
        value = self.config()
        value['benchmarks'] = ['characterbench']
        value['benchmark_options'] = {'characterbench': {'limit_cases': 8}}
        value['concurrency']['benchmarks'] = {'characterbench': 8}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            unified.prepare(value, run)
            captured = []
            def transport(role, payload, timeout):
                captured.append(payload)
                return response('{"response":"Hi."}' if role['model'] == 'actor' else '3')
            with patch.dict(os.environ, {'TEST_URL': 'http://fixture', 'TEST_KEY': 'fixture'}), \
                 patch.object(unified.benchmark_api, 'transport', side_effect=transport):
                result = unified.run(run, True)
                resumed = unified.run(run, True)
            self.assertEqual(result['results']['characterbench']['completed'], 8)
            self.assertEqual(resumed['requests'], 0)
            self.assertEqual(len(captured), 16)
            for payload in captured:
                self.assertEqual(payload['max_tokens'],8192)
                self.assertEqual(payload['temperature'],.7 if payload['model']=='actor' else 0)

    def test_rmt_prepares_historical_long_dialogues(self):
        value = self.config()
        value['benchmarks'] = ['rmtbench']
        value['concurrency']['benchmarks'] = {'rmtbench': 2}
        value['benchmark_options'] = {'rmtbench': {'limit_cases': 1}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'run'
            unified.prepare(value, directory)
            child = read_json(directory / 'rmtbench/manifest.json')
            self.assertEqual(child['config']['layout'], 'paper_four_blocks')
            self.assertEqual(child['config']['repeats'], 2)
            self.assertEqual(child['config']['seed'], 42)
            rows = read_json(directory / 'rmtbench/cases.json')
            self.assertEqual(len(rows), 1)
            self.assertGreater(len(rows[0]['turns']), 20)

    def test_reviewed_rmt_report_accepts_pinned_long_protocol(self):
        from src.evaluation.roleplay_benchmarks.reports import describe_run
        value = self.config()
        value['benchmarks'] = ['rmtbench']
        value['concurrency']['benchmarks'] = {'rmtbench': 2}
        value['benchmark_options'] = {'rmtbench': {'limit_cases': 1}}
        value['assessment'] = {'version': '20261018', 'coser_role_thoughts': 'include'}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'run'
            unified.prepare(value, directory)
            report = describe_run(directory/'rmtbench')
            self.assertFalse(report['eligible_for_comparison'])
            self.assertFalse(report['coverage']['complete'])
            self.assertEqual(report['native_scores'], {'en': {}, 'zh': {}})


if __name__ == "__main__":
    unittest.main()
