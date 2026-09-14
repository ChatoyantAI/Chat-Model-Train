import contextlib
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from src.evaluation.roleplay_benchmarks import community
from src.evaluation.roleplay_benchmarks.benchmark_api import Client, CallFailed, RequestLimit, transport
from src.evaluation.roleplay_benchmarks.community_protocols import (
    build_cases, rmt_turns, parse_rmt, parse_mini, MINI_DIMS, rmt_judge_jobs, mini_chunks)
from src.evaluation.roleplay_benchmarks.paper_templates import load_templates
from src.evaluation.roleplay_benchmarks.reports import protocol_id, describe_run
from src.evaluation.roleplay_benchmarks.sources import ROOT, read_json, write_json


def config(name):
    result = read_json(ROOT / f"configs/evaluation/roleplay_benchmarks/{name}.example.json")
    for role in result["roles"].values():
        role["model"] = "test-model"
        role["revision"] = "test-revision"
    result.update(languages=["en"], limit_cases=1, concurrency=2, max_requests_per_run=2000, max_attempts=2)
    if name == "minimax":
        result.update(generated_messages=4, chunk_messages=2)
    return result


def response(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def six_scores(value=80):
    return {dim: {"score": value, "reason": "Test fixture"} for dim in MINI_DIMS}


class PublicProtocolTests(unittest.TestCase):
    def test_binary_polarities_and_numeric_scores(self):
        self.assertEqual(parse_rmt("Explanation. [[No]]", "CM"), 1)
        self.assertEqual(parse_rmt("Explanation. [[Yes]]", "CM"), 0)
        self.assertEqual(parse_rmt("Explanation. [[Yes]]", "UPA"), 1)
        self.assertEqual(parse_rmt("Explanation. Score: [[4]]", "EE"), 4)
        self.assertEqual(parse_rmt("Explanation.\n**Score: [[3]]**", "EE"), 3)
        self.assertEqual(parse_rmt("Explanation.\n**[[No]]**", "CM"), 1)
        for text in ("[[3]]**", "**Score: [[3]]** unfinished", "**Score: [[6]]**"):
            with self.assertRaises(ValueError):
                parse_rmt(text, "EE")
        for text in ("[[0]]", "[[6]]", "[[Yes]]", "Score: 3", "[[3]] unfinished"):
            with self.assertRaises(ValueError):
                parse_rmt(text, "EE")

    def test_mini_rejects_missing_boolean_nan_duplicate_scores(self):
        self.assertEqual(parse_mini(json.dumps(six_scores()))["logic"]["score"], 80)
        for bad in (True, float("nan"), 101, -1, "80"):
            with self.assertRaises(ValueError):
                parse_mini(json.dumps(six_scores(bad)))
        with self.assertRaises(ValueError):
            parse_mini('{}')
        text = json.dumps(six_scores())
        with self.assertRaises(ValueError):
            parse_mini('{"basics":{"score":80,"reason":"x"},' + text[1:])

    def test_upa_tail_mapping(self):
        row = {"user_preference": [{"chat": ["intro", "ask one", "ask two"], "question": ["Q1", "Q2"], "answer": ["A1", "A2"]}]}
        turns = rmt_turns(row, "user_preference", 0)
        self.assertNotIn("reference_answer", turns[0])
        self.assertEqual([t["reference_answer"] for t in turns[1:]], ["A1", "A2"])

    def test_config_guardrails(self):
        cfg = config("minimax")
        community.validate(cfg)
        for delta in ({"api_key": "secret"}, {"generated_messages": 3}, {"max_requests_per_run": 0}, {"languages": ["en", "en"]}):
            with self.assertRaises(ValueError):
                community.validate({**cfg, **delta})

    def test_retry_parse_failure_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def sender(*args):
                calls.append(args)
                return response("invalid" if len(calls) == 1 else "Score: [[3]]")
            client = Client(config("rmtbench"), sender, lambda _: None)
            parse = lambda text: parse_rmt(text, "EE")
            self.assertEqual(client.call(tmp, "judge", [], parse)[1], 3)
            self.assertEqual(client.call(tmp, "judge", [], parse)[1], 3)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(list(Path(tmp).glob("*.attempt-*.json"))), 2)

    def test_429_retry_after_and_401_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            headers = Message()
            headers["Retry-After"] = "7"
            sleeps = []
            count = []
            def sender(*args):
                count.append(1)
                if len(count) == 1:
                    raise HTTPError("https://example.test/secret", 429, "secret", headers, None)
                return response("ok")
            client = Client(config("rmtbench"), sender, sleeps.append)
            self.assertEqual(client.call(tmp, "actor", [])[0], "ok")
            self.assertEqual(sleeps, [7])
            def fail(*args):
                raise HTTPError("https://example.test/secret", 401, "secret", headers, None)
            client = Client(config("rmtbench"), fail, sleeps.append)
            with self.assertRaises(CallFailed):
                client.call(tmp, "actor", [{"role": "user", "content": "different"}])
            self.assertEqual(client.count, 1)
            self.assertTrue(all("secret" not in file.read_text() for file in Path(tmp).glob("*.json")))

    def test_truncation_not_accepted_even_if_score_parses(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = Client(config("rmtbench"), lambda *a: response("[[3]]", "length"), lambda _: None)
            with self.assertRaises(CallFailed):
                client.call(tmp, "judge", [], lambda t: parse_rmt(t, "EE"))

    def test_request_limit_includes_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config("rmtbench")
            cfg["max_requests_per_run"] = 1
            client = Client(cfg, lambda *a: response("bad"), lambda _: None)
            with self.assertRaises(RequestLimit):
                client.call(tmp, "judge", [], lambda t: parse_rmt(t, "EE"))
            self.assertEqual(client.count, 1)

    def test_real_http_transport_to_local_fixture(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                raw = json.dumps(response("ok")).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"ACTOR_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "ACTOR_API_KEY": "fixture-only"}):
                role = config("rmtbench")["roles"]["actor"]
                self.assertEqual(transport(role, {"model": "test", "messages": []}, 5)["choices"][0]["message"]["content"], "ok")
            self.assertEqual(captured[0][0], "/v1/chat/completions")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


@unittest.skipUnless((ROOT / "data/external/roleplay_benchmarks/rmtbench-paper/2507.20352v2.html").exists(), "Download paper/source assets first")
class FullFixtureTests(unittest.TestCase):
    def test_published_templates_and_upa_fix(self):
        templates = load_templates()
        self.assertEqual(set(templates["dimensions"]), {"EE", "EC", "PA", "CU"})
        self.assertIn("<|Answer Start|>\n{answer}", templates["UPA"])
        self.assertIn("{response}", templates["quality"])

    def test_all_rmt_versions_deterministic_and_preserve_blocks(self):
        base = ROOT / "data/external/roleplay_benchmarks/rmtbench"
        for version in ("claude35", "claude45"):
            cfg = {**config("rmtbench"), "limit_cases": None, "version": version, "languages": ["en", "zh"]}
            cases = build_cases(cfg, base)
            self.assertEqual(len(cases), 320)
            self.assertEqual(cases, build_cases(cfg, base))
            for case in cases:
                self.assertEqual(len(case["layout"]), 4)
                groups = {}
                for turn in case["turns"]:
                    groups.setdefault((turn["field"], turn["block_index"]), []).append(turn["block_turn"])
                for indices in groups.values():
                    self.assertEqual(indices, list(range(len(indices))))

    def prepare_run(self, tmp, name):
        cfg = config(name)
        path = Path(tmp) / "config.json"
        write_json(path, cfg)
        directory = Path(tmp) / name
        community.prepare(path, directory)
        return directory

    def test_rmt_full_pipeline_resume_and_answer_not_in_actor(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            directory = self.prepare_run(tmp, "rmtbench")
            captured = []
            def sender(role, payload, timeout):
                captured.append(payload)
                text = payload["messages"][-1]["content"]
                if "Score: [[number]]" in text:
                    return response("reason\nScore: [[4]]")
                if "<|Model Response Start|>" in text:
                    return response("reason\n[[No]]" if "AI assistant" in text else "reason\n[[Yes]]")
                self.assertNotIn("reference_answer", json.dumps(payload))
                self.assertNotIn("<|Answer Start|>", json.dumps(payload))
                return response("Generated actor response")
            with patch("src.evaluation.roleplay_benchmarks.benchmark_api.transport", side_effect=AssertionError("Unexpected network")):
                self.assertTrue(community.run(directory, "evaluate")["dry_run"])
                partial = community.run(directory, "evaluate", True, 2, sender)
                self.assertFalse(partial["coverage"]["complete"])
                result = community.run(directory, "evaluate", True, 2000, sender)
                self.assertTrue(result["coverage"]["complete"])
                count = len(captured)
                self.assertEqual(community.run(directory, "evaluate", True, 2000, sender)["requests_this_run"], 0)
                self.assertEqual(len(captured), count)
                report = describe_run(directory)
                self.assertTrue(report["eligible_for_comparison"])
                self.assertEqual(report["native_scores"]["en"]["EE"]["mean"], 4)
            # Earlier actor response appears in subsequent actor history.
            actor_calls = [p for p in captured if p["messages"][0]["role"] == "system"]
            self.assertTrue(any(m["content"] == "Generated actor response" for m in actor_calls[1]["messages"][:-1]))

    def test_mini_dynamic_simulator_and_aggregation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            directory = self.prepare_run(tmp, "minimax")
            seen = []
            def sender(role, payload, timeout):
                seen.append(payload)
                if "Evaluate existing" in payload["messages"][0]["content"]:
                    body = json.loads(payload["messages"][1]["content"])
                    self.assertEqual(len(body["target_chunk"]), 2)
                    self.assertGreaterEqual(len(body["prior_context"]), 2)
                    return response(json.dumps(six_scores()))
                if "simulate the USER" in payload["messages"][0]["content"]:
                    self.assertEqual(json.loads(payload["messages"][1]["content"])["dialogue"][-1]["content"], "Actor changed the scene")
                    return response("User reacts to new scene")
                return response("Actor changed the scene")
            result = community.run(directory, "evaluate", True, 100, sender)
            self.assertEqual(result["requests_this_run"], 6)
            self.assertTrue(result["coverage"]["complete"])
            summary = read_json(directory / "summary.json")
            self.assertEqual(summary["scores"]["en"]["overall"], 80)
            self.assertNotIn("cost", summary["usage"])
            messages = read_json(next((directory / "cases").glob("*/trajectory.json")))["messages"]
            self.assertEqual(len(messages), 6)
            self.assertEqual(len(list(mini_chunks(messages, 2))), 2)

    def test_snapshot_and_protocol_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            directory = self.prepare_run(tmp, "minimax")
            manifest = read_json(directory / "manifest.json")
            modified = json.loads(json.dumps(manifest))
            modified["config"]["roles"]["actor"]["model"] = "another-actor"
            self.assertEqual(protocol_id(manifest), protocol_id(modified))
            modified["config"]["roles"]["judge"]["model"] = "another-judge"
            self.assertNotEqual(protocol_id(manifest), protocol_id(modified))
            write_json(directory / "prompts.json", {})
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                community.run(directory, "evaluate")

    def test_deepinfra_cost_fallback_and_no_double_count(self):
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            directory = self.prepare_run(tmp, "minimax")
            costs = iter([
                {"estimated_cost": 0.01},
                {"cost": 0.02, "estimated_cost": 99},
                {"cost": None, "estimated_cost": 0.03},
                {"cost": 0, "estimated_cost": 99},
                {},
                {"cost": True, "estimated_cost": 0.04},
            ])
            def sender(role, payload, timeout):
                text = json.dumps(six_scores()) if "Evaluate existing" in payload["messages"][0]["content"] else "next message"
                result = response(text)
                result["usage"].update(next(costs))
                return result
            community.run(directory, "evaluate", True, 100, sender)
            summary = read_json(directory / "summary.json")
            self.assertAlmostEqual(summary["usage"]["cost"], 0.10)
            self.assertEqual(summary["attempts_with_reported_cost"], 5)

    def test_mini_partial_scores_not_zero_or_full_result(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            directory = self.prepare_run(tmp, "minimax")
            def sender(role, payload, timeout):
                if "Evaluate existing" in payload["messages"][0]["content"]:
                    return response(json.dumps(six_scores(90)))
                return response("next message")
            result = community.run(directory, "evaluate", True, 5, sender)
            self.assertFalse(result["coverage"]["complete"])
            self.assertEqual(result["coverage"]["valid_judgments"], 1)
            summary = read_json(directory / "summary.json")
            self.assertNotIn("overall", summary["scores"]["en"])
            result = community.run(directory, "evaluate", True, 20, sender)
            self.assertEqual(result["requests_this_run"], 1)
            self.assertTrue(result["coverage"]["complete"])
            self.assertEqual(read_json(directory / "summary.json")["scores"]["en"]["overall"], 90)


if __name__ == "__main__":
    unittest.main()
