from __future__ import annotations

import gzip
import http.cookiejar
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from src.evaluation.roleplay_annotation_platform.build_rubric import build_rubric
from src.evaluation.roleplay_annotation_platform.prepare_data import (
    _validate_record,
    build_samples_db,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRIC_MARKDOWN = (
    REPO_ROOT / "docs" / "evaluation" / "ROLEPLAY_EVALUATION_DIMENSIONS_V0_2.md"
)
RUBRIC_JSON = (
    REPO_ROOT / "src" / "evaluation" / "roleplay_annotation_platform" / "rubric.json"
)
PLATFORM_JS = (
    REPO_ROOT / "src" / "evaluation" / "roleplay_annotation_platform" / "static" / "app.js"
)


def sample_record(evaluation_id: str, candidate: str) -> dict:
    return {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "rubric_version": "roleplay-rubric-v0.1",
        "trigger_type": "message",
        "input_messages": [
            {"role": "system", "content": "You are Character A."},
            {"role": "assistant", "content": "Greeting text"},
            {"role": "user", "content": "Continue."},
        ],
        "output_message": {"role": "assistant", "content": candidate},
    }


class RubricAndDataBuildTests(unittest.TestCase):
    def test_v02_rubric_is_migrated_without_omitting_examples_or_explanations(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        items = [item for dimension in rubric["dimensions"] for item in dimension["items"]]
        cases = [case for item in items for case in item["cases"]]
        examples = [example for case in cases for example in case["examples"]]
        self.assertEqual(5, len(rubric["dimensions"]))
        self.assertEqual(22, len(items))
        self.assertEqual(25, len(cases))
        self.assertEqual(79, len(examples))
        self.assertTrue(all(example["reply"] for example in examples))
        self.assertTrue(all(example["explanation"].startswith("说明：") for example in examples))
        self.assertTrue(all(item["decision_steps"] for item in items))
        self.assertTrue(all(item["na_condition"] for item in items))
        self.assertTrue(all(item["reference_markdown"] in rubric["source_markdown"] for item in items))
        self.assertEqual(
            "7f415a513dced7d5e5d0cc14ece33d700781ea3dd90b9ac454cdfadfee3ffd58",
            rubric["source_sha256"],
        )
        self.assertEqual(RUBRIC_MARKDOWN.read_text(encoding="utf-8"), rubric["source_markdown"])

    def test_v02_uses_each_items_exact_score_scale(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        items = {
            item["id"]: item
            for dimension in rubric["dimensions"]
            for item in dimension["items"]
        }
        self.assertEqual(["0", "2"], items["C1"]["scale"])
        self.assertEqual(["0", "1", "2"], items["H1"]["scale"])
        self.assertEqual(["0", "1", "2", "3"], items["H2"]["scale"])
        self.assertEqual(
            "四问全答，且三个以上感知渠道的具体细节",
            items["H2"]["anchors"]["3"],
        )
        self.assertIn("写出用到的渠道和对应细节", items["H2"]["decision_steps"][-1])
        self.assertEqual(11, sum(len(group["items"]) for group in rubric["gates"]))

    def test_blind_data_validator_rejects_ground_truth_keys(self) -> None:
        record = sample_record("eval_x", "candidate")
        record["model_name"] = "secret-model"
        with self.assertRaisesRegex(ValueError, "blind-data violation"):
            _validate_record(record, 1)

    def test_all_score_standards_are_visible_before_selection(self) -> None:
        script = PLATFORM_JS.read_text(encoding="utf-8")
        self.assertIn("description.textContent = item.anchors[score]", script)
        self.assertIn("先读标准，再选择一项", script)
        self.assertIn('[...item.scale, "N/A"]', script)
        self.assertIn("item.decision_steps", script)
        self.assertIn("example.explanation", script)
        self.assertIn("state.rubric.source_markdown", script)
        self.assertNotIn("选择分数后显示对应判定标准", script)


class PlatformApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tempdir.name)
        source = cls.root / "samples.jsonl.gz"
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(sample_record("eval_one", "Candidate one")) + "\n")
            handle.write(json.dumps(sample_record("eval_two", "Candidate two")) + "\n")
        cls.sample_db = cls.root / "samples.sqlite3"
        build_samples_db(source, cls.sample_db)
        cls.labels_csv = cls.root / "evaluation_labels.csv"
        cls.labels_csv.write_text(
            "evaluation_id,pair_group_id,model_id,model_name,human_selected,source_completed_at\n"
            "eval_one,pair_one,1,model-one,1,2026-08-26 12:00:00\n"
            "eval_two,pair_one,2,model-two,0,2026-08-26 12:00:00\n",
            encoding="utf-8",
        )
        cls.rubric_json = cls.root / "rubric.json"
        cls.rubric_json.write_text(
            RUBRIC_JSON.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        os.environ["ROLEPLAY_SAMPLE_DB"] = str(cls.sample_db)
        os.environ["ROLEPLAY_RESULTS_DB"] = str(cls.root / "results" / "annotations.sqlite3")
        os.environ["ROLEPLAY_RESULTS_DIR"] = str(cls.root / "results" / "by_user")
        os.environ["ROLEPLAY_RUBRIC_JSON"] = str(cls.rubric_json)
        os.environ["ROLEPLAY_LABELS_CSV"] = str(cls.labels_csv)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        environment = os.environ.copy()
        cls.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.evaluation.roleplay_annotation_platform.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
                "--log-level",
                "warning",
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if cls.server.poll() is not None:
                raise RuntimeError("local annotation server exited during startup")
            try:
                with urllib.request.urlopen(f"{cls.base_url}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise RuntimeError("local annotation server did not become ready")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server.kill()
        cls.tempdir.cleanup()

    @classmethod
    def client(cls):
        cookies = http.cookiejar.CookieJar()
        return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))

    @classmethod
    def request_json(cls, client, path: str, method: str = "GET", payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{cls.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with client.open(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    @staticmethod
    def gates() -> dict:
        return {
            "G-D1": False,
            "G-D2": False,
            "G-D3": False,
            "G-S1": "no",
            "G-S2": "N/A",
            "G-S3": "no",
            "G-P1": False,
            "G-P2": False,
            "G-P3": False,
            "G-P4": False,
            "G-P5": False,
        }

    def score_payload(self, value: int = 2, evidence: str = "") -> dict:
        rubric = json.loads(RUBRIC_JSON.read_text(encoding="utf-8"))
        return {
            item["id"]: {"value": value, "evidence": evidence}
            for dimension in rubric["dimensions"]
            for item in dimension["items"]
        }

    def test_same_samples_but_independent_user_results(self) -> None:
        alice = self.client()
        status, _ = self.request_json(
            alice, "/api/login", "POST", {"username": "alice"}
        )
        self.assertEqual(200, status)
        status, sample = self.request_json(alice, "/api/sample/1")
        self.assertEqual(200, status)
        self.assertEqual("Candidate one", sample["candidate"]["content"])
        self.assertNotIn("evaluation_id", sample)
        self.assertEqual("greeting", sample["messages"][1]["display_role"])
        status, saved = self.request_json(
            alice,
            "/api/annotation",
            "POST",
            {
                "seq": 1,
                "status": "submitted",
                "gates": self.gates(),
                "scores": self.score_payload(),
                "gate_notes": "",
                "overall_notes": "",
            },
        )
        self.assertEqual(200, status, saved)
        self.assertEqual(1, saved["progress"]["submitted"])

        bob = self.client()
        self.request_json(bob, "/api/login", "POST", {"username": "bob"})
        status, sample = self.request_json(bob, "/api/sample/1")
        self.assertEqual(200, status)
        self.assertEqual("Candidate one", sample["candidate"]["content"])
        self.assertIsNone(sample["annotation"])
        self.assertEqual(0, sample["progress"]["submitted"])

        snapshot = self.root / "results" / "by_user" / "alice" / "000001.json"
        self.assertTrue(snapshot.is_file())
        record = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual("eval_one", record["evaluation_id"])
        self.assertEqual("submitted", record["status"])

    def test_score_three_allows_empty_evidence(self) -> None:
        scores = self.score_payload()
        scores["H2"] = {"value": 3, "evidence": ""}
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "charlie"})
        status, response = self.request_json(
            client,
            "/api/annotation",
            "POST",
            {
                "seq": 1,
                "status": "submitted",
                "gates": self.gates(),
                "scores": scores,
                "gate_notes": "",
                "overall_notes": "",
            },
        )
        self.assertEqual(200, status)
        self.assertEqual("submitted", response["status"])

    def test_binary_item_rejects_a_score_not_in_its_scale(self) -> None:
        scores = self.score_payload()
        scores["C1"] = {"value": 1, "evidence": ""}
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "delta"})
        status, response = self.request_json(
            client,
            "/api/annotation",
            "POST",
            {
                "seq": 1,
                "status": "submitted",
                "gates": self.gates(),
                "scores": scores,
                "gate_notes": "",
                "overall_notes": "",
            },
        )
        self.assertEqual(400, status)
        self.assertIn("C1 分数无效", response["detail"])

    def test_battle_replays_pair_and_hides_model_identity(self) -> None:
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "viewer"})
        status, response = self.request_json(client, "/api/battle/1")
        self.assertEqual(200, status)
        self.assertEqual(1, response["pair_seq"])
        self.assertEqual(1, response["total"])
        self.assertTrue(response["context_match"])
        self.assertEqual({"A", "B"}, {candidate["label"] for candidate in response["candidates"]})
        self.assertEqual(
            {"Candidate one", "Candidate two"},
            {candidate["content"] for candidate in response["candidates"]},
        )
        self.assertEqual(1, sum(candidate["online_selected"] for candidate in response["candidates"]))
        serialized = json.dumps(response)
        self.assertNotIn("model-one", serialized)
        self.assertNotIn("model-two", serialized)
        self.assertNotIn("pair_one", serialized)
        self.assertNotIn("eval_one", serialized)

        status, models = self.request_json(client, "/api/battle/1/models")
        self.assertEqual(200, status)
        self.assertEqual(
            {"model-one", "model-two"},
            {model["model_name"] for model in models["models"]},
        )


if __name__ == "__main__":
    unittest.main()
