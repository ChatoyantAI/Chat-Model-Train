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
    REPO_ROOT / "docs" / "evaluation" / "ROLEPLAY_EVALUATION_DIMENSIONS_V0_1.md"
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
    def test_rubric_has_all_items_and_real_reply_examples(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        items = [item for dimension in rubric["dimensions"] for item in dimension["items"]]
        self.assertEqual(5, len(rubric["dimensions"]))
        self.assertEqual(25, len(items))
        self.assertTrue(all(set(item["example"]) == {"background", "0", "1", "2", "3"} for item in items))
        self.assertTrue(all(item["example"]["3"] for item in items))

    def test_blind_data_validator_rejects_ground_truth_keys(self) -> None:
        record = sample_record("eval_x", "candidate")
        record["model_name"] = "secret-model"
        with self.assertRaisesRegex(ValueError, "blind-data violation"):
            _validate_record(record, 1)


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
        cls.rubric_json = cls.root / "rubric.json"
        cls.rubric_json.write_text(
            json.dumps(build_rubric(RUBRIC_MARKDOWN), ensure_ascii=False),
            encoding="utf-8",
        )
        os.environ["ROLEPLAY_SAMPLE_DB"] = str(cls.sample_db)
        os.environ["ROLEPLAY_RESULTS_DB"] = str(cls.root / "results" / "annotations.sqlite3")
        os.environ["ROLEPLAY_RESULTS_DIR"] = str(cls.root / "results" / "by_user")
        os.environ["ROLEPLAY_RUBRIC_JSON"] = str(cls.rubric_json)
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
        }

    def score_payload(self, value: int = 2, evidence: str = "") -> dict:
        rubric = build_rubric(RUBRIC_MARKDOWN)
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

    def test_score_three_requires_evidence(self) -> None:
        scores = self.score_payload()
        scores["H1"] = {"value": 3, "evidence": ""}
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
        self.assertEqual(400, status)
        self.assertIn("H1", response["detail"])


if __name__ == "__main__":
    unittest.main()
