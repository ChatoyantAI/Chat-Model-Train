from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zlib
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.evaluation.roleplay_annotation_platform import filter_v2_app
from src.evaluation.roleplay_annotation_platform.single_sample_filter_v2 import (
    rubric_items,
)


ROOT = Path(__file__).resolve().parents[2]


class FilterV2AppTests(unittest.TestCase):
    def test_isolated_scored_review_and_two_phase_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_db = root / "samples.sqlite3"
            result_db = root / "v2-results" / "reviews.sqlite3"
            result_dir = root / "v2-results" / "by_user"
            with closing(sqlite3.connect(sample_db)) as connection:
                connection.execute(
                    "CREATE TABLE samples(seq INTEGER PRIMARY KEY, evaluation_id TEXT, "
                    "trigger_type TEXT, payload_zlib BLOB)"
                )
                payload = {
                    "input_messages": [
                        {"role": "system", "content": "You are Rowan."},
                        {"role": "assistant", "content": "Welcome."},
                        {"role": "user", "content": "Stay here."},
                    ],
                    "output_message": {"role": "assistant", "content": "I will."},
                }
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?)",
                    (
                        1,
                        "eval-1",
                        "single_reply",
                        zlib.compress(json.dumps(payload).encode("utf-8")),
                    ),
                )
                connection.commit()

            filter_v2_app.APP_STATE.clear()
            rubric_path = (
                ROOT
                / "src/evaluation/roleplay_annotation_platform/single_sample_filter_rubric_v2.json"
            )
            rubric_doc = (
                ROOT / "docs/evaluation/ROLEPLAY_SINGLE_SAMPLE_FILTER_RUBRIC_V2.md"
            )
            with (
                patch.object(filter_v2_app, "SAMPLE_DB", sample_db),
                patch.object(filter_v2_app, "RESULT_DB", result_db),
                patch.object(filter_v2_app, "RESULT_DIR", result_dir),
                patch.object(filter_v2_app, "RUBRIC_PATH", rubric_path),
                patch.object(filter_v2_app, "RUBRIC_DOC", rubric_doc),
                TestClient(filter_v2_app.app) as client,
            ):
                health = client.get("/health").json()
                self.assertEqual(
                    "roleplay-single-sample-filter-v2.2", health["protocol_version"]
                )
                self.assertEqual("isolated", health["result_db"])
                self.assertEqual(
                    200,
                    client.post(
                        "/api/login", json={"username": "reviewer-v21"}
                    ).status_code,
                )
                rubric = client.get("/api/rubric").json()
                items = rubric_items(rubric)
                self.assertEqual(20, len(items))
                preview = client.get("/api/prompt-preview/1").json()
                self.assertNotIn("Authorization", json.dumps(preview))
                self.assertEqual(0, preview["gate"]["temperature"])
                self.assertGreaterEqual(len(preview["score_requests"]), 5)
                self.assertTrue(
                    all(
                        batch["temperature"] == 1 for batch in preview["score_requests"]
                    )
                )
                self.assertNotIn(
                    "reason", json.dumps(preview["score_requests"][0]["schema"])
                )

                body = {
                    "seq": 1,
                    "protocol_version": rubric["version"],
                    "status": "submitted",
                    "gates": {gate["id"]: False for gate in rubric["gates"]},
                    "items": {item["id"]: "2" for item in items},
                    "notes": "",
                }
                saved = client.post("/api/review", json=body)
                self.assertEqual(200, saved.status_code)
                self.assertEqual("keep_candidate", saved.json()["decision"])

            snapshot = json.loads(
                (result_dir / "reviewer-v21" / "000001.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "roleplay-single-sample-filter-v2.2", snapshot["protocol_version"]
            )
            self.assertIsNotNone(snapshot["score_summary"]["quality_score"])

    def test_gate_submission_can_stop_before_scores(self) -> None:
        rubric = filter_v2_app.load_filter_rubric_v2()
        filter_v2_app.APP_STATE["rubric"] = rubric
        gates = {gate["id"]: False for gate in rubric["gates"]}
        gates["F-G2"] = True
        normalized_gates, items, decision = filter_v2_app.normalize_human_review(
            gates, {}, submitting=True
        )
        self.assertTrue(normalized_gates["F-G2"])
        self.assertEqual({}, items)
        self.assertEqual("needs_gate_verification", decision)

    def test_low_score_routes_to_item_verification(self) -> None:
        rubric = filter_v2_app.load_filter_rubric_v2()
        filter_v2_app.APP_STATE["rubric"] = rubric
        gates = {gate["id"]: False for gate in rubric["gates"]}
        values = {item["id"]: "2" for item in rubric_items(rubric)}
        values["L3"] = "1"
        _, _, decision = filter_v2_app.normalize_human_review(
            gates, values, submitting=True
        )
        self.assertEqual("needs_item_verification", decision)


if __name__ == "__main__":
    unittest.main()
