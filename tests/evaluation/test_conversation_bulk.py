import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.evaluation.roleplay_annotation_platform import conversation_filter_app as web
from src.evaluation.roleplay_annotation_platform.conversation_ranking import (
    rank_records,
    csv_text,
)
from src.evaluation.roleplay_annotation_platform.conversation_filter_data import (
    DDL,
    insert_conversation,
)
from src.evaluation.roleplay_annotation_platform.llm_judge import load_judge_config
from tests.evaluation.test_conversation_filter import (
    fixture,
    output,
    FakeCatalog,
    no_start,
)


class RankingTests(unittest.TestCase):
    def test_ties_na_zero_refusal_missing_and_csv_injection(self):
        one = output()
        one["scores"]["C1"]["value"] = 0
        one["scores"]["C2"]["value"] = 3
        one["scores"]["C3"]["value"] = 3
        two = output()
        two["scores"]["H2"]["value"] = "N/A"
        refusal = output()
        for row in refusal["scores"].values():
            row["reason"] = "无法进行评分"
        rows = [
            {
                "seq": 1,
                "status": "completed",
                "output_json": json.dumps(one),
                "character_name": "=formula()",
            },
            {"seq": 2, "status": "completed", "output_json": json.dumps(two)},
            {"seq": 3, "status": "completed", "output_json": json.dumps(refusal)},
            {"seq": 4, "status": "failed", "output_json": None},
        ]
        report = rank_records(rows)
        self.assertEqual(2, report["ranked_count"])
        self.assertEqual([1, 1], [r["rank"] for r in report["rows"]])
        self.assertEqual([2.0, 2.0], [r["mean_score"] for r in report["rows"]])
        self.assertEqual(0, report["rows"][0]["min_score"])
        self.assertEqual(19, report["rows"][1]["numeric_count"])
        self.assertEqual(1, report["suspected_refusal_count"])
        self.assertEqual(1, report["score_counts"]["N/A"])
        self.assertIn("'=formula()", csv_text(report))
        self.assertIsNone(report["excluded"][1]["mean_score"])

    def test_local_uncertainty_is_not_whole_answer_refusal(self):
        scored = output()
        scored["data_limitations"] = [
            "A2 是 context_only，无法评价该轮质量。",
            "图片链接正确性无法评估。",
        ]
        scored["scores"]["C5"].update(
            value="N/A", reason="只有一个角色，无法评分多角色归属。"
        )
        report = rank_records(
            [{"seq": 1, "status": "completed", "output_json": scored}]
        )
        self.assertEqual(1, report["ranked_count"])
        self.assertFalse(report["rows"][0]["suspected_refusal"])
        self.assertTrue(report["rows"][0]["review_notes"])


class BulkTests(unittest.TestCase):
    def test_prepared_100_sliding_slots_budget_and_ranking_ownership(self):
        real_start = web.Runtime.start
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.sqlite3"
            with sqlite3.connect(source) as c:
                c.executescript(DDL)
                for seq in range(1, 102):
                    p = fixture()
                    p["conversation_id"] = f"c-{seq}"
                    insert_conversation(c, seq, p)
                c.execute(
                    "INSERT INTO metadata VALUES ('manifest',?)",
                    (
                        json.dumps(
                            {
                                "ready": True,
                                "dataset_sha256": "data1",
                                "report": {},
                                "languages": {},
                            }
                        ),
                    ),
                )
            c.close()
            config = replace(
                load_judge_config(),
                api_key="mock-only",
                models=("test/model",),
                tatu_api_key="",
            )
            headers = {"X-Requested-With": "conversation-filter"}
            web.STATE.clear()
            with (
                patch.dict(os.environ, {"DEEPINFRA_API_KEY": ""}),
                patch.object(web, "SAMPLE_DB", source),
                patch.object(web, "RESULT_DB", root / "jobs.sqlite3"),
                patch.object(web, "load_judge_config", return_value=config),
                patch.object(web, "OpenRouterModelCatalog", FakeCatalog),
                patch.object(web.Runtime, "start", no_start),
                TestClient(web.app) as client,
            ):
                client.post(
                    "/api/login", json={"username": "bulk"}, headers=headers
                ).raise_for_status()
                body = {
                    "model": "test/model",
                    "scope": {"mode": "all"},
                    "budget": 50,
                    "concurrency": 100,
                    "max_attempts": 4,
                    "acknowledge_snapshot": True,
                    "nonce": "bulk-1",
                    "start_paused": True,
                }
                response = client.post("/api/jobs", json=body, headers=headers)
                self.assertEqual(200, response.status_code, response.text)
                job = response.json()["job_id"]
                self.assertTrue(response.json()["prepared"])
                self.assertEqual(
                    job,
                    client.post("/api/jobs", json=body, headers=headers).json()[
                        "job_id"
                    ],
                )
                self.assertEqual(
                    400,
                    client.post(
                        "/api/jobs",
                        json={**body, "concurrency": 101, "nonce": "invalid"},
                        headers=headers,
                    ).status_code,
                )
                runtime = web.STATE["runtime"]
                self.assertEqual(100, runtime.client._transport._pool._max_connections)
                self.assertIsNone(runtime.claim())
                info = client.get(f"/api/jobs/{job}").json()
                self.assertEqual("paused", info["status"])
                self.assertEqual(4, info["max_attempts"])
                self.assertEqual({"queued": 101}, info["counts"])
                self.assertEqual(
                    0, client.get(f"/api/jobs/{job}/ranking").json()["ranked_count"]
                )
                client.post(
                    f"/api/jobs/{job}/control",
                    json={"action": "resume"},
                    headers=headers,
                ).raise_for_status()

                async def simulation():
                    all_started = asyncio.Event()
                    replenished = asyncio.Event()
                    release_first = asyncio.Event()
                    release_rest = asyncio.Event()
                    active = 0
                    peak = 0
                    started = []

                    async def evaluate(saved_job, item):
                        nonlocal active, peak
                        active += 1
                        peak = max(peak, active)
                        started.append(item["seq"])
                        if len(started) == 100:
                            all_started.set()
                        if len(started) == 101:
                            replenished.set()
                        await (
                            release_first if item["seq"] == 1 else release_rest
                        ).wait()
                        with web.results() as c:
                            c.execute(
                                "UPDATE job_items SET status='completed',output_json=? WHERE job_id=? AND seq=?",
                                (json.dumps(output()), saved_job["id"], item["seq"]),
                            )
                        active -= 1

                    with patch.object(runtime, "evaluate", evaluate):
                        await real_start(runtime)
                        try:
                            await asyncio.wait_for(all_started.wait(), 10)
                            self.assertEqual(100, len(started))
                            release_first.set()
                            await asyncio.wait_for(replenished.wait(), 10)
                            self.assertFalse(release_rest.is_set())
                            self.assertEqual(100, active)
                            self.assertEqual(100, peak)
                        finally:
                            for task in runtime.tasks:
                                task.cancel()
                            await asyncio.gather(*runtime.tasks, return_exceptions=True)
                            runtime.tasks = []
                    return peak

                self.assertEqual(100, client.portal.call(simulation))
                ranking = client.get(f"/api/jobs/{job}/ranking").json()
                self.assertEqual(1, ranking["ranked_count"])
                self.assertEqual(2, ranking["rows"][0]["mean_score"])
                exported = client.get(f"/api/jobs/{job}/ranking?format=csv")
                self.assertEqual(200, exported.status_code)
                self.assertIn("text/csv", exported.headers["content-type"])
                # Resume remains bounded, and unknown in-flight work is not replayed.
                with web.results() as c:
                    c.execute("UPDATE jobs SET spent=50 WHERE id=?", (job,))
                self.assertIsNone(runtime.claim())
                self.assertEqual(
                    "paused", client.get(f"/api/jobs/{job}").json()["status"]
                )
                web.initialize()
                self.assertEqual(
                    100, client.get(f"/api/jobs/{job}").json()["counts"]["interrupted"]
                )
                client.post(
                    "/api/login", json={"username": "other"}, headers=headers
                ).raise_for_status()
                self.assertEqual(
                    404, client.get(f"/api/jobs/{job}/ranking?format=full").status_code
                )


if __name__ == "__main__":
    unittest.main()
