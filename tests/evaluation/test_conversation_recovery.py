import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_annotation_platform import conversation_filter_app as web
from src.evaluation.roleplay_annotation_platform.conversation_filter import (
    rubric_payload,
)
from src.evaluation.roleplay_annotation_platform.conversation_filter_data import (
    DDL,
    insert_conversation,
)
from src.evaluation.roleplay_annotation_platform.supervise_conversation_run import (
    recover_failures,
)
from src.evaluation.roleplay_annotation_platform.export_conversation_run import (
    export_run,
)
from tests.evaluation.test_conversation_filter import fixture, output


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.result = self.root / "jobs.sqlite3"
        self.source = self.root / "source.sqlite3"
        with patch.object(web, "RESULT_DB", self.result):
            web.initialize()
        with sqlite3.connect(self.source) as c:
            c.executescript(DDL)
            for seq in range(1, 5):
                packet = fixture()
                packet["conversation_id"] = f"c-{seq}"
                insert_conversation(c, seq, packet)
            c.execute(
                "INSERT INTO metadata VALUES('manifest',?)",
                (json.dumps({"dataset_sha256": "d1"}),),
            )
        c.close()
        with sqlite3.connect(self.result) as c:
            c.execute(
                "INSERT INTO jobs(id,username,nonce,model,status,created,updated,scope_json,template_json,rubric_hash,data_hash,total,budget,concurrency) VALUES ('job','test','nonce','deepinfra::zai-org/GLM-5.3-Flash','completed_errors','date','date','{}','{}',?,'d1',4,90,100)",
                (rubric_payload()["sha256"],),
            )
            c.executemany(
                "INSERT INTO job_items(job_id,seq,position,status,attempts,remaining,error) VALUES ('job',?,?,'failed',4,0,'网络请求失败：ReadTimeout')",
                ((seq, seq) for seq in range(1, 5)),
            )
            c.execute(
                "UPDATE job_items SET status='completed',output_json=?,error='' WHERE seq=1",
                (json.dumps(output()),),
            )
            c.execute("UPDATE job_items SET error='HTTP 400：unsupported' WHERE seq=3")
            c.execute("UPDATE job_items SET attempts=12 WHERE seq=4")
        c.close()

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_transient_failures_requeued_and_total_cap(self):
        result = recover_failures(self.result, self.source, "job")
        self.assertEqual(
            {
                "action": "requeued",
                "count": 1,
                "previous_status": "completed_errors",
                "new_unknown_cost_attempts": 0,
            },
            result,
        )
        with sqlite3.connect(self.result) as c:
            self.assertEqual(
                ("queued", 4, 4),
                c.execute(
                    "SELECT status,attempts,remaining FROM job_items WHERE seq=2"
                ).fetchone(),
            )
            self.assertEqual(
                "completed",
                c.execute("SELECT status FROM job_items WHERE seq=1").fetchone()[0],
            )
        c.close()
        self.assertEqual(
            "wait", recover_failures(self.result, self.source, "job")["action"]
        )

    def test_preserve_manual_pause_and_budget(self):
        with sqlite3.connect(self.result) as c:
            c.execute(
                "UPDATE jobs SET status='paused',message='已暂停派发；在途请求会继续保存'"
            )
        c.close()
        self.assertEqual(
            "wait", recover_failures(self.result, self.source, "job")["action"]
        )
        with sqlite3.connect(self.result) as c:
            c.execute(
                "UPDATE jobs SET status='running',spent=89,unknown_cost_attempts=11"
            )
        c.close()
        self.assertEqual(
            "预算保护", recover_failures(self.result, self.source, "job")["reason"]
        )
        with sqlite3.connect(self.result) as c:
            self.assertEqual(
                "paused", c.execute("SELECT status FROM jobs").fetchone()[0]
            )
        c.close()

    def test_recovery_uses_task_budget_instead_of_a_fixed_run_limit(self):
        with sqlite3.connect(self.result) as c:
            c.execute("UPDATE jobs SET status='running',budget=5,spent=6")
        c.close()
        self.assertEqual(
            "预算保护", recover_failures(self.result, self.source, "job")["reason"]
        )
        with sqlite3.connect(self.result) as c:
            c.execute("UPDATE jobs SET status='completed_errors',budget=125,spent=91")
        c.close()
        self.assertEqual(
            "requeued", recover_failures(self.result, self.source, "job")["action"]
        )

    def test_export_complete_groups_and_idempotent_source_unchanged(self):
        low = output()
        low["scores"]["L2"]["value"] = 1
        refusal = output()
        for row in refusal["scores"].values():
            row["reason"] = "无法评估该内容"
        with sqlite3.connect(self.result) as c:
            c.execute(
                "UPDATE job_items SET status='completed',output_json=? WHERE seq=2",
                (json.dumps(low),),
            )
            c.execute(
                "UPDATE job_items SET status='completed',output_json=? WHERE seq=3",
                (json.dumps(refusal),),
            )
        c.close()
        original = self.source.read_bytes()
        exported = export_run(self.result, self.source, "job", self.root / "exports")
        self.assertEqual(
            {"keep": 1, "review": 1, "unreviewable": 2}, exported["groups"]
        )
        self.assertEqual(4, exported["ranked_count"] + exported["unranked_count"])
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(
            exported, export_run(self.result, self.source, "job", self.root / "exports")
        )
        with gzip.open(
            self.root / "exports/job/keep.conversations.jsonl.gz",
            "rt",
            encoding="utf-8",
        ) as f:
            record = json.loads(f.readline())
        self.assertEqual(["target-1", "target-3"], record["candidate_message_ids"])
        self.assertEqual("context_only", record["conversation"]["messages"][3]["scope"])


if __name__ == "__main__":
    unittest.main()
