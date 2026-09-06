"""Build, read and verify a tiny training release using only synthetic data."""

from contextlib import closing, redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.evaluation.roleplay_annotation_platform import conversation_filter_app as web
from src.evaluation.roleplay_annotation_platform.conversation_filter import (
    rubric_payload,
)
from src.evaluation.roleplay_annotation_platform.conversation_filter_data import (
    DDL,
    insert_conversation,
)
from src.evaluation.roleplay_annotation_platform.conversation_platform_context import (
    assemble_platform_context,
)
from src.evaluation.roleplay_annotation_platform.export_conversation_run import (
    export_run,
)
from src.evaluation.roleplay_annotation_platform.export_target_training_bundle import (
    build_bundle,
    verify_release,
)
from src.evaluation.roleplay_annotation_platform.select_ranked_conversation_targets import (
    sha256_file,
)
from tests.evaluation.test_conversation_filter import fixture, output


class TargetBundleEndToEnd(unittest.TestCase):
    def test_frozen_rank_to_shards_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            (root / "data").mkdir(parents=True)
            source = root / "data/targets.parquet"
            rows = [
                dict(
                    conversation_id="conversation-1",
                    character_id="role-1",
                    message_id=f"target-{i}",
                    query_increase_id=i * 2,
                    target_increase_id=i * 2 + 1,
                    query_text=f"Question {i}",
                    target_text=f"Answer {i}",
                    language="en",
                    model="hidden-model",
                    turn_n=i,
                    query_chars=len(f"Question {i}"),
                    target_chars=len(f"Answer {i}"),
                )
                for i in (1, 3)
            ]
            pq.write_table(pa.Table.from_pylist(rows), source)
            packet = fixture()
            packet["greeting"]["content"] = "Welcome aboard."
            packet = assemble_platform_context(
                packet, {"chat_mod": 250, "is_test_account": False}
            )
            # Exercise non-ASCII provenance and reports on Windows and Linux.
            packet["character_name"] = "领航员"
            data_db = root / "data/conversations-v1-platform.sqlite3"
            with closing(sqlite3.connect(data_db)) as conn, conn:
                conn.executescript(DDL)
                insert_conversation(conn, 1, packet)
                conn.execute(
                    "INSERT INTO metadata VALUES (?,?)",
                    (
                        "manifest",
                        json.dumps(
                            {
                                "dataset_sha256": "synthetic-data-v1",
                                "source_sha256": sha256_file(source),
                                "source_file": str(source),
                                "report": {"candidates": 2},
                            }
                        ),
                    ),
                )
            result_db = root / "results/jobs.sqlite3"
            with patch.object(web, "RESULT_DB", result_db):
                web.initialize()
            with closing(sqlite3.connect(result_db)) as conn, conn:
                conn.execute(
                    "INSERT INTO jobs(id,username,nonce,model,status,created,updated,scope_json,template_json,rubric_hash,data_hash,total,budget,concurrency) VALUES ('job','tester','nonce','mock','completed','date','date','{}','{}',?,'synthetic-data-v1',1,5,1)",
                    (rubric_payload()["sha256"],),
                )
                conn.execute(
                    "INSERT INTO job_items(job_id,seq,position,status,output_json) VALUES ('job',1,1,'completed',?)",
                    (json.dumps(output()),),
                )
            export_run(result_db, data_db, "job", root / "exports")
            destination = Path(temporary) / "release"
            with redirect_stdout(io.StringIO()):
                manifest = build_bundle(
                    root, "job", destination, limit=2, character_cap=2, shard_size=1
                )
            self.assertEqual([1, 1], [s["records"] for s in manifest["shards"]])
            self.assertFalse(manifest["target_independently_scored"])
            verify_release(destination)
            records = []
            for shard in manifest["shards"]:
                with gzip.open(
                    destination / shard["path"], "rt", encoding="utf-8"
                ) as handle:
                    records.extend(json.loads(line) for line in handle)
            self.assertEqual(
                ["target-1", "target-3"], [r["sample_id"] for r in records]
            )
            self.assertEqual(["Answer 1", "Answer 3"], [r["target"] for r in records])
            self.assertTrue(
                all(
                    r["messages"][0]["role"] == "system"
                    and r["messages"][-1]["role"] == "user"
                    for r in records
                )
            )
            self.assertEqual(
                manifest,
                build_bundle(
                    root, "job", destination, limit=2, character_cap=2, shard_size=1
                ),
            )
            with self.assertRaisesRegex(ValueError, "Existing release differs"):
                build_bundle(
                    root, "job", destination, limit=2, character_cap=3, shard_size=1
                )
            previous = destination / "audit/selection/target_ranking.jsonl.gz"
            with redirect_stdout(io.StringIO()):
                compared = build_bundle(
                    root,
                    "job",
                    Path(temporary) / "compared",
                    limit=2,
                    character_cap=2,
                    shard_size=1,
                    previous_ranking=previous,
                )
            self.assertEqual(
                sha256_file(previous),
                compared["build_signature"]["previous_ranking_sha256"],
            )
            audit = json.loads(
                (Path(temporary) / "compared/audit/verification.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(2, audit["overlap_with_previous_ranking"])
            self.assertEqual(0, audit["replaced_from_previous_ranking"])


if __name__ == "__main__":
    unittest.main()
