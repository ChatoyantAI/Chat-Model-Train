"""User-authorized, fixed-job failure recovery and completion export companion.

Does not call a model itself. Requeues only failed/invalid/interrupted items;
never retries a valid low score or an explicit judge refusal.
"""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from .conversation_filter import rubric_payload
from .conversation_ranking import REFUSAL
from .export_conversation_run import export_run, read_job

TRANSIENT = re.compile(
    r"HTTP (?:408|429|500|502|503|504)\b|网络请求失败|响应处理失败|后台处理意外中断|服务重启"
)
AUTOMATIC_PAUSES = ("服务已重启", "后台处理意外中断")
MAX_TOTAL_ATTEMPTS = 12
UNKNOWN_COST_RESERVE = 0.10  # conservative budget bookkeeping, NOT a billed-price claim


def recover_failures(result_db, sample_db, job_id):
    with closing(
        sqlite3.connect(f"file:{Path(sample_db).as_posix()}?mode=ro", uri=True)
    ) as source:
        info = json.loads(
            source.execute(
                "SELECT value FROM metadata WHERE key='manifest'"
            ).fetchone()[0]
        )
    with closing(sqlite3.connect(result_db, timeout=30)) as c, c:
        c.row_factory = sqlite3.Row
        c.execute("BEGIN IMMEDIATE")
        job = dict(c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
        dispatch_budget = job["budget"]
        c.execute("""CREATE TABLE IF NOT EXISTS conversation_recovery_log(
            id INTEGER PRIMARY KEY,job_id TEXT NOT NULL,created TEXT NOT NULL,action TEXT NOT NULL,
            count INTEGER NOT NULL,details TEXT NOT NULL)""")

        def record(action, count=0, details=""):
            c.execute(
                "INSERT INTO conversation_recovery_log(job_id,created,action,count,details) VALUES (?,?,?,?,?)",
                (
                    job_id,
                    datetime.now(timezone.utc).isoformat(),
                    action,
                    count,
                    details,
                ),
            )

        if (
            job["data_hash"]
            != (info.get("dataset_sha256") or info.get("source_sha256"))
            or job["rubric_hash"] != rubric_payload()["sha256"]
        ):
            return {"action": "wait", "reason": "版本已变化，不混用新旧评分"}
        if job["status"] == "cancelled":
            return {"action": "wait", "reason": "用户已取消"}
        risk = job["spent"] + job["unknown_cost_attempts"] * UNKNOWN_COST_RESERVE
        if risk >= dispatch_budget:
            if job["status"] in ("queued", "running"):
                c.execute(
                    "UPDATE jobs SET status='paused',message=? WHERE id=?",
                    (
                        f"预算保护：已知费用加未知请求保守预留已达到 ${dispatch_budget:g}，保留在途余量",
                        job_id,
                    ),
                )
                record(
                    "budget_pause",
                    details=f"known={job['spent']}; unknown={job['unknown_cost_attempts']}; reserve={UNKNOWN_COST_RESERVE}",
                )
            return {
                "action": "wait",
                "reason": "预算保护",
                "known_cost": job["spent"],
                "unknown_cost_attempts": job["unknown_cost_attempts"],
            }
        if job["status"] == "paused" and not job["message"].startswith(
            AUTOMATIC_PAUSES
        ):
            return {"action": "wait", "reason": "保留用户暂停、权限/额度或费用暂停"}
        automatic_resumes = c.execute(
            "SELECT COUNT(*) FROM conversation_recovery_log WHERE job_id=? AND action='resume_interruption'",
            (job_id,),
        ).fetchone()[0]
        if job["status"] == "paused" and automatic_resumes >= 3:
            return {"action": "wait", "reason": "重复中断达到恢复上限，需要排查"}
        # Let the ordinary four-attempt queue finish its current pass first.
        running = c.execute(
            "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status='running'",
            (job_id,),
        ).fetchone()[0]
        queued = c.execute(
            "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status='queued'",
            (job_id,),
        ).fetchone()[0]
        if running or (queued and job["status"] != "paused"):
            return {"action": "wait", "reason": "主队列仍在推进"}
        candidates = c.execute(
            "SELECT * FROM job_items WHERE job_id=? AND status IN ('failed','invalid','interrupted') AND attempts<? ORDER BY position",
            (job_id, MAX_TOTAL_ATTEMPTS),
        ).fetchall()
        selected = []
        unknown = 0
        for item in candidates:
            latest = c.execute(
                "SELECT raw_json FROM attempts WHERE job_id=? AND seq=? ORDER BY id DESC LIMIT 1",
                (job_id, item["seq"]),
            ).fetchone()
            raw = json.loads(latest[0]) if latest else {}
            if raw.get("refusal") or REFUSAL.search(str(raw.get("content") or "")):
                continue
            if item["status"] == "failed" and not TRANSIENT.search(item["error"]):
                continue
            if item["status"] == "interrupted":
                recorded = c.execute(
                    "SELECT 1 FROM attempts WHERE job_id=? AND seq=? AND attempt=?",
                    (job_id, item["seq"], item["attempts"]),
                ).fetchone()
                unknown += int(not recorded)
            selected.append(
                (min(4, MAX_TOTAL_ATTEMPTS - item["attempts"]), job_id, item["seq"])
            )
        if risk + unknown * UNKNOWN_COST_RESERVE >= dispatch_budget:
            return {"action": "wait", "reason": "恢复中断请求的费用预留不足"}
        if not selected and not (job["status"] == "paused" and queued):
            return {
                "action": "none",
                "reason": "没有可补跑的技术失败；保留拒评、不可重试错误和达到上限项目",
            }
        c.executemany(
            "UPDATE job_items SET status='queued',remaining=?,ready_at=0 WHERE job_id=? AND seq=?",
            selected,
        )
        c.execute(
            "UPDATE jobs SET status='queued',updated=?,unknown_cost_attempts=unknown_cost_attempts+?,message='自动补跑技术失败；保留原尝试，累计每段最多 12 次' WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), unknown, job_id),
        )
        record(
            "retry_failed",
            len(selected),
            f"new_unknown={unknown}; max_total_attempts={MAX_TOTAL_ATTEMPTS}",
        )
        if job["status"] == "paused":
            record("resume_interruption", len(selected))
        return {
            "action": "requeued",
            "count": len(selected),
            "previous_status": job["status"],
            "new_unknown_cost_attempts": unknown,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-db", required=True, type=Path)
    parser.add_argument("--sample-db", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authorize-failure-retries", action="store_true")
    args = parser.parse_args()
    if not args.authorize_failure_retries:
        raise SystemExit("此补跑需用户明确授权 --authorize-failure-retries")
    os.umask(0o027)
    args.output_root.mkdir(parents=True, exist_ok=True)
    previous = None
    while True:
        action = recover_failures(args.result_db, args.sample_db, args.job_id)
        job, counts = read_job(args.result_db, args.job_id)
        state = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "job_id": args.job_id,
            "status": job["status"],
            "counts": counts,
            "spent": job["spent"],
            "unknown_cost_attempts": job["unknown_cost_attempts"],
            "recovery": action,
        }
        snapshot = args.output_root / (args.job_id + "-monitor.json")
        temporary = snapshot.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(snapshot)
        if action != previous:
            print(json.dumps(state, ensure_ascii=False), flush=True)
            previous = action
        if (
            job["status"] in ("completed", "completed_errors", "cancelled")
            and not any(counts.get(s, 0) for s in ("running", "queued"))
            and action["action"] != "requeued"
        ):
            result = export_run(
                args.result_db, args.sample_db, args.job_id, args.output_root
            )
            print(
                json.dumps({"action": "exported", **result}, ensure_ascii=False),
                flush=True,
            )
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
