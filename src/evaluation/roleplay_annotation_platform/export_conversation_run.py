"""Read-only completion companion: exports data locally on the scoring server.

It never calls a model, resumes a job, changes a score, or deletes source data.
"""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sqlite3
import time
import zlib

from .conversation_filter import screening
from .conversation_ranking import rank_records, csv_text


def read_job(result_db, job_id):
    with closing(
        sqlite3.connect(
            f"file:{Path(result_db).as_posix()}?mode=ro", uri=True, timeout=30
        )
    ) as c:
        c.row_factory = sqlite3.Row
        job = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if job is None:
            raise ValueError("指定任务不存在")
        counts = dict(
            c.execute(
                "SELECT status,COUNT(*) FROM job_items WHERE job_id=? GROUP BY status",
                (job_id,),
            )
        )
        return dict(job), counts


def export_run(result_db, sample_db, job_id, output_root):
    """Only export terminal snapshots. Files appear together after atomic rename."""
    job, counts = read_job(result_db, job_id)
    if job["status"] not in ("completed", "completed_errors", "cancelled") or any(
        counts.get(s, 0) for s in ("queued", "running")
    ):
        raise ValueError("任务尚有在途或排队项目，不生成终态文件")
    destination = Path(output_root) / job_id
    if destination.exists():
        previous = json.loads(
            (destination / "SUMMARY.json").read_text(encoding="utf-8")
        )
        if previous["job_updated"] != job["updated"]:
            raise ValueError(
                "任务在导出后发生变化；保留已有快照，需另选输出目录重新导出"
            )
        return previous
    stage = Path(output_root) / (job_id + ".building")
    stage.mkdir(parents=True, exist_ok=True)
    with closing(
        sqlite3.connect(
            f"file:{Path(result_db).as_posix()}?mode=ro", uri=True, timeout=30
        )
    ) as c:
        c.row_factory = sqlite3.Row
        rows = [
            dict(r)
            for r in c.execute(
                "SELECT seq,status,output_json,error FROM job_items WHERE job_id=? ORDER BY position",
                (job_id,),
            )
        ]
    with closing(
        sqlite3.connect(
            f"file:{Path(sample_db).as_posix()}?mode=ro", uri=True, timeout=30
        )
    ) as source:
        source.row_factory = sqlite3.Row
        metadata = {
            r["seq"]: dict(r)
            for r in source.execute(
                "SELECT seq,conversation_id,character_name,language,candidate_count FROM conversations"
            )
        }
        report = rank_records([{**r, **metadata[r["seq"]]} for r in rows])
        report.update(
            job_id=job_id,
            job_status=job["status"],
            rubric_hash=job["rubric_hash"],
            data_hash=job["data_hash"],
        )
        refused = {r["seq"] for r in report["excluded"] if r["suspected_refusal"]}
        rank_by_seq = {r["seq"]: r for r in report["rows"]}
        groups = Counter()
        candidates = Counter()
        with (
            gzip.open(
                stage / "keep.conversations.jsonl.gz", "wt", encoding="utf-8"
            ) as keep,
            gzip.open(
                stage / "review.index.jsonl.gz", "wt", encoding="utf-8"
            ) as review,
            gzip.open(
                stage / "unreviewable.index.jsonl.gz", "wt", encoding="utf-8"
            ) as unreviewable,
            gzip.open(
                stage / "all_scores.jsonl.gz", "wt", encoding="utf-8"
            ) as all_scores,
        ):
            for row in rows:
                output = json.loads(row["output_json"]) if row["output_json"] else None
                screen = (
                    screening(output, minimum=2, exclude_major=True) if output else None
                )
                if row["status"] != "completed" or not output or row["seq"] in refused:
                    group = "unreviewable"
                elif screen["label"] == "eligible":
                    group = "keep"
                else:
                    group = "review"
                groups[group] += 1
                candidates[group] += metadata[row["seq"]]["candidate_count"]
                record = {
                    **metadata[row["seq"]],
                    "job_id": job_id,
                    "group": group,
                    "status": row["status"],
                    "error": row["error"],
                    "result": output,
                    "screening": screen,
                    "suspected_refusal": row["seq"] in refused,
                    "rank": rank_by_seq.get(row["seq"], {}).get("rank"),
                    "diagnostic_mean": rank_by_seq.get(row["seq"], {}).get(
                        "mean_score"
                    ),
                    "rubric_hash": job["rubric_hash"],
                    "data_hash": job["data_hash"],
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                all_scores.write(line)
                if group == "keep":
                    packed = source.execute(
                        "SELECT payload_zlib FROM conversations WHERE seq=?",
                        (row["seq"],),
                    ).fetchone()[0]
                    conversation = json.loads(zlib.decompress(packed))
                    record["conversation"] = conversation
                    record["candidate_message_ids"] = [
                        m["candidate_message_id"]
                        for m in conversation["messages"]
                        if m["scope"] == "candidate"
                    ]
                    keep.write(json.dumps(record, ensure_ascii=False) + "\n")
                elif group == "review":
                    review.write(line)
                else:
                    unreviewable.write(line)
    if sum(groups.values()) != job["total"]:
        raise ValueError("导出分组总数与任务不一致")
    summary = {
        "job_id": job_id,
        "job_updated": job["updated"],
        "job_status": job["status"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total": job["total"],
        "counts": counts,
        "groups": dict(groups),
        "candidate_counts": dict(candidates),
        "spent_usd": job["spent"],
        "unknown_cost_attempts": job["unknown_cost_attempts"],
        "ranked_count": report["ranked_count"],
        "unranked_count": report["excluded_count"],
        "suspected_refusal_count": report["suspected_refusal_count"],
        "score_counts": report["score_counts"],
        "mean_score_range": report["mean_score_range"],
        "rubric_hash": job["rubric_hash"],
        "data_hash": job["data_hash"],
        "output_directory": str(destination),
        "ranking_method": report["method"],
        "note": "keep 是机器建议保留，不是逐条人工验收；N/A 不计 0；保留原始数据，待复核与无法评审分开。",
    }
    (stage / "ranking.csv").write_text(csv_text(report), encoding="utf-8", newline="")
    (stage / "ranking.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )
    (stage / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (stage / "README.md").write_text(
        "# 全量会话筛选结果\n\n" + summary["note"] + "\n\n"
        f"任务 `{job_id}`，状态 `{job['status']}`。共 {job['total']} 段。\n\n"
        f"分组：{json.dumps(dict(groups), ensure_ascii=False)}。\n\n"
        "- keep.conversations.jsonl.gz：建议保留；完整原会话、候选消息 ID、20 项评分和依据。\n"
        "- review.index.jsonl.gz：存在低分、重大问题或依据警告，需要复核；含会话 ID 和评分。\n"
        "- unreviewable.index.jsonl.gz：调用失败、无有效结果、取消或疑似拒评；不能当质量 0 分。\n"
        "- all_scores.jsonl.gz：全部任务条目的状态、评分、分组与关联标识。\n"
        "- ranking.csv / ranking.json：均分排名及全部分项、N/A、低分数、复核提示。\n"
        "- SUMMARY.json：总数、分组、费用、版本及数据哈希。\n\n"
        "建议保留条件沿用页面：每个数字项至少 2，无 major 问题，无 warnings/data_limitations，非疑似拒评。"
        "N/A 不扣分。此口径较保守；资料限制可能为模型误报，review 不等于垃圾数据。"
        "所有 keep 条目仍需理解为会话级筛选，不是每条 candidate 的独立通过证明。"
        "其他分组的原对话仍在只读 conversations-v1-platform.sqlite3，可按 seq/conversation_id 关联。\n\n"
        + report["note"]
        + "\n",
        encoding="utf-8",
    )
    # All outputs are new derived files. Never overwrite an existing finished export.
    stage.rename(destination)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-db", required=True, type=Path)
    parser.add_argument("--sample-db", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    os.umask(0o027)
    while True:
        job, counts = read_job(args.result_db, args.job_id)
        if job["status"] in ("completed", "completed_errors", "cancelled") and not any(
            counts.get(s, 0) for s in ("queued", "running")
        ):
            print(
                json.dumps(
                    export_run(
                        args.result_db, args.sample_db, args.job_id, args.output_root
                    ),
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return
        if not args.wait:
            raise SystemExit("任务未结束；使用 --wait 可只读等待并在终态导出。")
        time.sleep(60)


if __name__ == "__main__":
    main()
