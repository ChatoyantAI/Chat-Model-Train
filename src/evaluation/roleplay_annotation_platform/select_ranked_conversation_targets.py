"""Select existing coarse-filter targets by frozen conversation ranks, without LLM calls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile
import zlib

METHOD = "conversation-rank-target-prefix-v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def q(p):
        i = (len(values) - 1) * p
        n = int(i)
        return values[n] + (values[min(n + 1, len(values) - 1)] - values[n]) * (i - n)

    return {
        "count": len(values),
        "min": values[0],
        "p25": q(0.25),
        "median": q(0.5),
        "p75": q(0.75),
        "p90": q(0.9),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": values[-1],
        "mean": statistics.mean(values),
    }


def concentration(counter):
    sizes = sorted(counter.values(), reverse=True)
    total = sum(sizes)
    return {
        "groups": len(sizes),
        "rows": total,
        "top_share": {
            str(n): sum(sizes[:n]) / total if total else None
            for n in (1, 5, 10, 50, 100)
        },
        "max_group": max(sizes, default=0),
        "group_size": quantiles(sizes),
    }


def plan_selection(rows, limit, character_cap=None):
    """Mean descending; stable seq tie-break; a target has no independent judge score."""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if character_cap is not None and (
        type(character_cap) is not int or character_cap < 1
    ):
        raise ValueError("character_cap must be a positive integer")
    usable = [
        r
        for r in rows
        if r.get("status") == "completed"
        and not r.get("suspected_refusal")
        and r.get("complete_scores")
        and r.get("numeric_count", 0) > 0
    ]
    if len({r["seq"] for r in usable}) != len(usable):
        raise ValueError("duplicate conversation sequence")
    usable.sort(key=lambda r: (-Fraction(r["score_sum"], r["numeric_count"]), r["seq"]))
    remaining = limit
    plan = []
    character_counts = Counter()
    for row in usable:
        count = row["candidate_count"]
        if type(count) is not int or count < 1:
            raise ValueError("invalid candidate count")
        if character_cap is not None and not row.get("character_id"):
            raise ValueError("character_id is required for capped selection")
        character = row.get("character_id")
        take = min(
            count,
            remaining,
            character_cap - character_counts[character]
            if character_cap is not None
            else count,
        )
        if take <= 0:
            continue
        plan.append((row, take))
        character_counts[character] += take
        remaining -= take
        if not remaining:
            return plan
    raise ValueError(f"Not enough rankable targets: need {limit}, short {remaining}")


class Profile:
    def __init__(self):
        self.counts = {
            k: Counter()
            for k in (
                "language",
                "model",
                "character",
                "user",
                "conversation",
                "media_type",
                "character_nsfw",
            )
        }
        self.lengths = defaultdict(list)
        self.target_hashes = Counter()
        self.pair_hashes = Counter()
        self.checks = Counter()

    def add(self, row, user_key):
        dims = {
            "language": row.get("language"),
            "model": row.get("model"),
            "character": row.get("character_id"),
            "user": user_key,
            "conversation": row["conversation_id"],
            "media_type": row.get("media_type"),
            "character_nsfw": row.get("character_nsfw"),
        }
        for key, value in dims.items():
            self.counts[key]["<missing>" if value is None else str(value)] += 1
        target = row["target_text"]
        query = row["query_text"]
        self.lengths["target_chars"].append(len(target))
        self.lengths["query_chars"].append(len(query))
        self.lengths["target_whitespace_words"].append(len(target.split()))
        self.lengths["turn_n"].append(row["turn_n"])
        self.target_hashes[hashlib.sha256(target.encode()).hexdigest()] += 1
        self.pair_hashes[
            hashlib.sha256(
                json.dumps([query, target], ensure_ascii=False).encode()
            ).hexdigest()
        ] += 1
        self.checks["stored_target_chars_mismatch"] += row.get("target_chars") != len(
            target
        )
        self.checks["stored_query_chars_mismatch"] += row.get("query_chars") != len(
            query
        )

    def summary(self):
        return {
            "concentration": {k: concentration(c) for k, c in self.counts.items()},
            "lengths": {k: quantiles(v) for k, v in self.lengths.items()},
            "exact_duplicates": {
                "target_text_extra_rows": sum(
                    n - 1 for n in self.target_hashes.values()
                ),
                "query_target_extra_rows": sum(
                    n - 1 for n in self.pair_hashes.values()
                ),
            },
            "checks": dict(self.checks),
        }


def export_selection(
    root: Path, job_id: str, limit: int, output: Path, character_cap=None
):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    export = root / "exports" / job_id
    ranking = json.loads((export / "ranking.json").read_text(encoding="utf-8"))
    frozen = json.loads((export / "SUMMARY.json").read_text(encoding="utf-8"))
    if frozen["job_status"] != "completed":
        raise ValueError("Use a completed, frozen job")
    signature = {
        "method": METHOD,
        "job_id": job_id,
        "limit": limit,
        "ranking_sha256": sha256_file(export / "ranking.json"),
        "data_hash": ranking["data_hash"],
        "rubric_hash": ranking["rubric_hash"],
    }
    if character_cap is not None:
        signature.update(
            method="conversation-rank-character-cap-v1", character_cap=character_cap
        )
    if output.exists():
        previous = json.loads((output / "SUMMARY.json").read_text(encoding="utf-8"))
        if previous["selection_signature"] != signature:
            raise ValueError(
                "Output already exists with different inputs; not overwriting"
            )
        return previous
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    index, selected_conversations, expected = [], [], {}
    source_db = root / "data/conversations-v1-platform.sqlite3"
    result_db = root / "results/jobs.sqlite3"
    with (
        closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as samples,
        closing(sqlite3.connect(f"file:{result_db}?mode=ro", uri=True)) as results,
        gzip.open(stage / "contexts.jsonl.gz", "wt", encoding="utf-8") as contexts,
    ):
        samples.row_factory = sqlite3.Row
        meta = {
            r["conversation_id"]: dict(r)
            for r in samples.execute(
                "SELECT seq,conversation_id,user_key,character_id,character_name,candidate_count,turn_count,language FROM conversations"
            )
        }
        planning_rows = [
            {**r, "character_id": meta[r["conversation_id"]]["character_id"]}
            for r in ranking["rows"]
        ]
        plan = plan_selection(planning_rows, limit, character_cap=character_cap)
        manifest = json.loads(
            samples.execute(
                "SELECT value FROM metadata WHERE key='manifest'"
            ).fetchone()[0]
        )
        assert manifest["dataset_sha256"] == ranking["data_hash"]
        source = Path(manifest["source_file"])
        assert sha256_file(source) == manifest["source_sha256"], (
            "Coarse-filter source changed"
        )
        job = results.execute(
            "SELECT status,updated FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        assert tuple(job) == ("completed", frozen["job_updated"]), (
            "Job changed after frozen export"
        )
        for position, (row, take) in enumerate(plan, 1):
            packet = json.loads(
                zlib.decompress(
                    samples.execute(
                        "SELECT payload_zlib FROM conversations WHERE seq=?",
                        (row["seq"],),
                    ).fetchone()[0]
                )
            )
            candidates = [
                m
                for m in packet["messages"]
                if m["role"] == "assistant" and m["scope"] == "candidate"
            ]
            assert len(candidates) == row["candidate_count"]
            selected_ids = []
            result = json.loads(
                results.execute(
                    "SELECT output_json FROM job_items WHERE job_id=? AND seq=?",
                    (job_id, row["seq"]),
                ).fetchone()[0]
            )
            major = any(i.get("severity") == "major" for i in result.get("issues", []))
            for m in candidates[:take]:
                message_id = m["candidate_message_id"]
                assert message_id not in expected, "Duplicate target ID"
                entry = {
                    "selection_order": len(index) + 1,
                    "message_id": message_id,
                    "conversation_id": packet["conversation_id"],
                    "conversation_seq": row["seq"],
                    "conversation_rank": row["rank"],
                    "conversation_selection_order": position,
                    "conversation_mean_score": row["mean_score"],
                    "conversation_score_sum": row["score_sum"],
                    "conversation_numeric_items": row["numeric_count"],
                    "conversation_na_items": row["na_count"],
                    "conversation_low_items": row["low_count"],
                    "conversation_major_issue": major,
                    "conversation_review_notes": row["review_notes"],
                    "conversation_scores": row["scores"],
                    "target_independently_scored": False,
                    "assistant_turn_id": m["id"],
                    "target_increase_id": m["source_increase_id"],
                    "character_id": packet["character_id"],
                    "character_name": packet["character_name"],
                    "user_key": packet["user_key"],
                    "language": m.get("generation_language", packet["language"]),
                    "model": m.get("source_model"),
                    "target_chars": len(m["content"]),
                }
                expected[message_id] = {
                    "index": len(index),
                    "entry": entry,
                    "target_sha256": hashlib.sha256(m["content"].encode()).hexdigest(),
                }
                index.append(entry)
                selected_ids.append(message_id)
            selected_conversations.append(
                {
                    **row,
                    "selection_order": position,
                    "selected_targets": take,
                    "partial_conversation_selection": take < len(candidates),
                    "major_issue": major,
                    "character_id": packet["character_id"],
                    "user_key": packet["user_key"],
                    "data_limitation_count": len(result.get("data_limitations", [])),
                    "warning_count": len(result.get("warnings", [])),
                }
            )
            contexts.write(
                json.dumps(
                    {
                        "conversation_seq": row["seq"],
                        "conversation": packet,
                        "selected_candidate_message_ids": selected_ids,
                        "conversation_result": result,
                        "job_id": job_id,
                        "data_hash": ranking["data_hash"],
                        "rubric_hash": ranking["rubric_hash"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    assert len(index) == limit
    print(
        json.dumps(
            {
                "stage": "selection_planned",
                "targets": len(index),
                "conversations": len(plan),
            }
        ),
        flush=True,
    )
    pool, chosen = Profile(), Profile()
    found = set()
    tables = []
    source_ids = set()
    rankable_cids = {r["conversation_id"] for r in ranking["rows"]}
    rankable_targets = 0
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=4096):
        flags = []
        for row in batch.to_pylist():
            mid = row["message_id"]
            assert mid not in source_ids, "Duplicate ID in source"
            source_ids.add(mid)
            conversation = meta[row["conversation_id"]]
            assert str(row["character_id"]) == str(conversation["character_id"])
            pool.add(row, conversation["user_key"])
            rankable_targets += row["conversation_id"] in rankable_cids
            item = expected.get(mid)
            flags.append(item is not None)
            if item:
                entry = item["entry"]
                assert mid not in found
                assert row["conversation_id"] == entry["conversation_id"]
                assert row["target_increase_id"] == entry["target_increase_id"]
                assert (
                    hashlib.sha256(row["target_text"].encode()).hexdigest()
                    == item["target_sha256"]
                ), "Target changed since evaluation"
                assert row["language"] == entry["language"]
                found.add(mid)
                chosen.add(row, conversation["user_key"])
                entry["model"] = row["model"]
                entry["source_turn_n"] = row["turn_n"]
                entry["query_chars"] = len(row["query_text"])
        if any(flags):
            tables.append(pa.Table.from_batches([batch]).filter(pa.array(flags)))
    assert (
        found == set(expected) and len(source_ids) == manifest["report"]["candidates"]
    )
    if character_cap is not None:
        assert max(chosen.counts["character"].values()) <= character_cap
    table = pa.concat_tables(tables)
    order = pc.index_in(
        pa.array([e["message_id"] for e in index], type=table["message_id"].type),
        value_set=table["message_id"],
    )
    assert order.null_count == 0
    table = table.take(order)
    pq.write_table(table, stage / "targets.parquet", compression="zstd")
    with gzip.open(stage / "targets.jsonl.gz", "wt", encoding="utf-8") as f:
        for batch in table.to_batches(max_chunksize=4096):
            for row in batch.to_pylist():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gzip.open(stage / "target_ranking.jsonl.gz", "wt", encoding="utf-8") as f:
        for e in index:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    names = {r["character_id"]: r["character_name"] for r in meta.values()}
    distributions = {}
    for key, before in pool.counts.items():
        after = chosen.counts[key]
        distributions[key] = [
            {
                "id": k,
                "name": names.get(k, k) if key == "character" else k,
                "pool_targets": before[k],
                "selected_targets": after[k],
                "pool_share": before[k] / len(source_ids),
                "selected_share": after[k] / limit,
                "selection_rate": after[k] / before[k],
            }
            for k in sorted(before, key=lambda k: (-after[k], -before[k], k))
        ]
    cutoff = Fraction(plan[-1][0]["score_sum"], plan[-1][0]["numeric_count"])
    tied = [
        r
        for r in ranking["rows"]
        if Fraction(r["score_sum"], r["numeric_count"]) == cutoff
    ]
    flags = {
        "targets_from_review_flagged_conversations": sum(
            bool(e["conversation_review_notes"]) for e in index
        ),
        "targets_from_low_score_conversations": sum(
            e["conversation_low_items"] > 0 for e in index
        ),
        "targets_from_major_issue_conversations": sum(
            e["conversation_major_issue"] for e in index
        ),
        "targets_from_more_than_10_na_conversations": sum(
            e["conversation_na_items"] > 10 for e in index
        ),
    }
    stats = {
        "pool": pool.summary(),
        "selected": chosen.summary(),
        "distributions": distributions,
        "conversation_means": quantiles(r["mean_score"] for r, _ in plan),
        "target_weighted_conversation_means": quantiles(
            e["conversation_mean_score"] for e in index
        ),
        "conversation_low_items": dict(Counter(r["low_count"] for r, _ in plan)),
        "conversation_na_items": dict(Counter(r["na_count"] for r, _ in plan)),
        "quality_flags": flags,
    }
    stats["per_item_selected_conversations"] = {
        k: dict(Counter(str(r["scores"][k]) for r, _ in plan))
        for k in plan[0][0]["scores"]
    }
    stats["per_item_selected_targets"] = {
        k: {
            v: sum(t for r, t in plan if str(r["scores"][k]) == v)
            for v in ("0", "1", "2", "3", "N/A")
        }
        for k in plan[0][0]["scores"]
    }
    # Independently reread final artifacts: IDs/order/schema/content and context references.
    reread = pq.read_table(stage / "targets.parquet")
    assert reread.num_rows == limit and reread.schema.equals(parquet.schema_arrow)
    assert reread.equals(table) and reread["message_id"].to_pylist() == [
        e["message_id"] for e in index
    ]
    with gzip.open(stage / "targets.jsonl.gz", "rb") as f:
        json_rows = [json.loads(line) for line in f]
    assert json_rows == reread.to_pylist()
    context_ids = []
    with gzip.open(stage / "contexts.jsonl.gz", "rb") as f:
        for line in f:
            record = json.loads(line)
            assert set(record["selected_candidate_message_ids"]) <= {
                m.get("candidate_message_id")
                for m in record["conversation"]["messages"]
                if m["scope"] == "candidate"
            }
            context_ids.extend(record["selected_candidate_message_ids"])
    assert context_ids == [e["message_id"] for e in index]
    for key, rows in distributions.items():
        assert sum(r["pool_targets"] for r in rows) == len(source_ids)
        assert sum(r["selected_targets"] for r in rows) == limit
    summary = {
        "selection_signature": signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_targets": limit,
        "selected_conversations": len(plan),
        "selected_characters": len(chosen.counts["character"]),
        "selected_users": len(chosen.counts["user"]),
        "source_targets": len(source_ids),
        "rankable_source_targets": rankable_targets,
        "source_conversations": len(meta),
        "rankable_conversations": ranking["ranked_count"],
        "excluded_conversations": ranking["excluded_count"],
        "cutoff_mean": float(cutoff),
        "cutoff_exact": str(cutoff),
        "cutoff_tie_conversations": len(tied),
        "cutoff_tie_pool_targets": sum(r["candidate_count"] for r in tied),
        "cutoff_tie_selected_targets": sum(
            t for r, t in plan if Fraction(r["score_sum"], r["numeric_count"]) == cutoff
        ),
        "last_conversation_seq": plan[-1][0]["seq"],
        "last_conversation_taken": plan[-1][1],
        "last_conversation_available": plan[-1][0]["candidate_count"],
        "quality_flags": flags,
        "source_file": str(source),
        "source_sha256": manifest["source_sha256"],
        "original_source_columns": parquet.schema_arrow.names,
        "pyarrow_version": pa.__version__,
        "output_directory": str(output),
        "paid_requests": 0,
        "verification": "passed",
    }
    if character_cap is not None:
        summary["character_cap"] = character_cap
    dump(stage / "SUMMARY.json", summary)
    dump(stage / "statistics.json", stats)
    dump(stage / "selected_conversations.json", selected_conversations)
    cap_note = (
        f"同一 character_id 跨用户、跨语言合计最多 {character_cap} 条 target；达到上限后跳过，继续从其他角色中按分数补足。没有用户配额。"
        if character_cap is not None
        else "不使用角色/用户配额。"
    )
    desc = f"""# 按会话评分选出的前 {limit:,} 条 target

共 {len(plan):,} 段会话、{len(chosen.counts["character"]):,} 个角色、{len(chosen.counts["user"]):,} 位用户；截断会话均分 {float(cutoff):.6f}/3（精确值 {cutoff}）。未调用模型，未改原始文件。

## 排序和截断

使用任务 {job_id} 冻结排名：适用数字项等权均分降序，N/A 不按 0。排除未完成、无有效数字结果、疑似整段拒评。其余警告、低分、重大问题保留为标记，不作为本次 Top-N 的一票否决，也不强行用之前 keep=480 段的条件凑数。

同均分沿用会话 seq 升序。{cap_note} 同一会话内按原始回合顺序取粗筛候选。截断分数并列有 {len(tied)} 段会话，本次从该同分组取 {summary["cutoff_tie_selected_targets"]} 条。最后 # {summary["last_conversation_seq"]} 取 {summary["last_conversation_taken"]} / {summary["last_conversation_available"]} 条；角色上限也可能导致中间会话只选部分 target。未声称同分先取者质量更好。精确取满 {limit:,} 条唯一 message_id。

## 文件格式

- targets.parquet：原始粗筛文件的选中行，列名、类型、正文完全保留，按 selection_order 排列。
- targets.jsonl.gz：同一批原始行的 JSONL，gzip 压缩，UTF-8，每行一条 target。
- target_ranking.jsonl.gz：每条 target 的顺序、会话分数/分项、来源、用户散列、风险标记与回合关联；没有单条独立评分。
- contexts.jsonl.gz：每个入选会话保存一次冻结上下文、整套评分和入选 target ID；不要把未入选背景当作训练 target。
- selected_conversations.json：所有入选会话的排名和入选条数。
- statistics.json：全粗筛池与入选集的角色、用户、语言、模型、长度、集中度和精确重复对比。
- SUMMARY.json / FILES.json：版本、哈希、截断规则、文件大小与校验和。

## 使用限制

分数是候选回复组的会话评分，不是单条 target 的独立分数。{flags["targets_from_low_score_conversations"]:,} 条来自有低分项的会话，{flags["targets_from_major_issue_conversations"]:,} 条来自有 major 标记的会话；Top-N 不等于逐条合格证明。高 N/A 会话的均分可能偏高，本次不偷偷新增惩罚，明确保留适用项数和标记。

未做训练/验证集切分，未自动去重。角色/用户分布处理仅限上述配额规则；后续切分须避免同用户/同会话泄漏。长度按 Unicode 字符重算，不依赖旧 target_chars 字段；空格分词不是模型 Token 数，也不适合跨语言直接比较。统计里的模型是 target 来源模型，不是 Judge 渠道。角色名为原始资料，仅供标识。

数据口径：对话为完整离线快照至最后候选位置；角色/Greeting 按既有平台方式拼接，不保证历史原请求逐字一致。source 中 query_text 保留原样；本导出不重新改写用户文本或拼接训练 prompt。
"""
    (stage / "README.md").write_text(desc, encoding="utf-8")
    files = [
        {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(stage.iterdir())
        if p.is_file()
    ]
    dump(stage / "FILES.json", files)
    stage.rename(output)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--character-cap", type=int)
    args = ap.parse_args()
    import os

    os.umask(0o027)
    print(
        json.dumps(
            export_selection(
                args.root, args.job_id, args.limit, args.output, args.character_cap
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
