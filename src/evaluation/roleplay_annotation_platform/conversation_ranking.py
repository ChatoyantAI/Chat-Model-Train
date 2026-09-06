"""Transparent, read-only aggregation of conversation scores; never a Gate."""

from collections import Counter
from fractions import Fraction
import csv
import io
import json
import re
import statistics

from .conversation_filter import ITEMS

METHOD = "conversation-numeric-mean-v1"
NOTE = (
    "诊断排名：适用数字项等权平均（0～3），N/A 不折成 0；同均分并列。"
    "不同适用项目组合不完全可比；低分项、重大问题和待复核标记必须一起看。"
    "不改原始分数、不自动删除数据，不表示逐轮通过。疑似拒评仅单列供人工核对，不重试绕过。"
)
# This only flags wording in an existing *judge answer*. It is not a safety
# classifier for source conversations, and never affects dispatch or billing.
REFUSAL = re.compile(
    r"(?:无法|不能|不予|拒绝|不进行|不提供).{0,20}(?:评估|评审|评价|评分)"
    r"|不可评价|不计分|拒评"
    r"|(?:cannot|unable to|won.t|will not|refuse to).{0,40}(?:evaluat|assess|scor|rating)",
    re.I,
)


def full_refusal(output):
    """Do not mistake a local N/A explanation or a context-only note for refusal.

    Whole-answer refusal needs corroboration across score reasons. A single
    uncertain field or an excluded background turn is not such evidence.
    """
    scores = (output or {}).get("scores") or {}
    reasons = [
        str(row.get("reason", "")).strip()
        for row in scores.values()
        if isinstance(row, dict)
    ]
    if not reasons:
        return False
    matched = sum(bool(REFUSAL.search(reason)) for reason in reasons)
    referred = sum(
        bool(
            re.fullmatch(
                r"(?:同上|如上|见上|same as above|as above)[。.!！\s]*", reason, re.I
            )
        )
        for reason in reasons
    )
    return matched >= max(3, len(reasons) / 2) or (
        matched >= 1 and matched + referred >= max(3, len(reasons) / 2)
    )


def score_summary(output):
    output = output or {}
    supplied = output.get("scores") or {}
    values = {item["id"]: supplied.get(item["id"], {}).get("value") for item in ITEMS}
    numeric = [v for v in values.values() if type(v) is int and 0 <= v <= 3]
    counts = Counter(v for v in values.values() if type(v) is int and 0 <= v <= 3)
    na = [k for k, v in values.items() if v == "N/A"]
    notes = []
    if output.get("warnings"):
        notes.append("引用或结构警告")
    if output.get("data_limitations"):
        notes.append("模型报告资料限制，需核对是否真实缺失")
    if any(i.get("severity") == "major" for i in output.get("issues", [])):
        notes.append("模型报告重大问题")
    suspected = full_refusal(output)
    if suspected:
        notes.append("疑似拒评措辞；分数可能是占位，暂不排名，需人工核对")
    complete = len(numeric) + len(na) == len(ITEMS)
    if not complete:
        notes.append("评分项缺失或非法")
    return {
        "scores": values,
        "numeric_count": len(numeric),
        "na_count": len(na),
        "applicability": ",".join(k for k, v in values.items() if type(v) is int),
        "mean_score": sum(numeric) / len(numeric) if numeric else None,
        "score_sum": sum(numeric),
        "min_score": min(numeric) if numeric else None,
        "zero_count": counts[0],
        "one_count": counts[1],
        "two_count": counts[2],
        "three_count": counts[3],
        "low_count": counts[0] + counts[1],
        "review_notes": notes,
        "suspected_refusal": suspected,
        "complete_scores": complete,
    }


def rank_records(records):
    ranked, excluded = [], []
    states = Counter()
    per_item = {i["id"]: {"name": i["name"], "counts": Counter()} for i in ITEMS}
    for record in records:
        row = dict(record)
        states[row["status"]] += 1
        saved = row.pop("output_json", None)
        try:
            output = json.loads(saved) if isinstance(saved, str) else saved
        except (ValueError, TypeError):
            output = None
        row.update(score_summary(output))
        row["rank"] = None
        if (
            row["status"] != "completed"
            or not row["numeric_count"]
            or not row["complete_scores"]
            or row["suspected_refusal"]
        ):
            row["exclusion_reason"] = (
                "未完成有效数字评分"
                if row["status"] != "completed"
                or not row["complete_scores"]
                or not row["numeric_count"]
                else "疑似拒评，占位分不参与排名"
            )
            excluded.append(row)
        else:
            ranked.append(row)
    ranked.sort(key=lambda r: (-Fraction(r["score_sum"], r["numeric_count"]), r["seq"]))
    previous = None
    for index, row in enumerate(ranked, 1):
        value = Fraction(row["score_sum"], row["numeric_count"])
        row["rank"] = previous[1] if previous and previous[0] == value else index
        previous = value, row["rank"]
        for item, score in row["scores"].items():
            per_item[item]["counts"][str(score)] += 1
    means = [r["mean_score"] for r in ranked]
    counts = {
        str(v): sum(
            r[{0: "zero_count", 1: "one_count", 2: "two_count", 3: "three_count"}[v]]
            for r in ranked
        )
        for v in range(4)
    }
    counts["N/A"] = sum(r["na_count"] for r in ranked)
    return {
        "method": METHOD,
        "note": NOTE,
        "counts": dict(states),
        "total": sum(states.values()),
        "ranked_count": len(ranked),
        "excluded_count": len(excluded),
        "suspected_refusal_count": sum(r["suspected_refusal"] for r in excluded),
        "review_flagged_ranked_count": sum(bool(r["review_notes"]) for r in ranked),
        "score_counts": counts,
        "per_item": per_item,
        "mean_score_range": [min(means), max(means)] if means else None,
        "mean_score_median": statistics.median(means) if means else None,
        "distinct_means": len(
            {Fraction(r["score_sum"], r["numeric_count"]) for r in ranked}
        ),
        "rows": ranked,
        "excluded": sorted(excluded, key=lambda r: r["seq"]),
    }


def csv_text(report):
    buffer = io.StringIO(newline="")
    fields = [
        "rank",
        "seq",
        "conversation_id",
        "character_name",
        "language",
        "candidate_count",
        "status",
        "mean_score",
        "numeric_count",
        "na_count",
        "min_score",
        "zero_count",
        "one_count",
        "two_count",
        "three_count",
        "low_count",
        "review_notes",
        "exclusion_reason",
        "applicability",
    ] + [i["id"] for i in ITEMS]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in report["rows"] + report["excluded"]:
        flat = {**row, **row["scores"], "review_notes": "；".join(row["review_notes"])}
        # Safe to open as a spreadsheet even if role names are attacker controlled.
        for key, value in flat.items():
            if isinstance(value, str) and value.lstrip().startswith(
                ("=", "+", "-", "@")
            ):
                flat[key] = "'" + value
        writer.writerow(flat)
    return "\ufeff" + buffer.getvalue()
