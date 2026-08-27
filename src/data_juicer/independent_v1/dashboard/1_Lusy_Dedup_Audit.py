from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


FULL_ROOT = Path(
    "/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816/"
    "02_data_juicer/01_independent_v1"
)
FULL_DASHBOARD = FULL_ROOT / "07_dashboard"
SMOKE_ROOT = Path(
    "/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816/"
    "02_data_juicer/00_smoke_dedup_v1/dashboard"
)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl(path: Path, limit: int = 1_000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
                if len(records) >= limit:
                    break
    except (OSError, json.JSONDecodeError):
        return []
    return records


def integer(value: Any) -> str:
    return "—" if value is None else f"{int(value):,}"


def percentage(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def elapsed(started_at: str | None, finished_at: str | None = None) -> str:
    if not started_at:
        return "—"
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at) if finished_at else datetime.now(timezone.utc)
        seconds = max(0, int((end - start).total_seconds()))
    except ValueError:
        return "—"
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, _ = divmod(remainder, 60)
    if days:
        return f"{days}天 {hours}小时"
    if hours:
        return f"{hours}小时 {minutes}分"
    return f"{minutes}分钟"


def flatten_pair(record: dict[str, Any]) -> dict[str, Any]:
    left = record.get("dup1") or {}
    right = record.get("dup2") or {}
    labels = {"exact": "精确", "minhash": "MinHash", "simhash": "SimHash"}
    return {
        "算子": labels.get(record.get("operator"), record.get("operator")),
        "对话 A": left.get("conversation_id"),
        "对话 B": right.get("conversation_id"),
        "A bucket": left.get("conversation_bucket"),
        "B bucket": right.get("conversation_bucket"),
        "A 轮数": left.get("turn_count"),
        "B 轮数": right.get("turn_count"),
        "A 语言": left.get("output_language"),
        "B 语言": right.get("output_language"),
        "A 预览": left.get("dialogue_preview"),
        "B 预览": right.get("dialogue_preview"),
    }


st.set_page_config(page_title="Lusy 全量去重与复读审计", page_icon="🧪", layout="wide")
st.title("Lusy RP · Data-Juicer 全量独立初筛")
st.caption("Data-Juicer v1.5.5 · CPU 后台运行 · 原始 01_constructed 保持只读")


@st.fragment(run_every=30)
def render_dashboard() -> None:
    status = read_json(FULL_DASHBOARD / "run_status.json")
    summary = read_json(FULL_DASHBOARD / "summary.json")

    if not status:
        st.info("全量任务状态文件尚未出现。页面每 30 秒自动刷新。")
        return

    state_labels = {
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败",
    }
    stage_labels = {
        "derive_views": "派生视图",
        "exact": "精确去重",
        "minhash": "MinHash",
        "simhash": "SimHash",
        "repetition": "复读统计",
        "summarize": "生成汇总",
        "complete": "全部完成",
    }
    progress = status.get("progress") or {}
    completed_buckets = int(progress.get("buckets_completed") or 0)
    total_buckets = int(progress.get("buckets_total") or 256)

    top1, top2, top3, top4, top5 = st.columns(5)
    top1.metric("任务状态", state_labels.get(status.get("state"), status.get("state", "—")))
    top2.metric("当前阶段", stage_labels.get(status.get("current_stage"), status.get("current_stage", "—")))
    top3.metric("派生 bucket", f"{completed_buckets} / {total_buckets}")
    top4.metric("已读取对话", integer(progress.get("records_completed")))
    top5.metric("已运行", elapsed(status.get("started_at"), status.get("finished_at")))

    st.progress(
        min(1.0, completed_buckets / total_buckets) if total_buckets else 0.0,
        text="bucket 级派生进度；进入 Data-Juicer 全局算子后，下方阶段表会更新",
    )

    if status.get("state") == "failed":
        st.error(f"后台任务失败：{status.get('last_error')}")
    else:
        st.warning(
            "MinHash、SimHash 的输出是独立候选分支，不是最终删除名单；本轮忽略 fork 关系，"
            "每条完整对话都按独立样本参与全局比较。"
        )

    tab_progress, tab_results, tab_pairs, tab_repetition, tab_contract, tab_smoke = st.tabs(
        ["运行进度", "算子结果", "重复对证据", "单对话复读", "口径与参数", "旧 smoke 归档"]
    )

    with tab_progress:
        stage_rows = []
        for key in ["derive_views", "exact", "minhash", "simhash", "repetition", "summarize"]:
            stage = (status.get("stages") or {}).get(key) or {}
            stage_rows.append(
                {
                    "阶段": stage.get("label") or stage_labels.get(key, key),
                    "状态": {"pending": "等待", "running": "运行中", "completed": "完成", "failed": "失败", "skipped": "跳过"}.get(
                        stage.get("status"), stage.get("status", "等待")
                    ),
                    "输入": stage.get("input_records"),
                    "输出/已处理": stage.get("output_records"),
                    "bucket": (
                        f"{stage.get('buckets_completed', 0)} / {stage.get('buckets_total', 256)}"
                        if key == "derive_views"
                        else "—"
                    ),
                    "耗时": elapsed(stage.get("started_at"), stage.get("finished_at")),
                    "说明": stage.get("message"),
                }
            )
        st.dataframe(pd.DataFrame(stage_rows), width="stretch", hide_index=True)
        st.caption(f"最后状态更新时间（UTC）：{status.get('updated_at', '—')}。页面每 30 秒自动刷新。")

        profile = (summary or {}).get("profile") or {}
        counts = profile.get("counts") or {}
        if counts:
            st.subheader("原始独立样本画像")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("完整对话", integer(counts.get("records")))
            c2.metric("user→assistant turns", integer(counts.get("turns")))
            c3.metric("Greeting 与后续回复完全相同", integer(counts.get("greeting_equals_later_reply")))
            c4.metric("存在 assistant 精确复读的对话", integer(counts.get("rows_with_assistant_exact_repeat")))

    with tab_results:
        if not summary:
            st.info("第一个 bucket 阶段结束后会生成汇总；全局算子完成一个，就会多显示一行结果。")
        else:
            operators = summary.get("operators") or {}
            labels = {
                "exact": "完整 messages 精确去重",
                "minhash": "多语言 MinHash 近似去重",
                "simhash": "字符 SimHash 模板去重",
            }
            rows = []
            for key in ["exact", "minhash", "simhash"]:
                item = operators.get(key) or {}
                rows.append(
                    {
                        "分支": labels[key],
                        "比较输入": item.get("input"),
                        "保留": item.get("retained"),
                        "候选移除": item.get("removed"),
                        "候选移除率": item.get("removed_ratio"),
                        "短文本旁路": item.get("short_passthrough"),
                        "证据对": item.get("trace_pairs"),
                        "文本视图": item.get("view"),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
                column_config={"候选移除率": st.column_config.NumberColumn(format="percent")},
            )
            st.caption(
                "流程是：全量派生 → 精确去重；MinHash 与 SimHash 都从精确去重后的同一输入独立运行，"
                "不会让一种模糊算法先删除另一种算法要比较的样本。"
            )

    with tab_pairs:
        pairs = read_jsonl(FULL_DASHBOARD / "duplicate_pairs.jsonl", limit=1_000)
        if not pairs:
            st.info("尚无完成算子的 tracer 证据对。")
        else:
            choices = list(dict.fromkeys(record.get("operator") for record in pairs))
            labels = {"exact": "精确重复", "minhash": "MinHash 近似重复", "simhash": "SimHash 模板相似"}
            choice = st.selectbox("选择算子", choices, format_func=lambda key: labels.get(key, key))
            rows = [flatten_pair(record) for record in pairs if record.get("operator") == choice]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=620)

    with tab_repetition:
        repetition = (summary or {}).get("repetition")
        top_records = read_jsonl(FULL_DASHBOARD / "repetition_top_records.jsonl", limit=250)
        if not repetition:
            st.info("复读算子尚未完成。本轮阈值为 0..1，只打分，不因复读率直接删除。")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("已打分对话", integer(repetition.get("records")))
            c2.metric("字符 10-gram P99", percentage((repetition.get("character") or {}).get("p99")))
            c3.metric("字词 10-gram P99", percentage((repetition.get("word") or {}).get("p99")))
            c4.metric("字符复读率 > 20%", integer((repetition.get("character") or {}).get("over_0_20")))
            if top_records:
                st.dataframe(pd.DataFrame(top_records), width="stretch", hide_index=True, height=620)

    with tab_contract:
        st.json(
            {
                "运行范围": status.get("scope"),
                "样本单位": status.get("record_unit"),
                "原始 JSONL 保持只读": bool(status.get("source_is_untouched")),
                "仅 CPU": status.get("cpu_only"),
                "参数": status.get("parameters"),
                "解释限制": (summary or {}).get("important_interpretation"),
            },
            expanded=True,
        )

    with tab_smoke:
        smoke = read_json(SMOKE_ROOT / "summary.json")
        if smoke:
            st.caption("2026-08-17 的小样本实验，仅用于验证算子和可视化链路；不代表全量重复率。")
            st.json(smoke, expanded=False)
        else:
            st.info("旧 smoke 汇总不存在。")


render_dashboard()
