from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


RUN_ROOT = Path(
    "/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816/"
    "02_data_juicer/02_semantic_audit_v1"
)
DASHBOARD_ROOT = RUN_ROOT / "07_dashboard"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def integer(value: Any) -> str:
    return f"{int(value or 0):,}"


def duration(seconds: Any) -> str:
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分钟"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时"
    return f"{seconds / 86400:.2f} 天"


st.set_page_config(page_title="Lusy 语义向量审查", page_icon="🧠", layout="wide")
st.title("Lusy v2.2 · Turn 级语义向量")
st.caption("只读派生产物；不修改冻结数据，不依据 embedding 自动删除样本。")

status = read_json(DASHBOARD_ROOT / "run_status.json") or {}
benchmark = read_json(DASHBOARD_ROOT / "embedding_benchmark.json") or status.get("benchmark") or {}

top = st.columns(5)
top[0].metric("状态", status.get("status", "not started"))
top[1].metric("阶段", status.get("stage", "—"))
top[2].metric("对话", integer(status.get("expected_dialogues")))
top[3].metric("每类 Turn", integer(status.get("expected_turns")))
top[4].metric("更新时间", str(status.get("updated_at", "—"))[:19])

expected_turns = int(status.get("expected_turns") or 0)
derive_turns = int(status.get("derive_turns") or 0)
st.subheader("Turn 视图")
st.progress(min(1.0, derive_turns / expected_turns) if expected_turns else 0.0)
derive_cols = st.columns(3)
derive_cols[0].metric("已派生 Turn", integer(derive_turns))
derive_cols[1].metric("完成分片", f"{int(status.get('derive_completed_shards') or 0)}/256")
derive_cols[2].metric("目标 Turn", integer(expected_turns))

st.subheader("Embedding 进度")
progress = status.get("embedding_progress") or {}
for field, label in [("assistant", "Assistant"), ("user", "Previous User")]:
    item = progress.get(field) or {}
    rows = int(item.get("rows") or 0)
    cols = st.columns([2, 2, 2, 3])
    cols[0].metric(f"{label} 向量", integer(rows))
    cols[1].metric("分片", f"{int(item.get('completed_shards') or 0)}/256")
    cols[2].metric("累计编码时间", duration(item.get("elapsed_seconds")))
    cols[3].progress(min(1.0, rows / expected_turns) if expected_turns else 0.0)

if benchmark:
    st.subheader("真实文本吞吐与 ETA")
    assistant = benchmark.get("assistant") or {}
    user = benchmark.get("user") or {}
    bench_cols = st.columns(5)
    bench_cols[0].metric("模型", benchmark.get("model", "—"))
    bench_cols[1].metric("维度", integer(benchmark.get("dimension")))
    bench_cols[2].metric("Assistant 条/秒", f"{float(assistant.get('records_per_second') or 0):.1f}")
    bench_cols[3].metric("User 条/秒", f"{float(user.get('records_per_second') or 0):.1f}")
    bench_cols[4].metric("保守总 ETA", duration(benchmark.get("estimated_total_seconds_with_25pct_margin")))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "字段": "assistant_text",
                    "样本数": assistant.get("records"),
                    "条/秒": assistant.get("records_per_second"),
                    "token/秒": assistant.get("content_tokens_per_second"),
                    "padding 利用率": assistant.get("padding_efficiency"),
                },
                {
                    "字段": "user_text",
                    "样本数": user.get("records"),
                    "条/秒": user.get("records_per_second"),
                    "token/秒": user.get("content_tokens_per_second"),
                    "padding 利用率": user.get("padding_efficiency"),
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        f"FP16 原始向量预计 {float(benchmark.get('raw_vector_gib_total') or 0):.1f} GiB；"
        "ETA 来自真实 RP turn、共享 GPU 现状和当前 max_length/batch，不包含后续 FAISS 建索引。"
    )
else:
    st.info("完成 bucket-000 的真实文本基准测试后，这里会显示吞吐和全量 ETA。")

with st.expander("版本契约"):
    st.json(
        {
            "input": "independent_v1 exact survivors + frozen v2.2 JSONL",
            "unit": "one accepted user/assistant pair",
            "assistant_vector": "L2-normalized BGE-M3 CLS embedding",
            "user_vector": "L2-normalized BGE-M3 CLS embedding",
            "dtype": "float16",
            "max_length": status.get("embedding_max_length", 512),
            "deletion": "none",
            "next_stage": "within-dialog cosine + global FAISS candidate recall",
        }
    )

if st.button("刷新"):
    st.rerun()

