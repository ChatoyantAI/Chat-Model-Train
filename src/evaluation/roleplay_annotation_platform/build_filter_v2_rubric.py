"""Build the standalone probabilistic single-sample rubric from rubric.json.

The V2 service owns its filtering protocol and hard gates, while the scored
dimensions deliberately reuse the latest reviewed role-play scale.  Keeping
this as a deterministic build step prevents the web copy from silently
drifting away from the main rubric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = MODULE_DIR / "rubric.json"
TARGET_PATH = MODULE_DIR / "single_sample_filter_rubric_v2.json"
PROTOCOL_VERSION = "roleplay-single-sample-filter-v2.2"


ITEM_FIELDS = (
    "id",
    "name",
    "starred",
    "scale_label",
    "scale",
    "allows_na",
    "anchors",
    "intro",
    "decision_steps",
    "decision_notes",
    "not_counted",
    "na_condition",
    "cases",
    "reference_markdown",
)


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build() -> dict[str, Any]:
    source = _read(SOURCE_PATH)
    previous = _read(TARGET_PATH)
    gates = previous.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("the existing V2 rubric must contain its hard gates")

    dimensions: list[dict[str, Any]] = []
    for dimension in source.get("dimensions", []):
        items: list[dict[str, Any]] = []
        for item in dimension.get("items", []):
            record = {field: item[field] for field in ITEM_FIELDS if field in item}
            items.append(record)
        dimensions.append(
            {
                "id": dimension["id"],
                "label": dimension.get("label", ""),
                "name": dimension["name"],
                "description": dimension.get("description", ""),
                "items": items,
            }
        )

    return {
        "version": PROTOCOL_VERSION,
        "source_rubric_version": source.get("version"),
        "source_rubric_sha256": source.get("source_sha256"),
        "purpose": (
            "使用逐项分数及评分 Token 概率分布衡量单条角色扮演回复，"
            "用于质量排序、过滤和争议复核；不比较其他候选。"
        ),
        "gate_semantics": {
            "true": "明确发现该 Gate 所描述的不可用错误",
            "false": "没有发现该 Gate 错误",
        },
        "score_semantics": {
            "0": "明显失败或与要求相反",
            "1": "已经涉及该能力，但存在明确缺陷",
            "2": "完整、自然、达到高质量训练样本要求",
            "3": "先达到 2 分，并且该能力本身成为本轮突出的质量优点",
            "N/A": "只有该原子项明确写出的不适用条件成立时使用",
        },
        "score_codes": {
            "A": "0",
            "B": "1",
            "C": "2",
            "D": "3",
            "X": "N/A",
        },
        "aggregation": {
            "item": "评分 Token 合法档位分布的条件期望；N/A 概率单独表示适用性",
            "group": "组内适用项目按各自量表最大值归一化后等权平均",
            "overall": "有适用项目的 H/C/M/L/N 五个大类等权平均，再映射到 0-100",
        },
        "gates": gates,
        "dimensions": dimensions,
    }


def main() -> None:
    payload = build()
    TARGET_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
