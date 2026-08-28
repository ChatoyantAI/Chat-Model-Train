from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DIMENSION_IDS = {
    "维度一": "intimacy",
    "维度二": "character_consistency",
    "维度三": "context_memory",
    "维度四": "language",
    "维度五": "narrative",
}

SAFETY_VALUES = {
    "G-S1": {"是": "yes", "否": "no", "不清楚": "unclear"},
    "G-S2": {"通过": "pass", "失败": "fail", "N/A": "N/A"},
    "G-S3": {"是": "yes", "否": "no", "N/A": "N/A"},
}

SAFETY_DEFAULTS = {"G-S1": "no", "G-S2": "N/A", "G-S3": "no"}


def split_row(line: str) -> list[str]:
    return [part.strip() for part in line.strip().strip("|").split("|")]


def find_line(lines: list[str], value: str, start: int = 0) -> int:
    try:
        return lines.index(value, start)
    except ValueError as error:
        raise ValueError(f"missing required heading: {value}") from error


def first_matching_line(lines: list[str], pattern: str, start: int, end: int) -> int:
    compiled = re.compile(pattern)
    for index in range(start, end):
        if compiled.match(lines[index]):
            return index
    raise ValueError(f"missing required content matching {pattern!r}")


def parse_table(lines: list[str], start: int) -> tuple[list[str], list[list[str]], int]:
    if start >= len(lines) or not lines[start].startswith("|"):
        raise ValueError(f"expected Markdown table at line {start + 1}")
    headers = split_row(lines[start])
    if start + 1 >= len(lines) or not lines[start + 1].startswith("|"):
        raise ValueError(f"missing Markdown table separator at line {start + 2}")
    rows: list[list[str]] = []
    index = start + 2
    while index < len(lines) and lines[index].startswith("|"):
        row = split_row(lines[index])
        if len(row) != len(headers):
            raise ValueError(f"table width mismatch at line {index + 1}")
        rows.append(row)
        index += 1
    return headers, rows, index


def exact_block(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end]).strip()


def nonempty_text(lines: list[str], start: int, end: int) -> str:
    return "\n".join(line.strip() for line in lines[start:end] if line.strip()).strip()


def parse_cases(lines: list[str], start: int, end: int) -> list[dict[str, Any]]:
    case_pattern = re.compile(
        r"^\*\*(?P<label>案例[^*]*)\*\*\s*"
        r"(?P<background_label>背景(?:（[^）]+）)?)：(?P<background>.*)$"
    )
    score_pattern = re.compile(r"^\*\*([0-3]) 分([^*]*)\*\*$")
    case_starts = [index for index in range(start, end) if case_pattern.match(lines[index])]
    cases: list[dict[str, Any]] = []

    for position, case_start in enumerate(case_starts):
        case_end = case_starts[position + 1] if position + 1 < len(case_starts) else end
        case_match = case_pattern.match(lines[case_start])
        assert case_match is not None
        example_starts = [
            index
            for index in range(case_start + 1, case_end)
            if score_pattern.match(lines[index])
        ]
        examples: list[dict[str, Any]] = []
        for example_position, example_start in enumerate(example_starts):
            example_end = (
                example_starts[example_position + 1]
                if example_position + 1 < len(example_starts)
                else case_end
            )
            score_match = score_pattern.match(lines[example_start])
            assert score_match is not None
            content_lines = lines[example_start + 1 : example_end]
            quote_lines: list[str] = []
            explanation_lines: list[str] = []
            for line in content_lines:
                if line.startswith(">"):
                    quote_lines.append(line[1:].removeprefix(" "))
                elif line.strip():
                    explanation_lines.append(line.strip())
            while quote_lines and not quote_lines[0]:
                quote_lines.pop(0)
            while quote_lines and not quote_lines[-1]:
                quote_lines.pop()
            examples.append(
                {
                    "score": score_match.group(1),
                    "label": f"{score_match.group(1)} 分{score_match.group(2)}",
                    "reply": "\n".join(quote_lines),
                    "explanation": "\n".join(explanation_lines),
                    "source_markdown": exact_block(lines, example_start, example_end),
                }
            )
        if not examples:
            raise ValueError(f"case at line {case_start + 1} has no score examples")
        cases.append(
            {
                "label": case_match.group("label"),
                "background_label": case_match.group("background_label"),
                "background": case_match.group("background").strip(),
                "examples": examples,
                "source_markdown": exact_block(lines, case_start, case_end),
            }
        )
    return cases


def parse_item_section(
    lines: list[str], item: dict[str, Any], start: int, end: int
) -> None:
    body_start = start + 1
    decision_index = first_matching_line(lines, r"^\*\*判定步骤\*\*$", body_start, end)
    case_index = first_matching_line(lines, r"^\*\*案例", decision_index + 1, end)
    not_counted_index = next(
        (
            index
            for index in range(decision_index + 1, case_index)
            if lines[index].startswith("**不计入**")
        ),
        None,
    )
    na_index = next(
        (
            index
            for index in range(decision_index + 1, case_index)
            if lines[index].startswith("**N/A**")
        ),
        None,
    )
    if na_index is None:
        raise ValueError(f"missing N/A condition for {item['id']}")

    instruction_end = not_counted_index if not_counted_index is not None else na_index
    steps: list[str] = []
    decision_notes: list[str] = []
    for line in lines[decision_index + 1 : instruction_end]:
        step_match = re.match(r"^\d+\.\s+(.*)$", line)
        if step_match:
            steps.append(step_match.group(1))
        elif line.strip():
            decision_notes.append(line.strip())
    if not steps:
        raise ValueError(f"missing decision steps for {item['id']}")

    def labelled_text(label_index: int | None, section_end: int, label: str) -> str:
        if label_index is None:
            return ""
        first = lines[label_index].removeprefix(label).lstrip("：: ")
        rest = [first] if first else []
        rest.extend(line.strip() for line in lines[label_index + 1 : section_end] if line.strip())
        return "\n".join(rest).strip()

    not_counted_end = na_index if not_counted_index is not None else na_index
    item.update(
        {
            "intro": nonempty_text(lines, body_start, decision_index),
            "decision_steps": steps,
            "decision_notes": "\n".join(decision_notes),
            "not_counted": labelled_text(
                not_counted_index, not_counted_end, "**不计入**"
            ),
            "na_condition": labelled_text(na_index, case_index, "**N/A**"),
            "guidance_markdown": exact_block(lines, body_start, case_index),
            "cases": parse_cases(lines, case_index, end),
            "reference_markdown": exact_block(lines, start, end),
        }
    )


def parse_gates(lines: list[str]) -> tuple[str, list[dict[str, Any]], str]:
    gate_start = find_line(lines, "## Gate 层")
    first_group_start = find_line(lines, "### 数据有效性 Gate", gate_start)
    dimension_start = first_matching_line(lines, r"^## 维度一：", gate_start, len(lines))
    gate_intro = nonempty_text(lines, gate_start + 1, first_group_start)
    group_specs = [
        ("数据有效性 Gate", "boolean"),
        ("安全与上线资格 Gate", "enum"),
        ("产品体验失败 Gate", "boolean"),
    ]
    groups: list[dict[str, Any]] = []
    group_heading_indices = [
        find_line(lines, f"### {name}", gate_start) for name, _ in group_specs
    ]

    for position, ((name, gate_type), heading_index) in enumerate(
        zip(group_specs, group_heading_indices)
    ):
        section_end = (
            group_heading_indices[position + 1]
            if position + 1 < len(group_heading_indices)
            else dimension_start
        )
        table_start = first_matching_line(lines, r"^\|", heading_index + 1, section_end)
        headers, rows, table_end = parse_table(lines, table_start)
        items: list[dict[str, Any]] = []
        for row in rows:
            gate_id = row[0]
            if gate_type == "enum":
                value_map = SAFETY_VALUES.get(gate_id)
                if value_map is None:
                    raise ValueError(f"missing enum mapping for {gate_id}")
                labels = [part.strip() for part in row[2].split(" / ")]
                items.append(
                    {
                        "id": gate_id,
                        "question": row[1],
                        "type": "enum",
                        "options": [
                            {"value": value_map[label], "label": label} for label in labels
                        ],
                        "default": SAFETY_DEFAULTS[gate_id],
                        "result_text": row[2],
                    }
                )
            elif name == "数据有效性 Gate":
                items.append(
                    {
                        "id": gate_id,
                        "question": row[1],
                        "type": "boolean",
                        "default": False,
                        "yes_result": row[2],
                        "no_result": row[3],
                    }
                )
            else:
                items.append(
                    {
                        "id": gate_id,
                        "question": row[1],
                        "type": "boolean",
                        "default": False,
                        "yes_condition": row[2],
                    }
                )
        groups.append(
            {
                "group": name,
                "headers": headers,
                "items": items,
                "source_markdown": exact_block(lines, heading_index, table_end),
            }
        )

    last_table_start = first_matching_line(
        lines, r"^\|", group_heading_indices[-1] + 1, dimension_start
    )
    _, _, last_table_end = parse_table(lines, last_table_start)
    gate_case = nonempty_text(lines, last_table_end, dimension_start)
    return gate_intro, groups, gate_case


def build_rubric(markdown_path: Path) -> dict[str, Any]:
    source_bytes = markdown_path.read_bytes()
    source_markdown = source_bytes.decode("utf-8").replace("\r\n", "\n")
    lines = source_markdown.splitlines()

    score_heading = find_line(lines, "## 原子项统一评分")
    first_table_start = first_matching_line(lines, r"^\|", score_heading + 1, len(lines))
    score_headers, score_rows, first_table_end = parse_table(lines, first_table_start)
    second_table_start = first_matching_line(lines, r"^\|", first_table_end, len(lines))
    usage_headers, usage_rows, second_table_end = parse_table(lines, second_table_start)
    rules_label = find_line(lines, "规则：", second_table_end)
    gate_heading = find_line(lines, "## Gate 层", rules_label)
    rules = [
        match.group(1)
        for line in lines[rules_label + 1 : gate_heading]
        if (match := re.match(r"^\d+\.\s+(.*)$", line))
    ]
    last_rule_index = max(
        index
        for index in range(rules_label + 1, gate_heading)
        if re.match(r"^\d+\.\s+", lines[index])
    )
    case_note = nonempty_text(lines, last_rule_index + 1, gate_heading)
    gate_intro, gates, gate_case = parse_gates(lines)

    dimensions: list[dict[str, Any]] = []
    dimension_starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^## 维度[一二三四五]：", line)
    ]
    item_lookup: dict[str, dict[str, Any]] = {}
    for dimension_position, dimension_start in enumerate(dimension_starts):
        dimension_end = (
            dimension_starts[dimension_position + 1]
            if dimension_position + 1 < len(dimension_starts)
            else len(lines)
        )
        heading_match = re.match(
            r"^## (维度[一二三四五])：(.+)$", lines[dimension_start]
        )
        assert heading_match is not None
        dimension_label, dimension_name = heading_match.groups()
        table_start = first_matching_line(lines, r"^\| ID \|", dimension_start + 1, dimension_end)
        headers, rows, table_end = parse_table(lines, table_start)
        items: list[dict[str, Any]] = []
        for row in rows:
            id_match = re.match(r"^([A-Z]\d+)(?:\s+★)?$", row[0])
            if not id_match:
                raise ValueError(f"invalid item ID: {row[0]}")
            item_id = id_match.group(1)
            scale = [part.strip() for part in row[2].split("/")]
            all_anchors = dict(zip(("0", "1", "2", "3", "N/A"), row[3:8]))
            item = {
                "id": item_id,
                "name": row[1],
                "starred": "★" in row[0],
                "scale_label": row[2],
                "scale": scale,
                "anchors": {score: all_anchors[score] for score in [*scale, "N/A"]},
                "all_anchors": all_anchors,
                "summary_row_markdown": "| " + " | ".join(row) + " |",
            }
            if item_id in item_lookup:
                raise ValueError(f"duplicate item ID: {item_id}")
            item_lookup[item_id] = item
            items.append(item)

        item_starts = [
            index
            for index in range(table_end, dimension_end)
            if re.match(r"^### [A-Z]\d+\s+", lines[index])
        ]
        for item_position, item_start in enumerate(item_starts):
            item_end = (
                item_starts[item_position + 1]
                if item_position + 1 < len(item_starts)
                else dimension_end
            )
            heading_match = re.match(r"^### ([A-Z]\d+)\s+", lines[item_start])
            assert heading_match is not None
            item_id = heading_match.group(1)
            if item_id not in item_lookup:
                raise ValueError(f"unknown item section: {item_id}")
            parse_item_section(lines, item_lookup[item_id], item_start, item_end)

        missing_sections = [item["id"] for item in items if "decision_steps" not in item]
        if missing_sections:
            raise ValueError(f"missing item sections: {missing_sections}")
        dimensions.append(
            {
                "id": DIMENSION_IDS[dimension_label],
                "label": dimension_label,
                "name": dimension_name,
                "description": nonempty_text(lines, dimension_start + 1, table_start),
                "headers": headers,
                "items": items,
                "summary_table_markdown": exact_block(lines, table_start, table_end),
            }
        )

    item_ids = [item["id"] for dimension in dimensions for item in dimension["items"]]
    if len(dimensions) != 5 or len(item_ids) != len(set(item_ids)):
        raise ValueError("expected five dimensions and unique atomic item IDs")
    parsed_example_count = sum(
        len(case["examples"])
        for dimension in dimensions
        for item in dimension["items"]
        for case in item["cases"]
    )
    source_example_count = sum(
        bool(re.match(r"^\*\*[0-3] 分", line)) for line in lines
    )
    if parsed_example_count != source_example_count:
        raise ValueError(
            f"example loss detected: parsed {parsed_example_count}, source {source_example_count}"
        )

    return {
        "version": "roleplay-rubric-v0.2",
        "title": lines[0].removeprefix("# "),
        "intro": nonempty_text(lines, 1, score_heading),
        "source_filename": markdown_path.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_markdown": source_markdown,
        "score_headers": score_headers,
        "score_scale": {row[0]: row[1] for row in score_rows},
        "score_scale_rows": [dict(zip(score_headers, row)) for row in score_rows],
        "scale_usage_headers": usage_headers,
        "scale_usage": [dict(zip(usage_headers, row)) for row in usage_rows],
        "rules": rules,
        "case_note": case_note,
        "gate_intro": gate_intro,
        "gates": gates,
        "gate_case": gate_case,
        "dimensions": dimensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build web rubric JSON from Markdown")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rubric = build_rubric(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rubric, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    item_count = sum(len(dimension["items"]) for dimension in rubric["dimensions"])
    example_count = sum(
        len(case["examples"])
        for dimension in rubric["dimensions"]
        for item in dimension["items"]
        for case in item["cases"]
    )
    print(
        f"wrote {args.output}: {len(rubric['dimensions'])} dimensions, "
        f"{item_count} items, {example_count} examples"
    )


if __name__ == "__main__":
    main()
