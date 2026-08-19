#!/usr/bin/env python3
"""Build a compact fork/branch lineage index from immutable v1 bundles.

The index is descriptive.  It never deletes a branch: exact/subset relations are
recommendations for a later quality-aware dedup stage, because copied histories
can cross user/persona/greeting contexts and forks are not always supersets.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


VERSION = "lusy-fork-lineage-index-v1"
DEFAULT_SOURCE = Path("/home/gyr/lusychat-training/lusy-chat-4m-v1/bundles")


def text(value: Any) -> str:
    return str(value or "").strip()


def normalized_content(value: Any) -> str:
    return text(value).replace("\r\n", "\n").replace("\r", "\n")


def safe_time(value: Any) -> str:
    if value is None:
        return ""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def turn_signature(turn: dict[str, Any]) -> str:
    payload = {
        "user": normalized_content(turn.get("user_content")),
        "user_time": safe_time(turn.get("user_create_time")),
        "assistant": normalized_content(turn.get("assistant_content")),
        "assistant_time": safe_time(turn.get("assistant_create_time")),
    }
    return sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def is_subsequence(shorter: list[str], longer: list[str]) -> bool:
    if not shorter:
        return True
    cursor = iter(longer)
    return all(any(candidate == item for candidate in cursor) for item in shorter)


def common_prefix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def classify_relation(child: list[str], parent: list[str]) -> tuple[str, int, int]:
    prefix = common_prefix_length(child, parent)
    common = len(set(child).intersection(parent))
    if child == parent:
        relation = "exact_copy"
    elif len(child) < len(parent) and child == parent[: len(child)]:
        relation = "child_prefix_of_parent"
    elif len(parent) < len(child) and parent == child[: len(parent)]:
        relation = "parent_prefix_of_child"
    elif len(child) < len(parent) and is_subsequence(child, parent):
        relation = "child_subsequence_of_parent"
    elif len(parent) < len(child) and is_subsequence(parent, child):
        relation = "parent_subsequence_of_child"
    elif common:
        relation = "partial_overlap"
    else:
        relation = "disjoint"
    return relation, prefix, common


def context_fingerprint(row: dict[str, Any]) -> dict[str, str]:
    return {
        "user_id": text(row.get("user_id")),
        "character_id": text(row.get("character_id")),
        "persona_id": text(row.get("persona_id")),
        "greeting_sha256": sha256_text(normalized_content(row.get("initial_greeting"))),
    }


def recommendation(relation: str, compatible: bool) -> str:
    if not compatible:
        return "keep_context_changed_branch"
    if relation in {
        "exact_copy",
        "child_prefix_of_parent",
        "child_subsequence_of_parent",
    }:
        return "collapse_candidate_to_parent_after_quality_gate"
    if relation in {"parent_prefix_of_child", "parent_subsequence_of_child"}:
        return "prefer_child_maximal_branch_after_quality_gate"
    return "keep_diverged_branch"


def source_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("conversation_bucket=*/part-000000.parquet"))
    if len(paths) != 256:
        raise RuntimeError(f"expected 256 source files, found {len(paths)}")
    return paths


def load_fork_edges(paths: list[Path]) -> tuple[dict[str, str], int]:
    edges: dict[str, str] = {}
    rows = 0
    for path in paths:
        # Use ParquetFile directly.  Dataset discovery would also materialize the
        # hive directory key ``conversation_bucket`` and collide with the same
        # physical uint16 column stored in each file.
        table = pq.ParquetFile(path).read(columns=["conversation_id", "parent_id"])
        rows += table.num_rows
        for conversation_id, parent_id in zip(
            table["conversation_id"].to_pylist(), table["parent_id"].to_pylist()
        ):
            cid, parent = text(conversation_id), text(parent_id)
            if parent:
                edges[cid] = parent
    return edges, rows


def resolve_lineage(conversation_id: str, edges: dict[str, str]) -> tuple[str, int, bool]:
    current = conversation_id
    seen: set[str] = set()
    depth = 0
    while current in edges:
        if current in seen:
            return current, depth, True
        seen.add(current)
        current = edges[current]
        depth += 1
    return current, depth, False


def load_relevant_rows(paths: list[Path], relevant: set[str]) -> dict[str, dict[str, Any]]:
    columns = [
        "conversation_id",
        "parent_id",
        "user_id",
        "character_id",
        "persona_id",
        "initial_greeting",
        "is_root_conversation",
        "is_fork_conversation",
        "conversation_status",
        "conversation_station",
        "turns",
    ]
    relevant_array = pa.array(sorted(relevant), type=pa.large_string())
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        table = pq.ParquetFile(path).read(columns=columns)
        mask = pc.is_in(table["conversation_id"], value_set=relevant_array)
        filtered = table.filter(mask)
        for row in filtered.to_pylist():
            result[text(row.get("conversation_id"))] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")

    paths = source_paths(args.source_root)
    edges, source_rows = load_fork_edges(paths)
    relevant = set(edges).union(edges.values())
    rows = load_relevant_rows(paths, relevant)

    child_map: dict[str, list[str]] = collections.defaultdict(list)
    for child, parent in edges.items():
        child_map[parent].append(child)

    entries: dict[str, dict[str, Any]] = {}
    relation_counts = collections.Counter()
    recommendation_counts = collections.Counter()
    missing_parent_rows = 0
    cycles = 0
    for conversation_id in sorted(relevant):
        row = rows.get(conversation_id)
        if row is None:
            continue
        lineage_id, depth, cycle = resolve_lineage(conversation_id, edges)
        cycles += int(cycle)
        parent_id = edges.get(conversation_id)
        entry: dict[str, Any] = {
            "lineage_id": lineage_id,
            "fork_depth": depth,
            "fork_child_ids": sorted(child_map.get(conversation_id) or []),
            "fork_child_count": len(child_map.get(conversation_id) or []),
        }
        if parent_id:
            parent = rows.get(parent_id)
            if parent is None:
                missing_parent_rows += 1
                relation = "parent_missing"
                prefix = common = 0
                differences = ["parent_missing"]
                compatible = False
            else:
                child_signatures = [
                    turn_signature(turn)
                    for turn in sorted(
                        row.get("turns") or [],
                        key=lambda item: int(item.get("turn_order") or 0),
                    )
                ]
                parent_signatures = [
                    turn_signature(turn)
                    for turn in sorted(
                        parent.get("turns") or [],
                        key=lambda item: int(item.get("turn_order") or 0),
                    )
                ]
                relation, prefix, common = classify_relation(
                    child_signatures, parent_signatures
                )
                child_context = context_fingerprint(row)
                parent_context = context_fingerprint(parent)
                differences = [
                    field
                    for field in child_context
                    if child_context[field] != parent_context[field]
                ]
                compatible = not differences
            dedup = recommendation(relation, compatible)
            entry.update(
                {
                    "parent_conversation_id": parent_id,
                    "relation_to_parent": relation,
                    "shared_prefix_turns": prefix,
                    "common_turns": common,
                    "context_compatible_with_parent": compatible,
                    "context_differences": differences,
                    "dedup_recommendation": dedup,
                }
            )
            relation_counts[relation] += 1
            recommendation_counts[dedup] += 1
        entries[conversation_id] = entry

    payload = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.source_root),
        "source_rows": source_rows,
        "source_files": len(paths),
        "fork_edges": len(edges),
        "relevant_rows_found": len(rows),
        "entries_count": len(entries),
        "missing_parent_rows": missing_parent_rows,
        "lineage_cycles": cycles,
        "relation_counts": dict(sorted(relation_counts.items())),
        "recommendation_counts": dict(sorted(recommendation_counts.items())),
        "policy": (
            "descriptive_keep_all_structurally_valid_branches; do not delete based on "
            "fork relation before downstream quality and duplicate adjudication"
        ),
        "signature_policy": (
            "normalized current user+assistant content plus original create timestamps; "
            "message UUID/increase_id intentionally excluded"
        ),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "fork_edges": len(edges),
                "entries": len(entries),
                "relation_counts": payload["relation_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
