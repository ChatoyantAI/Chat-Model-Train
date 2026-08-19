#!/usr/bin/env python3
"""Build causal RP views in an isolated staging root.

This never mutates the source tree.  It aligns the old neutral view with the
latest-target prompt snapshot, preserves user/role inputs byte-for-byte at the
logical row level, and replaces only recent_scene_text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from recent_view_safe import extract_safe_dialogue_views


SCHEMA_VERSION = "rp-mds-causal-recent-view-v2"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_rows(
    path: Path,
    columns: list[str],
    *,
    selection_class: str | None = None,
    batch_size: int = 512,
) -> Iterator[dict]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        for row in batch.to_pylist():
            if (
                selection_class is None
                or row.get("selection_class") == selection_class
            ):
                yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output_root = args.output_root.resolve()
    output_path = (
        output_root / "02_texts/neutral_no_summary_views.parquet"
    )
    report_path = output_root / "00_manifest/CAUSAL_VIEW_BUILD_REPORT.json"
    if output_root.exists():
        raise SystemExit(f"refusing existing staging root: {output_root}")
    output_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    old_view_path = root / "02_texts/neutral_no_summary_views.parquet"
    dialogue_path = root / "01_dialogues/dialogue_index.parquet"
    denylist_path = root / "00_manifest/EMBEDDING_DENYLIST.json"
    denylist = {
        str(value)
        for value in json.loads(
            denylist_path.read_text(encoding="utf-8")
        ).get("dialogue_ids", [])
    }

    old_parquet = pq.ParquetFile(old_view_path)
    old_columns = old_parquet.schema_arrow.names
    additions = pa.schema(
        [
            ("representative_target_id", pa.string()),
            ("representative_target_increase_id", pa.int64()),
            ("target_assistant_source_index", pa.int32()),
            ("current_user_source_index", pa.int32()),
            (
                "recent_history_user_source_indices",
                pa.list_(pa.int32()),
            ),
            (
                "recent_history_assistant_source_indices",
                pa.list_(pa.int32()),
            ),
            ("recent_max_source_index", pa.int32()),
            ("source_message_count", pa.int32()),
            ("target_boundary_source", pa.string()),
            ("recent_view_schema_version", pa.string()),
            ("causal_valid", pa.bool_()),
        ]
    )
    duplicate_names = set(old_columns) & set(additions.names)
    if duplicate_names:
        raise RuntimeError(
            f"source already contains causal audit fields: {duplicate_names}"
        )
    output_schema = pa.schema(
        list(old_parquet.schema_arrow) + list(additions)
    )

    old_rows = parquet_rows(old_view_path, old_columns)
    dialogue_rows = parquet_rows(
        dialogue_path,
        [
            "dialogue_id",
            "selection_class",
            "representative_target_id",
            "representative_target_increase_id",
            "messages",
        ],
        selection_class="neutral_cluster",
    )

    temp_path = output_path.with_name(output_path.name + ".partial")
    writer = pq.ParquetWriter(
        temp_path,
        schema=output_schema,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
    )
    buffer: list[dict] = []
    counters = Counter()
    started = time.time()
    try:
        for old_row, dialogue_row in zip_longest(old_rows, dialogue_rows):
            if old_row is None or dialogue_row is None:
                raise RuntimeError("old view/dialogue index row-count mismatch")
            if old_row["dialogue_id"] != dialogue_row["dialogue_id"]:
                raise RuntimeError(
                    "old view/dialogue index ordering mismatch: "
                    f"{old_row['dialogue_id']} != {dialogue_row['dialogue_id']}"
                )
            dialogue_id = str(old_row["dialogue_id"])
            counters["neutral_source_rows"] += 1
            if dialogue_id in denylist:
                counters["denylisted_skipped"] += 1
                continue

            safe = extract_safe_dialogue_views(dialogue_row["messages"])
            if old_row["user_texts"] != safe.user_texts:
                raise RuntimeError(
                    f"user trajectory changed for dialogue {dialogue_id}"
                )
            source_indices = [
                safe.current_user_index,
                *[
                    index
                    for pair in safe.history_pair_indices
                    for index in pair
                ],
            ]
            recent_max_index = max(source_indices)
            if recent_max_index >= safe.target_assistant_index:
                raise RuntimeError(
                    f"non-causal index boundary for dialogue {dialogue_id}"
                )
            target_content = str(
                dialogue_row["messages"][safe.target_assistant_index].get(
                    "content"
                )
                or ""
            ).strip()
            exact_source_duplicate = any(
                str(dialogue_row["messages"][index].get("content") or "").strip()
                == target_content
                for index in source_indices
            )
            if exact_source_duplicate:
                # Text equality is only a diagnostic.  Repeated text from an
                # earlier causal source index is not the current target index
                # and therefore is not target leakage.
                counters["exact_target_text_duplicate_in_causal_sources"] += 1

            row = dict(old_row)
            row["recent_scene_text"] = safe.recent_scene_text
            row.update(
                {
                    "representative_target_id": str(
                        dialogue_row["representative_target_id"]
                    ),
                    "representative_target_increase_id": int(
                        dialogue_row["representative_target_increase_id"]
                    ),
                    "target_assistant_source_index": int(
                        safe.target_assistant_index
                    ),
                    "current_user_source_index": int(
                        safe.current_user_index
                    ),
                    "recent_history_user_source_indices": [
                        int(user_index)
                        for user_index, _ in safe.history_pair_indices
                    ],
                    "recent_history_assistant_source_indices": [
                        int(assistant_index)
                        for _, assistant_index in safe.history_pair_indices
                    ],
                    "recent_max_source_index": int(recent_max_index),
                    "source_message_count": len(dialogue_row["messages"] or []),
                    "target_boundary_source": safe.target_boundary_source,
                    "recent_view_schema_version": SCHEMA_VERSION,
                    "causal_valid": True,
                }
            )
            buffer.append(row)
            counters["causal_rows"] += 1
            counters[
                f"boundary_{safe.target_boundary_source}"
            ] += 1
            counters[
                f"history_pairs_{len(safe.history_pair_indices)}"
            ] += 1
            if len(buffer) >= 512:
                writer.write_table(
                    pa.Table.from_pylist(buffer, schema=output_schema)
                )
                buffer.clear()
        if buffer:
            writer.write_table(
                pa.Table.from_pylist(buffer, schema=output_schema)
            )
    except BaseException:
        writer.close()
        raise
    else:
        writer.close()

    if counters["neutral_source_rows"] != 185_042:
        raise RuntimeError(
            f"unexpected neutral source rows: {counters['neutral_source_rows']}"
        )
    if counters["causal_rows"] != 185_040:
        raise RuntimeError(
            f"unexpected causal output rows: {counters['causal_rows']}"
        )
    os.replace(temp_path, output_path)

    report = {
        "schema_version": SCHEMA_VERSION,
        "definition": (
            "latest target prompt snapshot; all visible user messages for "
            "user trajectory; latest three complete historical user-assistant "
            "pairs plus current user for recent scene; current target assistant "
            "and all following content excluded"
        ),
        "source_old_view": str(old_view_path),
        "source_old_view_sha256": sha256_file(old_view_path),
        "source_dialogue_index": str(dialogue_path),
        "source_dialogue_index_sha256": sha256_file(dialogue_path),
        "denylist": str(denylist_path),
        "denylist_sha256": sha256_file(denylist_path),
        "counters": dict(sorted(counters.items())),
        "invariants": {
            "user_texts_logically_identical_all_rows": True,
            "role_text_copied_without_modification": True,
            "recent_max_source_index_lt_target_assistant_source_index_all_rows": True,
            "target_assistant_source_index_never_referenced": True,
            "exact_target_text_duplicate_in_recent_sources_diagnostic": int(
                counters[
                    "exact_target_text_duplicate_in_causal_sources"
                ]
            ),
            "causal_valid_all_rows": True,
        },
        "output": str(output_path),
        "output_rows": pq.ParquetFile(output_path).metadata.num_rows,
        "output_sha256": sha256_file(output_path),
        "output_schema": str(output_schema),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    temp_report = report_path.with_name(report_path.name + ".partial")
    temp_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
