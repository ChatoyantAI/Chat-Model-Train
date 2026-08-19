#!/usr/bin/env python3
"""Build deterministic, resumable embedding tasks for RP dialogue MDS.

This reads only the already-filtered neutral/no-summary dialogue views.  Core
teacher/liked dialogues and effective-summary dialogues are intentionally not
inputs to this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT_DEFAULT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
INDEX_COLUMNS = [
    "dialogue_id",
    "language_bucket",
    "candidate_target_count",
    "replay_mode",
    "representative_topic",
    "representative_turn_bucket",
    "candidate_topic_counts_json",
    "candidate_turn_bucket_counts_json",
    "candidate_length_bucket_counts_json",
    "candidate_guideline_counts_json",
]


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet_writer(path: Path, schema: pa.Schema) -> tuple[pq.ParquetWriter, Path]:
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        temp.unlink()
    return (
        pq.ParquetWriter(
            temp,
            schema=schema,
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
        ),
        temp,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--denylist", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    source = root / "02_texts" / "neutral_no_summary_views.parquet"
    text_dir = root / "02_texts"
    embed_dir = root / "03_embeddings"
    manifest_dir = root / "00_manifest"
    text_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    denylist_path = args.denylist or (manifest_dir / "EMBEDDING_DENYLIST.json")
    if denylist_path.exists():
        denylist_payload = json.loads(denylist_path.read_text(encoding="utf-8"))
        denylist_ids = set(str(value) for value in denylist_payload.get("dialogue_ids", []))
    else:
        denylist_payload = {"dialogue_ids": []}
        denylist_ids = set()

    map_path = text_dir / "dialogue_embedding_map.parquet"
    index_path = embed_dir / "embedding_index.parquet"
    user_task_path = text_dir / "user_unique_texts.parquet"
    role_task_path = text_dir / "role_unique_texts.parquet"
    recent_task_path = text_dir / "recent_texts.parquet"
    report_path = manifest_dir / "EMBEDDING_TASK_BUILD_REPORT.json"

    started = time.time()
    parquet_file = pq.ParquetFile(source)
    expected_rows = parquet_file.metadata.num_rows

    map_schema = pa.schema(
        [
            ("row_index", pa.int32()),
            ("dialogue_id", pa.string()),
            ("user_text_indices", pa.list_(pa.int32())),
            ("role_text_index", pa.int32()),
            ("recent_text_index", pa.int32()),
        ]
    )
    index_schema = pa.schema(
        [("row_index", pa.int32())]
        + [(field.name, field.type) for field in parquet_file.schema_arrow if field.name in INDEX_COLUMNS]
    )
    recent_schema = pa.schema(
        [
            ("text_index", pa.int32()),
            ("dialogue_id", pa.string()),
            ("text", pa.string()),
        ]
    )

    map_writer, map_temp = atomic_parquet_writer(map_path, map_schema)
    index_writer, index_temp = atomic_parquet_writer(index_path, index_schema)
    recent_writer, recent_temp = atomic_parquet_writer(recent_task_path, recent_schema)

    user_to_index: dict[str, int] = {}
    user_unique: list[str] = []
    role_to_index: dict[str, int] = {}
    role_unique: list[str] = []
    total_user_messages = 0
    empty_user_messages = 0
    row_offset = 0
    source_rows_seen = 0
    denied_dialogues_found: set[str] = set()
    denied_candidate_targets = 0

    scan_columns = INDEX_COLUMNS + ["user_texts", "role_text", "recent_scene_text"]
    try:
        for batch in parquet_file.iter_batches(columns=scan_columns, batch_size=args.batch_size):
            data = batch.to_pydict()
            source_rows_seen += batch.num_rows
            keep_positions: list[int] = []
            for position, dialogue_id in enumerate(data["dialogue_id"]):
                normalized_id = str(dialogue_id)
                if normalized_id in denylist_ids:
                    denied_dialogues_found.add(normalized_id)
                    denied_candidate_targets += int(data["candidate_target_count"][position])
                else:
                    keep_positions.append(position)
            if not keep_positions:
                continue
            data = {
                name: [values[position] for position in keep_positions]
                for name, values in data.items()
            }
            size = len(keep_positions)
            row_indices = list(range(row_offset, row_offset + size))
            dialogue_ids = [str(value) for value in data["dialogue_id"]]

            mapped_users: list[list[int]] = []
            mapped_roles: list[int] = []
            recent_texts: list[str] = []

            for texts, role_text, recent_text in zip(
                data["user_texts"],
                data["role_text"],
                data["recent_scene_text"],
                strict=True,
            ):
                current_users: list[int] = []
                for raw_text in texts or []:
                    total_user_messages += 1
                    text = str(raw_text or "").strip()
                    if not text:
                        empty_user_messages += 1
                        continue
                    text_index = user_to_index.get(text)
                    if text_index is None:
                        text_index = len(user_unique)
                        user_to_index[text] = text_index
                        user_unique.append(text)
                    current_users.append(text_index)
                if not current_users:
                    raise RuntimeError(f"dialogue row {row_offset + len(mapped_users)} has no user text")
                mapped_users.append(current_users)

                clean_role = str(role_text or "").strip()
                if not clean_role:
                    raise RuntimeError(f"dialogue row {row_offset + len(mapped_roles)} has no role text")
                role_index = role_to_index.get(clean_role)
                if role_index is None:
                    role_index = len(role_unique)
                    role_to_index[clean_role] = role_index
                    role_unique.append(clean_role)
                mapped_roles.append(role_index)

                clean_recent = str(recent_text or "").strip()
                if not clean_recent:
                    raise RuntimeError(f"dialogue row {row_offset + len(recent_texts)} has no recent text")
                recent_texts.append(clean_recent)

            map_writer.write_table(
                pa.Table.from_pydict(
                    {
                        "row_index": row_indices,
                        "dialogue_id": dialogue_ids,
                        "user_text_indices": mapped_users,
                        "role_text_index": mapped_roles,
                        "recent_text_index": row_indices,
                    },
                    schema=map_schema,
                )
            )

            index_payload = {"row_index": row_indices}
            for name in INDEX_COLUMNS:
                index_payload[name] = data[name]
            index_writer.write_table(pa.Table.from_pydict(index_payload, schema=index_schema))

            recent_writer.write_table(
                pa.Table.from_pydict(
                    {
                        "text_index": row_indices,
                        "dialogue_id": dialogue_ids,
                        "text": recent_texts,
                    },
                    schema=recent_schema,
                )
            )
            row_offset += size
    except BaseException:
        map_writer.close()
        index_writer.close()
        recent_writer.close()
        raise
    else:
        map_writer.close()
        index_writer.close()
        recent_writer.close()
        os.replace(map_temp, map_path)
        os.replace(index_temp, index_path)
        os.replace(recent_temp, recent_task_path)

    if source_rows_seen != expected_rows:
        raise RuntimeError(f"source row count mismatch: {source_rows_seen} != {expected_rows}")
    missing_denylist = denylist_ids - denied_dialogues_found
    if missing_denylist:
        raise RuntimeError(f"denylist IDs not present in source: {sorted(missing_denylist)}")
    expected_output_rows = expected_rows - len(denied_dialogues_found)
    if row_offset != expected_output_rows:
        raise RuntimeError(f"output row count mismatch: {row_offset} != {expected_output_rows}")

    task_schema = pa.schema([("text_index", pa.int32()), ("text", pa.string())])
    for path, texts in [(user_task_path, user_unique), (role_task_path, role_unique)]:
        temp = path.with_name(path.name + ".partial")
        pq.write_table(
            pa.Table.from_pydict(
                {"text_index": list(range(len(texts))), "text": texts},
                schema=task_schema,
            ),
            temp,
            compression="zstd",
            compression_level=6,
            use_dictionary=False,
            write_statistics=True,
            row_group_size=65536,
        )
        os.replace(temp, path)

    report = {
        "schema_version": "rp-mds-embedding-tasks-v1",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_dialogues": source_rows_seen,
        "dialogues": row_offset,
        "denylist_path": str(denylist_path),
        "denylist_sha256": sha256_file(denylist_path) if denylist_path.exists() else None,
        "denied_dialogue_ids": sorted(denied_dialogues_found),
        "denied_dialogues": len(denied_dialogues_found),
        "denied_candidate_targets": denied_candidate_targets,
        "total_user_messages": total_user_messages,
        "empty_user_messages": empty_user_messages,
        "unique_user_texts": len(user_unique),
        "unique_role_texts": len(role_unique),
        "recent_texts": row_offset,
        "user_exact_duplicate_fraction": 1 - len(user_unique) / max(total_user_messages, 1),
        "ordering": "first occurrence in neutral_no_summary_views.parquet",
        "outputs": {},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    for path in [map_path, index_path, user_task_path, role_task_path, recent_task_path]:
        report["outputs"][str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    temp_report = report_path.with_name(report_path.name + ".partial")
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
