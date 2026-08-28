"""Batch builder for blind, single-response roleplay Judge inputs."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import UTC, datetime
import gzip
import json
import os
from pathlib import Path
import secrets
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .context_builder import (
    PRODUCTION_CODE_COMMIT,
    ContextBuildError,
    build_candidate_context,
    restore_history,
)
from .io import GroupReader, file_metadata, iter_json_lines, load_runtime_snapshot


RUBRIC_VERSION = "roleplay-rubric-v0.1"
JUDGE_INPUT_FILENAME = "judge_inputs.jsonl.gz"
LABEL_FILENAME = "evaluation_labels.csv"
AUDIT_FILENAME = "evaluation_context_audit.jsonl.gz"
MANIFEST_FILENAME = "manifest.json"

FORBIDDEN_JUDGE_KEYS = frozenset(
    {
        "comparison_id",
        "pair_group_id",
        "candidate_id",
        "candidate_side",
        "left",
        "right",
        "model",
        "model_id",
        "model_name",
        "human_selected",
        "preferred_side",
        "winner",
        "loser",
        "selection",
    }
)


class BuildError(RuntimeError):
    """Raised when a complete Judge dataset cannot be built safely."""


def _write_json_line(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def _trigger_type(value: str | None) -> str:
    if value == "user_message":
        return "message"
    if value in ("continue", "auto_reply"):
        return str(value)
    raise BuildError(f"unsupported trigger type: {value!r}")


def _assert_no_forbidden_judge_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_JUDGE_KEYS:
                raise BuildError(f"forbidden Judge key {key!r} at {path}")
            _assert_no_forbidden_judge_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_judge_keys(child, f"{path}[{index}]")


def _repair_pair_adjacency(
    ordered: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Swap rare random collisions so siblings never appear consecutively."""

    for index in range(1, len(ordered)):
        if ordered[index][1] != ordered[index - 1][1]:
            continue
        replacement_index = next(
            (
                candidate_index
                for candidate_index in range(index + 1, len(ordered))
                if ordered[candidate_index][1] != ordered[index - 1][1]
            ),
            None,
        )
        if replacement_index is None:
            raise BuildError("cannot separate adjacent candidates from the same pair")
        ordered[index], ordered[replacement_index] = (
            ordered[replacement_index],
            ordered[index],
        )
    if any(
        ordered[index][1] == ordered[index - 1][1]
        for index in range(1, len(ordered))
    ):
        raise BuildError("pair adjacency repair failed")
    return ordered


def _write_shuffled_judge_inputs(
    database: sqlite3.Connection,
    path: Path,
) -> tuple[int, bool]:
    ordered = [
        (str(row[0]), str(row[1]))
        for row in database.execute(
            "SELECT evaluation_id, pair_group_id "
            "FROM staged_records ORDER BY shuffle_key, evaluation_id"
        )
    ]
    _repair_pair_adjacency(ordered)

    lookup = database.cursor()
    written = 0
    last_pair_group_id: str | None = None
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for evaluation_id, pair_group_id in ordered:
            if pair_group_id == last_pair_group_id:
                raise BuildError("same-pair candidates became adjacent during output")
            row = lookup.execute(
                "SELECT payload FROM staged_records WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
            if row is None:
                raise BuildError(f"missing staged payload for {evaluation_id}")
            handle.write(str(row[0]))
            handle.write("\n")
            written += 1
            last_pair_group_id = pair_group_id
    return written, True


def _target_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "judge_inputs": output_dir / JUDGE_INPUT_FILENAME,
        "labels": output_dir / LABEL_FILENAME,
        "context_audit": output_dir / AUDIT_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }


def _assert_output_targets_available(
    output_paths: Iterable[Path], *, overwrite: bool
) -> None:
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise BuildError(f"output files already exist: {names}; pass overwrite=True")


def build_judge_dataset(
    *,
    raw_dir: Path,
    runtime_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    rubric_version: str = RUBRIC_VERSION,
) -> Mapping[str, Any]:
    """Build all Judge inputs, isolated labels, audit records, and the manifest."""

    raw_dir = raw_dir.resolve()
    runtime_path = runtime_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir == raw_dir:
        raise BuildError("output_dir must not be the frozen raw data directory")
    required = {
        "comparisons": raw_dir / "comparisons.jsonl.gz",
        "context_messages": raw_dir / "context_messages.jsonl.gz",
        "image_contexts": raw_dir / "message_image_contexts.jsonl.gz",
        "mutation_events": raw_dir / "message_mutation_events.jsonl.gz",
        "source_manifest": raw_dir / "manifest.json",
        "runtime": runtime_path,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise BuildError("missing required input files: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = _target_paths(output_dir)
    if set(final_paths.values()) & set(required.values()):
        raise BuildError("an output path would overwrite a frozen input file")
    _assert_output_targets_available(final_paths.values(), overwrite=overwrite)

    try:
        runtime_by_comparison = load_runtime_snapshot(runtime_path)
        source_manifest = json.loads(
            required["source_manifest"].read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BuildError(str(exc)) from exc

    context_reader = GroupReader(required["context_messages"])
    image_reader = GroupReader(required["image_contexts"])
    mutation_reader = GroupReader(required["mutation_events"])

    comparisons_seen = 0
    valid_comparisons = 0
    excluded_comparisons = 0
    judge_records = 0
    label_records = 0
    audit_records = 0
    used_runtime_ids: set[str] = set()
    evaluation_ids: set[str] = set()
    template_counts: Counter[str] = Counter()
    trigger_counts: Counter[str] = Counter()
    summary_used_count = 0
    current_snapshot_count = 0

    with tempfile.TemporaryDirectory(
        prefix=".roleplay-judge-build-", dir=output_dir
    ) as temporary_directory:
        temporary_dir = Path(temporary_directory)
        temp_paths = _target_paths(temporary_dir)
        staging_database_path = temporary_dir / "shuffle.sqlite3"
        database = sqlite3.connect(staging_database_path)
        database.execute(
            "CREATE TABLE staged_records ("
            "evaluation_id TEXT PRIMARY KEY, "
            "pair_group_id TEXT NOT NULL, "
            "shuffle_key BLOB NOT NULL, "
            "payload TEXT NOT NULL)"
        )

        label_fields = (
            "evaluation_id",
            "pair_group_id",
            "model_id",
            "model_name",
            "human_selected",
            "source_completed_at",
        )

        try:
            with (
                temp_paths["labels"].open("w", encoding="utf-8", newline="") as labels_handle,
                gzip.open(
                    temp_paths["context_audit"],
                    "wt",
                    encoding="utf-8",
                    newline="",
                ) as audit_handle,
            ):
                label_writer = csv.DictWriter(labels_handle, fieldnames=label_fields)
                label_writer.writeheader()

                for comparison in iter_json_lines(required["comparisons"]):
                    comparisons_seen += 1
                    comparison_id = str(comparison["comparison_id"])
                    context_rows = context_reader.take(comparison_id)
                    image_rows = image_reader.take(comparison_id)
                    mutation_rows = mutation_reader.take(comparison_id)

                    if not comparison["scoring"].get("eligible"):
                        excluded_comparisons += 1
                        continue
                    valid_comparisons += 1

                    runtime = runtime_by_comparison.get(comparison_id)
                    if runtime is None:
                        raise BuildError(
                            f"missing runtime snapshot for comparison {comparison_id}"
                        )
                    used_runtime_ids.add(comparison_id)
                    restored = restore_history(
                        comparison,
                        context_rows,
                        image_rows,
                        mutation_rows,
                    )
                    pair_group_id = f"pair_{uuid4().hex}"

                    for candidate_side in ("left", "right"):
                        candidate = comparison["candidates"][candidate_side]
                        content = str(candidate.get("content") or "")
                        if not content.strip():
                            raise BuildError(
                                f"eligible comparison {comparison_id} has an empty candidate"
                            )
                        evaluation_id = f"eval_{uuid4().hex}"
                        if evaluation_id in evaluation_ids:
                            raise BuildError("evaluation_id collision")
                        evaluation_ids.add(evaluation_id)

                        context_result = build_candidate_context(
                            comparison,
                            candidate_side,
                            restored,
                            runtime,
                        )
                        judge_record = {
                            "schema_version": 1,
                            "evaluation_id": evaluation_id,
                            "rubric_version": rubric_version,
                            "trigger_type": _trigger_type(
                                comparison["trigger"].get("type")
                            ),
                            "input_messages": list(context_result.input_messages),
                            "output_message": {
                                "role": "assistant",
                                "content": content,
                            },
                        }
                        _assert_no_forbidden_judge_keys(judge_record)
                        payload = json.dumps(
                            judge_record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        database.execute(
                            "INSERT INTO staged_records "
                            "(evaluation_id, pair_group_id, shuffle_key, payload) "
                            "VALUES (?, ?, ?, ?)",
                            (
                                evaluation_id,
                                pair_group_id,
                                secrets.token_bytes(16),
                                payload,
                            ),
                        )

                        human_selected = int(
                            comparison["selection"].get("preferred_side")
                            == candidate_side
                        )
                        label_writer.writerow(
                            {
                                "evaluation_id": evaluation_id,
                                "pair_group_id": pair_group_id,
                                "model_id": candidate.get("model_id"),
                                "model_name": candidate.get("model_name"),
                                "human_selected": human_selected,
                                "source_completed_at": comparison["timestamps"].get(
                                    "completed_at"
                                ),
                            }
                        )
                        label_records += 1

                        audit = dict(context_result.audit)
                        audit_record = {
                            "schema_version": 1,
                            "evaluation_id": evaluation_id,
                            "source_comparison_id": comparison_id,
                            "model_id": candidate.get("model_id"),
                            "model_name": candidate.get("model_name"),
                            **audit,
                            "runtime_snapshot": runtime.get("snapshot"),
                            "reconstruction_description": (
                                "High-fidelity context reconstruction from frozen data "
                                "using the collection-period production logic."
                            ),
                        }
                        _write_json_line(audit_handle, audit_record)
                        audit_records += 1

                        template_counts[str(audit["template"])] += 1
                        trigger_counts[str(judge_record["trigger_type"])] += 1
                        summary_used_count += int(bool(audit["summary_used"]))
                        current_snapshot_count += int(
                            bool(audit["current_snapshot_fields_present"])
                        )
                        judge_records += 1

                    if valid_comparisons % 500 == 0:
                        database.commit()

            database.commit()
            context_reader.assert_finished()
            image_reader.assert_finished()
            mutation_reader.assert_finished()

            unused_runtime = set(runtime_by_comparison) - used_runtime_ids
            if unused_runtime:
                raise BuildError(
                    f"runtime snapshot contains {len(unused_runtime)} unused comparisons"
                )
            if judge_records != valid_comparisons * 2:
                raise BuildError("Judge record count is not exactly two per comparison")
            if label_records != judge_records or audit_records != judge_records:
                raise BuildError("Judge, label, and audit record counts differ")
            if len(evaluation_ids) != judge_records:
                raise BuildError("evaluation_id values are not unique")

            written, no_adjacent_pairs = _write_shuffled_judge_inputs(
                database, temp_paths["judge_inputs"]
            )
            if written != judge_records:
                raise BuildError(
                    f"wrote {written} Judge records, expected {judge_records}"
                )

            files = {
                "judge_inputs": file_metadata(
                    temp_paths["judge_inputs"], judge_records
                ),
                "evaluation_labels": file_metadata(
                    temp_paths["labels"], label_records
                ),
                "evaluation_context_audit": file_metadata(
                    temp_paths["context_audit"], audit_records
                ),
            }
            source_counts = source_manifest.get("counts", {})
            manifest = {
                "schema_version": 1,
                "campaign_id": source_manifest.get("campaign_id"),
                "frozen_at_utc": source_manifest.get("cutoff_utc"),
                "production_code_commit": PRODUCTION_CODE_COMMIT,
                "rubric_version": rubric_version,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "reconstruction_description": (
                    "High-fidelity context reconstruction from frozen data using "
                    "the collection-period production logic."
                ),
                "counts": {
                    "source_comparisons": comparisons_seen,
                    "valid_comparisons": valid_comparisons,
                    "excluded_comparisons": excluded_comparisons,
                    "judge_inputs": judge_records,
                    "labels": label_records,
                    "context_audits": audit_records,
                },
                "source_manifest_counts": source_counts,
                "distribution": {
                    "trigger_type": dict(sorted(trigger_counts.items())),
                    "template": dict(sorted(template_counts.items())),
                    "summary_used": summary_used_count,
                    "current_snapshot_fields_present": current_snapshot_count,
                },
                "blinding": {
                    "evaluation_id": "cryptographically random UUID4-derived ID",
                    "pair_group_id_excluded_from_judge_inputs": True,
                    "model_identity_excluded_from_judge_inputs": True,
                    "human_selection_excluded_from_judge_inputs": True,
                    "cryptographic_random_shuffle": True,
                    "same_pair_candidates_adjacent": not no_adjacent_pairs,
                },
                "inputs": {
                    "frozen_manifest": str(required["source_manifest"]),
                    "runtime_snapshot": {
                        **file_metadata(runtime_path, len(runtime_by_comparison)),
                        "historical_snapshot": False,
                    },
                },
                "files": files,
                "known_reconstruction_limitations": [
                    "Production did not persist the final model-specific input_messages.",
                    "Character and Persona fields are frozen export-time snapshots, not guaranteed generation-time versions.",
                    "chat_mod, user gender, test-account status, prompt experiment assignment, and model context windows come from a current snapshot rather than a historical snapshot.",
                    "Generation-time active history is reconstructed from frozen rows, timestamps, and mutation events.",
                ],
            }
            temp_paths["manifest"].write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (ContextBuildError, OSError, ValueError, sqlite3.Error) as exc:
            raise BuildError(str(exc)) from exc
        finally:
            database.close()

        for key in ("judge_inputs", "labels", "context_audit", "manifest"):
            os.replace(temp_paths[key], final_paths[key])

    return manifest
