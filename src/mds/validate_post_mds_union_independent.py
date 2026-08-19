#!/usr/bin/env python3
"""Independent, read-only validation of the formal causal-v2 post-MDS union.

This validator does not import the union builder and does not trust the builder
report for row membership. It independently reconstructs every expected ratio
from the canonical candidate map, K1000 MMR selection, bypass partitions, and
denylist. Validation reports are written only after every check passes.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
UNION = ROOT / "06_union"
RATIOS = (50, 60, 70, 80)
EXPECTED_COUNTS = {
    50: {
        "dialogues": 97_172,
        "targets": 355_963,
        "neutral_dialogues": 92_891,
        "neutral_targets": 327_607,
    },
    60: {
        "dialogues": 115_746,
        "targets": 419_691,
        "neutral_dialogues": 111_465,
        "neutral_targets": 391_335,
    },
    70: {
        "dialogues": 134_247,
        "targets": 482_092,
        "neutral_dialogues": 129_966,
        "neutral_targets": 453_736,
    },
    80: {
        "dialogues": 152_672,
        "targets": 542_203,
        "neutral_dialogues": 148_391,
        "neutral_targets": 513_847,
    },
}
EXPECTED_PARTITIONS = {
    "neutral_mmr": {"dialogues": 185_040, "targets": 625_008},
    "core_bypass": {"dialogues": 3_851, "targets": 25_643},
    "summary_bypass": {"dialogues": 430, "targets": 2_713},
    "denylisted": {"dialogues": 2, "targets": 6},
}
EXPECTED_BUILDER_SHA = (
    "3d6f16081a309c24b0b87525d467a9c48c54547be194d5a1d2f510afb5a2d2b9"
)
EXPECTED_SELECTION_SHA = (
    "326ae297b0ce8dcbd3f53acf7b56c4196c59fd54e9f8a7dcb837e463312b9d0c"
)
EXPECTED_APPROVAL_SHA = (
    "5ea1958ef84fe50c35c43fa6c2b9b3da7974f06c9aabd2240fa5e26fda216401"
)
EXPECTED_TEST_EVIDENCE_SHA = (
    "ed1290fd10d3b696fa423948cb2af2679eb88847abc674f590ad7d4f5434c1be"
)
DURABLE_TEST_EVIDENCE_PATH = (
    ROOT
    / "07_validation"
    / "post_mds_union_builder_independent"
    / "TEST_RESULT.json"
)
EXPECTED_REPORT_SCHEMA = "post-mds-union-causal-v2-exactlang-k1000-v1"
EXPECTED_SELECTION_SCHEMA = "rp-mds-mmr-exact-language-causal-v2"

CANDIDATE_COLUMNS = [
    "target_id",
    "target_message_id",
    "target_increase_id",
    "dialogue_id",
    "conversation_id",
    "user_id",
    "character_id",
    "pool",
    "core_protected",
    "language",
    "language_bucket",
    "topic",
    "turn_bucket",
    "response_length_bucket",
    "guideline_bucket",
    "replay_mode",
]
NATIVE_TARGET_ID_COLUMNS = [
    "target_message_id",
    "dialogue_id",
    "conversation_id",
    "user_id",
    "character_id",
]
NATIVE_DIALOGUE_ID_COLUMNS = [
    "conversation_id",
    "user_id",
    "character_id",
]


class ValidationFailure(RuntimeError):
    pass


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationFailure(f"JSON root is not an object: {path}")
    return payload


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValidationFailure(f"missing {label}: {resolved}")
    return resolved


def require_columns(path: Path, required: Iterable[str], label: str) -> pa.Schema:
    schema = pq.ParquetFile(path).schema_arrow
    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise ValidationFailure(f"{label} missing columns: {missing}")
    return schema


def require_arrow_strings(
    schema: pa.Schema,
    columns: Iterable[str],
    label: str,
) -> None:
    for column in columns:
        if not pa.types.is_string(schema.field(column).type):
            raise ValidationFailure(
                f"{label}.{column} is not Arrow string: "
                f"{schema.field(column).type}"
            )


def as_string(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValidationFailure(f"{column} contains null native IDs")
        frame[column] = frame[column].astype("string")
        if bool(frame[column].str.len().eq(0).any()):
            raise ValidationFailure(f"{column} contains empty native IDs")


def unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if bool(frame[column].duplicated().any()):
        raise ValidationFailure(f"{label}.{column} is not unique")


def equal_series(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    left_obj = left.astype("object")
    right_obj = right.astype("object")
    both_missing = pd.isna(left_obj) & pd.isna(right_obj)
    equal = left_obj.eq(right_obj) | both_missing
    return bool(equal.all())


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().items()
    }


checks: dict[str, bool] = {}
check_details: dict[str, Any] = {}


def check(name: str, condition: bool, detail: Any = None) -> None:
    checks[name] = bool(condition)
    if detail is not None:
        check_details[name] = detail
    if not condition:
        raise ValidationFailure(f"check failed: {name}; detail={detail}")


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    started = datetime.now(timezone.utc)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)

    report_path = require_file(UNION / "POST_MDS_UNION_REPORT.json", "union report")
    primary_path = require_file(UNION / "PRIMARY_SELECTION.json", "primary selection")
    checksums_path = require_file(UNION / "SHA256SUMS.json", "checksum manifest")
    validation_json_path = UNION / "INDEPENDENT_VALIDATION.json"
    validation_md_path = UNION / "INDEPENDENT_VALIDATION.md"
    if validation_json_path.exists() or validation_md_path.exists():
        raise ValidationFailure("independent validation outputs already exist")

    expected_builder_outputs = {
        report_path.name,
        primary_path.name,
        checksums_path.name,
        *(
            f"{kind}_retention_{ratio}.parquet"
            for ratio in RATIOS
            for kind in ("dialogues", "targets")
        ),
    }
    current_files = {path.name for path in UNION.iterdir() if path.is_file()}
    check(
        "builder_output_file_set_exact_before_validation",
        current_files == expected_builder_outputs,
        {
            "missing": sorted(expected_builder_outputs - current_files),
            "unexpected": sorted(current_files - expected_builder_outputs),
        },
    )
    check(
        "no_partial_files",
        not any(".partial." in path.name for path in UNION.iterdir()),
    )
    check(
        "no_union_staging_directories",
        not list(ROOT.glob(".06_union.staging.*")),
    )

    report = read_json(report_path)
    primary = read_json(primary_path)
    checksum_manifest = read_json(checksums_path)

    builder_path = require_file(
        Path(str(report.get("builder_script", ""))),
        "frozen union builder",
    )
    selection_path = require_file(
        ROOT
        / "05_selection"
        / "selection_mmr_causal_v2_exactlang_k1000.parquet",
        "K1000 selection",
    )
    selection_report_path = require_file(
        ROOT
        / "00_manifest"
        / "MMR_SELECTION_CAUSAL_V2_EXACTLANG_K1000_REPORT.json",
        "K1000 selection report",
    )
    candidate_path = require_file(
        ROOT / "01_dialogues" / "candidate_assistant_map.parquet",
        "canonical candidate map",
    )
    core_path = require_file(
        ROOT / "08_core_bypass" / "core_dialogues.parquet",
        "core bypass",
    )
    summary_path = require_file(
        ROOT / "08_core_bypass" / "summary_dialogues.parquet",
        "summary bypass",
    )
    denylist_path = require_file(
        ROOT / "00_manifest" / "EMBEDDING_DENYLIST.json",
        "denylist",
    )
    approval_path = require_file(
        ROOT
        / "00_manifest"
        / "POST_MDS_UNION_BUILDER_INDEPENDENT_VALIDATION.json",
        "builder approval",
    )
    durable_test_evidence_path = require_file(
        DURABLE_TEST_EVIDENCE_PATH,
        "durable builder test evidence mirror",
    )

    check("report_status_pass", report.get("status") == "PASS")
    check(
        "report_schema_exact",
        report.get("schema_version") == EXPECTED_REPORT_SCHEMA,
    )
    check(
        "report_ratio_contract_exact",
        report.get("retention_ratios") == list(RATIOS),
    )
    check(
        "report_primary_k1000_selected60",
        report.get("recommended_k") == 1000
        and report.get("production_seed") == 20260724
        and report.get("primary_ratio") == 60
        and report.get("primary_flag") == "selected_60"
        and report.get("exact_language_isolation") is True,
    )
    check(
        "builder_sha_exact",
        sha256_file(builder_path) == EXPECTED_BUILDER_SHA
        and report.get("builder_script_sha256") == EXPECTED_BUILDER_SHA,
    )
    check(
        "selection_sha_exact",
        sha256_file(selection_path) == EXPECTED_SELECTION_SHA,
    )
    check(
        "approval_sha_exact",
        sha256_file(approval_path) == EXPECTED_APPROVAL_SHA,
    )
    approval = read_json(approval_path)
    approval_test_evidence = approval.get("test_evidence", {})
    check(
        "approval_test_evidence_contract_exact",
        isinstance(approval_test_evidence, dict)
        and approval_test_evidence.get("source")
        == "/tmp/lusy_union_builder_independent_test_result.json"
        and approval_test_evidence.get("sha256") == EXPECTED_TEST_EVIDENCE_SHA,
    )
    check(
        "durable_test_evidence_mirror_sha_exact",
        sha256_file(durable_test_evidence_path) == EXPECTED_TEST_EVIDENCE_SHA,
    )

    selection_report = read_json(selection_report_path)
    check(
        "mmr_report_k1000_selected60_provenance",
        selection_report.get("schema_version") == EXPECTED_SELECTION_SCHEMA
        and selection_report.get("artifact_status") == "formal"
        and selection_report.get("pipeline_id") == "causal_v2_exactlang"
        and selection_report.get("k") == 1000
        and selection_report.get("seed") == 20260724
        and selection_report.get("primary_selection_column") == "selected_60"
        and selection_report.get("cross_language_mmr") is False
        and selection_report.get("superseded_inputs_used") is False
        and Path(str(selection_report.get("output_path", ""))).resolve()
        == selection_path
        and selection_report.get("output_sha256") == EXPECTED_SELECTION_SHA,
    )
    check(
        "no_k2000_or_superseded_primary_provenance",
        "k2000" not in str(report.get("input_schemas", {}).get("selection", {}).get("path", ""))
        and "98_superseded" not in json.dumps(report.get("input_schemas", {})),
    )

    # Independently hash every committed input and output.
    committed_inputs = checksum_manifest.get("inputs")
    committed_outputs = checksum_manifest.get("outputs")
    check(
        "checksum_manifest_shape",
        checksum_manifest.get("schema_version")
        == f"{EXPECTED_REPORT_SCHEMA}-checksums"
        and isinstance(committed_inputs, dict)
        and isinstance(committed_outputs, dict),
    )
    actual_input_hashes: dict[str, str] = {}
    for raw_path, committed_sha in committed_inputs.items():
        path = require_file(Path(raw_path), f"checksum input {raw_path}")
        actual_sha = sha256_file(path)
        actual_input_hashes[str(path)] = actual_sha
        check(
            f"input_checksum::{path.name}",
            committed_sha == actual_sha,
        )
        check(
            f"input_not_superseded::{path.name}",
            "98_superseded" not in path.parts,
        )

    expected_output_paths = {
        str(UNION / f"{kind}_retention_{ratio}.parquet")
        for ratio in RATIOS
        for kind in ("dialogues", "targets")
    } | {str(primary_path)}
    check(
        "checksum_output_path_set_exact",
        set(committed_outputs) == expected_output_paths,
        {
            "missing": sorted(expected_output_paths - set(committed_outputs)),
            "unexpected": sorted(set(committed_outputs) - expected_output_paths),
        },
    )
    actual_output_hashes: dict[str, str] = {}
    for raw_path, committed in committed_outputs.items():
        path = require_file(Path(raw_path), f"checksum output {raw_path}")
        actual_sha = sha256_file(path)
        actual_output_hashes[str(path)] = actual_sha
        check(
            f"output_checksum::{path.name}",
            isinstance(committed, dict)
            and committed.get("sha256") == actual_sha,
        )
        if path.suffix == ".parquet":
            rows = int(pq.ParquetFile(path).metadata.num_rows)
            check(
                f"output_checksum_row_count::{path.name}",
                committed.get("rows") == rows,
            )

    check(
        "report_checksum_manifest_reference",
        report.get("checksums_manifest", {}).get("path") == str(checksums_path)
        and report.get("checksums_manifest", {}).get("sha256")
        == sha256_file(checksums_path),
    )
    check(
        "report_primary_reference",
        report.get("primary_selection", {}).get("path") == str(primary_path)
        and report.get("primary_selection", {}).get("sha256")
        == sha256_file(primary_path),
    )

    check(
        "primary_json_contract",
        primary.get("schema_version")
        == f"{EXPECTED_REPORT_SCHEMA}-primary-selection"
        and primary.get("ratio") == 60
        and primary.get("flag") == "selected_60"
        and primary.get("k") == 1000
        and primary.get("production_seed") == 20260724
        and primary.get("exact_language_isolation") is True
        and primary.get("diagnostic_target_mass_flags_used") is False
        and primary.get("dialogues") == EXPECTED_COUNTS[60]["dialogues"]
        and primary.get("targets") == EXPECTED_COUNTS[60]["targets"],
    )

    candidate_schema = require_columns(
        candidate_path,
        CANDIDATE_COLUMNS,
        "candidate map",
    )
    require_arrow_strings(
        candidate_schema,
        [
            "target_id",
            "target_message_id",
            "dialogue_id",
            "conversation_id",
            "user_id",
            "character_id",
        ],
        "candidate map",
    )
    candidate = pq.read_table(
        candidate_path,
        columns=CANDIDATE_COLUMNS,
    ).to_pandas()
    as_string(
        candidate,
        [
            "target_id",
            "target_message_id",
            "dialogue_id",
            "conversation_id",
            "user_id",
            "character_id",
        ],
    )
    unique(candidate, "target_id", "candidate map")
    unique(candidate, "target_message_id", "candidate map")
    check("canonical_candidate_target_count", len(candidate) == 653_370)
    check(
        "canonical_candidate_dialogue_count",
        candidate["dialogue_id"].nunique() == 189_323,
    )

    candidate_counts = candidate.groupby("dialogue_id", sort=False).size()
    for column in [
        "conversation_id",
        "user_id",
        "character_id",
        "replay_mode",
    ]:
        consistency = candidate.groupby("dialogue_id", sort=False)[column].nunique(
            dropna=False
        )
        check(
            f"canonical_dialogue_{column}_consistent",
            bool(consistency.eq(1).all()),
        )
    candidate_dialogue = (
        candidate.sort_values(
            ["dialogue_id", "target_increase_id", "target_id"],
            kind="stable",
        )
        .drop_duplicates("dialogue_id")
        .set_index("dialogue_id")
    )
    candidate_target = candidate.set_index("target_id", drop=False)

    core = pq.read_table(
        core_path,
        columns=["dialogue_id", "candidate_target_count", "contains_core"],
    ).to_pandas()
    summary = pq.read_table(
        summary_path,
        columns=["dialogue_id", "candidate_target_count", "contains_core"],
    ).to_pandas()
    as_string(core, ["dialogue_id"])
    as_string(summary, ["dialogue_id"])
    unique(core, "dialogue_id", "core bypass")
    unique(summary, "dialogue_id", "summary bypass")
    core_ids = set(core["dialogue_id"].tolist())
    summary_ids = set(summary["dialogue_id"].tolist())
    summary_core_ids = set(
        summary.loc[summary["contains_core"].astype(bool), "dialogue_id"].tolist()
    )
    check("core_summary_disjoint", not core_ids & summary_ids)

    denylist = read_json(denylist_path)
    deny_ids = set(denylist.get("dialogue_ids", []))
    check("denylist_exact_count", len(deny_ids) == 2)
    check("denylist_disjoint_bypass", not deny_ids & (core_ids | summary_ids))

    protected_ids = set(
        candidate.loc[candidate["core_protected"].astype(bool), "dialogue_id"].tolist()
    )
    check(
        "protected_partition_contract_exact",
        protected_ids == core_ids | summary_core_ids
        and len(protected_ids) == 4_023
        and len(summary_core_ids) == 172
        and not protected_ids & deny_ids,
    )

    selection_schema = require_columns(
        selection_path,
        [
            "dialogue_id",
            "candidate_target_count",
            "exact_language",
            "cluster_id",
            "cluster_uid",
            "mmr_rank",
            "mmr_score",
            *[f"selected_{ratio}" for ratio in RATIOS],
        ],
        "K1000 selection",
    )
    require_arrow_strings(
        selection_schema,
        ["dialogue_id", "exact_language", "cluster_uid"],
        "K1000 selection",
    )
    selection_metadata = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (selection_schema.metadata or {}).items()
    }
    check(
        "selection_schema_metadata_primary",
        selection_metadata.get("lusy_schema_version")
        == EXPECTED_SELECTION_SCHEMA
        and selection_metadata.get("lusy_pipeline_id") == "causal_v2_exactlang"
        and selection_metadata.get("lusy_seed") == "20260724"
        and selection_metadata.get("lusy_primary_column") == "selected_60"
        and selection_metadata.get("lusy_language_isolation")
        == "exact_code_no_cross_language",
    )
    selection_columns = [
        "dialogue_id",
        "candidate_target_count",
        "exact_language",
        "cluster_id",
        "cluster_uid",
        "mmr_rank",
        "mmr_score",
        *[f"selected_{ratio}" for ratio in RATIOS],
    ]
    selection = pq.read_table(
        selection_path,
        columns=selection_columns,
    ).to_pandas()
    as_string(selection, ["dialogue_id", "exact_language", "cluster_uid"])
    unique(selection, "dialogue_id", "K1000 selection")
    check("neutral_selection_dialogue_count", len(selection) == 185_040)
    check(
        "neutral_selection_target_count",
        int(selection["candidate_target_count"].sum()) == 625_008,
    )
    check(
        "selection_core_summary_deny_disjoint",
        not set(selection["dialogue_id"]) & (core_ids | summary_ids | deny_ids),
    )
    cluster_language_cardinality = selection.groupby("cluster_id")[
        "exact_language"
    ].nunique(dropna=False)
    check(
        "selection_clusters_exact_language_only",
        bool(cluster_language_cardinality.eq(1).all()),
    )
    cluster_size = selection.groupby("cluster_id")["dialogue_id"].transform("size")
    previous_mask: pd.Series | None = None
    for ratio in RATIOS:
        ceiling = (cluster_size.astype("int64") * ratio + 99) // 100
        ceiling = ceiling.where(cluster_size.gt(5), cluster_size)
        expected_mask = selection["mmr_rank"].astype("int64").le(ceiling)
        observed_mask = selection[f"selected_{ratio}"].astype(bool)
        check(
            f"selection_flag_recomputed::{ratio}",
            bool(observed_mask.eq(expected_mask).all()),
        )
        if previous_mask is not None:
            check(
                f"selection_flags_nested::{ratio}",
                not bool((previous_mask & ~observed_mask).any()),
            )
        previous_mask = observed_mask
    selection_index = selection.set_index("dialogue_id", drop=False)

    check(
        "canonical_partition_complete",
        set(candidate_counts.index)
        == set(selection["dialogue_id"]) | core_ids | summary_ids | deny_ids,
    )
    observed_partition_counts = {
        "neutral_mmr": {
            "dialogues": len(selection),
            "targets": int(selection["candidate_target_count"].sum()),
        },
        "core_bypass": {
            "dialogues": len(core_ids),
            "targets": int(candidate_counts.reindex(list(core_ids)).sum()),
        },
        "summary_bypass": {
            "dialogues": len(summary_ids),
            "targets": int(candidate_counts.reindex(list(summary_ids)).sum()),
        },
        "denylisted": {
            "dialogues": len(deny_ids),
            "targets": int(candidate_counts.reindex(list(deny_ids)).sum()),
        },
    }
    check(
        "fixed_partition_counts_exact",
        observed_partition_counts == EXPECTED_PARTITIONS,
        observed_partition_counts,
    )

    ratio_results: dict[str, Any] = {}
    previous_dialogue_ids: set[str] | None = None
    previous_target_ids: set[str] | None = None

    for ratio in RATIOS:
        dialogue_path = require_file(
            UNION / f"dialogues_retention_{ratio}.parquet",
            f"ratio {ratio} dialogue manifest",
        )
        target_path = require_file(
            UNION / f"targets_retention_{ratio}.parquet",
            f"ratio {ratio} target manifest",
        )
        dialogue_columns = [
            "dialogue_id",
            "conversation_id",
            "user_id",
            "character_id",
            "candidate_target_count",
            "source_partition",
            "selection_reason",
            "retention_ratio",
            "cluster_id",
            "cluster_uid",
            "mmr_rank",
            "mmr_score",
        ]
        target_columns = [
            *CANDIDATE_COLUMNS,
            "source_partition",
            "selection_reason",
            "retention_ratio",
            "cluster_id",
            "cluster_uid",
            "mmr_rank",
            "mmr_score",
        ]
        dialogue_schema = require_columns(
            dialogue_path,
            dialogue_columns,
            f"ratio {ratio} dialogue manifest",
        )
        target_schema = require_columns(
            target_path,
            target_columns,
            f"ratio {ratio} target manifest",
        )
        require_arrow_strings(
            dialogue_schema,
            [
                "dialogue_id",
                "conversation_id",
                "user_id",
                "character_id",
                "source_partition",
                "selection_reason",
            ],
            f"ratio {ratio} dialogue manifest",
        )
        require_arrow_strings(
            target_schema,
            [
                "target_id",
                "target_message_id",
                "dialogue_id",
                "conversation_id",
                "user_id",
                "character_id",
                "source_partition",
                "selection_reason",
            ],
            f"ratio {ratio} target manifest",
        )

        dialogue = pq.read_table(
            dialogue_path,
            columns=dialogue_columns,
        ).to_pandas()
        target = pq.read_table(
            target_path,
            columns=target_columns,
        ).to_pandas()
        as_string(
            dialogue,
            [
                "dialogue_id",
                "conversation_id",
                "user_id",
                "character_id",
                "source_partition",
                "selection_reason",
            ],
        )
        as_string(
            target,
            [
                "target_id",
                "target_message_id",
                "dialogue_id",
                "conversation_id",
                "user_id",
                "character_id",
                "source_partition",
                "selection_reason",
            ],
        )
        unique(dialogue, "dialogue_id", f"ratio {ratio} dialogue manifest")
        unique(target, "target_id", f"ratio {ratio} target manifest")
        unique(target, "target_message_id", f"ratio {ratio} target manifest")

        dialogue_ids = set(dialogue["dialogue_id"].tolist())
        target_ids = set(target["target_id"].tolist())
        neutral_ids = set(
            selection.loc[
                selection[f"selected_{ratio}"].astype(bool),
                "dialogue_id",
            ].tolist()
        )
        expected_dialogue_ids = neutral_ids | core_ids | summary_ids
        expected_target_ids = set(
            candidate.loc[
                candidate["dialogue_id"].isin(expected_dialogue_ids),
                "target_id",
            ].tolist()
        )

        check(
            f"ratio_{ratio}_dialogue_membership_exact",
            dialogue_ids == expected_dialogue_ids,
        )
        check(
            f"ratio_{ratio}_whole_dialogue_target_expansion_exact",
            target_ids == expected_target_ids,
        )
        check(
            f"ratio_{ratio}_row_counts_exact",
            len(dialogue) == EXPECTED_COUNTS[ratio]["dialogues"]
            and len(target) == EXPECTED_COUNTS[ratio]["targets"],
            {"dialogues": len(dialogue), "targets": len(target)},
        )
        check(
            f"ratio_{ratio}_retention_ratio_constant",
            bool(dialogue["retention_ratio"].eq(ratio).all())
            and bool(target["retention_ratio"].eq(ratio).all()),
        )

        source_dialogues = value_counts(dialogue, "source_partition")
        source_targets = value_counts(target, "source_partition")
        expected_source_dialogues = {
            "neutral_mmr": EXPECTED_COUNTS[ratio]["neutral_dialogues"],
            "core_bypass": EXPECTED_PARTITIONS["core_bypass"]["dialogues"],
            "summary_bypass": EXPECTED_PARTITIONS["summary_bypass"]["dialogues"],
        }
        expected_source_targets = {
            "neutral_mmr": EXPECTED_COUNTS[ratio]["neutral_targets"],
            "core_bypass": EXPECTED_PARTITIONS["core_bypass"]["targets"],
            "summary_bypass": EXPECTED_PARTITIONS["summary_bypass"]["targets"],
        }
        check(
            f"ratio_{ratio}_dialogue_partition_counts_exact",
            source_dialogues == expected_source_dialogues,
            source_dialogues,
        )
        check(
            f"ratio_{ratio}_target_partition_counts_exact",
            source_targets == expected_source_targets,
            source_targets,
        )
        partition_sets = {
            name: set(
                dialogue.loc[
                    dialogue["source_partition"].eq(name),
                    "dialogue_id",
                ].tolist()
            )
            for name in ("neutral_mmr", "core_bypass", "summary_bypass")
        }
        check(
            f"ratio_{ratio}_partition_membership_exact",
            partition_sets["neutral_mmr"] == neutral_ids
            and partition_sets["core_bypass"] == core_ids
            and partition_sets["summary_bypass"] == summary_ids,
        )
        check(
            f"ratio_{ratio}_denylist_zero_reflow",
            not dialogue_ids & deny_ids
            and not set(target["dialogue_id"].tolist()) & deny_ids,
        )

        target_dialogue_counts = target.groupby("dialogue_id").size()
        manifest_counts = dialogue.set_index("dialogue_id")[
            "candidate_target_count"
        ].astype("int64")
        check(
            f"ratio_{ratio}_per_dialogue_target_counts_exact",
            target_dialogue_counts.reindex(manifest_counts.index)
            .astype("int64")
            .equals(manifest_counts),
        )
        check(
            f"ratio_{ratio}_target_dialogue_referential_integrity",
            set(target["dialogue_id"].tolist()) == dialogue_ids,
        )

        # Native IDs must match canonical values exactly for every target.
        canonical_target_rows = candidate_target.reindex(target["target_id"].tolist())
        check(
            f"ratio_{ratio}_canonical_target_reindex_complete",
            not canonical_target_rows["target_id"].isna().any(),
        )
        for column in NATIVE_TARGET_ID_COLUMNS:
            check(
                f"ratio_{ratio}_native_target_id_exact::{column}",
                equal_series(
                    target[column].reset_index(drop=True),
                    canonical_target_rows[column].reset_index(drop=True),
                ),
            )
        check(
            f"ratio_{ratio}_target_increase_id_exact",
            equal_series(
                target["target_increase_id"].reset_index(drop=True),
                canonical_target_rows["target_increase_id"].reset_index(drop=True),
            ),
        )

        canonical_dialogue_rows = candidate_dialogue.reindex(
            dialogue["dialogue_id"].tolist()
        )
        check(
            f"ratio_{ratio}_canonical_dialogue_reindex_complete",
            not canonical_dialogue_rows["target_id"].isna().any(),
        )
        for column in NATIVE_DIALOGUE_ID_COLUMNS:
            check(
                f"ratio_{ratio}_native_dialogue_id_exact::{column}",
                equal_series(
                    dialogue[column].reset_index(drop=True),
                    canonical_dialogue_rows[column].reset_index(drop=True),
                ),
            )

        source_by_dialogue = dialogue.set_index("dialogue_id")["source_partition"]
        expected_target_source = source_by_dialogue.reindex(
            target["dialogue_id"].tolist()
        ).reset_index(drop=True)
        check(
            f"ratio_{ratio}_target_partition_matches_dialogue",
            equal_series(
                target["source_partition"].reset_index(drop=True),
                expected_target_source,
            ),
        )

        neutral_dialogue = dialogue.loc[
            dialogue["source_partition"].eq("neutral_mmr")
        ].copy()
        canonical_selection_rows = selection_index.reindex(
            neutral_dialogue["dialogue_id"].tolist()
        )
        check(
            f"ratio_{ratio}_neutral_cluster_id_exact",
            equal_series(
                neutral_dialogue["cluster_id"].reset_index(drop=True),
                canonical_selection_rows["cluster_id"].reset_index(drop=True),
            ),
        )
        check(
            f"ratio_{ratio}_neutral_mmr_rank_exact",
            equal_series(
                neutral_dialogue["mmr_rank"].reset_index(drop=True),
                canonical_selection_rows["mmr_rank"].reset_index(drop=True),
            ),
        )
        check(
            f"ratio_{ratio}_neutral_cluster_uid_exact",
            equal_series(
                neutral_dialogue["cluster_uid"].reset_index(drop=True),
                canonical_selection_rows["cluster_uid"].reset_index(drop=True),
            ),
        )
        check(
            f"ratio_{ratio}_neutral_mmr_score_exact",
            bool(
                np.array_equal(
                    neutral_dialogue["mmr_score"].to_numpy(),
                    canonical_selection_rows["mmr_score"].to_numpy(),
                    equal_nan=True,
                )
            ),
        )

        if previous_dialogue_ids is not None:
            check(
                f"ratio_{ratio}_dialogue_set_nested",
                previous_dialogue_ids.issubset(dialogue_ids),
            )
        if previous_target_ids is not None:
            check(
                f"ratio_{ratio}_target_set_nested",
                previous_target_ids.issubset(target_ids),
            )
        previous_dialogue_ids = dialogue_ids
        previous_target_ids = target_ids

        report_ratio = report.get("ratios", {}).get(str(ratio), {})
        check(
            f"ratio_{ratio}_builder_report_matches_independent_counts",
            report_ratio.get("dialogues") == len(dialogue)
            and report_ratio.get("targets") == len(target)
            and report_ratio.get("neutral_selected_dialogues")
            == len(neutral_ids)
            and report_ratio.get("neutral_selected_targets")
            == expected_source_targets["neutral_mmr"],
        )
        check(
            f"ratio_{ratio}_builder_report_output_hashes_exact",
            report_ratio.get("dialogue_manifest", {}).get("sha256")
            == actual_output_hashes[str(dialogue_path)]
            and report_ratio.get("target_manifest", {}).get("sha256")
            == actual_output_hashes[str(target_path)],
        )

        ratio_results[str(ratio)] = {
            "dialogues": len(dialogue),
            "targets": len(target),
            "dialogues_by_partition": source_dialogues,
            "targets_by_partition": source_targets,
            "dialogue_sha256": actual_output_hashes[str(dialogue_path)],
            "target_sha256": actual_output_hashes[str(target_path)],
            "whole_dialogue_target_expansion": True,
            "native_ids_exact": True,
            "denylist_reflow": 0,
        }

        del dialogue
        del target
        del canonical_target_rows
        del canonical_dialogue_rows

    check(
        "primary_manifest_paths_and_hashes_exact",
        Path(primary["dialogue_manifest"]["path"]).resolve()
        == (UNION / "dialogues_retention_60.parquet").resolve()
        and Path(primary["target_manifest"]["path"]).resolve()
        == (UNION / "targets_retention_60.parquet").resolve()
        and primary["dialogue_manifest"]["sha256"]
        == actual_output_hashes[str(UNION / "dialogues_retention_60.parquet")]
        and primary["target_manifest"]["sha256"]
        == actual_output_hashes[str(UNION / "targets_retention_60.parquet")],
    )

    check(
        "all_builder_report_invariants_true",
        isinstance(report.get("invariants"), dict)
        and bool(report["invariants"])
        and all(value is True for value in report["invariants"].values()),
    )
    check(
        "all_checks_pass",
        bool(checks) and all(checks.values()),
    )

    finished = datetime.now(timezone.utc)
    validation = {
        "schema_version": "post-mds-union-independent-output-validation-v1",
        "status": "PASS",
        "pass": True,
        "created_at_utc": finished.isoformat(),
        "elapsed_seconds": round((finished - started).total_seconds(), 3),
        "validator_script": str(script_path),
        "validator_script_sha256": script_sha,
        "root": str(ROOT),
        "union_dir": str(UNION),
        "builder_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "schema_version": report["schema_version"],
            "status": report["status"],
        },
        "frozen_provenance": {
            "builder_path": str(builder_path),
            "builder_sha256": EXPECTED_BUILDER_SHA,
            "selection_path": str(selection_path),
            "selection_sha256": EXPECTED_SELECTION_SHA,
            "selection_schema": EXPECTED_SELECTION_SCHEMA,
            "recommended_k": 1000,
            "production_seed": 20260724,
            "primary_flag": "selected_60",
            "approval_path": str(approval_path),
            "approval_sha256": EXPECTED_APPROVAL_SHA,
            "builder_test_evidence": {
                "approval_source":
                "/tmp/lusy_union_builder_independent_test_result.json",
                "durable_mirror_path": str(durable_test_evidence_path),
                "sha256": EXPECTED_TEST_EVIDENCE_SHA,
                "byte_identity_bound_by_sha256": True,
            },
        },
        "canonical_universe": {
            "candidate_dialogues": int(candidate["dialogue_id"].nunique()),
            "candidate_targets": len(candidate),
            "partition_counts": observed_partition_counts,
            "protected_dialogues": len(protected_ids),
            "summary_contains_core_dialogues": len(summary_core_ids),
        },
        "ratios": ratio_results,
        "primary": {
            "ratio": 60,
            "flag": "selected_60",
            "dialogues": EXPECTED_COUNTS[60]["dialogues"],
            "targets": EXPECTED_COUNTS[60]["targets"],
        },
        "checks": checks,
        "check_details": check_details,
        "passed_checks": sum(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "input_hashes_recomputed": actual_input_hashes,
        "output_hashes_recomputed": actual_output_hashes,
        "checksum_manifest": {
            "path": str(checksums_path),
            "sha256": sha256_file(checksums_path),
            "builder_outputs_only": True,
            "independent_validation_files_intentionally_not_listed": True,
        },
    }

    markdown_lines = [
        "# Post-MDS Union 独立输出验收",
        "",
        "## 结论",
        "",
        f"- 状态：**PASS**",
        f"- 独立检查：**{validation['passed_checks']}/{validation['passed_checks']}**",
        f"- Primary：**K1000 / `selected_60` / seed 20260724**",
        f"- Builder SHA256：`{EXPECTED_BUILDER_SHA}`",
        f"- Selection SHA256：`{EXPECTED_SELECTION_SHA}`",
        f"- Builder 测试证据耐久镜像：`{durable_test_evidence_path}`",
        f"- 测试证据 SHA256：`{EXPECTED_TEST_EVIDENCE_SHA}`",
        f"- Denylist 回流：**0 dialogues / 0 targets**",
        f"- Whole-dialogue 展开：**四档全量精确匹配 canonical candidate map**",
        f"- Native ID：**逐 target、逐 dialogue 全量一致**",
        "",
        "## 四档结果",
        "",
        "| Retention | Dialogues | Targets | Neutral | Core | Summary |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in RATIOS:
        item = ratio_results[str(ratio)]
        markdown_lines.append(
            "| "
            f"{ratio}% | {item['dialogues']:,} | {item['targets']:,} | "
            f"{item['dialogues_by_partition']['neutral_mmr']:,} | "
            f"{item['dialogues_by_partition']['core_bypass']:,} | "
            f"{item['dialogues_by_partition']['summary_bypass']:,} |"
        )
    markdown_lines.extend(
        [
            "",
            "## 独立复核范围",
            "",
            "- 重新计算所有输入与 builder 输出 SHA256，并核对 `SHA256SUMS.json`。",
            "- 从 K1000 MMR `selected_50/60/70/80`、core、summary 重新构造 dialogue 集。",
            "- 从 canonical candidate map 重新展开每个被选 dialogue 的全部 targets。",
            "- 对每个 target 的 message/conversation/user/character/dialogue ID 逐值核对。",
            "- 对四档 dialogue/target 集执行严格嵌套检查。",
            "- 对 neutral/core/summary partition 的集合和固定计数全量核对。",
            "- 对 denylist 执行 dialogue 与 target 两层零回流检查。",
            "- 核对 `PRIMARY_SELECTION.json` 只指向 K1000 `selected_60`。",
            "",
            "## 固定分区",
            "",
            "| Partition | Dialogues | Targets |",
            "|---|---:|---:|",
            f"| neutral MMR universe | {EXPECTED_PARTITIONS['neutral_mmr']['dialogues']:,} | {EXPECTED_PARTITIONS['neutral_mmr']['targets']:,} |",
            f"| core bypass | {EXPECTED_PARTITIONS['core_bypass']['dialogues']:,} | {EXPECTED_PARTITIONS['core_bypass']['targets']:,} |",
            f"| effective-summary bypass | {EXPECTED_PARTITIONS['summary_bypass']['dialogues']:,} | {EXPECTED_PARTITIONS['summary_bypass']['targets']:,} |",
            f"| denylist | {EXPECTED_PARTITIONS['denylisted']['dialogues']:,} | {EXPECTED_PARTITIONS['denylisted']['targets']:,} |",
            "",
            "Protected contract：4,023 个 protected dialogues = "
            "3,851 core + 172 summary-with-core；neutral 与 denylist 重叠均为 0。",
            "",
            "## Checksum 说明",
            "",
            "`SHA256SUMS.json` 是 builder 原始输出清单；本独立验收文件在 builder 完成后追加，"
            "因此有意不进入该清单。其自身哈希由外部交接报告记录。",
            "",
        ]
    )
    markdown = "\n".join(markdown_lines)
    validation_text = (
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n"
    )

    # Materialize only after every read-only validation has succeeded.
    json_temp = validation_json_path.with_name(
        f".{validation_json_path.name}.partial.{os.getpid()}"
    )
    md_temp = validation_md_path.with_name(
        f".{validation_md_path.name}.partial.{os.getpid()}"
    )
    json_temp.write_text(validation_text, encoding="utf-8")
    md_temp.write_text(markdown, encoding="utf-8")
    os.replace(json_temp, validation_json_path)
    os.replace(md_temp, validation_md_path)

    print(
        json.dumps(
            {
                "status": "PASS",
                "passed_checks": validation["passed_checks"],
                "json": {
                    "path": str(validation_json_path),
                    "sha256": sha256_file(validation_json_path),
                },
                "markdown": {
                    "path": str(validation_md_path),
                    "sha256": sha256_file(validation_md_path),
                },
                "ratios": ratio_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValidationFailure, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
