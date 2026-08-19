#!/usr/bin/env python3
"""Build auditable causal-v2 exact-language post-MDS union manifests.

The builder never reads or writes production databases.  It joins four immutable
Parquet/JSON inputs:

* the neutral MMR selection table;
* mandatory core-bypass dialogue IDs;
* mandatory effective-summary-bypass dialogue IDs;
* the canonical candidate-assistant target map;
* the embedding denylist.

For every requested retention ratio it materializes one dialogue manifest and
one target manifest.  A selected neutral dialogue contributes every canonical
target assigned to that replay dialogue.  Mandatory bypass dialogues contribute
every canonical target at every ratio.  A denylisted dialogue is never allowed
back into any partition.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT_DEFAULT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
RATIOS_DEFAULT = (50, 60, 70, 80)
SCHEMA_VERSION = "post-mds-union-causal-v2-exactlang-k1000-v1"
MMR_SCHEMA_VERSION = "rp-mds-mmr-exact-language-causal-v2"
CLUSTER_SCHEMA_VERSION = "rp-mds-exact-language-kmeans-causal-v2"
EXPECTED_PRIMARY_K = 1000
EXPECTED_PRODUCTION_SEED = 20260724
EXPECTED_MMR_LAMBDA = 0.5
EXPECTED_FORMAL_UNIVERSE = {
    "candidate_dialogues": 189323,
    "candidate_targets": 653370,
    "neutral_dialogues": 185040,
    "neutral_targets": 625008,
    "core_dialogues": 3851,
    "core_targets": 25643,
    "summary_dialogues": 430,
    "summary_targets": 2713,
    "summary_core_dialogues": 172,
    "protected_dialogues": 4023,
    "deny_dialogues": 2,
    "deny_targets": 6,
}
EXPECTED_FORMAL_UNION = {
    50: {"dialogues": 97172, "targets": 355963},
    60: {"dialogues": 115746, "targets": 419691},
    70: {"dialogues": 134247, "targets": 482092},
    80: {"dialogues": 152672, "targets": 542203},
}
LEGACY_SELECTION_COLUMNS = {
    "paper_alpha_50",
    "paper_alpha_50_keep_small",
    *(f"target_mass_selected_{ratio}" for ratio in RATIOS_DEFAULT),
    *(f"dialogue_guardrail_{ratio}" for ratio in RATIOS_DEFAULT),
}
MMR_PRIMARY_SEMANTICS = (
    "if cluster_size<=5 keep all; otherwise "
    "ceil(ratio*dialogue_count); always retain whole dialogues"
)
MMR_SCHEMA_METADATA = {
    "lusy_schema_version": MMR_SCHEMA_VERSION,
    "lusy_pipeline_id": "causal_v2_exactlang",
    "lusy_seed": str(EXPECTED_PRODUCTION_SEED),
    "lusy_selection_unit": "dialogue_count_prefix_within_cluster",
    "lusy_selection_guardrail": "clusters_le_5_keep_all",
    "lusy_primary_column": "selected_60",
    "lusy_paper_exact_column": "paper_alpha_50_exact",
    "lusy_language_isolation": "exact_code_no_cross_language",
}

CANDIDATE_REQUIRED = (
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
)
BYPASS_REQUIRED = (
    "dialogue_id",
    "candidate_target_count",
    "selection_class",
    "contains_core",
    "has_effective_summary",
)
SELECTION_REQUIRED_BASE = (
    "dialogue_id",
    "candidate_target_count",
    "cluster_id",
    "mmr_rank",
    "mmr_score",
)
CONSISTENT_DIALOGUE_FIELDS = (
    "conversation_id",
    "user_id",
    "character_id",
    "replay_mode",
)
SELECTION_METADATA = (
    "row_index",
    "coarse_language_bucket",
    "exact_language",
    "replay_mode",
    "representative_topic",
    "representative_turn_bucket",
    "cluster_id",
    "language_cluster_id",
    "cluster_uid",
    "centroid_similarity",
    "mmr_rank",
    "mmr_score",
    "max_similarity_to_previous",
)


class ValidationError(RuntimeError):
    """An input violates a required union invariant."""


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def reject_forbidden_formal_path(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    for subtree_name in ("98_superseded",):
        subtree = (root / subtree_name).resolve()
        if subtree in resolved.parents or resolved == subtree:
            raise ValidationError(f"{label} points under forbidden {subtree_name}")
    return resolved


def validate_json_file_reference(
    reference: Any,
    root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(reference, dict):
        raise ValidationError(f"{label} reference is not an object")
    raw_path = Path(str(reference.get("path", "")))
    path = raw_path if raw_path.is_absolute() else root / raw_path
    path = reject_forbidden_formal_path(path, root, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} evidence is missing: {path}")
    actual_sha = sha256_file(path)
    if reference.get("sha256") != actual_sha:
        raise ValidationError(f"{label} evidence checksum mismatch: {path}")
    return {"path": str(path), "sha256": actual_sha}


def require_columns(
    path: Path,
    required: Iterable[str],
    label: str,
) -> pa.Schema:
    schema = pq.ParquetFile(path).schema_arrow
    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise ValidationError(f"{label} is missing required columns: {missing}")
    return schema


def require_arrow_strings(
    schema: pa.Schema,
    columns: Iterable[str],
    label: str,
) -> None:
    for column in columns:
        field = schema.field(column)
        if not pa.types.is_string(field.type):
            raise ValidationError(
                f"{label}.{column} must be Arrow string before any pandas "
                f"conversion; got {field.type}"
            )


def read_parquet_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=list(columns)).to_pandas()


def normalize_id_column(frame: pd.DataFrame, column: str, label: str) -> None:
    if frame[column].isna().any():
        raise ValidationError(f"{label}.{column} contains null IDs")
    frame[column] = frame[column].astype("string")
    empty = frame[column].str.len().fillna(0).eq(0)
    if bool(empty.any()):
        raise ValidationError(f"{label}.{column} contains empty IDs")


def bounded_examples(values: Iterable[Any], limit: int = 10) -> list[str]:
    return [str(value) for value in list(values)[:limit]]


def validate_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    duplicate_mask = frame[column].duplicated(keep=False)
    if bool(duplicate_mask.any()):
        examples = bounded_examples(frame.loc[duplicate_mask, column].drop_duplicates())
        raise ValidationError(
            f"{label}.{column} is not unique; examples={examples}"
        )


def count_json(values: pd.Series) -> str:
    counts = Counter(str(value) for value in values.fillna("<null>").tolist())
    return json.dumps(dict(sorted(counts.items())), ensure_ascii=False)


def parse_ratios(raw: str) -> tuple[int, ...]:
    try:
        ratios = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratios must be comma-separated integers") from exc
    if not ratios or any(ratio <= 0 or ratio > 100 for ratio in ratios):
        raise argparse.ArgumentTypeError("ratios must be within 1..100")
    if tuple(sorted(set(ratios))) != ratios:
        raise argparse.ArgumentTypeError("ratios must be unique and strictly increasing")
    return ratios


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=65536,
    )
    os.replace(temporary, path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def schema_snapshot(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path),
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.num_row_groups),
        "columns": [
            {"name": field.name, "type": str(field.type)}
            for field in parquet.schema_arrow
        ],
        "sha256": sha256_file(path),
    }


def build_dialogue_metadata(candidate: pd.DataFrame) -> pd.DataFrame:
    ordered = candidate.sort_values(
        ["dialogue_id", "target_increase_id", "target_id"],
        kind="stable",
    )
    for column in CONSISTENT_DIALOGUE_FIELDS:
        distinct = ordered.groupby("dialogue_id", sort=False, dropna=False)[
            column
        ].nunique(dropna=False)
        inconsistent = distinct[distinct.ne(1)]
        if not inconsistent.empty:
            raise ValidationError(
                f"candidate map has inconsistent {column} within dialogue; "
                f"examples={bounded_examples(inconsistent.index)}"
            )

    base = ordered.drop_duplicates("dialogue_id", keep="first")[
        [
            "dialogue_id",
            "conversation_id",
            "user_id",
            "character_id",
            "replay_mode",
        ]
    ].copy()
    counts = (
        ordered.groupby("dialogue_id", sort=False)
        .agg(
            candidate_target_count=("target_id", "size"),
            core_protected_target_count=("core_protected", "sum"),
        )
        .reset_index()
    )
    aggregates = (
        ordered.groupby("dialogue_id", sort=False)
        .agg(
            candidate_pool_counts_json=("pool", count_json),
            candidate_language_counts_json=("language_bucket", count_json),
            candidate_topic_counts_json=("topic", count_json),
            candidate_turn_bucket_counts_json=("turn_bucket", count_json),
            candidate_length_bucket_counts_json=(
                "response_length_bucket",
                count_json,
            ),
            candidate_guideline_counts_json=("guideline_bucket", count_json),
        )
        .reset_index()
    )
    metadata = base.merge(
        counts,
        on="dialogue_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        aggregates,
        on="dialogue_id",
        how="inner",
        validate="one_to_one",
    )
    return metadata


def validate_bypass(
    frame: pd.DataFrame,
    label: str,
    expected_class: str,
    candidate_counts: pd.Series,
) -> set[str]:
    normalize_id_column(frame, "dialogue_id", label)
    validate_unique(frame, "dialogue_id", label)
    if not frame["selection_class"].eq(expected_class).all():
        bad = frame.loc[
            frame["selection_class"].ne(expected_class),
            ["dialogue_id", "selection_class"],
        ].head(10)
        raise ValidationError(
            f"{label}.selection_class must be {expected_class}; "
            f"examples={bad.to_dict(orient='records')}"
        )
    if expected_class == "core_bypass":
        if not frame["contains_core"].fillna(False).astype(bool).all():
            raise ValidationError("core bypass contains rows without contains_core=true")
    elif expected_class == "summary_bypass":
        if not frame["has_effective_summary"].fillna(False).astype(bool).all():
            raise ValidationError(
                "summary bypass contains rows without has_effective_summary=true"
            )

    ids = set(frame["dialogue_id"].tolist())
    missing = ids - set(candidate_counts.index)
    if missing:
        raise ValidationError(
            f"{label} references dialogues absent from candidate map; "
            f"examples={bounded_examples(sorted(missing))}"
        )
    expected = frame.set_index("dialogue_id")["candidate_target_count"].astype("int64")
    actual = candidate_counts.reindex(expected.index).astype("int64")
    mismatch = expected.ne(actual)
    if bool(mismatch.any()):
        examples = [
            {
                "dialogue_id": dialogue_id,
                "bypass_count": int(expected.loc[dialogue_id]),
                "canonical_count": int(actual.loc[dialogue_id]),
            }
            for dialogue_id in expected.index[mismatch][:10]
        ]
        raise ValidationError(f"{label} target counts disagree with candidate map: {examples}")
    return ids


def validate_selection(
    selection: pd.DataFrame,
    selection_schema: pa.Schema,
    ratios: tuple[int, ...],
    candidate_counts: pd.Series,
) -> set[str]:
    normalize_id_column(selection, "dialogue_id", "selection")
    validate_unique(selection, "dialogue_id", "selection")
    ids = set(selection["dialogue_id"].tolist())
    missing = ids - set(candidate_counts.index)
    if missing:
        raise ValidationError(
            "selection references dialogues absent from candidate map; "
            f"examples={bounded_examples(sorted(missing))}"
        )
    expected = selection.set_index("dialogue_id")["candidate_target_count"].astype("int64")
    actual = candidate_counts.reindex(expected.index).astype("int64")
    mismatch = expected.ne(actual)
    if bool(mismatch.any()):
        examples = [
            {
                "dialogue_id": dialogue_id,
                "selection_count": int(expected.loc[dialogue_id]),
                "canonical_count": int(actual.loc[dialogue_id]),
            }
            for dialogue_id in expected.index[mismatch][:10]
        ]
        raise ValidationError(
            f"selection target counts disagree with candidate map: {examples}"
        )

    previous: pd.Series | None = None
    if selection["cluster_id"].isna().any():
        raise ValidationError("selection.cluster_id contains nulls")
    selection["cluster_id"] = selection["cluster_id"].astype("int64")
    cluster_size = selection.groupby("cluster_id", sort=False)["dialogue_id"].transform(
        "size"
    )
    if selection["exact_language"].isna().any():
        raise ValidationError("selection.exact_language contains nulls")
    language_cardinality = selection.groupby(
        "cluster_id",
        sort=False,
    )["exact_language"].nunique(dropna=False)
    if not language_cardinality.eq(1).all():
        bad = language_cardinality[language_cardinality.ne(1)].index.tolist()
        raise ValidationError(
            "selection contains cross-language clusters; "
            f"clusters={bounded_examples(bad)}"
        )
    if selection["mmr_rank"].isna().any():
        raise ValidationError("selection.mmr_rank contains nulls")
    selection["mmr_rank"] = selection["mmr_rank"].astype("int64")
    if bool(selection["mmr_rank"].le(0).any()):
        raise ValidationError("selection.mmr_rank must be positive")
    expected_rank = (
        selection.groupby("cluster_id", sort=False).cumcount() + 1
    )
    # Ranking rows need not be physically sorted, so compare the set 1..N per
    # cluster rather than the current row order.
    rank_integrity = (
        selection.groupby("cluster_id", sort=False)["mmr_rank"]
        .apply(
            lambda values: sorted(int(value) for value in values.tolist())
            == list(range(1, len(values) + 1))
        )
    )
    if not bool(rank_integrity.all()):
        bad = rank_integrity[~rank_integrity].index.tolist()
        raise ValidationError(
            "selection.mmr_rank is not exactly 1..cluster_size; "
            f"clusters={bounded_examples(bad)}"
        )
    del expected_rank
    if selection["mmr_score"].isna().any() or not selection["mmr_score"].map(
        lambda value: math.isfinite(float(value))
    ).all():
        raise ValidationError("selection.mmr_score contains non-finite values")

    for ratio in ratios:
        column = f"selected_{ratio}"
        if selection[column].isna().any():
            raise ValidationError(f"selection.{column} contains nulls")
        if not pa.types.is_boolean(selection_schema.field(column).type):
            raise ValidationError(f"selection.{column} must be boolean")
        current = selection[column].astype(bool)
        ceiling = (cluster_size.astype("int64") * ratio + 99) // 100
        ceiling = ceiling.where(cluster_size.gt(5), cluster_size)
        expected_primary = selection["mmr_rank"].le(ceiling)
        mismatch = current.ne(expected_primary)
        if bool(mismatch.any()):
            raise ValidationError(
                f"selection.{column} is not the causal-v2 dialogue-count MMR "
                "prefix with <=5 keep-all; "
                f"examples={bounded_examples(selection.loc[mismatch, 'dialogue_id'])}"
            )
        if previous is not None:
            lost = previous & ~current
            if bool(lost.any()):
                raise ValidationError(
                    f"selection flags are not nested at {ratio}; "
                    f"examples={bounded_examples(selection.loc[lost, 'dialogue_id'])}"
                )
        previous = current
    if "paper_alpha_50_exact" in selection.columns:
        paper_expected = selection["mmr_rank"].le(
            (cluster_size.astype("int64") + 1) // 2
        )
        paper_mismatch = selection["paper_alpha_50_exact"].astype(bool).ne(
            paper_expected
        )
        if bool(paper_mismatch.any()):
            raise ValidationError(
                "paper_alpha_50_exact is not strict ceil(0.5*cluster_size)"
            )
    return ids


def validate_mmr_report(
    report_path: Path,
    selection_path: Path,
    selection_schema: pa.Schema,
    selection: pd.DataFrame,
    ratios: tuple[int, ...],
    root: Path,
    formal_root: bool,
) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid MMR report JSON: {report_path}") from exc
    if report.get("schema_version") != MMR_SCHEMA_VERSION:
        raise ValidationError(
            f"MMR report schema must be {MMR_SCHEMA_VERSION}; "
            "legacy/target-inclusive reports are forbidden"
        )
    if report.get("artifact_status") != "formal":
        raise ValidationError("MMR report must be artifact_status=formal")
    if report.get("pipeline_id") != "causal_v2_exactlang":
        raise ValidationError("MMR report pipeline_id must be causal_v2_exactlang")
    if report.get("k") != EXPECTED_PRIMARY_K:
        raise ValidationError(
            f"MMR primary must be K={EXPECTED_PRIMARY_K}; "
            f"received K={report.get('k')}"
        )
    if report.get("seed") != EXPECTED_PRODUCTION_SEED:
        raise ValidationError(
            f"MMR seed must be frozen at {EXPECTED_PRODUCTION_SEED}"
        )
    mmr_lambda = report.get("lambda")
    if (
        not isinstance(mmr_lambda, (int, float))
        or float(mmr_lambda) != EXPECTED_MMR_LAMBDA
    ):
        raise ValidationError(
            f"MMR lambda must be exactly {EXPECTED_MMR_LAMBDA}"
        )
    if report.get("cross_language_mmr") is not False:
        raise ValidationError("MMR report permits cross-language ranking")
    if report.get("superseded_inputs_used") is not False:
        raise ValidationError("MMR report used a superseded input")
    raw_metadata = selection_schema.metadata or {}
    metadata = {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in raw_metadata.items()
    }
    for key, expected in MMR_SCHEMA_METADATA.items():
        if metadata.get(key) != expected:
            raise ValidationError(
                f"MMR selection schema metadata {key!r} must be {expected!r}"
            )
    metadata_script_sha = metadata.get("lusy_selection_script_sha256")
    report_script_sha = (
        report.get("selection_script", {}).get("sha256")
        if isinstance(report.get("selection_script"), dict)
        else None
    )
    if (
        not isinstance(metadata_script_sha, str)
        or len(metadata_script_sha) != 64
        or metadata_script_sha != report_script_sha
    ):
        raise ValidationError(
            "MMR selection script checksum metadata does not match its report"
        )
    selection_script_path = Path(
        str(report.get("selection_script", {}).get("path", ""))
    ).resolve()
    selection_script_path = reject_forbidden_formal_path(
        selection_script_path,
        root,
        "MMR selection script",
    )
    if not selection_script_path.is_file():
        raise FileNotFoundError(
            f"MMR selection script is missing: {selection_script_path}"
        )
    if sha256_file(selection_script_path) != report_script_sha:
        raise ValidationError(
            "MMR selection script entity checksum does not match report"
        )
    if formal_root:
        expected_selection_script = (
            root / "scripts" / "select_rp_mds_mmr_causal_v2_exactlang.py"
        ).resolve()
        if selection_script_path != expected_selection_script:
            raise ValidationError(
                "formal MMR selection script must be exactly "
                f"{expected_selection_script}; got {selection_script_path}"
            )
    if report.get("primary_selection_column") != "selected_60":
        raise ValidationError("MMR report primary selection must be selected_60")
    if report.get("paper_exact_column") != "paper_alpha_50_exact":
        raise ValidationError(
            "MMR report paper exact column must be paper_alpha_50_exact"
        )
    committed_path = Path(str(report.get("output_path", ""))).resolve()
    if committed_path != selection_path.resolve():
        raise ValidationError(
            "MMR report output_path does not commit the requested selection Parquet"
        )
    actual_sha = sha256_file(selection_path)
    if report.get("output_sha256") != actual_sha:
        raise ValidationError("MMR selection checksum does not match its report")
    semantics = report.get("retention_semantics")
    if not isinstance(semantics, dict) or semantics.get(
        "selected_50_60_70_80"
    ) != MMR_PRIMARY_SEMANTICS:
        raise ValidationError(
            "MMR report does not declare causal-v2 dialogue-count selection "
            "with <=5 keep-all"
        )
    if report.get("all_rows_ranked") is not True:
        raise ValidationError("MMR report says not all rows are ranked")
    if report.get("all_mmr_scores_finite") is not True:
        raise ValidationError("MMR report says MMR scores are not all finite")
    configured = report.get("clusters_configured")
    empty = report.get("clusters_empty")
    ranked = report.get("clusters_nonempty_ranked")
    if not all(isinstance(value, int) for value in (configured, empty, ranked)):
        raise ValidationError("MMR report cluster-completeness fields are invalid")
    if configured != empty + ranked:
        raise ValidationError(
            "MMR report cluster completeness does not satisfy configured=empty+ranked"
        )
    if report.get("input_rows") != len(selection):
        raise ValidationError("MMR report input_rows disagrees with selection")
    candidate_targets = int(selection["candidate_target_count"].sum())
    if report.get("input_candidate_targets") != candidate_targets:
        raise ValidationError(
            "MMR report input_candidate_targets disagrees with selection"
        )
    ratio_report = report.get("ratios")
    if not isinstance(ratio_report, dict):
        raise ValidationError("MMR report ratios is missing")
    for ratio in ratios:
        actual_mask = selection[f"selected_{ratio}"].astype(bool)
        committed = ratio_report.get(str(ratio))
        if not isinstance(committed, dict):
            raise ValidationError(f"MMR report is missing primary ratio {ratio}")
        if committed.get("selected_dialogues") != int(actual_mask.sum()):
            raise ValidationError(
                f"MMR report selected_dialogues disagrees at ratio {ratio}"
            )
        actual_targets = int(
            selection.loc[actual_mask, "candidate_target_count"].sum()
        )
        if committed.get("selected_candidate_targets") != actual_targets:
            raise ValidationError(
                f"MMR report selected_candidate_targets disagrees at ratio {ratio}"
            )
    provenance = report.get("input_provenance")
    expected_provenance = {
        "fused_embedding",
        "clusters",
        "clustering_report",
    }
    if not isinstance(provenance, dict) or not expected_provenance.issubset(provenance):
        raise ValidationError("MMR report input_provenance is incomplete")
    for key in expected_provenance:
        item = provenance[key]
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
        ):
            raise ValidationError(f"MMR report provenance item is invalid: {key}")
        path = reject_forbidden_formal_path(
            Path(item["path"]),
            root,
            f"MMR provenance {key}",
        )
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValidationError(
                f"MMR report provenance checksum is stale: {key}"
            )
    if metadata.get("lusy_cluster_sha256") != provenance["clusters"]["sha256"]:
        raise ValidationError(
            "MMR selection metadata cluster checksum does not match provenance"
        )
    if metadata.get("lusy_fused_sha256") != provenance["fused_embedding"]["sha256"]:
        raise ValidationError(
            "MMR selection metadata fused checksum does not match provenance"
        )
    if formal_root:
        exact_provenance = {
            "fused_embedding": (
                root
                / "03_embeddings"
                / "fused_causal_v2_u745_r005_s25.f16.npy"
            ),
            "clusters": (
                root
                / "04_clusters"
                / "clusters_causal_v2_exactlang_k1000.parquet"
            ),
            "clustering_report": (
                root
                / "00_manifest"
                / "CLUSTERING_CAUSAL_V2_EXACTLANG_REPORT.json"
            ),
        }
        for key, expected_path in exact_provenance.items():
            actual_path = Path(provenance[key]["path"]).resolve()
            if actual_path != expected_path.resolve():
                raise ValidationError(
                    f"formal MMR provenance {key} must be exactly "
                    f"{expected_path}; got {actual_path}"
                )
    clustering_report_path = Path(
        provenance["clustering_report"]["path"]
    ).resolve()
    clustering_report = json.loads(
        clustering_report_path.read_text(encoding="utf-8")
    )
    if clustering_report.get("schema_version") != CLUSTER_SCHEMA_VERSION:
        raise ValidationError("clustering report schema is not causal-v2")
    if clustering_report.get("artifact_status") != "formal":
        raise ValidationError("clustering report is not formal")
    if clustering_report.get("pipeline_id") != "causal_v2_exactlang":
        raise ValidationError(
            "clustering report pipeline_id must be causal_v2_exactlang"
        )
    if clustering_report.get("seed") != EXPECTED_PRODUCTION_SEED:
        raise ValidationError("clustering seed is not the frozen production seed")
    if clustering_report.get("cross_language_kmeans") is not False:
        raise ValidationError("clustering report permits cross-language KMeans")
    if clustering_report.get("superseded_inputs_used") is not False:
        raise ValidationError("clustering report used superseded inputs")
    cluster_run = clustering_report.get("runs", {}).get(
        str(EXPECTED_PRIMARY_K)
    )
    if not isinstance(cluster_run, dict):
        raise ValidationError("clustering report lacks the K1000 formal run")
    if cluster_run.get("output_sha256") != provenance["clusters"]["sha256"]:
        raise ValidationError(
            "MMR cluster provenance does not match formal K1000 run"
        )
    if Path(str(cluster_run.get("output_path", ""))).resolve() != Path(
        provenance["clusters"]["path"]
    ).resolve():
        raise ValidationError(
            "MMR cluster provenance path does not match formal K1000 run"
        )
    if int(cluster_run.get("cross_language_cluster_count", -1)) != 0:
        raise ValidationError("formal K1000 contains cross-language clusters")
    return report


def validate_stability_contract(
    recommendation_path: Path,
    validation_path: Path,
    root: Path,
) -> dict[str, Any]:
    recommendation = json.loads(
        recommendation_path.read_text(encoding="utf-8")
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if recommendation.get("schema_version") != (
        "lusy-rp-mds-causal-v2-stability-k-recommendation-v1"
    ):
        raise ValidationError("stability recommendation schema is invalid")
    if recommendation.get("artifact_status") != "final_validation":
        raise ValidationError("stability recommendation is not final")
    if recommendation.get("recommended_k") != EXPECTED_PRIMARY_K:
        raise ValidationError("stability recommendation does not select K1000")
    if recommendation.get("production_seed") != EXPECTED_PRODUCTION_SEED:
        raise ValidationError(
            "stability recommendation does not freeze seed 20260724"
        )
    if recommendation.get("selection_column") != "selected_60":
        raise ValidationError(
            "stability recommendation does not select selected_60"
        )
    if recommendation.get("exact_language_isolation") is not True:
        raise ValidationError(
            "stability recommendation lacks exact-language isolation"
        )
    if validation.get("schema_version") != (
        "lusy-rp-mds-causal-v2-cross-seed-stability-validation-v1"
    ):
        raise ValidationError("stability validation schema is invalid")
    validation_checks = validation.get("checks")
    if (
        validation.get("pass") is not True
        or not isinstance(validation_checks, dict)
        or not validation_checks
        or not all(value is True for value in validation_checks.values())
    ):
        raise ValidationError("stability validation is not fully PASS")
    validated_recommendation = validate_json_file_reference(
        validation.get("recommendation"),
        root,
        "stability recommendation",
    )
    if Path(validated_recommendation["path"]).resolve() != recommendation_path:
        raise ValidationError(
            "stability validation commits another recommendation file"
        )
    summary = validate_json_file_reference(
        recommendation.get("summary"),
        root,
        "stability summary",
    )
    return {
        "recommendation": {
            "path": str(recommendation_path),
            "sha256": sha256_file(recommendation_path),
        },
        "validation": {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
        },
        "summary": summary,
        "recommended_k": EXPECTED_PRIMARY_K,
        "production_seed": EXPECTED_PRODUCTION_SEED,
        "selection_column": "selected_60",
    }


def validate_postrun_contract(
    postrun_path: Path,
    selection_path: Path,
    root: Path,
) -> dict[str, Any]:
    report = json.loads(postrun_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != (
        "rp-mds-formal-causal-v2-exact-language-postrun-validation-v1"
    ):
        raise ValidationError("formal post-run validation schema is invalid")
    if report.get("status") != "PASS":
        raise ValidationError("formal post-run validation is not PASS")
    if report.get("failed_checks"):
        raise ValidationError("formal post-run validation contains failures")
    postrun_checks = report.get("checks")
    if (
        not isinstance(postrun_checks, dict)
        or not postrun_checks
        or not all(value is True for value in postrun_checks.values())
    ):
        raise ValidationError("formal post-run validation has a failed check")
    k1000 = report.get("selections", {}).get("1000", {})
    committed_path = Path(str(k1000.get("output_path", ""))).resolve()
    if committed_path != selection_path.resolve():
        raise ValidationError(
            "post-run validation commits a different K1000 selection"
        )
    actual_sha = sha256_file(selection_path)
    if k1000.get("output_sha256") != actual_sha:
        raise ValidationError(
            "post-run validation K1000 selection checksum mismatch"
        )
    forbidden = (root / "98_superseded").resolve()
    if forbidden in committed_path.parents:
        raise ValidationError("post-run validation points under 98_superseded")
    return {
        "path": str(postrun_path),
        "sha256": sha256_file(postrun_path),
        "status": "PASS",
        "checks_passed": len(report.get("checks", {})),
        "selection_sha256": actual_sha,
    }


def validate_builder_approval(
    approval_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("schema_version") != (
        "post-mds-union-causal-v2-builder-independent-validation-v1"
    ):
        raise ValidationError("independent builder approval schema is invalid")
    approval_checks = approval.get("checks")
    if (
        approval.get("pass") is not True
        or not isinstance(approval_checks, dict)
        or not approval_checks
        or not all(value is True for value in approval_checks.values())
    ):
        raise ValidationError("independent builder approval is not fully PASS")
    current_script_sha = sha256_file(Path(__file__).resolve())
    if approval.get("builder_script_sha256") != current_script_sha:
        raise ValidationError(
            "builder changed after independent approval; revalidation required"
        )
    if approval.get("selection_sha256") != sha256_file(selection_path):
        raise ValidationError(
            "independent approval commits a different selection"
        )
    if approval.get("recommended_k") != EXPECTED_PRIMARY_K:
        raise ValidationError("independent approval does not approve K1000")
    if approval.get("primary_flag") != "selected_60":
        raise ValidationError(
            "independent approval does not approve selected_60"
        )
    return {
        "path": str(approval_path),
        "sha256": sha256_file(approval_path),
        "builder_script_sha256": current_script_sha,
        "selection_sha256": sha256_file(selection_path),
        "pass": True,
    }


def prepare_outputs(
    output_dir: Path,
    ratios: tuple[int, ...],
    replace_existing: bool,
) -> Path:
    if output_dir.exists():
        if output_dir.is_dir() and not any(output_dir.iterdir()):
            raise FileExistsError(
                "refusing an already-created formal output directory; "
                "directory-level atomic promotion requires it to be absent"
            )
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if replace_existing:
        raise ValidationError(
            "causal-v2 output generation never replaces an existing directory"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    planned = [
        output_dir / f"{kind}_retention_{ratio}.parquet"
        for ratio in ratios
        for kind in ("dialogues", "targets")
    ]
    planned.extend(
        [
            output_dir / "POST_MDS_UNION_REPORT.json",
            output_dir / "SHA256SUMS.json",
            output_dir / "PRIMARY_SELECTION.json",
        ]
    )
    existing = [path for path in planned if path.exists()]
    if existing and not replace_existing:
        raise FileExistsError(
            "refusing to replace existing union outputs without "
            f"--replace-existing: {bounded_examples(existing)}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging.",
            dir=output_dir.parent,
        )
    ).resolve()
    expected_parent = output_dir.parent.resolve()
    if staging.parent != expected_parent or not staging.name.startswith(
        f".{output_dir.name}.staging."
    ):
        raise ValidationError(f"unsafe staging directory: {staging}")

    def cleanup_staging() -> None:
        if (
            staging.exists()
            and staging.parent == expected_parent
            and staging.name.startswith(f".{output_dir.name}.staging.")
        ):
            shutil.rmtree(staging)

    atexit.register(cleanup_staging)
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument("--core-bypass", type=Path)
    parser.add_argument("--summary-bypass", type=Path)
    parser.add_argument("--candidate-map", type=Path)
    parser.add_argument("--denylist", type=Path)
    parser.add_argument("--stability-recommendation", type=Path)
    parser.add_argument("--stability-validation", type=Path)
    parser.add_argument("--postrun-validation", type=Path)
    parser.add_argument("--builder-approval", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--ratios",
        type=parse_ratios,
        default=RATIOS_DEFAULT,
        help="strictly increasing comma-separated retention ratios",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="atomically replace only the known output files",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run every input/provenance/partition check without writing outputs",
    )
    parser.add_argument(
        "--primary-ratio",
        type=int,
        default=60,
        help="ratio designated as the default downstream Judge candidate view",
    )
    args = parser.parse_args()

    started = time.time()
    root = args.root.resolve()
    selection_path = require_file(
        args.selection
        or root
        / "05_selection"
        / "selection_mmr_causal_v2_exactlang_k1000.parquet",
        "MMR selection",
    )
    selection_report_path = require_file(
        args.selection_report
        or root
        / "00_manifest"
        / "MMR_SELECTION_CAUSAL_V2_EXACTLANG_K1000_REPORT.json",
        "MMR selection report",
    )
    core_path = require_file(
        args.core_bypass or root / "08_core_bypass" / "core_dialogues.parquet",
        "core bypass",
    )
    summary_path = require_file(
        args.summary_bypass
        or root / "08_core_bypass" / "summary_dialogues.parquet",
        "summary bypass",
    )
    candidate_path = require_file(
        args.candidate_map
        or root / "01_dialogues" / "candidate_assistant_map.parquet",
        "candidate assistant map",
    )
    denylist_path = require_file(
        args.denylist or root / "00_manifest" / "EMBEDDING_DENYLIST.json",
        "embedding denylist",
    )
    recommendation_path = require_file(
        args.stability_recommendation
        or root
        / "07_validation"
        / "stability"
        / "STABILITY_K_RECOMMENDATION.json",
        "stability K recommendation",
    )
    stability_validation_path = require_file(
        args.stability_validation
        or root
        / "07_validation"
        / "stability"
        / "STABILITY_VALIDATION_20260725.json",
        "stability validation",
    )
    postrun_validation_path = require_file(
        args.postrun_validation
        or root
        / "07_validation"
        / "formal_causal_v2_exactlang_postrun"
        / "VALIDATION_REPORT.json",
        "formal causal-v2 post-run validation",
    )
    output_dir = (args.output_dir or root / "06_union").resolve()
    ratios: tuple[int, ...] = args.ratios
    if ratios != RATIOS_DEFAULT:
        raise ValidationError(
            "causal-v2 union ratios are frozen at exactly 50,60,70,80"
        )
    if args.primary_ratio not in ratios:
        raise ValidationError("--primary-ratio must be one of --ratios")
    if args.primary_ratio != 60:
        raise ValidationError(
            "causal-v2 stability decision commits selected_60 as primary"
        )
    formal_root = root == ROOT_DEFAULT.resolve()
    if formal_root:
        exact_formal_paths = {
            "selection": (
                root
                / "05_selection"
                / "selection_mmr_causal_v2_exactlang_k1000.parquet"
            ),
            "selection_report": (
                root
                / "00_manifest"
                / "MMR_SELECTION_CAUSAL_V2_EXACTLANG_K1000_REPORT.json"
            ),
            "core": root / "08_core_bypass" / "core_dialogues.parquet",
            "summary": root / "08_core_bypass" / "summary_dialogues.parquet",
            "candidate": (
                root / "01_dialogues" / "candidate_assistant_map.parquet"
            ),
            "denylist": root / "00_manifest" / "EMBEDDING_DENYLIST.json",
            "recommendation": (
                root
                / "07_validation"
                / "stability"
                / "STABILITY_K_RECOMMENDATION.json"
            ),
            "stability_validation": (
                root
                / "07_validation"
                / "stability"
                / "STABILITY_VALIDATION_20260725.json"
            ),
            "postrun_validation": (
                root
                / "07_validation"
                / "formal_causal_v2_exactlang_postrun"
                / "VALIDATION_REPORT.json"
            ),
        }
        actual_formal_paths = {
            "selection": selection_path,
            "selection_report": selection_report_path,
            "core": core_path,
            "summary": summary_path,
            "candidate": candidate_path,
            "denylist": denylist_path,
            "recommendation": recommendation_path,
            "stability_validation": stability_validation_path,
            "postrun_validation": postrun_validation_path,
        }
        for label, expected in exact_formal_paths.items():
            actual = reject_forbidden_formal_path(
                actual_formal_paths[label],
                root,
                label,
            )
            if actual != expected.resolve():
                raise ValidationError(
                    f"formal {label} must be exactly {expected}; got {actual}"
                )
    selection_required = (
        *SELECTION_REQUIRED_BASE,
        "exact_language",
        *(f"selected_{r}" for r in ratios),
        "paper_alpha_50_exact",
    )
    selection_schema = require_columns(
        selection_path,
        selection_required,
        "MMR selection",
    )
    legacy_columns_present = sorted(
        LEGACY_SELECTION_COLUMNS & set(selection_schema.names)
    )
    if legacy_columns_present:
        raise ValidationError(
            "legacy target-mass/dialogue-guardrail selection columns are "
            f"forbidden in causal-v2 union input: {legacy_columns_present}"
        )
    require_arrow_strings(
        selection_schema,
        ["dialogue_id", "exact_language"],
        "MMR selection",
    )
    flag_columns = [
        *(f"selected_{ratio}" for ratio in ratios),
        "paper_alpha_50_exact",
    ]
    for column in flag_columns:
        if not pa.types.is_boolean(selection_schema.field(column).type):
            raise ValidationError(f"MMR selection flag {column} must be boolean")
    candidate_schema = require_columns(
        candidate_path,
        CANDIDATE_REQUIRED,
        "candidate assistant map",
    )
    core_schema = require_columns(core_path, BYPASS_REQUIRED, "core bypass")
    summary_schema = require_columns(
        summary_path,
        BYPASS_REQUIRED,
        "summary bypass",
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
        "candidate assistant map",
    )
    require_arrow_strings(core_schema, ["dialogue_id"], "core bypass")
    require_arrow_strings(summary_schema, ["dialogue_id"], "summary bypass")
    for label, schema in (
        ("core bypass", core_schema),
        ("summary bypass", summary_schema),
    ):
        for column in ("contains_core", "has_effective_summary"):
            if not pa.types.is_boolean(schema.field(column).type):
                raise ValidationError(
                    f"{label}.{column} must be Arrow boolean; "
                    f"got {schema.field(column).type}"
                )

    candidate = read_parquet_columns(candidate_path, CANDIDATE_REQUIRED)
    for column in (
        "target_id",
        "target_message_id",
        "dialogue_id",
        "conversation_id",
        "user_id",
        "character_id",
    ):
        normalize_id_column(candidate, column, "candidate_map")
    validate_unique(candidate, "target_id", "candidate_map")
    validate_unique(candidate, "target_message_id", "candidate_map")
    if candidate["target_increase_id"].isna().any():
        raise ValidationError("candidate_map.target_increase_id contains nulls")
    candidate["target_increase_id"] = candidate["target_increase_id"].astype("int64")
    if candidate["core_protected"].isna().any():
        raise ValidationError("candidate_map.core_protected contains nulls")
    candidate["core_protected"] = candidate["core_protected"].astype(bool)
    dialogue_metadata = build_dialogue_metadata(candidate)
    candidate_counts = dialogue_metadata.set_index("dialogue_id")[
        "candidate_target_count"
    ]

    bypass_columns = list(BYPASS_REQUIRED)
    core = read_parquet_columns(core_path, bypass_columns)
    summary = read_parquet_columns(summary_path, bypass_columns)
    core_ids = validate_bypass(
        core,
        "core_bypass",
        "core_bypass",
        candidate_counts,
    )
    summary_ids = validate_bypass(
        summary,
        "summary_bypass",
        "summary_bypass",
        candidate_counts,
    )
    overlap = core_ids & summary_ids
    if overlap:
        raise ValidationError(
            "core and summary bypass partitions overlap; "
            f"examples={bounded_examples(sorted(overlap))}"
        )
    if formal_root:
        core_target_count = int(
            candidate_counts.reindex(list(core_ids)).sum()
        )
        summary_target_count = int(
            candidate_counts.reindex(list(summary_ids)).sum()
        )
        if len(core_ids) != 3851 or core_target_count != 25643:
            raise ValidationError(
                "formal core bypass must be exactly 3,851 dialogues / "
                f"25,643 targets; got {len(core_ids)} / {core_target_count}"
            )
        if len(summary_ids) != 430 or summary_target_count != 2713:
            raise ValidationError(
                "formal effective-summary bypass must be exactly 430 "
                f"dialogues / 2,713 targets; got {len(summary_ids)} / "
                f"{summary_target_count}"
            )

    selection_columns = list(selection_schema.names)
    selection = read_parquet_columns(selection_path, selection_columns)
    if selection[flag_columns].isna().any().any():
        raise ValidationError("MMR selection contains null boolean flags")
    selection_ids = validate_selection(
        selection,
        selection_schema,
        ratios,
        candidate_counts,
    )
    mmr_report = validate_mmr_report(
        selection_report_path,
        selection_path,
        selection_schema,
        selection,
        ratios,
        root,
        formal_root,
    )
    stability_contract = validate_stability_contract(
        recommendation_path,
        stability_validation_path,
        root,
    )
    postrun_contract = validate_postrun_contract(
        postrun_validation_path,
        selection_path,
        root,
    )
    bypass_ids = core_ids | summary_ids
    overlap = selection_ids & bypass_ids
    if overlap:
        raise ValidationError(
            "neutral MMR selection overlaps mandatory bypass partitions; "
            f"examples={bounded_examples(sorted(overlap))}"
        )

    deny_payload = json.loads(denylist_path.read_text(encoding="utf-8"))
    deny_raw = deny_payload.get("dialogue_ids")
    if not isinstance(deny_raw, list) or any(not isinstance(item, str) for item in deny_raw):
        raise ValidationError("denylist.dialogue_ids must be a list of strings")
    deny_ids = set(deny_raw)
    mandatory_deny_overlap = bypass_ids & deny_ids
    if mandatory_deny_overlap:
        raise ValidationError(
            "denylist intersects mandatory bypass; cannot satisfy both invariants; "
            f"examples={bounded_examples(sorted(mandatory_deny_overlap))}"
        )
    selection_deny_overlap = selection_ids & deny_ids
    if selection_deny_overlap:
        raise ValidationError(
            "MMR selection contains denylisted dialogues (denylist recovery forbidden); "
            f"examples={bounded_examples(sorted(selection_deny_overlap))}"
        )

    candidate_dialogue_ids = set(dialogue_metadata["dialogue_id"].tolist())
    deny_present = candidate_dialogue_ids & deny_ids
    covered = selection_ids | bypass_ids | deny_present
    missing_partition = candidate_dialogue_ids - covered
    unknown_partition = covered - candidate_dialogue_ids
    if missing_partition or unknown_partition:
        raise ValidationError(
            "candidate dialogue partition is not complete/exact; "
            f"missing={bounded_examples(sorted(missing_partition))}, "
            f"unknown={bounded_examples(sorted(unknown_partition))}"
        )

    protected_dialogue_ids = set(
        candidate.loc[candidate["core_protected"], "dialogue_id"].tolist()
    )
    if summary["contains_core"].isna().any():
        raise ValidationError("summary_bypass.contains_core contains nulls")
    summary_core_ids = set(
        summary.loc[
            summary["contains_core"].astype(bool),
            "dialogue_id",
        ].tolist()
    )
    expected_protected_dialogue_ids = core_ids | summary_core_ids
    if protected_dialogue_ids != expected_protected_dialogue_ids:
        raise ValidationError(
            "core_protected dialogue set must exactly equal core bypass plus "
            "summary bypass rows with contains_core=true; "
            f"unexpected_protected={bounded_examples(sorted(protected_dialogue_ids - expected_protected_dialogue_ids))}, "
            f"missing_protected={bounded_examples(sorted(expected_protected_dialogue_ids - protected_dialogue_ids))}"
        )
    if (
        not core_ids.issubset(protected_dialogue_ids)
        or protected_dialogue_ids & selection_ids
        or protected_dialogue_ids & deny_present
    ):
        raise ValidationError(
            "core-protected set violates core/neutral/deny partition contract"
        )
    protected_in_neutral = protected_dialogue_ids & selection_ids
    if protected_in_neutral:
        raise ValidationError(
            "core-protected targets leaked into neutral MMR selection; "
            f"examples={bounded_examples(sorted(protected_in_neutral))}"
        )
    if formal_root:
        observed_universe = {
            "candidate_dialogues": len(candidate_dialogue_ids),
            "candidate_targets": int(len(candidate)),
            "neutral_dialogues": len(selection_ids),
            "neutral_targets": int(
                candidate_counts.reindex(list(selection_ids)).sum()
            ),
            "core_dialogues": len(core_ids),
            "core_targets": int(
                candidate_counts.reindex(list(core_ids)).sum()
            ),
            "summary_dialogues": len(summary_ids),
            "summary_targets": int(
                candidate_counts.reindex(list(summary_ids)).sum()
            ),
            "summary_core_dialogues": len(summary_core_ids),
            "protected_dialogues": len(protected_dialogue_ids),
            "deny_dialogues": len(deny_present),
            "deny_targets": int(
                candidate_counts.reindex(list(deny_present)).sum()
            ),
        }
        if observed_universe != EXPECTED_FORMAL_UNIVERSE:
            raise ValidationError(
                "formal causal-v2 universe counts drifted; "
                f"expected={EXPECTED_FORMAL_UNIVERSE}, "
                f"observed={observed_universe}"
            )

    neutral_metadata_columns = [
        column for column in SELECTION_METADATA if column in selection.columns
    ]
    neutral_metadata = selection[
        ["dialogue_id", *neutral_metadata_columns]
    ].copy()
    # Candidate metadata is canonical for shared columns; keep selection-only
    # attributes (cluster/rank) and avoid merge suffixes.
    neutral_metadata = neutral_metadata[
        [
            column
            for column in neutral_metadata.columns
            if column == "dialogue_id"
            or column
            not in {
                "language_bucket",
                "replay_mode",
                "representative_topic",
                "representative_turn_bucket",
            }
        ]
    ]

    approval_path = (
        args.builder_approval
        or root
        / "00_manifest"
        / "POST_MDS_UNION_BUILDER_INDEPENDENT_VALIDATION.json"
    ).resolve()
    if formal_root:
        expected_approval_path = (
            root
            / "00_manifest"
            / "POST_MDS_UNION_BUILDER_INDEPENDENT_VALIDATION.json"
        ).resolve()
        approval_path = reject_forbidden_formal_path(
            approval_path,
            root,
            "independent builder approval",
        )
        if approval_path != expected_approval_path:
            raise ValidationError(
                "formal independent builder approval must be exactly "
                f"{expected_approval_path}; got {approval_path}"
            )
    approval_contract: dict[str, Any] | None = None
    if approval_path.is_file():
        approval_contract = validate_builder_approval(
            approval_path,
            selection_path,
        )
    elif not args.validate_only:
        raise FileNotFoundError(
            "formal generation is locked until independent builder validation "
            f"exists: {approval_path}"
        )
    if formal_root:
        if output_dir != (root / "06_union").resolve():
            raise ValidationError(
                "formal union output directory must be exactly 06_union"
            )
        if args.replace_existing:
            raise ValidationError(
                "formal causal-v2 union refuses --replace-existing"
            )

    source_map = {
        **{dialogue_id: "core_bypass" for dialogue_id in core_ids},
        **{dialogue_id: "summary_bypass" for dialogue_id in summary_ids},
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder_script": str(Path(__file__).resolve()),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
        "created_unix_time": time.time(),
        "root": str(root),
        "output_dir": str(output_dir),
        "retention_ratios": list(ratios),
        "primary_ratio": args.primary_ratio,
        "primary_flag": f"selected_{args.primary_ratio}",
        "primary_selection_unit": "dialogue_count_per_cluster",
        "recommended_k": EXPECTED_PRIMARY_K,
        "production_seed": EXPECTED_PRODUCTION_SEED,
        "exact_language_isolation": True,
        "small_cluster_guardrail": "cluster_size<=5_keep_all",
        "ignored_diagnostic_selection_columns": ["paper_alpha_50_exact"],
        "selection_semantics": {
            "neutral": (
                "retain the whole replay dialogue when the MMR dialogue-fraction "
                "flag selected_<ratio> is true"
            ),
            "core_bypass": "retain every dialogue and every canonical target at every ratio",
            "summary_bypass": "retain every effective-summary dialogue and every canonical target at every ratio",
            "denylist": "denylist wins globally; overlap with mandatory bypass is a hard error",
        },
        "native_id_policy": (
            "preserve target_id, target_message_id, conversation_id, user_id, "
            "character_id and dialogue_id byte-for-byte as canonical strings"
        ),
        "input_schemas": {
            "selection": schema_snapshot(selection_path),
            "selection_report": {
                "path": str(selection_report_path),
                "sha256": sha256_file(selection_report_path),
                "schema_version": mmr_report["schema_version"],
            },
            "core_bypass": schema_snapshot(core_path),
            "summary_bypass": schema_snapshot(summary_path),
            "candidate_map": schema_snapshot(candidate_path),
            "stability_contract": stability_contract,
            "formal_postrun_validation": postrun_contract,
            "independent_builder_approval": approval_contract,
        },
        "denylist": {
            "path": str(denylist_path),
            "sha256": sha256_file(denylist_path),
            "version": deny_payload.get("version"),
            "listed_dialogues": len(deny_ids),
            "present_candidate_dialogues": len(deny_present),
            "present_candidate_targets": int(
                candidate["dialogue_id"].isin(deny_present).sum()
            ),
            "restored_dialogues": 0,
            "restored_targets": 0,
        },
        "canonical_universe": {
            "candidate_dialogues": len(candidate_dialogue_ids),
            "candidate_targets": int(len(candidate)),
            "core_protected_targets": int(candidate["core_protected"].sum()),
        },
        "input_partitions": {
            "neutral_mmr": {
                "dialogues": len(selection_ids),
                "targets": int(candidate_counts.reindex(list(selection_ids)).sum()),
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
                "dialogues": len(deny_present),
                "targets": int(candidate_counts.reindex(list(deny_present)).sum()),
            },
            "core_protected_contract": {
                "protected_dialogues": len(protected_dialogue_ids),
                "core_bypass_dialogues": len(core_ids),
                "summary_contains_core_dialogues": len(summary_core_ids),
                "protected_outside_core_or_summary": 0,
                "protected_in_neutral": 0,
                "protected_in_denylist": 0,
            },
        },
        "invariants": {
            "partition_pairwise_disjoint": True,
            "partition_complete_after_denylist": True,
            "candidate_target_id_unique": True,
            "candidate_target_message_id_unique": True,
            "bypass_target_counts_match_canonical_map": True,
            "selection_target_counts_match_canonical_map": True,
            "selection_flags_nested": True,
            "selection_flags_are_causal_v2_dialogue_prefixes": True,
            "small_clusters_le_5_all_kept": True,
            "paper_alpha_exact_not_used_for_union": True,
            "mmr_causal_v2_report_checksum_and_semantics_verified": True,
            "k1000_stability_recommendation_verified": True,
            "formal_postrun_validation_verified": True,
            "all_core_protected_targets_in_bypass": True,
            "denylist_never_restored": True,
        },
        "ratios": {},
    }
    if args.validate_only:
        preview: dict[str, Any] = {}
        for ratio in ratios:
            selected_neutral = set(
                selection.loc[
                    selection[f"selected_{ratio}"].astype(bool),
                    "dialogue_id",
                ]
                .astype("string")
                .tolist()
            )
            selected_dialogues = selected_neutral | bypass_ids
            preview[str(ratio)] = {
                "union_dialogues": len(selected_dialogues),
                "union_targets": int(
                    candidate_counts.reindex(list(selected_dialogues)).sum()
                ),
                "neutral_dialogues": len(selected_neutral),
                "neutral_targets": int(
                    candidate_counts.reindex(list(selected_neutral)).sum()
                ),
                "core_dialogues": len(core_ids),
                "summary_dialogues": len(summary_ids),
            }
        if formal_root:
            observed_union = {
                int(ratio): {
                    "dialogues": values["union_dialogues"],
                    "targets": values["union_targets"],
                }
                for ratio, values in preview.items()
            }
            if observed_union != EXPECTED_FORMAL_UNION:
                raise ValidationError(
                    "formal union preview counts drifted; "
                    f"expected={EXPECTED_FORMAL_UNION}, "
                    f"observed={observed_union}"
                )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "validate-only",
                    "formal_outputs_written": False,
                    "selection_sha256": sha256_file(selection_path),
                    "selection_report_sha256": sha256_file(selection_report_path),
                    "primary_ratio": args.primary_ratio,
                    "primary_flag": f"selected_{args.primary_ratio}",
                    "recommended_k": EXPECTED_PRIMARY_K,
                    "production_seed": EXPECTED_PRODUCTION_SEED,
                    "exact_language_isolation": True,
                    "small_cluster_guardrail": "cluster_size<=5_keep_all",
                    "formal_generation_unlocked": (
                        approval_contract is not None
                    ),
                    "stability_contract": stability_contract,
                    "formal_postrun_validation": postrun_contract,
                    "core_protected_contract": {
                        "protected_dialogues": len(protected_dialogue_ids),
                        "core_bypass_dialogues": len(core_ids),
                        "summary_contains_core_dialogues": len(
                            summary_core_ids
                        ),
                        "protected_outside_core_or_summary": 0,
                        "protected_in_neutral": 0,
                        "protected_in_denylist": 0,
                    },
                    "ratios": preview,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if approval_contract is None:
        raise ValidationError("independent builder approval unexpectedly absent")
    staging_dir = prepare_outputs(output_dir, ratios, args.replace_existing)
    output_checksums: dict[str, dict[str, Any]] = {}
    previous_dialogues: set[str] | None = None

    for ratio in ratios:
        selected_neutral = set(
            selection.loc[selection[f"selected_{ratio}"].astype(bool), "dialogue_id"]
            .astype("string")
            .tolist()
        )
        if selected_neutral & deny_ids:
            raise ValidationError(f"ratio {ratio} would restore a denylisted dialogue")
        selected_dialogues = selected_neutral | bypass_ids
        if previous_dialogues is not None and not previous_dialogues.issubset(
            selected_dialogues
        ):
            raise ValidationError(f"retention output is not nested at ratio {ratio}")
        previous_dialogues = selected_dialogues

        dialogue = dialogue_metadata.loc[
            dialogue_metadata["dialogue_id"].isin(selected_dialogues)
        ].copy()
        dialogue["source_partition"] = dialogue["dialogue_id"].map(source_map)
        dialogue["source_partition"] = dialogue["source_partition"].fillna(
            "neutral_mmr"
        )
        dialogue["selection_reason"] = dialogue["source_partition"].map(
            {
                "core_bypass": "mandatory_core_bypass",
                "summary_bypass": "mandatory_effective_summary_bypass",
                "neutral_mmr": f"mmr_selected_{ratio}",
            }
        )
        dialogue["retention_ratio"] = ratio
        dialogue = dialogue.merge(
            neutral_metadata,
            on="dialogue_id",
            how="left",
            validate="one_to_one",
        )
        dialogue["source_partition"] = pd.Categorical(
            dialogue["source_partition"],
            categories=["core_bypass", "summary_bypass", "neutral_mmr"],
            ordered=True,
        )
        dialogue = dialogue.sort_values(
            ["source_partition", "dialogue_id"],
            kind="stable",
        ).reset_index(drop=True)
        dialogue["source_partition"] = dialogue["source_partition"].astype("string")
        validate_unique(dialogue, "dialogue_id", f"dialogue_manifest_{ratio}")

        target = candidate.loc[
            candidate["dialogue_id"].isin(selected_dialogues)
        ].copy()
        target = target.merge(
            dialogue[
                [
                    "dialogue_id",
                    "source_partition",
                    "selection_reason",
                    "retention_ratio",
                    *[
                        column
                        for column in (
                            "cluster_id",
                            "cluster_uid",
                            "mmr_rank",
                            "mmr_score",
                        )
                        if column in dialogue.columns
                    ],
                ]
            ],
            on="dialogue_id",
            how="inner",
            validate="many_to_one",
        )
        target = target.sort_values(
            ["target_increase_id", "target_id"],
            kind="stable",
        ).reset_index(drop=True)
        validate_unique(target, "target_id", f"target_manifest_{ratio}")
        validate_unique(
            target,
            "target_message_id",
            f"target_manifest_{ratio}",
        )

        expected_target_count = int(dialogue["candidate_target_count"].sum())
        if len(target) != expected_target_count:
            raise ValidationError(
                f"ratio {ratio} target count mismatch: rows={len(target)}, "
                f"sum(dialogue candidate_target_count)={expected_target_count}"
            )
        if formal_root:
            observed_ratio = {
                "dialogues": int(len(dialogue)),
                "targets": int(len(target)),
            }
            if observed_ratio != EXPECTED_FORMAL_UNION[ratio]:
                raise ValidationError(
                    f"formal ratio {ratio} counts drifted; expected="
                    f"{EXPECTED_FORMAL_UNION[ratio]}, observed={observed_ratio}"
                )
        if set(target["dialogue_id"].tolist()) != set(dialogue["dialogue_id"].tolist()):
            raise ValidationError(
                f"ratio {ratio} dialogue/target referential integrity failed"
            )
        if bool(target["dialogue_id"].isin(deny_ids).any()):
            raise ValidationError(f"ratio {ratio} target manifest contains denylist")

        dialogue_path = output_dir / f"dialogues_retention_{ratio}.parquet"
        target_path = output_dir / f"targets_retention_{ratio}.parquet"
        dialogue_write_path = (
            staging_dir / f"dialogues_retention_{ratio}.parquet"
        )
        target_write_path = staging_dir / f"targets_retention_{ratio}.parquet"
        atomic_write_parquet(dialogue, dialogue_write_path)
        atomic_write_parquet(target, target_write_path)

        # Read back the materialized files before declaring them valid.
        dialogue_readback = pq.read_table(
            dialogue_write_path,
            columns=[
                "dialogue_id",
                "conversation_id",
                "user_id",
                "character_id",
                "candidate_target_count",
                "source_partition",
            ],
        ).to_pandas()
        target_readback = pq.read_table(
            target_write_path,
            columns=[
                "target_id",
                "target_message_id",
                "dialogue_id",
                "conversation_id",
                "user_id",
                "character_id",
            ],
        ).to_pandas()
        if len(dialogue_readback) != len(dialogue) or len(target_readback) != len(target):
            raise ValidationError(f"ratio {ratio} Parquet readback row count mismatch")
        validate_unique(
            dialogue_readback,
            "dialogue_id",
            f"dialogue_readback_{ratio}",
        )
        validate_unique(target_readback, "target_id", f"target_readback_{ratio}")
        validate_unique(
            target_readback,
            "target_message_id",
            f"target_readback_{ratio}",
        )
        for column in (
            "dialogue_id",
            "conversation_id",
            "user_id",
            "character_id",
        ):
            if dialogue_readback[column].tolist() != dialogue[column].tolist():
                raise ValidationError(
                    f"ratio {ratio} dialogue native ID changed during "
                    f"Parquet round-trip: {column}"
                )
        for column in (
            "target_id",
            "target_message_id",
            "dialogue_id",
            "conversation_id",
            "user_id",
            "character_id",
        ):
            if target_readback[column].tolist() != target[column].tolist():
                raise ValidationError(
                    f"ratio {ratio} target native ID changed during "
                    f"Parquet round-trip: {column}"
                )

        source_dialogues = {
            str(key): int(value)
            for key, value in dialogue["source_partition"].value_counts().items()
        }
        source_targets = {
            str(key): int(value)
            for key, value in target["source_partition"].value_counts().items()
        }
        expected_core_targets = int(candidate_counts.reindex(list(core_ids)).sum())
        expected_summary_targets = int(
            candidate_counts.reindex(list(summary_ids)).sum()
        )
        if source_dialogues.get("core_bypass", 0) != len(core_ids):
            raise ValidationError(f"ratio {ratio} did not retain every core dialogue")
        if source_dialogues.get("summary_bypass", 0) != len(summary_ids):
            raise ValidationError(f"ratio {ratio} did not retain every summary dialogue")
        if source_targets.get("core_bypass", 0) != expected_core_targets:
            raise ValidationError(f"ratio {ratio} did not retain every core target")
        if source_targets.get("summary_bypass", 0) != expected_summary_targets:
            raise ValidationError(f"ratio {ratio} did not retain every summary target")

        dialogue_sha = sha256_file(dialogue_write_path)
        target_sha = sha256_file(target_write_path)
        output_checksums[str(dialogue_path)] = {
            "sha256": dialogue_sha,
            "rows": int(len(dialogue)),
        }
        output_checksums[str(target_path)] = {
            "sha256": target_sha,
            "rows": int(len(target)),
        }
        report["ratios"][str(ratio)] = {
            "dialogues": int(len(dialogue)),
            "targets": int(len(target)),
            "neutral_selected_dialogues": len(selected_neutral),
            "neutral_selected_targets": int(
                candidate_counts.reindex(list(selected_neutral)).sum()
            ),
            "dialogues_by_partition": source_dialogues,
            "targets_by_partition": source_targets,
            "dialogue_target_count_sum": expected_target_count,
            "dialogue_manifest": {
                "path": str(dialogue_path),
                "sha256": dialogue_sha,
            },
            "target_manifest": {
                "path": str(target_path),
                "sha256": target_sha,
            },
            "checks": {
                "bypass_complete": True,
                "bypass_target_complete": True,
                "target_ids_unique": True,
                "target_message_ids_unique": True,
                "target_count_matches_dialogue_sum": True,
                "referential_integrity": True,
                "denylist_absent": True,
            },
        }

    checksum_path = output_dir / "SHA256SUMS.json"
    primary_ratio_report = report["ratios"][str(args.primary_ratio)]
    primary_selection_path = output_dir / "PRIMARY_SELECTION.json"
    atomic_write_json(
        {
            "schema_version": f"{SCHEMA_VERSION}-primary-selection",
            "ratio": args.primary_ratio,
            "flag": f"selected_{args.primary_ratio}",
            "selection_unit": "dialogue_count_per_cluster",
            "k": EXPECTED_PRIMARY_K,
            "production_seed": EXPECTED_PRODUCTION_SEED,
            "exact_language_isolation": True,
            "small_cluster_guardrail": "cluster_size<=5_keep_all",
            "dialogues": primary_ratio_report["dialogues"],
            "targets": primary_ratio_report["targets"],
            "dialogue_manifest": primary_ratio_report["dialogue_manifest"],
            "target_manifest": primary_ratio_report["target_manifest"],
            "diagnostic_target_mass_flags_used": False,
        },
        staging_dir / primary_selection_path.name,
    )
    output_checksums[str(primary_selection_path)] = {
        "sha256": sha256_file(staging_dir / primary_selection_path.name),
        "rows": None,
    }
    report["primary_selection"] = {
        "path": str(primary_selection_path),
        "sha256": sha256_file(staging_dir / primary_selection_path.name),
        "dialogues": primary_ratio_report["dialogues"],
        "targets": primary_ratio_report["targets"],
    }
    atomic_write_json(
        {
            "schema_version": f"{SCHEMA_VERSION}-checksums",
            "inputs": {
                str(selection_path): sha256_file(selection_path),
                str(selection_report_path): sha256_file(selection_report_path),
                str(core_path): sha256_file(core_path),
                str(summary_path): sha256_file(summary_path),
                str(candidate_path): sha256_file(candidate_path),
                str(denylist_path): sha256_file(denylist_path),
                str(recommendation_path): sha256_file(recommendation_path),
                str(stability_validation_path): sha256_file(
                    stability_validation_path
                ),
                str(postrun_validation_path): sha256_file(
                    postrun_validation_path
                ),
                str(approval_path): sha256_file(approval_path),
            },
            "outputs": output_checksums,
        },
        staging_dir / checksum_path.name,
    )
    report["checksums_manifest"] = {
        "path": str(checksum_path),
        "sha256": sha256_file(staging_dir / checksum_path.name),
    }
    report["elapsed_seconds"] = round(time.time() - started, 3)
    report["status"] = "PASS"
    report_path = output_dir / "POST_MDS_UNION_REPORT.json"
    atomic_write_json(report, staging_dir / report_path.name)
    if output_dir.exists():
        raise FileExistsError(
            "formal output appeared during build; refusing non-atomic promote"
        )
    os.replace(staging_dir, output_dir)

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "ratios": {
                    ratio: {
                        "dialogues": report["ratios"][str(ratio)]["dialogues"],
                        "targets": report["ratios"][str(ratio)]["targets"],
                    }
                    for ratio in ratios
                },
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
