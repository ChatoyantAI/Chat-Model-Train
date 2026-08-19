#!/usr/bin/env python3
"""Full deterministic MMR ranking for formal causal-v2 exact-language clusters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT_DEFAULT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
SEED_DEFAULT = 20260724
RATIOS = (50, 60, 70, 80)
SCHEMA_VERSION = "rp-mds-mmr-exact-language-causal-v2"
CLUSTER_SCHEMA_VERSION = "rp-mds-exact-language-kmeans-causal-v2"


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def reject_superseded(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    superseded = (root / "98_superseded").resolve()
    try:
        resolved.relative_to(superseded)
    except ValueError:
        return resolved
    raise RuntimeError(f"superseded input/output is forbidden: {resolved}")


def stable_u64(value: str, seed: int) -> int:
    payload = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "big",
    )


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def mmr_prefix(
    vectors: np.ndarray,
    anchor_similarity: np.ndarray,
    dialogue_ids: list[str],
    lambda_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a full deterministic local ranking and its MMR diagnostics."""
    size = len(dialogue_ids)
    if size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    if len(vectors) != size or len(anchor_similarity) != size:
        raise ValueError("MMR input row-count mismatch")
    if len(set(dialogue_ids)) != size:
        raise ValueError("dialogue IDs must be unique inside a cluster")
    stable_order = np.asarray(
        sorted(
            range(size),
            key=lambda row: (
                stable_u64(dialogue_ids[row], seed),
                dialogue_ids[row],
            ),
        ),
        dtype=np.int64,
    )
    vectors = normalize_rows(
        np.asarray(vectors[stable_order], dtype=np.float32)
    )
    anchor_similarity = np.asarray(
        anchor_similarity[stable_order],
        dtype=np.float32,
    )
    if not np.isfinite(vectors).all() or not np.isfinite(
        anchor_similarity
    ).all():
        raise ValueError("MMR inputs contain NaN or infinity")
    pair_similarity = vectors @ vectors.T
    selected = np.zeros(size, dtype=bool)
    maximum_similarity = np.full(size, -1.0, dtype=np.float32)
    order: list[int] = []
    scores: list[float] = []
    redundancies: list[float] = []
    while len(order) < size:
        if not order:
            candidate_scores = lambda_value * anchor_similarity
        else:
            candidate_scores = (
                lambda_value * anchor_similarity
                - (1.0 - lambda_value) * maximum_similarity
            )
        candidate_scores[selected] = -np.inf
        chosen = int(np.argmax(candidate_scores))
        if not np.isfinite(candidate_scores[chosen]):
            raise RuntimeError("MMR ranking terminated before all rows were ranked")
        order.append(chosen)
        scores.append(float(candidate_scores[chosen]))
        redundancies.append(
            -1.0 if len(order) == 1 else float(maximum_similarity[chosen])
        )
        selected[chosen] = True
        maximum_similarity = np.maximum(
            maximum_similarity,
            pair_similarity[:, chosen],
        )
    return (
        stable_order[np.asarray(order, dtype=np.int64)].astype(np.int32),
        np.asarray(scores, dtype=np.float32),
        np.asarray(redundancies, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--k", type=int, required=True, choices=(1000, 2000))
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path(
            "03_embeddings/fused_causal_v2_u745_r005_s25.f16.npy"
        ),
    )
    parser.add_argument(
        "--cluster",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--clustering-report",
        type=Path,
        default=Path(
            "00_manifest/CLUSTERING_CAUSAL_V2_EXACTLANG_REPORT.json"
        ),
    )
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--keep-all-small", type=int, default=5)
    parser.add_argument("--output-tag", default="causal_v2_exactlang")
    args = parser.parse_args()

    if args.seed != SEED_DEFAULT:
        raise ValueError(f"formal causal-v2 seed must be {SEED_DEFAULT}")
    if args.lambda_value != 0.5:
        raise ValueError("formal causal-v2 MMR lambda must be exactly 0.5")
    if args.keep_all_small != 5:
        raise ValueError("formal causal-v2 small-cluster threshold must be 5")

    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return reject_superseded(
            path if path.is_absolute() else root / path,
            root,
        )

    embedding_path = rooted(args.embedding)
    cluster_path = rooted(
        args.cluster
        if args.cluster is not None
        else Path(
            f"04_clusters/clusters_{args.output_tag}_k{args.k}.parquet"
        )
    )
    clustering_report_path = rooted(args.clustering_report)
    output_dir = rooted(root / "05_selection")
    manifest_dir = rooted(root / "00_manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"selection_mmr_{args.output_tag}_k{args.k}.parquet"
    )
    report_path = (
        manifest_dir
        / f"MMR_SELECTION_{args.output_tag.upper()}_K{args.k}_REPORT.json"
    )
    checksum_path = report_path.with_suffix(report_path.suffix + ".sha256")
    existing = [
        path
        for path in (output_path, report_path, checksum_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite formal artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    for required in (
        embedding_path,
        cluster_path,
        clustering_report_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"required formal input is missing: {required}")

    started = time.time()
    selection_script_path = Path(__file__).resolve()
    selection_script_sha = sha256_file(selection_script_path)
    embedding_sha = sha256_file(embedding_path)
    cluster_sha = sha256_file(cluster_path)
    clustering_report_sha = sha256_file(clustering_report_path)
    clustering_report = json.loads(
        clustering_report_path.read_text(encoding="utf-8")
    )
    if clustering_report.get("schema_version") != CLUSTER_SCHEMA_VERSION:
        raise RuntimeError("clustering report is not the formal causal-v2 schema")
    if clustering_report.get("artifact_status") != "formal":
        raise RuntimeError("clustering report is not marked formal")
    if clustering_report.get("cross_language_kmeans") is not False:
        raise RuntimeError("clustering report does not prohibit cross-language KMeans")
    if clustering_report.get("superseded_inputs_used") is not False:
        raise RuntimeError("clustering report used a superseded input")
    committed_fused = (
        clustering_report.get("input_provenance", {}).get("fused", {})
    )
    if committed_fused.get("sha256") != embedding_sha:
        raise RuntimeError("fused embedding differs from clustering provenance")
    if Path(str(committed_fused.get("path", ""))).resolve() != embedding_path:
        raise RuntimeError("clustering report commits a different fused path")
    run = clustering_report.get("runs", {}).get(str(args.k))
    if not isinstance(run, dict):
        raise RuntimeError(f"clustering report has no K={args.k} run")
    if run.get("output_sha256") != cluster_sha:
        raise RuntimeError("cluster parquet differs from clustering commit marker")
    if Path(str(run.get("output_path", ""))).resolve() != cluster_path:
        raise RuntimeError("clustering report points to a different cluster path")
    if int(run.get("cross_language_cluster_count", -1)) != 0:
        raise RuntimeError("cluster run reports cross-language clusters")

    embedding = np.load(embedding_path, mmap_mode="r")
    cluster_table = pq.read_table(cluster_path)
    rows = cluster_table.num_rows
    if embedding.ndim != 2 or embedding.shape[0] != rows:
        raise RuntimeError(
            f"embedding/cluster row mismatch: {embedding.shape}, {rows}"
        )
    required_columns = {
        "row_index",
        "dialogue_id",
        "candidate_target_count",
        "exact_language",
        "cluster_id",
        "cluster_uid",
        "centroid_similarity",
    }
    missing_columns = required_columns - set(cluster_table.column_names)
    if missing_columns:
        raise RuntimeError(
            f"cluster artifact is missing columns: {sorted(missing_columns)}"
        )
    metadata = cluster_table.schema.metadata or {}
    if metadata.get(b"lusy_schema_version", b"").decode(
        "utf-8"
    ) != CLUSTER_SCHEMA_VERSION:
        raise RuntimeError("cluster parquet schema metadata is not causal-v2")
    if metadata.get(b"lusy_language_isolation") != (
        b"exact_code_no_cross_language"
    ):
        raise RuntimeError("cluster parquet does not commit exact-language isolation")
    if metadata.get(b"lusy_fused_sha256", b"").decode(
        "ascii"
    ) != embedding_sha:
        raise RuntimeError("cluster parquet was derived from another fused embedding")

    row_indices = np.asarray(
        cluster_table["row_index"].to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if not np.array_equal(row_indices, np.arange(rows, dtype=np.int64)):
        raise RuntimeError("cluster rows are not in embedding order")
    dialogue_ids = [
        str(value) for value in cluster_table["dialogue_id"].to_pylist()
    ]
    if len(set(dialogue_ids)) != rows:
        raise RuntimeError("cluster artifact contains duplicate dialogue IDs")
    exact_languages = np.asarray(
        cluster_table["exact_language"].to_pylist(),
        dtype=object,
    )
    labels = np.asarray(
        cluster_table["cluster_id"].to_numpy(zero_copy_only=False),
        dtype=np.int32,
    )
    if np.any(labels < 0):
        raise RuntimeError("cluster artifact contains negative cluster IDs")
    target_weights = np.asarray(
        cluster_table["candidate_target_count"].to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int64,
    )
    if np.any(target_weights <= 0):
        raise RuntimeError("candidate_target_count must be positive")
    anchor_similarity = np.asarray(
        cluster_table["centroid_similarity"].to_numpy(
            zero_copy_only=False
        ),
        dtype=np.float32,
    )
    if not np.isfinite(anchor_similarity).all():
        raise RuntimeError("centroid similarity contains NaN or infinity")
    if np.any(anchor_similarity < -1.001) or np.any(
        anchor_similarity > 1.001
    ):
        raise RuntimeError("centroid similarity is outside the cosine range")
    unique_labels = np.unique(labels)
    configured_clusters = int(run.get("effective_k", -1))
    empty_clusters = int(run.get("empty_clusters", 0))
    if len(unique_labels) != configured_clusters - empty_clusters:
        raise RuntimeError("non-empty cluster count differs from commit marker")
    cluster_languages: dict[int, str] = {}
    for label in unique_labels:
        values = set(exact_languages[labels == label].tolist())
        if len(values) != 1:
            raise RuntimeError(
                f"cross-language MMR cluster detected: {int(label)} -> {values}"
            )
        cluster_languages[int(label)] = str(next(iter(values)))

    rank = np.full(rows, -1, dtype=np.int32)
    mmr_score = np.full(rows, np.nan, dtype=np.float32)
    prior_max_similarity = np.full(rows, np.nan, dtype=np.float32)
    cumulative_dialogue_fraction = np.zeros(rows, dtype=np.float32)
    selected_masks = {
        ratio: np.zeros(rows, dtype=bool) for ratio in RATIOS
    }
    paper_alpha_50_exact = np.zeros(rows, dtype=bool)
    cluster_stats: list[dict[str, Any]] = []
    language_selected: dict[str, dict[str, int]] = {}

    for cluster_number, label in enumerate(unique_labels, start=1):
        member_rows = np.flatnonzero(labels == label)
        local_order, scores, redundancies = mmr_prefix(
            vectors=np.asarray(embedding[member_rows], dtype=np.float32),
            anchor_similarity=anchor_similarity[member_rows],
            dialogue_ids=[dialogue_ids[row] for row in member_rows],
            lambda_value=args.lambda_value,
            seed=args.seed,
        )
        ordered_rows = member_rows[local_order]
        if len(ordered_rows) != len(member_rows):
            raise RuntimeError(f"incomplete MMR order for cluster {int(label)}")
        rank[ordered_rows] = np.arange(
            1,
            len(ordered_rows) + 1,
            dtype=np.int32,
        )
        mmr_score[ordered_rows] = scores
        prior_max_similarity[ordered_rows] = redundancies
        cumulative_dialogue_fraction[ordered_rows] = (
            np.arange(1, len(ordered_rows) + 1, dtype=np.float32)
            / len(ordered_rows)
        )
        paper_count = int(np.ceil(0.5 * len(member_rows)))
        paper_alpha_50_exact[ordered_rows[:paper_count]] = True
        selected_counts: dict[str, int] = {}
        selected_targets: dict[str, int] = {}
        small_cluster_guardrail = len(member_rows) <= args.keep_all_small
        language = cluster_languages[int(label)]
        language_selected.setdefault(
            language,
            {
                "input_dialogues": 0,
                "input_candidate_targets": 0,
                **{f"selected_{ratio}_dialogues": 0 for ratio in RATIOS},
                **{
                    f"selected_{ratio}_candidate_targets": 0
                    for ratio in RATIOS
                },
            },
        )
        language_selected[language]["input_dialogues"] += len(member_rows)
        language_selected[language]["input_candidate_targets"] += int(
            target_weights[member_rows].sum()
        )
        for ratio in RATIOS:
            count = (
                len(member_rows)
                if small_cluster_guardrail
                else int(np.ceil(len(member_rows) * ratio / 100.0))
            )
            chosen_rows = ordered_rows[:count]
            selected_masks[ratio][chosen_rows] = True
            selected_counts[str(ratio)] = count
            chosen_targets = int(target_weights[chosen_rows].sum())
            selected_targets[str(ratio)] = chosen_targets
            language_selected[language][
                f"selected_{ratio}_dialogues"
            ] += count
            language_selected[language][
                f"selected_{ratio}_candidate_targets"
            ] += chosen_targets
        cluster_stats.append(
            {
                "cluster_id": int(label),
                "exact_language": language,
                "dialogues": int(len(member_rows)),
                "candidate_targets": int(target_weights[member_rows].sum()),
                "ranked_dialogues": int(len(ordered_rows)),
                "small_cluster_keep_all": small_cluster_guardrail,
                "paper_alpha_50_exact_dialogues": paper_count,
                "selected_dialogues": selected_counts,
                "selected_candidate_targets": selected_targets,
            }
        )
        if cluster_number % 100 == 0 or cluster_number == len(unique_labels):
            print(
                f"MMR K={args.k} clusters "
                f"{cluster_number:,}/{len(unique_labels):,}",
                flush=True,
            )

    if not np.all(rank > 0) or not np.isfinite(mmr_score).all():
        raise RuntimeError("not every dialogue received a finite full MMR rank")
    if not np.isfinite(prior_max_similarity[rank > 1]).all():
        raise RuntimeError("non-first MMR redundancy contains NaN or infinity")
    for label in unique_labels:
        cluster_rank = np.sort(rank[labels == label])
        expected_rank = np.arange(
            1,
            len(cluster_rank) + 1,
            dtype=np.int32,
        )
        if not np.array_equal(cluster_rank, expected_rank):
            raise RuntimeError(f"rank is not 1..N for cluster {int(label)}")
    for ratio in RATIOS:
        if np.any(selected_masks[ratio] & (rank <= 0)):
            raise RuntimeError(f"selected_{ratio} contains an unranked row")
        for label in unique_labels:
            member = labels == label
            selected_rank = np.sort(rank[member & selected_masks[ratio]])
            if selected_rank.size and not np.array_equal(
                selected_rank,
                np.arange(1, selected_rank.size + 1, dtype=np.int32),
            ):
                raise RuntimeError(
                    f"selected_{ratio} is not a rank prefix in cluster {int(label)}"
                )

    output_table = cluster_table.append_column(
        "mmr_rank",
        pa.array(rank, type=pa.int32()),
    )
    output_table = output_table.append_column(
        "mmr_score",
        pa.array(mmr_score, type=pa.float32()),
    )
    output_table = output_table.append_column(
        "max_similarity_to_previous",
        pa.array(prior_max_similarity, type=pa.float32()),
    )
    output_table = output_table.append_column(
        "cumulative_dialogue_fraction_in_cluster",
        pa.array(cumulative_dialogue_fraction, type=pa.float32()),
    )
    for ratio in RATIOS:
        output_table = output_table.append_column(
            f"selected_{ratio}",
            pa.array(selected_masks[ratio], type=pa.bool_()),
        )
    output_table = output_table.append_column(
        "paper_alpha_50_exact",
        pa.array(paper_alpha_50_exact, type=pa.bool_()),
    )
    output_metadata = dict(output_table.schema.metadata or {})
    output_metadata.update(
        {
            b"lusy_schema_version": SCHEMA_VERSION.encode("ascii"),
            b"lusy_selection_unit": b"dialogue_count_prefix_within_cluster",
            b"lusy_selection_guardrail": b"clusters_le_5_keep_all",
            b"lusy_primary_column": b"selected_60",
            b"lusy_paper_exact_column": b"paper_alpha_50_exact",
            b"lusy_language_isolation": b"exact_code_no_cross_language",
            b"lusy_selection_script_sha256": selection_script_sha.encode(
                "ascii"
            ),
            b"lusy_fused_sha256": embedding_sha.encode("ascii"),
            b"lusy_cluster_sha256": cluster_sha.encode("ascii"),
        }
    )
    output_table = output_table.replace_schema_metadata(output_metadata)
    output_temp = output_path.with_name(
        f".{output_path.name}.partial-{os.getpid()}"
    )
    pq.write_table(
        output_table,
        output_temp,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=65536,
    )
    output_sha = sha256_file(output_temp)
    os.replace(output_temp, output_path)
    if output_sha != sha256_file(output_path):
        raise RuntimeError("MMR parquet checksum changed after atomic rename")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": "formal",
        "pipeline_id": args.output_tag,
        "k": args.k,
        "lambda": args.lambda_value,
        "seed": args.seed,
        "selection_script": {
            "path": str(selection_script_path),
            "sha256": selection_script_sha,
        },
        "primary_selection_column": "selected_60",
        "paper_exact_column": "paper_alpha_50_exact",
        "input_provenance": {
            "fused_embedding": {
                "path": str(embedding_path),
                "sha256": embedding_sha,
            },
            "clusters": {
                "path": str(cluster_path),
                "sha256": cluster_sha,
            },
            "clustering_report": {
                "path": str(clustering_report_path),
                "sha256": clustering_report_sha,
                "schema_version": clustering_report.get("schema_version"),
            },
        },
        "forbidden_input_subtree": str((root / "98_superseded").resolve()),
        "superseded_inputs_used": False,
        "language_isolation": "exact code; every cluster has cardinality 1",
        "cross_language_mmr": False,
        "exact_language_bucket_count": len(set(exact_languages.tolist())),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
        },
        "keep_all_small_clusters_le_dialogues": args.keep_all_small,
        "ranking_scope": (
            "within each exact-language, language-stratified semantic cluster"
        ),
        "ranking_formula": (
            "lambda*cosine(dialogue,centroid) - "
            "(1-lambda)*max_cosine(dialogue,previously_selected)"
        ),
        "first_item_redundancy_contribution": 0.0,
        "first_item_redundancy_output_sentinel": -1.0,
        "ranking_completeness": "all dialogues receive rank 1..cluster_size",
        "retention_semantics": {
            "selected_50_60_70_80": (
                "if cluster_size<=5 keep all; otherwise "
                "ceil(ratio*dialogue_count); always retain whole dialogues"
            ),
            "paper_alpha_50_exact": (
                "ceil(0.5*dialogue_count) without the <=5 keep-all guardrail"
            ),
        },
        "input_rows": rows,
        "input_candidate_targets": int(target_weights.sum()),
        "clusters_configured": configured_clusters,
        "clusters_empty": empty_clusters,
        "clusters_nonempty_ranked": int(len(unique_labels)),
        "all_rows_ranked": True,
        "all_mmr_scores_finite": True,
        "all_nonfirst_redundancies_finite": True,
        "memory_complexity": {
            "algorithm": "per-cluster dense pairwise cosine matrix",
            "asymptotic_peak": (
                "O(max_cluster_dialogues^2 + "
                "max_cluster_dialogues*embedding_dim)"
            ),
            "max_cluster_dialogues": int(
                max(stat["dialogues"] for stat in cluster_stats)
            ),
            "pairwise_matrix_peak_bytes_float32": int(
                4 * max(stat["dialogues"] for stat in cluster_stats) ** 2
            ),
            "cluster_vector_peak_bytes_float32": int(
                4
                * max(stat["dialogues"] for stat in cluster_stats)
                * embedding.shape[1]
            ),
        },
        "ratios": {},
        "paper_alpha_50_exact": {},
        "exact_language_selection": language_selected,
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "elapsed_seconds": round(time.time() - started, 3),
        "cluster_stats": cluster_stats,
    }
    for ratio in RATIOS:
        mask = selected_masks[ratio]
        report["ratios"][str(ratio)] = {
            "selected_dialogues": int(mask.sum()),
            "selected_dialogue_fraction": float(mask.mean()),
            "selected_candidate_targets": int(target_weights[mask].sum()),
            "selected_target_fraction": float(
                target_weights[mask].sum() / target_weights.sum()
            ),
            "clusters_covered": int(len(np.unique(labels[mask]))),
        }
    report["paper_alpha_50_exact"] = {
        "selected_dialogues": int(paper_alpha_50_exact.sum()),
        "selected_dialogue_fraction": float(paper_alpha_50_exact.mean()),
        "selected_candidate_targets": int(
            target_weights[paper_alpha_50_exact].sum()
        ),
        "selected_target_fraction": float(
            target_weights[paper_alpha_50_exact].sum()
            / target_weights.sum()
        ),
        "clusters_covered": int(
            len(np.unique(labels[paper_alpha_50_exact]))
        ),
    }
    atomic_json(report_path, report)
    checksum_temp = checksum_path.with_name(
        f".{checksum_path.name}.partial-{os.getpid()}"
    )
    checksum_temp.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    os.replace(checksum_temp, checksum_path)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"cluster_stats", "exact_language_selection"}
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
