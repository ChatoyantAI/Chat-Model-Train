#!/usr/bin/env python3
"""Formal causal-v2, exact-language MiniBatchKMeans for RP dialogue MDS.

This script deliberately does not use ``language_bucket`` from the legacy
embedding index.  It joins the formal neutral embedding universe back to
``dialogue_index.dominant_language`` and clusters every exact language code
independently.  Null/blank/und/other values form one explicit ``unknown``
bucket; no bucket ever shares a KMeans fit with another language.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sklearn
from sklearn.cluster import MiniBatchKMeans


ROOT_DEFAULT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
SEED_DEFAULT = 20260724
MAIN_LANGUAGES = ("en", "es", "pt", "fr", "zh", "de", "ja")
SCHEMA_VERSION = "rp-mds-exact-language-kmeans-causal-v2"


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


def canonical_exact_language(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower().replace("_", "-")
    if not text or text in {"und", "unk", "none", "null", "unknown", "other"}:
        return "unknown"
    return text


def language_order(counts: Counter[str]) -> list[str]:
    other_known = sorted(
        (
            language
            for language in counts
            if language not in MAIN_LANGUAGES and language != "unknown"
        ),
        key=lambda language: (-counts[language], language),
    )
    ordered = [
        language for language in MAIN_LANGUAGES if counts.get(language, 0) > 0
    ]
    ordered.extend(other_known)
    if counts.get("unknown", 0) > 0:
        ordered.append("unknown")
    return ordered


def safe_language_filename(language: str) -> str:
    safe = re.sub(r"[^a-z0-9.-]+", "_", language.lower()).strip("._")
    return safe or "unknown"


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def allocate_k(
    counts: Counter[str],
    requested_total: int,
    minimum_cluster_capacity: int,
) -> dict[str, int]:
    """Capped sqrt allocation with at least one cluster per exact language."""
    languages = language_order(counts)
    if not languages:
        raise ValueError("cannot allocate K for an empty universe")
    caps = {
        language: min(
            max(1, counts[language] // max(1, minimum_cluster_capacity)),
            counts[language],
        )
        for language in languages
    }
    effective_total = min(int(requested_total), sum(caps.values()))
    if effective_total < len(languages):
        raise RuntimeError(
            f"K={requested_total} cannot allocate at least one cluster to "
            f"{len(languages)} exact-language buckets"
        )
    allocation = {language: 1 for language in languages}
    remaining = effective_total - len(languages)
    weights = {language: math.sqrt(float(counts[language])) for language in languages}
    while remaining:
        active = [
            language
            for language in languages
            if allocation[language] < caps[language]
        ]
        if not active:
            break
        weight_sum = sum(weights[language] for language in active)
        quotas = {
            language: (
                remaining * weights[language] / weight_sum
                if weight_sum
                else remaining / len(active)
            )
            for language in active
        }
        base_added = 0
        for language in active:
            amount = min(
                caps[language] - allocation[language],
                int(math.floor(quotas[language])),
            )
            allocation[language] += amount
            base_added += amount
        remaining -= base_added
        if not remaining:
            break
        ranking = sorted(
            active,
            key=lambda language: (
                quotas[language] - math.floor(quotas[language]),
                weights[language],
                language,
            ),
            reverse=True,
        )
        progressed = False
        for language in ranking:
            if not remaining:
                break
            if allocation[language] < caps[language]:
                allocation[language] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("K allocation stalled")
    if remaining or sum(allocation.values()) != effective_total:
        raise RuntimeError("K allocation total mismatch")
    return allocation


def committed_output_for_path(
    assembly_report: dict[str, Any],
    expected_path: Path,
) -> tuple[str, dict[str, Any]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    for name, value in assembly_report.get("outputs", {}).items():
        if not isinstance(value, dict) or not value.get("path"):
            continue
        if Path(str(value["path"])).resolve() == expected_path.resolve():
            matches.append((str(name), value))
    if len(matches) != 1:
        raise RuntimeError(
            f"assembly report must commit exactly one output for {expected_path}; "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_committed_file(
    assembly_report: dict[str, Any],
    expected_path: Path,
) -> dict[str, Any]:
    name, commit = committed_output_for_path(assembly_report, expected_path)
    actual_sha = sha256_file(expected_path)
    if commit.get("sha256") != actual_sha:
        raise RuntimeError(f"assembly checksum mismatch for {name}: {expected_path}")
    return {
        "assembly_output_key": name,
        "path": str(expected_path),
        "sha256": actual_sha,
        "shape": commit.get("shape"),
        "dtype": commit.get("dtype"),
    }


def load_exact_language_rows(
    index: pa.Table,
    dialogue_path: Path,
    denylist_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    dialogue = pq.read_table(
        dialogue_path,
        columns=[
            "dialogue_id",
            "selection_class",
            "contains_core",
            "has_effective_summary",
            "dominant_language",
            "candidate_target_count",
        ],
    )
    neutral = dialogue.filter(
        pc.equal(dialogue["selection_class"], pa.scalar("neutral_cluster"))
    )
    source: dict[str, tuple[str, bool, bool, int]] = {}
    for dialogue_id, language, core, summary, target_count in zip(
        neutral["dialogue_id"].to_pylist(),
        neutral["dominant_language"].to_pylist(),
        neutral["contains_core"].to_pylist(),
        neutral["has_effective_summary"].to_pylist(),
        neutral["candidate_target_count"].to_pylist(),
        strict=True,
    ):
        dialogue_id = str(dialogue_id)
        if dialogue_id in source:
            raise RuntimeError(f"duplicate dialogue_id in dialogue index: {dialogue_id}")
        source[dialogue_id] = (
            canonical_exact_language(language),
            bool(core),
            bool(summary),
            int(target_count),
        )
    denylist = json.loads(denylist_path.read_text(encoding="utf-8"))
    denied = set(str(value) for value in denylist.get("dialogue_ids", []))
    embedding_ids = [str(value) for value in index["dialogue_id"].to_pylist()]
    embedding_set = set(embedding_ids)
    if len(embedding_set) != len(embedding_ids):
        raise RuntimeError("embedding index contains duplicate dialogue IDs")
    missing_source = embedding_set - set(source)
    neutral_not_embedded = set(source) - embedding_set
    if missing_source:
        raise RuntimeError(
            f"embedding universe contains {len(missing_source)} non-neutral dialogues"
        )
    if neutral_not_embedded != denied:
        raise RuntimeError(
            "formal neutral universe differs from dialogue_index only by the committed "
            f"denylist: actual_extra={len(neutral_not_embedded)}, denylist={len(denied)}"
        )
    exact_languages: list[str] = []
    index_target_counts = index["candidate_target_count"].to_pylist()
    for dialogue_id, embedded_target_count in zip(
        embedding_ids,
        index_target_counts,
        strict=True,
    ):
        language, contains_core, has_summary, source_target_count = source[dialogue_id]
        if contains_core or has_summary:
            raise RuntimeError(
                f"core/effective-summary bypass leaked into clustering: {dialogue_id}"
            )
        if int(embedded_target_count) != source_target_count:
            raise RuntimeError(f"candidate_target_count mismatch: {dialogue_id}")
        exact_languages.append(language)
    return np.asarray(exact_languages, dtype=object), {
        "selection_class_required": "neutral_cluster",
        "core_bypass_excluded": True,
        "effective_summary_bypass_excluded": True,
        "neutral_dialogues_before_denylist": neutral.num_rows,
        "denylisted_neutral_dialogues": len(denied),
        "formal_neutral_dialogues": len(embedding_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path(
            "03_embeddings/fused_causal_v2_u745_r005_s25.f16.npy"
        ),
    )
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=Path("03_embeddings/embedding_index.parquet"),
    )
    parser.add_argument(
        "--dialogue-index",
        type=Path,
        default=Path("01_dialogues/dialogue_index.parquet"),
    )
    parser.add_argument(
        "--denylist",
        type=Path,
        default=Path("00_manifest/EMBEDDING_DENYLIST.json"),
    )
    parser.add_argument(
        "--assembly-report",
        type=Path,
        default=Path(
            "00_manifest/CAUSAL_FUSED_U745_R005_S25_ASSEMBLY_REPORT.json"
        ),
    )
    parser.add_argument(
        "--user-embedding",
        type=Path,
        default=Path("03_embeddings/user_trajectory.f16.npy"),
    )
    parser.add_argument(
        "--role-embedding",
        type=Path,
        default=Path("03_embeddings/role.f16.npy"),
    )
    parser.add_argument(
        "--recent-embedding",
        type=Path,
        default=Path("03_embeddings/recent_causal_v2.f16.npy"),
    )
    parser.add_argument("--k-values", default="1000,2000")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-no-improvement", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--minimum-cluster-capacity", type=int, default=20)
    parser.add_argument("--output-tag", default="causal_v2_exactlang")
    args = parser.parse_args()

    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return reject_superseded(
            path if path.is_absolute() else root / path,
            root,
        )

    embedding_path = rooted(args.embedding)
    index_path = rooted(args.embedding_index)
    dialogue_path = rooted(args.dialogue_index)
    denylist_path = rooted(args.denylist)
    assembly_report_path = rooted(args.assembly_report)
    user_path = rooted(args.user_embedding)
    role_path = rooted(args.role_embedding)
    recent_path = rooted(args.recent_embedding)
    if recent_path != (root / "03_embeddings" / "recent_causal_v2.f16.npy").resolve():
        raise RuntimeError(
            "formal causal-v2 clustering must read exactly "
            "03_embeddings/recent_causal_v2.f16.npy"
        )
    output_dir = rooted(root / "04_clusters")
    manifest_dir = rooted(root / "00_manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    k_values = [
        int(value.strip()) for value in args.k_values.split(",") if value.strip()
    ]
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("k-values must contain positive integers")
    if len(k_values) != len(set(k_values)):
        raise ValueError("k-values must be unique")
    if args.seed != SEED_DEFAULT:
        raise ValueError(f"formal causal-v2 seed must be {SEED_DEFAULT}")

    for required in (
        embedding_path,
        index_path,
        dialogue_path,
        denylist_path,
        assembly_report_path,
        user_path,
        role_path,
        recent_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"required formal input is missing: {required}")

    assembly_report = json.loads(assembly_report_path.read_text(encoding="utf-8"))
    assembly_schema = str(assembly_report.get("schema_version", "")).lower()
    if "causal" not in assembly_schema or "v2" not in assembly_schema:
        raise RuntimeError(
            f"assembly report is not a causal-v2 commit marker: {assembly_schema!r}"
        )
    committed_inputs = {
        "fused": validate_committed_file(assembly_report, embedding_path),
        "user": validate_committed_file(assembly_report, user_path),
        "role": validate_committed_file(assembly_report, role_path),
        "recent_causal_v2": validate_committed_file(assembly_report, recent_path),
    }
    if "causal" not in committed_inputs["recent_causal_v2"]["assembly_output_key"]:
        raise RuntimeError(
            "assembly output key for recent_causal_v2 must explicitly contain 'causal'"
        )
    index_sha = sha256_file(index_path)
    index_commit = assembly_report.get("embedding_index", {})
    if index_commit.get("sha256") != index_sha:
        raise RuntimeError("embedding index does not match causal-v2 assembly report")
    if Path(str(index_commit.get("path", ""))).resolve() != index_path:
        raise RuntimeError("assembly report points to a different embedding index")

    embedding = np.load(embedding_path, mmap_mode="r")
    index = pq.read_table(index_path)
    rows = index.num_rows
    expected_fused_shape = committed_inputs["fused"].get("shape")
    if list(embedding.shape) != list(expected_fused_shape or []):
        raise RuntimeError(
            f"fused shape differs from assembly report: {embedding.shape} "
            f"!= {expected_fused_shape}"
        )
    if embedding.ndim != 2 or embedding.shape[0] != rows:
        raise RuntimeError(f"embedding/index row mismatch: {embedding.shape}, {rows}")
    row_indices = np.asarray(
        index["row_index"].to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if not np.array_equal(row_indices, np.arange(rows, dtype=np.int64)):
        raise RuntimeError("embedding index row_index is not contiguous")
    dialogue_ids = [str(value) for value in index["dialogue_id"].to_pylist()]
    target_counts = np.asarray(
        index["candidate_target_count"].to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if np.any(target_counts <= 0):
        raise RuntimeError("candidate_target_count must be positive")
    exact_languages, universe_report = load_exact_language_rows(
        index,
        dialogue_path,
        denylist_path,
    )
    counts: Counter[str] = Counter(str(value) for value in exact_languages)
    languages_in_order = language_order(counts)
    if sum(counts.values()) != rows:
        raise RuntimeError("exact-language census row mismatch")

    expected_outputs: list[Path] = []
    for requested_k in k_values:
        expected_outputs.extend(
            [
                output_dir
                / f"clusters_{args.output_tag}_k{requested_k}.parquet",
                output_dir
                / f"{args.output_tag}_k{requested_k}_centroids",
                manifest_dir
                / f"CLUSTER_{args.output_tag.upper()}_K{requested_k}_REPORT.json",
            ]
        )
    overall_path = (
        manifest_dir / f"CLUSTERING_{args.output_tag.upper()}_REPORT.json"
    )
    expected_outputs.append(overall_path)
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite formal artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    started_all = time.time()
    clustering_script_path = Path(__file__).resolve()
    overall_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": "formal",
        "pipeline_id": args.output_tag,
        "input_provenance": {
            **committed_inputs,
            "embedding_index": {
                "path": str(index_path),
                "sha256": index_sha,
            },
            "dialogue_index": {
                "path": str(dialogue_path),
                "sha256": sha256_file(dialogue_path),
            },
            "denylist": {
                "path": str(denylist_path),
                "sha256": sha256_file(denylist_path),
            },
            "assembly_report": {
                "path": str(assembly_report_path),
                "sha256": sha256_file(assembly_report_path),
                "schema_version": assembly_report.get("schema_version"),
            },
        },
        "forbidden_input_subtree": str((root / "98_superseded").resolve()),
        "superseded_inputs_used": False,
        "rows": rows,
        "candidate_targets": int(target_counts.sum()),
        "fused_shape": list(embedding.shape),
        "language_source": "dialogue_index.dominant_language",
        "language_normalization": (
            "strip + lowercase + '_' to '-'; null/blank/und/unk/other -> unknown"
        ),
        "language_isolation": "one independent KMeans fit per exact language code",
        "cross_language_kmeans": False,
        "exact_language_counts": {
            language: {
                "dialogues": counts[language],
                "candidate_targets": int(
                    target_counts[exact_languages == language].sum()
                ),
            }
            for language in languages_in_order
        },
        "exact_language_bucket_count": len(languages_in_order),
        "universe_guards": universe_report,
        "seed": args.seed,
        "clustering_script": {
            "path": str(clustering_script_path),
            "sha256": sha256_file(clustering_script_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "parameters": {
            "batch_size": args.batch_size,
            "max_iter": args.max_iter,
            "n_init": args.n_init,
            "max_no_improvement": args.max_no_improvement,
            "minimum_cluster_capacity": args.minimum_cluster_capacity,
            "allocation": (
                "sqrt(exact-language dialogue count), largest remainder, "
                "K_lang<=max(1,floor(N_lang/minimum_cluster_capacity))"
            ),
            "distance": (
                "Euclidean on L2-normalized fused vectors "
                "(cosine-equivalent assignment)"
            ),
        },
        "runs": {},
    }

    for requested_k in k_values:
        run_started = time.time()
        allocation = allocate_k(
            counts,
            requested_total=requested_k,
            minimum_cluster_capacity=args.minimum_cluster_capacity,
        )
        labels_all = np.full(rows, -1, dtype=np.int32)
        similarities_all = np.full(rows, np.nan, dtype=np.float32)
        local_labels_all = np.full(rows, -1, dtype=np.int32)
        language_reports: dict[str, Any] = {}
        centroid_path = (
            output_dir / f"{args.output_tag}_k{requested_k}_centroids"
        )
        centroid_temp = output_dir / (
            f".{centroid_path.name}.partial-{os.getpid()}"
        )
        centroid_temp.mkdir(parents=False, exist_ok=False)
        output_path = (
            output_dir / f"clusters_{args.output_tag}_k{requested_k}.parquet"
        )
        output_temp = output_path.with_name(
            f".{output_path.name}.partial-{os.getpid()}"
        )
        global_offset = 0
        try:
            for language in languages_in_order:
                mask_indices = np.flatnonzero(exact_languages == language)
                local_k = allocation[language]
                local_started = time.time()
                vectors = normalize_rows(
                    np.asarray(embedding[mask_indices], dtype=np.float32)
                )
                if not np.isfinite(vectors).all():
                    raise RuntimeError(
                        f"non-finite fused vectors in exact language {language}"
                    )
                model = MiniBatchKMeans(
                    n_clusters=local_k,
                    init="k-means++",
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    batch_size=min(args.batch_size, len(mask_indices)),
                    init_size=min(
                        len(mask_indices),
                        max(3 * local_k, args.batch_size),
                    ),
                    reassignment_ratio=0.01,
                    max_no_improvement=args.max_no_improvement,
                    random_state=args.seed,
                    compute_labels=True,
                    verbose=0,
                )
                local_labels = model.fit_predict(vectors).astype(
                    np.int32,
                    copy=False,
                )
                centroids = normalize_rows(
                    np.asarray(model.cluster_centers_, dtype=np.float32)
                )
                similarities = np.einsum(
                    "ij,ij->i",
                    vectors,
                    centroids[local_labels],
                ).astype(np.float32)
                labels_all[mask_indices] = local_labels + global_offset
                local_labels_all[mask_indices] = local_labels
                similarities_all[mask_indices] = similarities

                local_centroid_path = (
                    centroid_temp
                    / f"{safe_language_filename(language)}.f32.npy"
                )
                with local_centroid_path.open("wb") as handle:
                    np.save(handle, centroids, allow_pickle=False)

                dialogue_sizes = np.bincount(local_labels, minlength=local_k)
                target_sizes = np.bincount(
                    local_labels,
                    weights=target_counts[mask_indices],
                    minlength=local_k,
                )
                language_reports[language] = {
                    "dialogues": int(mask_indices.size),
                    "candidate_targets": int(
                        target_counts[mask_indices].sum()
                    ),
                    "k": int(local_k),
                    "global_cluster_offset": int(global_offset),
                    "inertia": float(model.inertia_),
                    "n_iter": int(model.n_iter_),
                    "n_steps": int(model.n_steps_),
                    "dialogue_cluster_size": quantiles(dialogue_sizes),
                    "target_cluster_size": quantiles(target_sizes),
                    "singleton_clusters": int(np.sum(dialogue_sizes == 1)),
                    "clusters_le_5": int(np.sum(dialogue_sizes <= 5)),
                    "empty_clusters": int(np.sum(dialogue_sizes == 0)),
                    "centroid_similarity": quantiles(similarities),
                    "centroid_file": local_centroid_path.name,
                    "centroid_sha256": sha256_file(local_centroid_path),
                    "elapsed_seconds": round(time.time() - local_started, 3),
                }
                print(
                    f"K={requested_k} exact_language={language} "
                    f"N={len(mask_indices):,} K_lang={local_k} "
                    f"elapsed={language_reports[language]['elapsed_seconds']}s",
                    flush=True,
                )
                global_offset += local_k

            if np.any(labels_all < 0) or np.any(local_labels_all < 0):
                raise RuntimeError(f"K={requested_k} has unassigned rows")
            if not np.isfinite(similarities_all).all():
                raise RuntimeError(
                    f"K={requested_k} has non-finite centroid similarities"
                )
            cluster_language_cardinality = {}
            for cluster_id in np.unique(labels_all):
                cluster_languages = set(
                    exact_languages[labels_all == cluster_id].tolist()
                )
                cluster_language_cardinality[int(cluster_id)] = len(
                    cluster_languages
                )
            if set(cluster_language_cardinality.values()) != {1}:
                raise RuntimeError("cross-language cluster detected")

            cluster_uids = [
                f"{language}:k{requested_k}:{local_cluster:05d}"
                for language, local_cluster in zip(
                    exact_languages,
                    local_labels_all,
                    strict=True,
                )
            ]
            output_table = index
            if "language_bucket" in output_table.column_names:
                output_table = output_table.rename_columns(
                    [
                        (
                            "coarse_language_bucket"
                            if name == "language_bucket"
                            else name
                        )
                        for name in output_table.column_names
                    ]
                )
            output_table = output_table.append_column(
                "exact_language",
                pa.array(exact_languages.tolist(), type=pa.string()),
            )
            output_table = output_table.append_column(
                "cluster_id",
                pa.array(labels_all, type=pa.int32()),
            )
            output_table = output_table.append_column(
                "language_cluster_id",
                pa.array(local_labels_all, type=pa.int32()),
            )
            output_table = output_table.append_column(
                "cluster_uid",
                pa.array(cluster_uids, type=pa.string()),
            )
            output_table = output_table.append_column(
                "centroid_similarity",
                pa.array(similarities_all, type=pa.float32()),
            )
            metadata = dict(output_table.schema.metadata or {})
            metadata.update(
                {
                    b"lusy_schema_version": SCHEMA_VERSION.encode("ascii"),
                    b"lusy_pipeline_id": args.output_tag.encode("ascii"),
                    b"lusy_language_source": b"dialogue_index.dominant_language",
                    b"lusy_language_isolation": b"exact_code_no_cross_language",
                    b"lusy_seed": str(args.seed).encode("ascii"),
                    b"lusy_clustering_script_sha256": sha256_file(
                        clustering_script_path
                    ).encode("ascii"),
                    b"lusy_fused_sha256": sha256_file(
                        embedding_path
                    ).encode("ascii"),
                }
            )
            output_table = output_table.replace_schema_metadata(metadata)
            pq.write_table(
                output_table,
                output_temp,
                compression="zstd",
                compression_level=6,
                use_dictionary=True,
                write_statistics=True,
                row_group_size=65536,
            )
            effective_k = global_offset
            global_dialogue_sizes = np.bincount(
                labels_all,
                minlength=effective_k,
            )
            global_target_sizes = np.bincount(
                labels_all,
                weights=target_counts,
                minlength=effective_k,
            )
            run_report = {
                "schema_version": SCHEMA_VERSION,
                "artifact_status": "formal",
                "requested_k": requested_k,
                "effective_k": effective_k,
                "allocation": allocation,
                "allocation_sum": sum(allocation.values()),
                "exact_language_bucket_count": len(languages_in_order),
                "cross_language_cluster_count": 0,
                "output_path": str(output_path),
                "output_sha256": sha256_file(output_temp),
                "centroid_directory": str(centroid_path),
                "dialogue_cluster_size": quantiles(global_dialogue_sizes),
                "target_cluster_size": quantiles(global_target_sizes),
                "singleton_clusters": int(
                    np.sum(global_dialogue_sizes == 1)
                ),
                "clusters_le_5": int(np.sum(global_dialogue_sizes <= 5)),
                "empty_clusters": int(np.sum(global_dialogue_sizes == 0)),
                "centroid_similarity": quantiles(similarities_all),
                "languages": language_reports,
                "elapsed_seconds": round(time.time() - run_started, 3),
            }
            os.replace(centroid_temp, centroid_path)
            os.replace(output_temp, output_path)
            if run_report["output_sha256"] != sha256_file(output_path):
                raise RuntimeError("cluster parquet checksum changed after rename")
            report_path = (
                manifest_dir
                / f"CLUSTER_{args.output_tag.upper()}_K{requested_k}_REPORT.json"
            )
            atomic_json(report_path, run_report)
            run_report["report_path"] = str(report_path)
            run_report["report_sha256"] = sha256_file(report_path)
            overall_report["runs"][str(requested_k)] = run_report
        except BaseException:
            if output_temp.exists():
                output_temp.unlink()
            if centroid_temp.exists():
                shutil.rmtree(centroid_temp)
            raise

    overall_report["elapsed_seconds"] = round(time.time() - started_all, 3)
    atomic_json(overall_path, overall_report)
    overall_sha_path = overall_path.with_suffix(overall_path.suffix + ".sha256")
    atomic_json(
        overall_sha_path,
        {
            "sha256": sha256_file(overall_path),
            "file": overall_path.name,
        },
    )
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "rows": rows,
                "candidate_targets": int(target_counts.sum()),
                "exact_language_bucket_count": len(languages_in_order),
                "runs": {
                    k: {
                        key: value
                        for key, value in report.items()
                        if key not in {"languages"}
                    }
                    for k, report in overall_report["runs"].items()
                },
                "overall_report": str(overall_path),
                "overall_report_sha256": sha256_file(overall_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
