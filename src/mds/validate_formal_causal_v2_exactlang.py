#!/usr/bin/env python3
"""Independent post-run validation for formal causal-v2 exact-language MDS."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
RATIOS = (50, 60, 70, 80)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        value = bool(condition)
        checks[name] = value
        if not value:
            failures.append(name)

    decision_path = ROOT / "00_manifest" / "CAUSAL_FUSION_WEIGHT_DECISION.json"
    decision_validation_path = (
        ROOT
        / "00_manifest"
        / "CAUSAL_FUSION_WEIGHT_DECISION_VALIDATION.json"
    )
    assembly_path = (
        ROOT
        / "00_manifest"
        / "CAUSAL_FUSED_U745_R005_S25_ASSEMBLY_REPORT.json"
    )
    clustering_path = (
        ROOT
        / "00_manifest"
        / "CLUSTERING_CAUSAL_V2_EXACTLANG_REPORT.json"
    )
    census_path = (
        ROOT / "07_validation" / "EXACT_LANGUAGE_CENSUS_CAUSAL_V2.json"
    )
    decision = json.loads(decision_path.read_text())
    decision_validation = json.loads(decision_validation_path.read_text())
    assembly = json.loads(assembly_path.read_text())
    clustering = json.loads(clustering_path.read_text())
    census = json.loads(census_path.read_text())

    check("decision_status_final", decision.get("artifact_status") == "final")
    check(
        "decision_validation_pass",
        decision_validation.get("pass") is True
        and all(decision_validation.get("checks", {}).values()),
    )
    check(
        "decision_validation_sha_current",
        decision_validation.get("decision_manifest_sha256")
        == sha256_file(decision_path),
    )
    weights = {
        name: float(decision["weights"][name])
        for name in ("user", "role", "recent")
    }
    check(
        "frozen_weights_exact",
        weights == {"user": 0.745, "role": 0.005, "recent": 0.25},
    )
    check("frozen_weights_sum_one", abs(sum(weights.values()) - 1) <= 1e-12)
    check(
        "assembly_schema",
        assembly.get("schema_version")
        == "lusy-rp-mds-causal-v2-fused-assembly-v1",
    )
    check("assembly_status_formal", assembly.get("artifact_status") == "formal")
    check(
        "assembly_decision_sha",
        assembly.get("fusion", {})
        .get("weight_decision", {})
        .get("sha256")
        == sha256_file(decision_path),
    )
    check(
        "assembly_independent_decision_validation",
        assembly.get("input_guards", {})
        .get("weight_decision_independent_validation", {})
        .get("pass")
        is True,
    )
    check(
        "assembly_target_excluded",
        assembly.get("input_guards", {}).get("target_assistant_excluded")
        is True,
    )
    check(
        "assembly_superseded_false",
        assembly.get("input_guards", {}).get("superseded_inputs_used")
        is False,
    )

    output_commits = assembly["outputs"]
    view_paths = {
        "user": Path(output_commits["user"]["path"]),
        "role": Path(output_commits["role"]["path"]),
        "recent": Path(output_commits["recent_causal_v2"]["path"]),
        "fused": Path(output_commits["fused"]["path"]),
    }
    expected_active = {
        "user": ROOT / "03_embeddings" / "user_trajectory.f16.npy",
        "role": ROOT / "03_embeddings" / "role.f16.npy",
        "recent": ROOT / "03_embeddings" / "recent_causal_v2.f16.npy",
        "fused": (
            ROOT
            / "03_embeddings"
            / "fused_causal_v2_u745_r005_s25.f16.npy"
        ),
    }
    for name, path in view_paths.items():
        check(f"{name}_formal_active_path", path.resolve() == expected_active[name])
        commit_key = "recent_causal_v2" if name == "recent" else name
        check(
            f"{name}_checksum",
            output_commits[commit_key]["sha256"] == sha256_file(path),
        )
        check(
            f"{name}_not_superseded",
            ROOT.joinpath("98_superseded").resolve()
            not in path.resolve().parents,
        )
        check(
            f"{name}_not_stage",
            ROOT.joinpath("99_smoke").resolve() not in path.resolve().parents,
        )

    user = np.load(view_paths["user"], mmap_mode="r")
    role = np.load(view_paths["role"], mmap_mode="r")
    recent = np.load(view_paths["recent"], mmap_mode="r")
    fused = np.load(view_paths["fused"], mmap_mode="r")
    check(
        "formal_shapes",
        user.shape
        == role.shape
        == recent.shape
        == (185040, 1024)
        and fused.shape == (185040, 3072),
    )
    check(
        "formal_dtypes_float16",
        user.dtype
        == role.dtype
        == recent.dtype
        == fused.dtype
        == np.float16,
    )
    sample_rng = np.random.default_rng(20260724)
    sample_indices = np.sort(
        sample_rng.choice(len(fused), size=4096, replace=False)
    )
    sample_blocks = []
    for name, array in (("user", user), ("role", role), ("recent", recent)):
        block = np.asarray(array[sample_indices], dtype=np.float32)
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        sample_blocks.append(np.sqrt(weights[name]) * block)
    expected_fused = np.concatenate(sample_blocks, axis=1)
    expected_fused /= np.maximum(
        np.linalg.norm(expected_fused, axis=1, keepdims=True),
        1e-12,
    )
    expected_fused = expected_fused.astype(np.float16)
    actual_fused = np.asarray(fused[sample_indices])
    check("fused_formula_bit_exact_4096", np.array_equal(expected_fused, actual_fused))
    sample_max_abs = float(
        np.max(
            np.abs(
                expected_fused.astype(np.float32)
                - actual_fused.astype(np.float32)
            )
        )
    )
    fused_nonfinite = 0
    fused_norm_min = float("inf")
    fused_norm_max = float("-inf")
    for start in range(0, len(fused), 8192):
        block = np.asarray(fused[start : start + 8192], dtype=np.float32)
        fused_nonfinite += int((~np.isfinite(block)).sum())
        norms = np.linalg.norm(block, axis=1)
        fused_norm_min = min(fused_norm_min, float(norms.min()))
        fused_norm_max = max(fused_norm_max, float(norms.max()))
    check("fused_all_finite", fused_nonfinite == 0)
    check(
        "fused_norms_valid",
        fused_norm_min >= 0.999
        and fused_norm_max <= 1.001,
    )

    check(
        "clustering_schema",
        clustering.get("schema_version")
        == "rp-mds-exact-language-kmeans-causal-v2",
    )
    check("clustering_status_formal", clustering.get("artifact_status") == "formal")
    check("clustering_cross_language_false", clustering.get("cross_language_kmeans") is False)
    check("clustering_superseded_false", clustering.get("superseded_inputs_used") is False)
    check("clustering_rows", clustering.get("rows") == 185040)
    check("clustering_targets", clustering.get("candidate_targets") == 625008)
    check("clustering_languages_21", clustering.get("exact_language_bucket_count") == 21)
    check(
        "clustering_fused_sha",
        clustering.get("input_provenance", {}).get("fused", {}).get("sha256")
        == sha256_file(view_paths["fused"]),
    )
    check(
        "clustering_script_current",
        clustering.get("clustering_script", {}).get("sha256")
        == sha256_file(
            ROOT / "scripts" / "cluster_rp_mds_causal_v2_exactlang.py"
        ),
    )
    check(
        "census_counts_match_clustering",
        clustering.get("exact_language_counts")
        == census.get("language_counts"),
    )

    cluster_summaries: dict[str, Any] = {}
    selection_summaries: dict[str, Any] = {}
    selected_60_sets: dict[int, set[str]] = {}
    for k in (1000, 2000):
        run = clustering["runs"][str(k)]
        run_report_path = Path(run["report_path"])
        run_report = json.loads(run_report_path.read_text())
        cluster_path = Path(run["output_path"])
        check(f"k{k}_run_report_sha", run["report_sha256"] == sha256_file(run_report_path))
        check(f"k{k}_cluster_sha", run["output_sha256"] == sha256_file(cluster_path))
        check(f"k{k}_effective_k", run["effective_k"] == k)
        check(f"k{k}_allocation_sum", sum(run["allocation"].values()) == k)
        check(
            f"k{k}_allocation_matches_census",
            run["allocation"] == census["allocations"][str(k)],
        )
        check(f"k{k}_cross_language_report_zero", run["cross_language_cluster_count"] == 0)

        table = pq.read_table(cluster_path)
        check(f"k{k}_cluster_rows", table.num_rows == 185040)
        meta = table.schema.metadata or {}
        check(
            f"k{k}_cluster_schema_metadata",
            meta.get(b"lusy_schema_version", b"").decode()
            == "rp-mds-exact-language-kmeans-causal-v2",
        )
        check(
            f"k{k}_cluster_language_metadata",
            meta.get(b"lusy_language_isolation")
            == b"exact_code_no_cross_language",
        )
        row_indices = np.asarray(table["row_index"].to_numpy(), dtype=np.int64)
        dialogue_ids = np.asarray(table["dialogue_id"].to_pylist(), dtype=object)
        target_counts = np.asarray(
            table["candidate_target_count"].to_numpy(),
            dtype=np.int64,
        )
        exact_languages = np.asarray(
            table["exact_language"].to_pylist(),
            dtype=object,
        )
        labels = np.asarray(table["cluster_id"].to_numpy(), dtype=np.int32)
        local_labels = np.asarray(
            table["language_cluster_id"].to_numpy(),
            dtype=np.int32,
        )
        centroid_similarity = np.asarray(
            table["centroid_similarity"].to_numpy(),
            dtype=np.float32,
        )
        check(
            f"k{k}_row_index_contiguous",
            np.array_equal(row_indices, np.arange(185040)),
        )
        check(f"k{k}_dialogue_ids_unique", len(set(dialogue_ids)) == 185040)
        check(f"k{k}_candidate_target_total", int(target_counts.sum()) == 625008)
        check(f"k{k}_centroid_similarity_finite", np.isfinite(centroid_similarity).all())
        observed_counts = Counter(str(value) for value in exact_languages)
        expected_counts = {
            language: values["dialogues"]
            for language, values in census["language_counts"].items()
        }
        check(f"k{k}_exact_language_counts", dict(observed_counts) == expected_counts)
        cross_language_clusters = 0
        for label in np.unique(labels):
            if len(set(exact_languages[labels == label].tolist())) != 1:
                cross_language_clusters += 1
        check(f"k{k}_cross_language_clusters_zero", cross_language_clusters == 0)
        unique_nonempty = len(np.unique(labels))
        check(
            f"k{k}_nonempty_cluster_count",
            unique_nonempty == k - int(run["empty_clusters"]),
        )
        for language, allocated in run["allocation"].items():
            language_local = local_labels[exact_languages == language]
            check(
                f"k{k}_{language}_local_label_bounds",
                bool(language_local.size)
                and int(language_local.min()) >= 0
                and int(language_local.max()) < int(allocated),
            )
        cluster_summaries[str(k)] = {
            "output_path": str(cluster_path),
            "output_sha256": run["output_sha256"],
            "elapsed_seconds": run["elapsed_seconds"],
            "effective_k": run["effective_k"],
            "nonempty_clusters": unique_nonempty,
            "empty_clusters": run["empty_clusters"],
            "singleton_clusters": run["singleton_clusters"],
            "clusters_le_5_including_empty": run["clusters_le_5"],
            "cross_language_clusters": cross_language_clusters,
            "centroid_similarity": run["centroid_similarity"],
        }

        selection_report_path = (
            ROOT
            / "00_manifest"
            / f"MMR_SELECTION_CAUSAL_V2_EXACTLANG_K{k}_REPORT.json"
        )
        selection_report = json.loads(selection_report_path.read_text())
        selection_path = Path(selection_report["output_path"])
        check(
            f"k{k}_selection_sha",
            selection_report["output_sha256"] == sha256_file(selection_path),
        )
        check(
            f"k{k}_selection_script_current",
            selection_report["selection_script"]["sha256"]
            == sha256_file(
                ROOT
                / "scripts"
                / "select_rp_mds_mmr_causal_v2_exactlang.py"
            ),
        )
        check(f"k{k}_all_rows_ranked_report", selection_report["all_rows_ranked"] is True)
        check(f"k{k}_cross_language_mmr_report", selection_report["cross_language_mmr"] is False)
        selected_table = pq.read_table(selection_path)
        selected_meta = selected_table.schema.metadata or {}
        check(
            f"k{k}_selection_schema_metadata",
            selected_meta.get(b"lusy_schema_version", b"").decode()
            == "rp-mds-mmr-exact-language-causal-v2",
        )
        check(
            f"k{k}_selection_guardrail_metadata",
            selected_meta.get(b"lusy_selection_guardrail")
            == b"clusters_le_5_keep_all",
        )
        selected_dialogue_ids = np.asarray(
            selected_table["dialogue_id"].to_pylist(),
            dtype=object,
        )
        selected_labels = np.asarray(
            selected_table["cluster_id"].to_numpy(),
            dtype=np.int32,
        )
        selected_languages = np.asarray(
            selected_table["exact_language"].to_pylist(),
            dtype=object,
        )
        rank = np.asarray(selected_table["mmr_rank"].to_numpy(), dtype=np.int32)
        scores = np.asarray(selected_table["mmr_score"].to_numpy(), dtype=np.float32)
        redundancy = np.asarray(
            selected_table["max_similarity_to_previous"].to_numpy(),
            dtype=np.float32,
        )
        check(
            f"k{k}_selection_row_identity",
            np.array_equal(selected_dialogue_ids, dialogue_ids)
            and np.array_equal(selected_labels, labels)
            and np.array_equal(selected_languages, exact_languages),
        )
        check(f"k{k}_rank_positive", np.all(rank > 0))
        check(f"k{k}_scores_finite", np.isfinite(scores).all())
        check(
            f"k{k}_nonfirst_redundancy_finite",
            np.isfinite(redundancy[rank > 1]).all(),
        )
        masks = {
            ratio: np.asarray(
                selected_table[f"selected_{ratio}"].to_numpy(),
                dtype=bool,
            )
            for ratio in RATIOS
        }
        paper = np.asarray(
            selected_table["paper_alpha_50_exact"].to_numpy(),
            dtype=bool,
        )
        check(
            f"k{k}_selected_masks_nested",
            np.all(~masks[50] | masks[60])
            and np.all(~masks[60] | masks[70])
            and np.all(~masks[70] | masks[80]),
        )
        cluster_rank_complete = True
        selection_prefix_complete = True
        small_keep_all_complete = True
        paper_exact_complete = True
        for label in np.unique(selected_labels):
            member = selected_labels == label
            member_rank = np.sort(rank[member])
            size = int(member.sum())
            if not np.array_equal(member_rank, np.arange(1, size + 1)):
                cluster_rank_complete = False
            for ratio in RATIOS:
                expected_count = (
                    size
                    if size <= 5
                    else int(math.ceil(size * ratio / 100))
                )
                chosen_rank = np.sort(rank[member & masks[ratio]])
                if not np.array_equal(
                    chosen_rank,
                    np.arange(1, expected_count + 1),
                ):
                    selection_prefix_complete = False
                if size <= 5 and expected_count != size:
                    small_keep_all_complete = False
            expected_paper = int(math.ceil(0.5 * size))
            paper_rank = np.sort(rank[member & paper])
            if not np.array_equal(
                paper_rank,
                np.arange(1, expected_paper + 1),
            ):
                paper_exact_complete = False
        check(f"k{k}_cluster_rank_1_to_n", cluster_rank_complete)
        check(f"k{k}_selected_prefix_exact", selection_prefix_complete)
        check(f"k{k}_small_clusters_all_kept", small_keep_all_complete)
        check(f"k{k}_paper_alpha_exact", paper_exact_complete)
        ratio_summary = {}
        for ratio in RATIOS:
            mask = masks[ratio]
            committed = selection_report["ratios"][str(ratio)]
            actual_dialogues = int(mask.sum())
            actual_targets = int(target_counts[mask].sum())
            check(
                f"k{k}_selected_{ratio}_dialogues_report",
                actual_dialogues == committed["selected_dialogues"],
            )
            check(
                f"k{k}_selected_{ratio}_targets_report",
                actual_targets == committed["selected_candidate_targets"],
            )
            ratio_summary[str(ratio)] = {
                "dialogues": actual_dialogues,
                "candidate_targets": actual_targets,
                "dialogue_fraction": actual_dialogues / 185040,
                "target_fraction": actual_targets / 625008,
            }
        selected_60_sets[k] = set(selected_dialogue_ids[masks[60]].tolist())
        selection_summaries[str(k)] = {
            "output_path": str(selection_path),
            "output_sha256": selection_report["output_sha256"],
            "elapsed_seconds": selection_report["elapsed_seconds"],
            "all_rows_ranked": True,
            "cross_language_mmr": False,
            "ratios": ratio_summary,
            "paper_alpha_50_exact": selection_report["paper_alpha_50_exact"],
        }

    overlap = selected_60_sets[1000] & selected_60_sets[2000]
    union = selected_60_sets[1000] | selected_60_sets[2000]
    selection_overlap = {
        "k1000_selected_60": len(selected_60_sets[1000]),
        "k2000_selected_60": len(selected_60_sets[2000]),
        "intersection": len(overlap),
        "union": len(union),
        "jaccard": len(overlap) / len(union),
        "overlap_over_smaller": len(overlap)
        / min(len(selected_60_sets[1000]), len(selected_60_sets[2000])),
    }
    check("selected_60_overlap_nontrivial", selection_overlap["jaccard"] > 0.4)

    core_dialogues = pq.read_table(
        ROOT / "08_core_bypass" / "core_dialogues.parquet",
        columns=["dialogue_id", "candidate_target_count"],
    )
    summary_dialogues = pq.read_table(
        ROOT / "08_core_bypass" / "summary_dialogues.parquet",
        columns=["dialogue_id", "candidate_target_count"],
    )
    bypass = {
        "core_dialogues": core_dialogues.num_rows,
        "core_targets": int(
            np.asarray(core_dialogues["candidate_target_count"].to_numpy()).sum()
        ),
        "summary_dialogues": summary_dialogues.num_rows,
        "summary_targets": int(
            np.asarray(summary_dialogues["candidate_target_count"].to_numpy()).sum()
        ),
    }
    union_projection = {}
    for k in (1000, 2000):
        neutral = selection_summaries[str(k)]["ratios"]["60"]
        union_projection[str(k)] = {
            "selected_60_plus_bypass_dialogues": (
                neutral["dialogues"]
                + bypass["core_dialogues"]
                + bypass["summary_dialogues"]
            ),
            "selected_60_plus_bypass_targets": (
                neutral["candidate_targets"]
                + bypass["core_targets"]
                + bypass["summary_targets"]
            ),
        }

    report = {
        "schema_version": (
            "rp-mds-formal-causal-v2-exact-language-postrun-validation-v1"
        ),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failed_checks": failures,
        "decision": {
            "path": str(decision_path),
            "sha256": sha256_file(decision_path),
            "chosen_label": decision["chosen_label"],
            "weights": weights,
        },
        "assembly": {
            "path": str(assembly_path),
            "sha256": sha256_file(assembly_path),
            "elapsed_seconds": assembly["elapsed_seconds"],
            "fused": output_commits["fused"],
            "sample_formula_max_abs_error": sample_max_abs,
            "fused_full_nonfinite_count": fused_nonfinite,
            "fused_full_norm_min": fused_norm_min,
            "fused_full_norm_max": fused_norm_max,
        },
        "clusters": cluster_summaries,
        "selections": selection_summaries,
        "selected_60_cross_k_overlap": selection_overlap,
        "bypass_not_clustered": bypass,
        "selected_60_union_projection": union_projection,
    }
    output_dir = ROOT / "07_validation" / "formal_causal_v2_exactlang_postrun"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "VALIDATION_REPORT.json"
    json_temp = json_path.with_name(json_path.name + ".partial")
    json_temp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(json_temp, json_path)
    markdown_path = output_dir / "VALIDATION_REPORT.md"
    lines = [
        "# Formal causal-v2 exact-language MDS validation",
        "",
        f"- Status: **{report['status']}**",
        f"- Checks: {sum(checks.values())}/{len(checks)} passed",
        f"- Failed: {', '.join(failures) if failures else 'none'}",
        f"- Frozen weights: U={weights['user']}, R={weights['role']}, S={weights['recent']}",
        (
            f"- Fused: `{output_commits['fused']['sha256']}`, "
            f"shape={output_commits['fused']['shape']}, "
            f"formula sample max abs error={sample_max_abs}"
        ),
        "",
        "## KMeans",
        "",
    ]
    for k in ("1000", "2000"):
        value = cluster_summaries[k]
        lines.append(
            f"- K={k}: nonempty={value['nonempty_clusters']}, "
            f"empty={value['empty_clusters']}, "
            f"cross-language={value['cross_language_clusters']}, "
            f"elapsed={value['elapsed_seconds']}s, "
            f"sha=`{value['output_sha256']}`"
        )
    lines.extend(["", "## MMR selected_60", ""])
    for k in ("1000", "2000"):
        value = selection_summaries[k]
        ratio = value["ratios"]["60"]
        lines.append(
            f"- K={k}: dialogues={ratio['dialogues']:,}, "
            f"targets={ratio['candidate_targets']:,}, "
            f"elapsed={value['elapsed_seconds']}s, "
            f"sha=`{value['output_sha256']}`"
        )
    lines.extend(
        [
            (
                f"- Cross-K Jaccard={selection_overlap['jaccard']:.6f}, "
                f"overlap/smaller={selection_overlap['overlap_over_smaller']:.6f}"
            ),
            "",
            "## Projected union at selected_60 + bypass",
            "",
            (
                f"- K=1000: {union_projection['1000']['selected_60_plus_bypass_dialogues']:,} "
                f"dialogues / {union_projection['1000']['selected_60_plus_bypass_targets']:,} targets"
            ),
            (
                f"- K=2000: {union_projection['2000']['selected_60_plus_bypass_dialogues']:,} "
                f"dialogues / {union_projection['2000']['selected_60_plus_bypass_targets']:,} targets"
            ),
            "",
        ]
    )
    markdown_temp = markdown_path.with_name(markdown_path.name + ".partial")
    markdown_temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(markdown_temp, markdown_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
