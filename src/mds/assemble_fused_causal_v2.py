#!/usr/bin/env python3
"""Assemble the formal causal-v2 fused RP dialogue embedding.

The selected weights come only from a frozen machine-readable decision
manifest.  On the formal root, the accepted U/R/index/recent checksums are
pinned so an archived target-inclusive recent view cannot be substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT_DEFAULT = Path("/home/gyr/lusychat-training/mds-dialogue-rp-v1")
SCHEMA_VERSION_FALLBACK = "rp-mds-dialogue-embeddings-causal-v2"
FORMAL_PINS = {
    "user": "4ff83143fa7d7472122dd6d70aae07953c9df164aee3e2d50017f6cb46dbf8f8",
    "role": "e2cab55755083d434a2b61f7a637d0908352fbb1e2e4f12a32a468e75d1c9350",
    "recent_causal_v2": "c6f8547a12a429706b2742208e34864da1162c910b7b3b3448af4556483272fd",
    "embedding_index": "0e01af9e769cf7829b3e57f022dd1ac5420154db34fe67d203be64370e568a1b",
}


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


def array_audit(path: Path, expected_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[0] != expected_rows:
        raise RuntimeError(
            f"invalid embedding shape for {path}: {array.shape}; "
            f"expected rows={expected_rows}"
        )
    if array.dtype != np.float16:
        raise RuntimeError(f"formal view must be float16: {path} -> {array.dtype}")
    norm_min = float("inf")
    norm_max = float("-inf")
    nonfinite = 0
    for start in range(0, expected_rows, 8192):
        block = np.asarray(array[start : start + 8192], dtype=np.float32)
        nonfinite += int((~np.isfinite(block)).sum())
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    if nonfinite:
        raise RuntimeError(f"view contains {nonfinite} non-finite values: {path}")
    if norm_min < 0.98 or norm_max > 1.02:
        raise RuntimeError(
            f"view norms are outside accepted range: {path}, "
            f"min={norm_min}, max={norm_max}"
        )
    return array, {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "nonfinite_count": nonfinite,
        "norm_min": norm_min,
        "norm_max": norm_max,
    }


def validate_referenced_report(reference: Any, root: Path) -> dict[str, Any]:
    if not isinstance(reference, dict):
        raise RuntimeError("weight decision report reference must be an object")
    raw_path = Path(str(reference.get("path", "")))
    path = reject_superseded(
        raw_path if raw_path.is_absolute() else root / raw_path,
        root,
    )
    if not path.is_file():
        raise FileNotFoundError(f"weight decision evidence is missing: {path}")
    actual_sha = sha256_file(path)
    if reference.get("sha256") != actual_sha:
        raise RuntimeError(f"weight decision evidence checksum mismatch: {path}")
    return {"path": str(path), "sha256": actual_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument(
        "--weight-decision",
        type=Path,
        default=Path("00_manifest/CAUSAL_FUSION_WEIGHT_DECISION.json"),
    )
    parser.add_argument(
        "--user",
        type=Path,
        default=Path("03_embeddings/user_trajectory.f16.npy"),
    )
    parser.add_argument(
        "--role",
        type=Path,
        default=Path("03_embeddings/role.f16.npy"),
    )
    parser.add_argument(
        "--recent",
        type=Path,
        default=Path("03_embeddings/recent_causal_v2.f16.npy"),
    )
    parser.add_argument(
        "--embedding-index",
        type=Path,
        default=Path("03_embeddings/embedding_index.parquet"),
    )
    parser.add_argument(
        "--causal-embedding-report",
        type=Path,
        default=Path(
            "00_manifest/causal_recent_v2/CAUSAL_RECENT_EMBEDDING_REPORT.json"
        ),
    )
    parser.add_argument(
        "--causal-formal-validation",
        type=Path,
        default=Path(
            "00_manifest/causal_recent_v2/FORMAL_CAUSAL_RECENT_V2_VALIDATION.json"
        ),
    )
    parser.add_argument(
        "--reuse-proof",
        type=Path,
        default=Path("00_manifest/causal_recent_v2/CAUSAL_REUSE_PROOF.json"),
    )
    parser.add_argument(
        "--model-provenance",
        type=Path,
        default=Path("00_manifest/BGE_M3_MODEL_PROVENANCE.json"),
    )
    parser.add_argument(
        "--decision-validation",
        type=Path,
        default=Path(
            "00_manifest/CAUSAL_FUSION_WEIGHT_DECISION_VALIDATION.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return reject_superseded(
            path if path.is_absolute() else root / path,
            root,
        )

    decision_path = rooted(args.weight_decision)
    user_path = rooted(args.user)
    role_path = rooted(args.role)
    recent_path = rooted(args.recent)
    index_path = rooted(args.embedding_index)
    causal_report_path = rooted(args.causal_embedding_report)
    formal_validation_path = rooted(args.causal_formal_validation)
    reuse_proof_path = rooted(args.reuse_proof)
    model_provenance_path = rooted(args.model_provenance)
    decision_validation_path = rooted(args.decision_validation)
    if recent_path != (root / "03_embeddings" / "recent_causal_v2.f16.npy").resolve():
        raise RuntimeError(
            "formal fusion must read exactly "
            "03_embeddings/recent_causal_v2.f16.npy"
        )
    for required in (
        decision_path,
        user_path,
        role_path,
        recent_path,
        index_path,
        causal_report_path,
        formal_validation_path,
        reuse_proof_path,
        model_provenance_path,
        decision_validation_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    started = time.time()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision_schema = str(decision.get("schema_version", "")).lower()
    if "causal" not in decision_schema or "v2" not in decision_schema:
        raise RuntimeError("weight decision is not a causal-v2 manifest")
    if decision.get("artifact_status") != "final":
        raise RuntimeError("weight decision must be frozen with artifact_status=final")
    raw_weights = decision.get("weights", {})
    if set(raw_weights) not in (
        {"user", "role", "recent"},
        {"user", "role", "recent", "sum"},
    ):
        raise RuntimeError(
            "weight decision must contain user/role/recent and optional sum"
        )
    weights = {
        name: float(raw_weights[name])
        for name in ("user", "role", "recent")
    }
    if any(value < 0.0 or value > 1.0 for value in weights.values()):
        raise RuntimeError(f"fusion weights are outside [0,1]: {weights}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise RuntimeError(f"fusion weights do not sum to 1: {weights}")
    if "sum" in raw_weights and abs(float(raw_weights["sum"]) - 1.0) > 1e-9:
        raise RuntimeError("weight decision's explicit sum is not 1")
    fusion_contract = decision.get("fusion_contract", {})
    if not isinstance(fusion_contract, dict):
        raise RuntimeError("weight decision lacks fusion_contract")
    contract_output = Path(
        str(fusion_contract.get("final_fused_path", ""))
    )
    contract_report = Path(
        str(fusion_contract.get("assembly_report_path", ""))
    )
    output_path = rooted(
        args.output if args.output is not None else contract_output
    )
    report_path = rooted(
        args.report if args.report is not None else contract_report
    )
    if output_path != reject_superseded(contract_output, root):
        raise RuntimeError("CLI fused output differs from frozen fusion contract")
    if report_path != reject_superseded(contract_report, root):
        raise RuntimeError("CLI assembly report differs from frozen fusion contract")
    expected_schema_version = str(
        fusion_contract.get(
            "assembly_report_schema",
            SCHEMA_VERSION_FALLBACK,
        )
    )
    if "causal" not in expected_schema_version.lower() or "v2" not in (
        expected_schema_version.lower()
    ):
        raise RuntimeError("fusion contract assembly schema is not causal-v2")
    contract_rows = int(fusion_contract.get("row_count", -1))
    contract_output_dimension = int(
        fusion_contract.get("output_dimension", -1)
    )
    if contract_rows <= 0 or contract_output_dimension <= 0:
        raise RuntimeError("fusion contract has non-positive dimensions")
    if fusion_contract.get("output_dtype") != "float16":
        raise RuntimeError("fusion contract output dtype is not float16")
    input_dimensions = fusion_contract.get("input_dimensions", {})
    contract_input_dimensions = {
        name: int(input_dimensions.get(name, -1))
        for name in ("user", "role", "recent")
    }
    if any(value <= 0 for value in contract_input_dimensions.values()):
        raise RuntimeError("fusion contract input dimensions are invalid")
    if contract_output_dimension != sum(contract_input_dimensions.values()):
        raise RuntimeError(
            "fusion contract output dimension is not the concatenated input size"
        )
    if root == ROOT_DEFAULT.resolve():
        if contract_rows != 185040:
            raise RuntimeError("formal fusion contract row count is not 185040")
        if contract_output_dimension != 3072:
            raise RuntimeError(
                "formal fusion contract output dimension is not 3072"
            )
        if set(contract_input_dimensions.values()) != {1024}:
            raise RuntimeError(
                "formal fusion contract input dimensions are not all 1024"
            )
    if output_path.exists() or report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite formal output: {output_path} or {report_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    recent_decision_sha = decision.get("recent_embedding_sha256")
    if recent_decision_sha != sha256_file(recent_path):
        raise RuntimeError(
            "weight decision was calibrated against a different recent embedding"
        )
    decision_evidence = {}
    for key in (
        "calibration_report",
        "calibration_markdown",
        "source_grid_report",
        "human_audit",
        "human_audit_markdown",
        "formal_causal_validation",
    ):
        if key in decision:
            decision_evidence[key] = validate_referenced_report(
                decision[key],
                root,
            )
    if "calibration_report" not in decision_evidence:
        raise RuntimeError("final weight decision lacks calibration_report evidence")
    if "human_audit" not in decision_evidence:
        raise RuntimeError("final weight decision lacks human_audit evidence")
    decision_recent_path = reject_superseded(
        Path(str(decision.get("recent_embedding_path", ""))),
        root,
    )
    if decision_recent_path != recent_path:
        raise RuntimeError("weight decision names a different recent embedding")
    decision_index_path = reject_superseded(
        Path(str(decision.get("embedding_index_path", ""))),
        root,
    )
    if decision_index_path != index_path:
        raise RuntimeError("weight decision names a different embedding index")
    if decision.get("embedding_index_sha256") != sha256_file(index_path):
        raise RuntimeError("weight decision embedding-index checksum mismatch")
    causal_views_path = reject_superseded(
        Path(str(decision.get("causal_views_path", ""))),
        root,
    )
    if not causal_views_path.is_file():
        raise FileNotFoundError("weight decision causal views are missing")
    if decision.get("causal_views_sha256") != sha256_file(causal_views_path):
        raise RuntimeError("weight decision causal-views checksum mismatch")
    decision_validation = json.loads(
        decision_validation_path.read_text(encoding="utf-8")
    )
    if decision_validation.get("pass") is not True:
        raise RuntimeError("independent weight-decision validation is not PASS")
    if not all(decision_validation.get("checks", {}).values()):
        raise RuntimeError(
            "independent weight-decision validation has a failed check"
        )
    if (
        decision_validation.get("decision_manifest_sha256")
        != sha256_file(decision_path)
    ):
        raise RuntimeError(
            "weight decision changed after independent validation"
        )
    formal_read_paths = {
        decision_path,
        user_path,
        role_path,
        recent_path,
        index_path,
        causal_report_path,
        formal_validation_path,
        reuse_proof_path,
        model_provenance_path,
        decision_validation_path,
        causal_views_path,
        *(
            Path(reference["path"]).resolve()
            for reference in decision_evidence.values()
        ),
    }
    for forbidden_raw in decision.get("forbidden_inputs", []):
        forbidden = Path(str(forbidden_raw)).resolve()
        for read_path in formal_read_paths:
            if read_path == forbidden:
                raise RuntimeError(f"formal input is explicitly forbidden: {read_path}")
            if forbidden.is_dir():
                try:
                    read_path.relative_to(forbidden)
                except ValueError:
                    pass
                else:
                    raise RuntimeError(
                        f"formal input is under forbidden subtree: {read_path}"
                    )

    index = pq.read_table(
        index_path,
        columns=["row_index", "dialogue_id", "candidate_target_count"],
    )
    rows = index.num_rows
    row_indices = np.asarray(
        index["row_index"].to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if not np.array_equal(row_indices, np.arange(rows, dtype=np.int64)):
        raise RuntimeError("embedding index row_index is not contiguous")
    dialogue_ids = [str(value) for value in index["dialogue_id"].to_pylist()]
    if len(set(dialogue_ids)) != rows:
        raise RuntimeError("embedding index contains duplicate dialogue IDs")
    if np.any(
        np.asarray(
            index["candidate_target_count"].to_numpy(zero_copy_only=False)
        )
        <= 0
    ):
        raise RuntimeError("embedding index has non-positive target counts")

    user, user_audit = array_audit(user_path, rows)
    role, role_audit = array_audit(role_path, rows)
    recent, recent_audit = array_audit(recent_path, rows)
    if not (user.shape == role.shape == recent.shape):
        raise RuntimeError(
            f"view shape mismatch: user={user.shape}, role={role.shape}, "
            f"recent={recent.shape}"
        )
    view_dimension = int(user.shape[1])
    if rows != contract_rows:
        raise RuntimeError("actual row count differs from fusion contract")
    if any(
        dimension != view_dimension
        for dimension in contract_input_dimensions.values()
    ):
        raise RuntimeError("actual view dimension differs from fusion contract")
    input_actual = {
        "user": user_audit["sha256"],
        "role": role_audit["sha256"],
        "recent_causal_v2": recent_audit["sha256"],
        "embedding_index": sha256_file(index_path),
    }
    if root == ROOT_DEFAULT.resolve() and input_actual != FORMAL_PINS:
        raise RuntimeError(
            "formal input pin mismatch; accepted U/R/recent/index were not reused"
        )

    causal_report = json.loads(causal_report_path.read_text(encoding="utf-8"))
    if causal_report.get("schema_version") != "rp-mds-causal-recent-embedding-v2":
        raise RuntimeError("causal embedding report schema mismatch")
    if causal_report.get("pass") is not True:
        raise RuntimeError("causal embedding report is not PASS")
    if causal_report.get("target_assistant_excluded") is not True:
        raise RuntimeError("causal embedding report does not exclude target assistant")
    if causal_report.get("embedding_sha256") != recent_audit["sha256"]:
        raise RuntimeError("causal embedding report checksum mismatch")
    if int(causal_report.get("rows", -1)) != rows:
        raise RuntimeError("causal embedding report row mismatch")
    formal_validation = json.loads(
        formal_validation_path.read_text(encoding="utf-8")
    )
    if formal_validation.get("pass") is not True:
        raise RuntimeError("formal causal recent validation is not PASS")
    if not all(formal_validation.get("checks", {}).values()):
        raise RuntimeError("formal causal recent validation has a failed gate")
    if (
        decision_evidence["formal_causal_validation"]["path"]
        != str(formal_validation_path)
    ):
        raise RuntimeError(
            "decision's formal causal validation differs from configured validation"
        )
    reuse_proof = json.loads(reuse_proof_path.read_text(encoding="utf-8"))
    if reuse_proof.get("pass") is not True:
        raise RuntimeError("U/R causal reuse proof is not PASS")
    if not all(reuse_proof.get("checks", {}).values()):
        raise RuntimeError("U/R causal reuse proof has a failed gate")
    model_provenance = json.loads(
        model_provenance_path.read_text(encoding="utf-8")
    )
    if model_provenance.get("repository") != "BAAI/bge-m3":
        raise RuntimeError("unexpected embedding model repository")
    if (
        model_provenance.get("revision")
        != "5617a9f61b028005a4858fdac845db406aefb181"
    ):
        raise RuntimeError("unexpected BGE-M3 model revision")
    if model_provenance.get("formal_runs_use_local_files_only") is not True:
        raise RuntimeError("model provenance does not enforce local-files-only")

    fused_shape = (rows, 3 * view_dimension)
    output_temp = output_path.with_name(
        f".{output_path.name}.partial-{os.getpid()}"
    )
    fused = np.lib.format.open_memmap(
        output_temp,
        mode="w+",
        dtype=np.float16,
        shape=fused_shape,
    )
    sqrt_weights = {
        name: np.sqrt(value) for name, value in weights.items()
    }
    fused_norm_min = float("inf")
    fused_norm_max = float("-inf")
    fused_nonfinite = 0
    try:
        for start in range(0, rows, args.chunk_size):
            end = min(rows, start + args.chunk_size)
            user_block = np.asarray(user[start:end], dtype=np.float32)
            role_block = np.asarray(role[start:end], dtype=np.float32)
            recent_block = np.asarray(recent[start:end], dtype=np.float32)
            user_block /= np.maximum(
                np.linalg.norm(user_block, axis=1, keepdims=True),
                1e-12,
            )
            role_block /= np.maximum(
                np.linalg.norm(role_block, axis=1, keepdims=True),
                1e-12,
            )
            recent_block /= np.maximum(
                np.linalg.norm(recent_block, axis=1, keepdims=True),
                1e-12,
            )
            block = np.concatenate(
                [
                    sqrt_weights["user"] * user_block,
                    sqrt_weights["role"] * role_block,
                    sqrt_weights["recent"] * recent_block,
                ],
                axis=1,
            )
            block /= np.maximum(
                np.linalg.norm(block, axis=1, keepdims=True),
                1e-12,
            )
            fused_nonfinite += int((~np.isfinite(block)).sum())
            norms = np.linalg.norm(block, axis=1)
            fused_norm_min = min(fused_norm_min, float(norms.min()))
            fused_norm_max = max(fused_norm_max, float(norms.max()))
            fused[start:end] = block.astype(np.float16)
            if start % (20 * args.chunk_size) == 0:
                print(f"fused rows {end:,}/{rows:,}", flush=True)
        fused.flush()
        del fused
        if fused_nonfinite:
            raise RuntimeError(
                f"fused construction produced {fused_nonfinite} non-finite values"
            )
        os.replace(output_temp, output_path)
    except BaseException:
        try:
            del fused
        except UnboundLocalError:
            pass
        if output_temp.exists():
            output_temp.unlink()
        raise

    fused_array, fused_audit = array_audit(output_path, rows)
    if fused_array.shape != fused_shape:
        raise RuntimeError("fused output shape changed after atomic rename")
    del fused_array
    script_path = Path(__file__).resolve()
    report = {
        "schema_version": expected_schema_version,
        "pipeline_schema_alias": SCHEMA_VERSION_FALLBACK,
        "artifact_status": "formal",
        "dialogues": rows,
        "dimension_per_view": view_dimension,
        "fused_dimension": 3 * view_dimension,
        "fusion": {
            "formula": (
                "concat(sqrt(w_user)*L2(U), sqrt(w_role)*L2(R), "
                "sqrt(w_recent)*L2(S_causal_v2)), then L2 normalize"
            ),
            "weights": weights,
            "weight_decision": {
                "path": str(decision_path),
                "sha256": sha256_file(decision_path),
                "schema_version": decision.get("schema_version"),
                "chosen_label": decision.get("chosen_label"),
                "evidence": decision_evidence,
            },
            "frozen_contract": fusion_contract,
        },
        "input_guards": {
            "accepted_formal_pins_enforced": root == ROOT_DEFAULT.resolve(),
            "target_assistant_excluded": True,
            "recent_source_required": str(
                root / "03_embeddings" / "recent_causal_v2.f16.npy"
            ),
            "superseded_inputs_used": False,
            "forbidden_input_subtree": str(
                (root / "98_superseded").resolve()
            ),
            "causal_embedding_report": {
                "path": str(causal_report_path),
                "sha256": sha256_file(causal_report_path),
                "pass": True,
            },
            "formal_causal_validation": {
                "path": str(formal_validation_path),
                "sha256": sha256_file(formal_validation_path),
                "pass": True,
            },
            "user_role_reuse_proof": {
                "path": str(reuse_proof_path),
                "sha256": sha256_file(reuse_proof_path),
                "pass": True,
            },
            "embedding_model_provenance": {
                "path": str(model_provenance_path),
                "sha256": sha256_file(model_provenance_path),
                "repository": model_provenance.get("repository"),
                "revision": model_provenance.get("revision"),
                "local_files_only": True,
            },
            "weight_decision_independent_validation": {
                "path": str(decision_validation_path),
                "sha256": sha256_file(decision_validation_path),
                "pass": True,
            },
        },
        "outputs": {
            "user": user_audit,
            "role": role_audit,
            "recent_causal_v2": recent_audit,
            "fused": fused_audit,
        },
        "embedding_index": {
            "path": str(index_path),
            "sha256": input_actual["embedding_index"],
            "rows": rows,
        },
        "assembly_script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
        },
        "fused_precast_norm_min": fused_norm_min,
        "fused_precast_norm_max": fused_norm_max,
        "fused_precast_nonfinite_count": fused_nonfinite,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json(report_path, report)
    checksum_path = report_path.with_suffix(report_path.suffix + ".sha256")
    checksum_temp = checksum_path.with_name(
        f".{checksum_path.name}.partial-{os.getpid()}"
    )
    checksum_temp.write_text(
        f"{sha256_file(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    os.replace(checksum_temp, checksum_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
