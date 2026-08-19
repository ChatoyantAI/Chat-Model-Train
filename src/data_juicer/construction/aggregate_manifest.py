#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import orjson


SCHEMA_VERSION = "lusy-rp-dialogue-data-juicer-v2.2"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, known_hash: str | None = None) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": known_hash or sha256_file(path),
    }


def merge_counts(rows: list[dict], key: str) -> dict[str, int]:
    result = collections.Counter()
    for row in rows:
        result.update(row.get(key) or {})
    return dict(sorted(result.items()))


def validate_and_count_dialogues(paths: list[Path]) -> dict:
    counts = collections.Counter()
    soft = collections.Counter()
    relations = collections.Counter()
    recommendations = collections.Counter()
    min_turns = None
    max_turns = 0
    for path in paths:
        bucket_ids: set[str] = set()
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                row = orjson.loads(line)
                cid = row.get("conversation_id")
                if not cid or cid in bucket_ids:
                    raise RuntimeError(f"missing/duplicate id: {path}:{line_number}:{cid}")
                bucket_ids.add(cid)
                if row.get("schema_version") != SCHEMA_VERSION:
                    raise RuntimeError(f"schema mismatch: {path}:{line_number}")
                messages = row.get("messages") or []
                if not messages or messages[0].get("role") != "system":
                    raise RuntimeError(f"messages prefix mismatch: {path}:{line_number}")
                if any(not str(message.get("content") or "").strip() for message in messages):
                    raise RuntimeError(f"empty message: {path}:{line_number}")
                contract = row.get("data_juicer_contract") or {}
                if contract.get("primary_field") != "messages":
                    raise RuntimeError(f"Data-Juicer contract missing: {path}:{line_number}")
                turns = int(row.get("turn_count") or 0)
                min_turns = turns if min_turns is None else min(min_turns, turns)
                max_turns = max(max_turns, turns)
                counts["dialogues"] += 1
                counts["messages"] += len(messages)
                counts["retained_turns"] += turns
                counts["dialogue_chars"] += int(row.get("dialogue_chars") or 0)
                policy = row.get("source_policy_audit") or {}
                for field in (
                    "conversation_is_inactive",
                    "source_user_is_deleted_user",
                    "requires_privacy_and_minor_scrub",
                    "minor_nsfw_hard_exclusion",
                    "requires_content_safety_and_pii_scrub",
                ):
                    soft[field] += int(bool(policy.get(field)))
                lineage = row.get("lineage") or {}
                relations[str(lineage.get("relation_to_parent") or "missing")] += 1
                recommendations[
                    str(lineage.get("dedup_recommendation") or "missing")
                ] += 1
                counts["fork_dialogues"] += int(bool(lineage.get("is_fork_conversation")))
                counts["dialogues_with_fork_children"] += int(
                    int(lineage.get("fork_child_count") or 0) > 0
                )
    return {
        **dict(counts),
        "min_turns": min_turns,
        "max_turns": max_turns,
        "soft_audit_flags": dict(sorted(soft.items())),
        "lineage_relations": dict(sorted(relations.items())),
        "lineage_recommendations": dict(sorted(recommendations.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--identity-snapshot", type=Path, required=True)
    parser.add_argument("--lineage-index", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--expected-buckets", type=int, default=256)
    parser.add_argument("--expected-source-rows", type=int, default=1_477_539)
    args = parser.parse_args()

    report_paths = sorted((args.root / "reports").glob("bucket-*.report.json"))
    dialogue_paths = sorted((args.root / "dialogues").glob("bucket-*.jsonl"))
    quarantine_paths = sorted((args.root / "quarantine").glob("bucket-*.quarantine.jsonl"))
    if not (
        len(report_paths)
        == len(dialogue_paths)
        == len(quarantine_paths)
        == args.expected_buckets
    ):
        raise SystemExit("expected exactly 256 report/dialogue/quarantine files")

    reports = [orjson.loads(path.read_bytes()) for path in report_paths]
    if [int(row["bucket"]) for row in reports] != list(range(args.expected_buckets)):
        raise SystemExit("bucket reports are not exactly 000..255")
    processed = sum(int(row["processed_rows"]) for row in reports)
    accepted = sum(int(row["accepted_rows"]) for row in reports)
    quarantined = sum(int(row["quarantined_rows"]) for row in reports)
    if processed != args.expected_source_rows or processed != accepted + quarantined:
        raise SystemExit(
            f"row accounting failed: {processed=} {accepted=} {quarantined=}"
        )
    for report in reports:
        for section in ("output", "quarantine"):
            path = Path(report[section]["path"])
            if not path.is_file() or path.stat().st_size != int(
                report[section]["size_bytes"]
            ):
                raise SystemExit(f"artifact missing or size mismatch: {path}")

    dialogue_counts = validate_and_count_dialogues(dialogue_paths)
    if int(dialogue_counts["dialogues"]) != accepted:
        raise SystemExit(
            f"dialogue scan mismatch: {dialogue_counts['dialogues']} != {accepted}"
        )
    if not args.identity_snapshot.is_file() or not args.lineage_index.is_file():
        raise SystemExit("identity snapshot or lineage index missing")

    implementation_paths = sorted(args.scripts_root.glob("*.py")) + [
        args.scripts_root.parent / "run_prepare_only.sh",
        args.scripts_root.parent / "vendor" / "build_dialogues_v2_1.py",
    ]
    if not all(path.is_file() for path in implementation_paths):
        raise SystemExit("frozen implementation is incomplete")

    manifest = {
        "manifest_version": "lusy-rp-data-juicer-preparation-manifest-v2.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_unit": "one immutable v1 bundle row per raw conversation record",
        "output_unit": "one structurally valid complete dialogue branch per JSONL row",
        "data_juicer_executed": False,
        "mds_used": False,
        "target_selection_used": False,
        "loss_mask_assigned": False,
        "counts": {
            "buckets": len(reports),
            "source_rows": processed,
            "accepted_dialogues": accepted,
            "quarantined_dialogues": quarantined,
            "quarantine_reason_counts": merge_counts(reports, "reason_counts"),
            "identity_sources": merge_counts(reports, "identity_sources"),
            "persona_sources": merge_counts(reports, "persona_sources"),
            "greeting_sources": merge_counts(reports, "greeting_sources"),
            "memory_modes": merge_counts(reports, "memory_modes"),
            "field_presence": merge_counts(reports, "field_presence"),
            "rows_with_residual_source_macros": sum(
                int(row["rows_with_residual_source_macros"]) for row in reports
            ),
            **dialogue_counts,
        },
        "hard_construction_gates": [
            "wrong_station",
            "assistant_or_user_version_anomaly",
            "untrainable_turn",
            "empty_or_missing_current_message",
            "missing_or_duplicate_event_id",
            "causal_order_or_turn_order_violation",
            "missing_conversation_id_or_greeting",
            "empty_after_cot_or_macro_render",
        ],
        "audit_only_not_hard_exclusions": [
            "conversation_status",
            "source_user_deleted_status",
            "requires_privacy_and_minor_scrub",
            "minor_nsfw_hard_exclusion",
            "requires_content_safety_and_pii_scrub",
            "fork_or_non_root_status",
        ],
        "contract": {
            "data_juicer_primary_field": "messages",
            "roles": ["system", "assistant_greeting", "user_assistant_pairs"],
            "identity_precedence": [
                "conversation persona.nickname",
                "conversation persona_id nickname",
                "default persona.nickname",
                "frozen account nickname",
                "User",
            ],
            "character_field_allowlist": [
                "name",
                "gender",
                "personality",
                "scenario",
                "intro",
                "qa_examples",
                "greeting",
            ],
            "excluded_character_fields": ["description", "prompt", "example_reply"],
            "runtime_memory_policy": (
                "latest valid effective memory plus every complete current turn after "
                "compressed_until_increase_id; otherwise full dialogue"
            ),
            "fork_policy": (
                "keep all structurally valid branches; retain lineage/relation/context "
                "metadata for later quality-aware dedup"
            ),
        },
        "source_files_modified": False,
        "identity_snapshot": artifact(args.identity_snapshot),
        "lineage_index": artifact(args.lineage_index),
        "implementation": [artifact(path) for path in implementation_paths],
        "bucket_artifacts": [
            {
                "bucket": int(report["bucket"]),
                "source": report["source"],
                "dialogues": report["output"],
                "quarantine": report["quarantine"],
            }
            for report in reports
        ],
        "bucket_reports": [artifact(path) for path in report_paths],
    }
    output = args.root / "CONSTRUCTION_MANIFEST.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite: {output}")
    temporary = output.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "status": "ok",
                "source_rows": processed,
                "accepted_dialogues": accepted,
                "quarantined_dialogues": quarantined,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
