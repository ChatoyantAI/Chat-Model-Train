#!/usr/bin/env python3
"""Lusy RP dialogue construction v2.2.

This is a deliberately small policy layer over the frozen, hash-verified v2.1
renderer.  The renderer and message assembly stay unchanged; only source
admission and provenance metadata change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vendor import build_dialogues_v2_1 as base


SCHEMA_VERSION = "lusy-rp-dialogue-data-juicer-v2.2"
RENDERER_VERSION = "rp-system-v1.4.1-dialogue-memory-card-v2.2"
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816/01_constructed"
)
DEFAULT_LINEAGE_INDEX = Path(
    "/home/gyr/lusychat-training/data-juicer-dialogue-v2.2-20260816/"
    "00_inputs/fork_lineage_index.json"
)

SOFT_AUDIT_REASONS = frozenset(
    {
        "inactive_conversation",
        "not_root_conversation",
        "deleted_user",
        "requires_privacy_and_minor_scrub",
        "minor_nsfw_hard_exclusion",
    }
)

_original_causal_structure_reasons = base.causal_structure_reasons
_original_materialize = base.materialize
_lineage_entries: dict[str, dict[str, Any]] = {}
_lineage_meta: dict[str, Any] = {}


def load_lineage_index(path: Path) -> None:
    global _lineage_entries, _lineage_meta
    if not path.is_file():
        _lineage_entries = {}
        _lineage_meta = {"status": "missing", "path": str(path)}
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    _lineage_entries = payload.get("entries") or {}
    _lineage_meta = {
        "status": "loaded",
        "path": str(path),
        "version": payload.get("version"),
        "entry_count": len(_lineage_entries),
        "policy": payload.get("policy"),
    }


def causal_structure_reasons(
    row: dict[str, Any], turns: list[dict[str, Any]]
) -> list[str]:
    """Keep structural failures hard; retain mutable/account/minor metadata."""
    return [
        reason
        for reason in _original_causal_structure_reasons(row, turns)
        if reason not in SOFT_AUDIT_REASONS
    ]


def source_policy_audit(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_status": int(row.get("conversation_status") or 0),
        "conversation_is_inactive": int(row.get("conversation_status") or 0) != 1,
        "source_user_is_deleted_user": bool(row.get("source_user_is_deleted_user")),
        "requires_privacy_and_minor_scrub": bool(
            row.get("requires_privacy_and_minor_scrub")
        ),
        "minor_nsfw_hard_exclusion": bool(row.get("minor_nsfw_hard_exclusion")),
        "requires_content_safety_and_pii_scrub": bool(
            row.get("requires_content_safety_and_pii_scrub")
        ),
        "metadata_policy": (
            "audit_only_not_a_construction_exclusion; downstream filtering remains required"
        ),
        "message_version_policy": (
            "canonical_current_status_1_text_versions_only; superseded_or_reroll_versions_not_restored"
        ),
    }


def lineage_audit(row: dict[str, Any]) -> dict[str, Any]:
    conversation_id = base.text(row.get("conversation_id"))
    parent_id = base.text(row.get("parent_id")) or None
    entry = _lineage_entries.get(conversation_id) or {}
    is_fork = bool(row.get("is_fork_conversation")) or bool(parent_id)
    lineage_id = entry.get("lineage_id") or (parent_id if is_fork else conversation_id)
    return {
        "lineage_id": lineage_id,
        "branch_id": conversation_id,
        "parent_conversation_id": parent_id,
        "is_root_conversation": bool(row.get("is_root_conversation")),
        "is_fork_conversation": is_fork,
        "fork_depth": entry.get("fork_depth", 1 if is_fork else 0),
        "fork_child_count": int(entry.get("fork_child_count") or 0),
        "fork_child_ids": entry.get("fork_child_ids") or [],
        "relation_to_parent": entry.get("relation_to_parent", "not_applicable"),
        "shared_prefix_turns": int(entry.get("shared_prefix_turns") or 0),
        "common_turns": int(entry.get("common_turns") or 0),
        "context_compatible_with_parent": entry.get("context_compatible_with_parent"),
        "context_differences": entry.get("context_differences") or [],
        "dedup_recommendation": entry.get("dedup_recommendation", "keep"),
        "retention_policy": (
            "keep_all_structurally_valid_branches; lineage metadata enables later "
            "quality-aware deduplication"
        ),
        "copied_history_loss_policy": (
            "unset; copied prefix must not be treated as independently verified all-loss data"
        ),
    }


def materialize(
    row: dict[str, Any],
    by_conversation: dict[str, dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    cards_by_character: dict[str, list[dict[str, Any]]],
    account_nicknames: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    record, quarantine = _original_materialize(
        row, by_conversation, by_id, cards_by_character, account_nicknames
    )
    policy = source_policy_audit(row)
    lineage = lineage_audit(row)
    quarantine["source_policy_audit"] = policy
    quarantine["lineage"] = lineage
    if record is None:
        return None, quarantine

    record["schema_version"] = SCHEMA_VERSION
    record["renderer_version"] = RENDERER_VERSION
    record["parent_conversation_id"] = lineage["parent_conversation_id"]
    record["lineage_id"] = lineage["lineage_id"]
    record["branch_id"] = lineage["branch_id"]
    record["source_policy_audit"] = policy
    record["lineage"] = lineage
    record["data_juicer_contract"] = {
        "format": "jsonl_one_complete_dialogue_per_row",
        "primary_field": "messages",
        "messages_schema": [{"role": "system|user|assistant", "content": "string"}],
        "flat_text_field": "dialogue_text",
        "assistant_only_text_field": "assistant_text",
        "loss_mask_assigned": False,
        "data_juicer_executed": False,
    }
    record["render_audit"]["source_admission_policy"] = (
        "v2.2_structural_only_hard_gate_with_soft_source_metadata"
    )
    record["render_audit"]["lineage_index"] = _lineage_meta
    record["render_audit"]["loss_policy"] = (
        "unset_dialogue_filtering_input_not_training_data"
    )
    return record, quarantine


def install_policy_layer(lineage_index: Path) -> None:
    load_lineage_index(lineage_index)
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.RENDERER_VERSION = RENDERER_VERSION
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.causal_structure_reasons = causal_structure_reasons
    base.materialize = materialize


def main() -> None:
    lineage_index = Path(
        os.environ.get("LUSY_FORK_LINEAGE_INDEX", str(DEFAULT_LINEAGE_INDEX))
    )
    install_policy_layer(lineage_index)
    base.main()


if __name__ == "__main__":
    main()
