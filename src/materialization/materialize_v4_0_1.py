#!/usr/bin/env python3
"""Create the non-overwriting v4.0.1 Thoughts-position correction package."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(os.environ.get("LUSY_TRAINING_ROOT", "/home/gyr/lusychat-training"))
PARENT = Path(
    os.environ.get("LUSY_V4_PARENT", ROOT / "lusy-sft-55k-20260810-v4.0")
)
STAGING = Path(
    os.environ.get(
        "LUSY_V4_STAGING",
        ROOT / ".staging/lusy-sft-55k-20260810-v4.0.1",
    )
)
TARGET = Path(
    os.environ.get("LUSY_V4_TARGET", ROOT / "lusy-sft-55k-20260810-v4.0.1")
)
V3 = Path(os.environ.get("LUSY_V3_SOURCE", ROOT / "lusy-sft-60k-20260802-v3.1"))

CONVERSATION_ID = "59df4f3b-1c66-4c77-856e-f3fe8a33d533"
TARGET_LINES = {
    178: "m011",
    15_958: "m003",
    15_959: "m017",
    15_960: "m029",
}
CANONICAL_LINE = 15_960
EXPECTED_TURNS = [f"m{index:03d}" for index in range(3, 30, 2)]
EXPECTED_UNIQUE_TURNS = 14
EXPECTED_OCCURRENCES = 28
EXPECTED_TARGET_OCCURRENCES = 4
EXPECTED_HISTORY_OCCURRENCES = 24

EXPECTED_COUNTS = {
    "rp.train.jsonl": 44_275,
    "rp.val.jsonl": 891,
    "general.train.jsonl": 9_800,
    "general.val.jsonl": 200,
    "combined.train.jsonl": 54_075,
    "combined.val.jsonl": 1_091,
}

EXPECTED_V3_HASHES = {
    "data/sft/rp.train.jsonl": "16bf13fec01264846d0502f53c0b8546fb9fdc7bd5f697c31d3a2834ebda36c8",
    "data/sft/rp.val.jsonl": "70fc44d14db2dc32a5ef3a2a79974d953f355dd9f0430498639560dadfc624fe",
    "data/sft/general.train.jsonl": "a02a17b7387cbd857e026118cce60b618226019ecc490204ce4218f893346d30",
    "data/sft/general.val.jsonl": "70e99c3843b13d1c80c6820af01e5b61acd988f6e4cf9ac99c036796f15db564",
    "audit/sft_records.parquet": "70a9850777a93c1d11ddb5500899eb0ddf4c86af66e83d0a5bea456d401178bf",
    "audit/split_groups.parquet": "06b4adb7cd3992b5d715d6b5d6d81b24e43d951ed1f97b8402d6ff7b91855911",
    "MANIFEST.json": "b0f6a8124d4b69c86d95e262bb828265a459161921dfa2a07952ab2729c213d2",
}

THOUGHT_RE = re.compile(
    r"^(?P<thought>\*💭Lia's Thoughts:[^\r\n]*\*)"
    r"(?P<separator>(?:[ \t]*\r?\n)+)"
    r"(?P<body>[\s\S]*)$"
)
TRAILING_NEWLINES_RE = re.compile(r"^(?P<core>[\s\S]*?)(?P<suffix>(?:\r?\n)*)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def training_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def read_sha256sums(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (root / "SHA256SUMS").open("r", encoding="utf-8") as handle:
        for line in handle:
            digest, relative = line.rstrip("\n").split("  ", 1)
            result[relative] = digest
    return result


def verify_package_sums(root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        actual = sha256_file(root / relative)
        assert_equal(actual, digest, f"package hash {root.name}/{relative}")


def verify_v3_hashes() -> None:
    for relative, expected in EXPECTED_V3_HASHES.items():
        assert_equal(sha256_file(V3 / relative), expected, f"v3.1 unchanged {relative}")


def move_thought_to_end(content: str) -> str:
    marker = "💭Lia's Thoughts:"
    assert_equal(content.count(marker), 1, "Thoughts marker count")
    match = THOUGHT_RE.match(content)
    if match is None:
        raise RuntimeError("Assistant content does not start with the expected Thoughts block")
    thought = match.group("thought")
    separator = match.group("separator")
    body = match.group("body")
    trailing = TRAILING_NEWLINES_RE.match(body)
    if trailing is None:
        raise RuntimeError("Unable to preserve trailing newlines")
    corrected = trailing.group("core") + separator + thought + trailing.group("suffix")
    if corrected.startswith(thought):
        raise RuntimeError("Thoughts block remained at the beginning")
    if not re.search(r"\*💭Lia's Thoughts:[^\r\n]*\*(?:\r?\n)*$", corrected):
        raise RuntimeError("Thoughts block was not moved to the end")
    assert_equal(corrected.count(marker), 1, "corrected Thoughts marker count")
    return corrected


def load_target_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if line_number in TARGET_LINES:
                records[line_number] = json.loads(raw)
    assert_equal(set(records), set(TARGET_LINES), "target RP line coverage")
    return records


def build_correction_map(records: dict[int, dict[str, Any]]) -> dict[str, dict[str, str]]:
    canonical = records[CANONICAL_LINE]
    messages = canonical["messages"]
    assert_equal(len(messages), 30, "canonical message count")
    greeting = messages[1]
    assert_equal(greeting["role"], "assistant", "Greeting role")
    if not re.search(r"\*💭Lia's Thoughts:[^\r\n]*\*(?:\r?\n)*$", greeting["content"]):
        raise RuntimeError("Creator Greeting does not establish Thoughts-at-end style")

    corrections: dict[str, dict[str, str]] = {}
    for index in range(3, 30, 2):
        turn_id = f"m{index:03d}"
        message = messages[index]
        assert_equal(message["role"], "assistant", f"{turn_id} role")
        before = message["content"]
        after = move_thought_to_end(before)
        corrections[turn_id] = {
            "before": before,
            "after": after,
            "before_sha256": sha256_text(before),
            "after_sha256": sha256_text(after),
        }
    assert_equal(sorted(corrections), EXPECTED_TURNS, "unique corrected turn IDs")
    return corrections


def correction_occurrence_counts(
    records: dict[int, dict[str, Any]], corrections: dict[str, dict[str, str]]
) -> tuple[Counter[str], Counter[str]]:
    target_counts: Counter[str] = Counter()
    history_counts: Counter[str] = Counter()
    for line_number, record in records.items():
        messages = record["messages"]
        expected_target = TARGET_LINES[line_number]
        assert_equal(f"m{len(messages)-1:03d}", expected_target, f"line {line_number} target")
        for index, message in enumerate(messages):
            turn_id = f"m{index:03d}"
            if turn_id not in corrections:
                continue
            assert_equal(message["content"], corrections[turn_id]["before"], f"{turn_id} before consistency")
            if message["train"]:
                target_counts[turn_id] += 1
            else:
                history_counts[turn_id] += 1
    assert_equal(sum(target_counts.values()), EXPECTED_TARGET_OCCURRENCES, "target occurrences")
    assert_equal(sum(history_counts.values()), EXPECTED_HISTORY_OCCURRENCES, "history occurrences")
    assert_equal(sum(target_counts.values()) + sum(history_counts.values()), EXPECTED_OCCURRENCES, "all occurrences")
    return target_counts, history_counts


def correct_record(
    record: dict[str, Any], corrections: dict[str, dict[str, str]], line_number: int
) -> dict[str, Any]:
    before_messages = record["messages"]
    corrected = json.loads(json.dumps(record, ensure_ascii=False))
    changed = 0
    for index, message in enumerate(corrected["messages"]):
        turn_id = f"m{index:03d}"
        if turn_id in corrections:
            assert_equal(message["role"], "assistant", f"line {line_number}/{turn_id} role")
            assert_equal(message["content"], corrections[turn_id]["before"], f"line {line_number}/{turn_id} before")
            message["content"] = corrections[turn_id]["after"]
            changed += 1
        else:
            assert_equal(message, before_messages[index], f"line {line_number}/{turn_id} unchanged message")
    assert_equal(changed, (len(corrected["messages"]) - 2) // 2, f"line {line_number} corrected assistant count")
    validate_training_record(corrected, f"corrected rp.train/{line_number}")
    return corrected


def validate_training_record(record: dict[str, Any], label: str) -> None:
    assert_equal(set(record), {"messages"}, f"{label} top-level keys")
    messages = record["messages"]
    if len(messages) < 2:
        raise RuntimeError(f"{label}: too few messages")
    for index, message in enumerate(messages):
        assert_equal(set(message), {"role", "content", "train"}, f"{label}/message {index} keys")
        if message["role"] not in {"system", "user", "assistant"}:
            raise RuntimeError(f"{label}/message {index}: invalid role")
        if not isinstance(message["content"], str) or not isinstance(message["train"], bool):
            raise RuntimeError(f"{label}/message {index}: invalid content/train type")
    true_indices = [index for index, message in enumerate(messages) if message["train"]]
    assert_equal(true_indices, [len(messages) - 1], f"{label} train mask")
    assert_equal(messages[-1]["role"], "assistant", f"{label} target role")


def rewrite_rp_train(
    source: Path,
    output: Path,
    corrected_records: dict[int, dict[str, Any]],
) -> dict[int, dict[str, str]]:
    record_hashes: dict[int, dict[str, str]] = {}
    with source.open("rb") as reader, output.open("wb") as writer:
        for line_number, raw in enumerate(reader, 1):
            if line_number not in corrected_records:
                writer.write(raw)
                continue
            old_payload = raw.rstrip(b"\r\n")
            new_payload = training_json_bytes(corrected_records[line_number])
            writer.write(new_payload + b"\n")
            record_hashes[line_number] = {
                "parent_output_record_sha256": sha256_bytes(old_payload),
                "corrected_output_record_sha256": sha256_bytes(new_payload),
                "corrected_output_target_sha256": sha256_text(
                    corrected_records[line_number]["messages"][-1]["content"]
                ),
            }
    assert_equal(line_number, EXPECTED_COUNTS["rp.train.jsonl"], "rp.train rows")
    assert_equal(set(record_hashes), set(TARGET_LINES), "corrected record hashes")
    return record_hashes


def evenly_interleave(primary: Path, secondary: Path, output: Path) -> None:
    primary_count = count_lines(primary)
    secondary_count = count_lines(secondary)
    total = primary_count + secondary_count
    with primary.open("rb") as first, secondary.open("rb") as second, output.open("wb") as target:
        used_primary = 0
        used_secondary = 0
        for position in range(1, total + 1):
            secondary_due = (position * secondary_count) // total > ((position - 1) * secondary_count) // total
            if secondary_due:
                line = second.readline()
                used_secondary += 1
            else:
                line = first.readline()
                used_primary += 1
            if not line:
                raise RuntimeError(f"Interleave input exhausted at position {position}")
            target.write(line)
        if first.read(1) or second.read(1):
            raise RuntimeError("Interleave left unread input rows")
    assert_equal(used_primary, primary_count, "combined RP rows")
    assert_equal(used_secondary, secondary_count, "combined General rows")


def compare_changed_lines(parent: Path, corrected: Path, expected: set[int], label: str) -> None:
    differences: list[int] = []
    with parent.open("rb") as first, corrected.open("rb") as second:
        line_number = 0
        while True:
            left = first.readline()
            right = second.readline()
            if not left and not right:
                break
            line_number += 1
            if left != right:
                differences.append(line_number)
    assert_equal(set(differences), expected, f"{label} changed line set")
    assert_equal(len(differences), len(expected), f"{label} changed line count")


def combined_positions(rp_count: int, general_count: int, selected_rp_lines: set[int]) -> set[int]:
    total = rp_count + general_count
    rp_line = 0
    positions: set[int] = set()
    for position in range(1, total + 1):
        general_due = (position * general_count) // total > ((position - 1) * general_count) // total
        if not general_due:
            rp_line += 1
            if rp_line in selected_rp_lines:
                positions.add(position)
    assert_equal(rp_line, rp_count, "combined RP traversal")
    return positions


def update_sft_audit(
    source: Path,
    output: Path,
    record_hashes: dict[int, dict[str, str]],
) -> None:
    table = pq.read_table(source)
    rows = table.to_pylist()
    touched: set[int] = set()
    for row in rows:
        if row["subset"] != "rp" or row["split"] != "train" or row["line_number"] not in TARGET_LINES:
            continue
        line_number = row["line_number"]
        assert_equal(row["conversation_id"], CONVERSATION_ID, f"audit line {line_number} conversation")
        assert_equal(row["target_turn_id"], TARGET_LINES[line_number], f"audit line {line_number} target turn")
        assert_equal(
            row["output_record_sha256"],
            record_hashes[line_number]["parent_output_record_sha256"],
            f"audit line {line_number} parent record hash",
        )
        row["output_record_sha256"] = record_hashes[line_number]["corrected_output_record_sha256"]
        row["output_target_sha256"] = record_hashes[line_number]["corrected_output_target_sha256"]
        row["materialization_action"] = "REPAIR+USER_APPROVED_THOUGHT_POSITION_CORRECTION"
        touched.add(line_number)
    assert_equal(touched, set(TARGET_LINES), "sft audit touched rows")
    corrected_table = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(corrected_table, output, compression="zstd", use_dictionary=True, write_statistics=True)
    assert_equal(corrected_table.num_rows, 55_166, "sft audit row count")


def write_correction_sidecars(
    corrections: dict[str, dict[str, str]],
    target_counts: Counter[str],
    history_counts: Counter[str],
    parent_manifest_sha: str,
    approved_at: str,
) -> None:
    rows: list[dict[str, Any]] = []
    for turn_id in EXPECTED_TURNS:
        correction = corrections[turn_id]
        rows.append(
            {
                "schema_version": "lusy-thought-position-correction-v1",
                "conversation_id": CONVERSATION_ID,
                "turn_id": turn_id,
                "before_sha256": correction["before_sha256"],
                "after_sha256": correction["after_sha256"],
                "target_occurrences": target_counts[turn_id],
                "history_occurrences": history_counts[turn_id],
                "total_occurrences": target_counts[turn_id] + history_counts[turn_id],
                "operation": "move_exact_leading_thought_block_to_response_end",
                "text_rewrite": False,
                "user_approved": True,
                "approved_at": approved_at,
                "parent_dataset_version": "4.0",
                "parent_manifest_sha256": parent_manifest_sha,
            }
        )
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        STAGING / "audit/thought_position_corrections.parquet",
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    write_json(
        STAGING / "audit/thought_position_correction_summary.json",
        {
            "schema_version": "lusy-thought-position-correction-summary-v1",
            "conversation_id": CONVERSATION_ID,
            "parent_dataset_version": "4.0",
            "corrected_dataset_version": "4.0.1",
            "unique_turns": EXPECTED_UNIQUE_TURNS,
            "turn_ids": EXPECTED_TURNS,
            "component_records": len(TARGET_LINES),
            "target_occurrences": sum(target_counts.values()),
            "history_occurrences": sum(history_counts.values()),
            "total_occurrences": sum(target_counts.values()) + sum(history_counts.values()),
            "operation": "move_exact_leading_thought_block_to_response_end",
            "text_rewrite": False,
            "system_modified": False,
            "user_modified": False,
            "greeting_modified": False,
            "approved_at": approved_at,
        },
    )


def verify_hash_chain(path: Path) -> dict[str, Any]:
    previous: str | None = None
    count = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            event = json.loads(raw)
            event_hash = event.pop("event_hash")
            assert_equal(event["sequence"], line_number - 1, f"ledger sequence line {line_number}")
            assert_equal(event["previous_event_hash"], previous, f"ledger previous line {line_number}")
            assert_equal(sha256_bytes(canonical_json_bytes(event)), event_hash, f"ledger event hash line {line_number}")
            previous = event_hash
            count += 1
    return {"events": count, "tail_hash": previous, "sha256": sha256_file(path)}


def append_ledger_event(path: Path, event_type: str, payload: dict[str, Any]) -> str:
    ledger = verify_hash_chain(path)
    event = {
        "schema_version": "lusy-materialization-event-v1",
        "sequence": ledger["events"],
        "timestamp": utc_now(),
        "event_type": event_type,
        "payload": payload,
        "previous_event_hash": ledger["tail_hash"],
    }
    event_hash = sha256_bytes(canonical_json_bytes(event))
    event["event_hash"] = event_hash
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(event) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return event_hash


def validate_all_component_records() -> int:
    total = 0
    for name in ("rp.train.jsonl", "rp.val.jsonl", "general.train.jsonl", "general.val.jsonl"):
        path = STAGING / "data/sft" / name
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                validate_training_record(json.loads(raw), f"{name}/{line_number}")
                total += 1
        assert_equal(line_number, EXPECTED_COUNTS[name], f"{name} full validation rows")
    assert_equal(total, 55_166, "unique component record validation count")
    return total


def validate_split_leakage() -> tuple[int, int]:
    rows = pq.read_table(STAGING / "audit/sft_records.parquet", columns=["subset", "split", "conversation_id", "group_id"]).to_pylist()
    rp_train = {row["conversation_id"] for row in rows if row["subset"] == "rp" and row["split"] == "train"}
    rp_val = {row["conversation_id"] for row in rows if row["subset"] == "rp" and row["split"] == "val"}
    general_train = {row["group_id"] for row in rows if row["subset"] == "general" and row["split"] == "train"}
    general_val = {row["group_id"] for row in rows if row["subset"] == "general" and row["split"] == "val"}
    return len(rp_train & rp_val), len(general_train & general_val)


def validate_drop_leakage() -> int:
    dropped = {
        json.loads(line)["conversation_id"]
        for line in (STAGING / "audit/drop_conversations.jsonl").read_text(encoding="utf-8").splitlines()
    }
    conversations = {
        value.as_py()
        for value in pq.read_table(
            STAGING / "audit/sft_records.parquet", columns=["subset", "conversation_id"]
        ).filter(pa.compute.equal(pq.read_table(
            STAGING / "audit/sft_records.parquet", columns=["subset"]
        )["subset"], "rp"))["conversation_id"]
    }
    return len(dropped & conversations)


def validate_corrected_thoughts(
    corrections: dict[str, dict[str, str]],
    parent_records: dict[int, dict[str, Any]],
    corrected_records: dict[int, dict[str, Any]],
) -> None:
    counts: Counter[str] = Counter()
    for line_number, record in corrected_records.items():
        parent_record = parent_records[line_number]
        for index, message in enumerate(record["messages"]):
            turn_id = f"m{index:03d}"
            parent_message = parent_record["messages"][index]
            if turn_id in corrections:
                assert_equal(message["content"], corrections[turn_id]["after"], f"corrected {line_number}/{turn_id}")
                if message["content"].startswith("*💭Lia's Thoughts:"):
                    raise RuntimeError(f"Thoughts still at beginning: {line_number}/{turn_id}")
                if not re.search(r"\*💭Lia's Thoughts:[^\r\n]*\*(?:\r?\n)*$", message["content"]):
                    raise RuntimeError(f"Thoughts not at end: {line_number}/{turn_id}")
                counts[turn_id] += 1
            else:
                assert_equal(message, parent_message, f"authorized-only message {line_number}/{turn_id}")
    assert_equal(sum(counts.values()), EXPECTED_OCCURRENCES, "validated corrected occurrences")
    assert_equal(set(counts), set(EXPECTED_TURNS), "validated corrected turn coverage")


def file_info(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(STAGING).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def main() -> None:
    if not PARENT.is_dir():
        raise RuntimeError(f"Parent dataset missing: {PARENT}")
    if TARGET.exists():
        raise RuntimeError(f"Target exists; refusing to overwrite: {TARGET}")
    if STAGING.exists():
        raise RuntimeError(f"Staging exists; refusing to overwrite: {STAGING}")

    parent_sums = read_sha256sums(PARENT)
    parent_manifest_sha = sha256_file(PARENT / "MANIFEST.json")
    assert_equal(parent_manifest_sha, "c40050acc4bcc4ebffa7694060fc63306be403862aa699744f2386de9e683e06", "parent manifest hash")
    log("verifying every parent v4.0 package hash and the frozen v3.1 hashes")
    verify_package_sums(PARENT, parent_sums)
    verify_v3_hashes()

    log("copying v4.0 to isolated v4.0.1 staging")
    shutil.copytree(PARENT, STAGING)
    approved_at = utc_now()

    rp_parent = PARENT / "data/sft/rp.train.jsonl"
    records = load_target_records(rp_parent)
    corrections = build_correction_map(records)
    repair_v2 = read_json(
        ROOT
        / ".codex-tmp/r82-final-review-materialization-v2/repairs"
        / f"{CONVERSATION_ID}.json"
    )
    repair_patches = {item["turn_id"]: item for item in repair_v2["proposed_patches"]}
    assert_equal(set(repair_patches), {"m003"}, "repair v2 patch coverage for corrected conversation")
    assert_equal(repair_patches["m003"]["rewrite"], corrections["m003"]["before"], "repair v2 m003 content before format correction")
    assert_equal(repair_patches["m003"]["after_sha256"], corrections["m003"]["before_sha256"], "repair v2 m003 SHA before format correction")
    target_counts, history_counts = correction_occurrence_counts(records, corrections)
    corrected_records = {
        line_number: correct_record(record, corrections, line_number)
        for line_number, record in records.items()
    }

    log("rewriting four RP component rows and rebuilding combined.train")
    rp_temp = STAGING / "data/sft/rp.train.corrected.tmp"
    record_hashes = rewrite_rp_train(rp_parent, rp_temp, corrected_records)
    os.replace(rp_temp, STAGING / "data/sft/rp.train.jsonl")
    combined_temp = STAGING / "data/sft/combined.train.corrected.tmp"
    evenly_interleave(
        STAGING / "data/sft/rp.train.jsonl",
        STAGING / "data/sft/general.train.jsonl",
        combined_temp,
    )
    os.replace(combined_temp, STAGING / "data/sft/combined.train.jsonl")

    compare_changed_lines(
        PARENT / "data/sft/rp.train.jsonl",
        STAGING / "data/sft/rp.train.jsonl",
        set(TARGET_LINES),
        "rp.train",
    )
    changed_combined_positions = combined_positions(44_275, 9_800, set(TARGET_LINES))
    compare_changed_lines(
        PARENT / "data/sft/combined.train.jsonl",
        STAGING / "data/sft/combined.train.jsonl",
        changed_combined_positions,
        "combined.train",
    )

    log("updating record audit and writing correction sidecars")
    audit_temp = STAGING / "audit/sft_records.corrected.tmp.parquet"
    update_sft_audit(PARENT / "audit/sft_records.parquet", audit_temp, record_hashes)
    os.replace(audit_temp, STAGING / "audit/sft_records.parquet")
    write_correction_sidecars(
        corrections,
        target_counts,
        history_counts,
        parent_manifest_sha,
        approved_at,
    )

    packaged_script = STAGING / "scripts/materialize_thought_position_correction_v4_0_1.py"
    shutil.copyfile(Path(__file__).resolve(), packaged_script)

    ledger_path = STAGING / "audit/materialization.events.jsonl"
    append_ledger_event(
        ledger_path,
        "USER_APPROVED_THOUGHT_POSITION_CORRECTION",
        {
            "conversation_id": CONVERSATION_ID,
            "parent_dataset_version": "4.0",
            "corrected_dataset_version": "4.0.1",
            "parent_manifest_sha256": parent_manifest_sha,
            "component_record_lines": sorted(TARGET_LINES),
            "unique_turns": EXPECTED_UNIQUE_TURNS,
            "turn_ids": EXPECTED_TURNS,
            "target_occurrences": EXPECTED_TARGET_OCCURRENCES,
            "history_occurrences": EXPECTED_HISTORY_OCCURRENCES,
            "total_occurrences": EXPECTED_OCCURRENCES,
            "operation": "move exact leading Thoughts block to assistant response end",
            "text_rewrite": False,
            "system_modified": False,
            "user_modified": False,
            "greeting_modified": False,
            "human_review_completed": False,
            "authority": "explicit user instruction in active Codex task",
        },
    )

    log("running full component and consistency validation")
    validated_records = validate_all_component_records()
    validate_corrected_thoughts(corrections, records, corrected_records)
    rp_leakage, general_leakage = validate_split_leakage()
    assert_equal(rp_leakage, 0, "RP train/val conversation leakage")
    assert_equal(general_leakage, 0, "General train/val group leakage")

    # DROP leakage is inherited from v4.0 and rechecked directly from the updated audit table.
    audit_rows = pq.read_table(
        STAGING / "audit/sft_records.parquet", columns=["subset", "conversation_id"]
    ).to_pylist()
    retained_rp = {row["conversation_id"] for row in audit_rows if row["subset"] == "rp"}
    dropped = {
        json.loads(line)["conversation_id"]
        for line in (STAGING / "audit/drop_conversations.jsonl").read_text(encoding="utf-8").splitlines()
    }
    drop_leakage = len(retained_rp & dropped)
    assert_equal(drop_leakage, 0, "DROP conversation leakage")

    for name, expected_rows in EXPECTED_COUNTS.items():
        assert_equal(count_lines(STAGING / "data/sft" / name), expected_rows, f"{name} rows")

    for name in ("general.train.jsonl", "general.val.jsonl"):
        assert_equal(
            sha256_file(STAGING / "data/sft" / name),
            sha256_file(V3 / "data/sft" / name),
            f"General byte identity {name}",
        )
    for relative in (
        "data/sft/rp.val.jsonl",
        "data/sft/general.train.jsonl",
        "data/sft/general.val.jsonl",
        "data/sft/combined.val.jsonl",
        "audit/applied_patches.parquet",
        "audit/repair_turn_decisions.parquet",
        "audit/conversation_dispositions.parquet",
        "audit/drop_conversations.jsonl",
    ):
        assert_equal(sha256_file(STAGING / relative), sha256_file(PARENT / relative), f"unchanged inherited file {relative}")

    append_ledger_event(
        ledger_path,
        "CORRECTION_VALIDATION_PASSED_ATOMIC_RENAME_AUTHORIZED",
        {
            "contentReady": True,
            "trainingReady": False,
            "full_component_records_validated": validated_records,
            "authorized_conversation_only": True,
            "unique_turns_corrected": EXPECTED_UNIQUE_TURNS,
            "all_occurrences_consistent": True,
            "all_corrected_thoughts_at_response_end": True,
            "greeting_unchanged": True,
            "system_and_user_unchanged": True,
            "rp_changed_component_records": len(TARGET_LINES),
            "combined_changed_records": len(changed_combined_positions),
            "drop_conversation_zero_leakage": True,
            "split_leakage": 0,
            "general_byte_identical": True,
            "parent_v4_0_unchanged": True,
            "v3_1_original_hashes_unchanged": True,
        },
    )
    ledger_summary = verify_hash_chain(ledger_path)
    assert_equal(ledger_summary["events"], 8, "ledger event count")

    parent_validation = read_json(PARENT / "audit/validation_report.json")
    validation = parent_validation
    output_file_info: dict[str, dict[str, Any]] = {}
    for name, rows in EXPECTED_COUNTS.items():
        output_file_info[name] = file_info(STAGING / "data/sft" / name, rows)
    validation["files"] = {
        name: {"bytes": output_file_info[name]["bytes"], "rows": rows, "sha256": output_file_info[name]["sha256"]}
        for name, rows in EXPECTED_COUNTS.items()
        if not name.startswith("combined.")
    }
    validation["combined_train"] = {
        "algorithm": "At position i (1-based), emit General iff floor(i*G/(R+G)) increases; otherwise emit RP",
        "bytes": output_file_info["combined.train.jsonl"]["bytes"],
        "rows": EXPECTED_COUNTS["combined.train.jsonl"],
        "sha256": output_file_info["combined.train.jsonl"]["sha256"],
        "rp_order_preserved": True,
        "general_order_preserved": True,
    }
    validation["combined_val"] = {
        "algorithm": "At position i (1-based), emit General iff floor(i*G/(R+G)) increases; otherwise emit RP",
        "bytes": output_file_info["combined.val.jsonl"]["bytes"],
        "rows": EXPECTED_COUNTS["combined.val.jsonl"],
        "sha256": output_file_info["combined.val.jsonl"]["sha256"],
        "rp_order_preserved": True,
        "general_order_preserved": True,
    }
    validation["materialization_ledger"] = ledger_summary
    validation["parent_v4_0_original_hashes_unchanged"] = True
    validation["thought_position_correction"] = {
        "conversation_id": CONVERSATION_ID,
        "component_record_lines": sorted(TARGET_LINES),
        "combined_record_lines": sorted(changed_combined_positions),
        "unique_turns": EXPECTED_UNIQUE_TURNS,
        "turn_ids": EXPECTED_TURNS,
        "target_occurrences": EXPECTED_TARGET_OCCURRENCES,
        "history_occurrences": EXPECTED_HISTORY_OCCURRENCES,
        "total_occurrences": EXPECTED_OCCURRENCES,
        "all_corrected_thoughts_at_response_end": True,
        "all_occurrences_consistent": True,
        "authorized_conversation_only": True,
        "greeting_unchanged": True,
        "system_unchanged": True,
        "user_messages_unchanged": True,
        "repair_v2_content_preserved_before_format_transform": True,
        "text_rewrite": False,
    }
    validation["training_blockers"] = [item.replace("v4.0", "v4.0.1") for item in validation["training_blockers"]]
    write_json(STAGING / "audit/validation_report.json", validation)

    readme = (PARENT / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("# LusyChat SFT 55k v4.0", "# LusyChat SFT 55k v4.0.1", 1)
    correction_section = f"""

## v4.0.1 correction

This non-overwriting child of v4.0 applies one explicit user-approved formatting correction to
conversation `{CONVERSATION_ID}`. Four RP component records contain 14 unique assistant turns
whose exact leading `*💭Lia's Thoughts: ...*` block is moved to the end of the same response.
The operation changes no text inside the Thoughts block or response body. All 28 target/history
occurrences are updated consistently. System, user, and Greeting content are unchanged.

The original v4.0 package remains immutable. See `audit/thought_position_corrections.parquet`,
`audit/thought_position_correction_summary.json`, and the appended hash-chain events.
"""
    readme = readme.replace("\n## Counts\n", correction_section + "\n## Counts\n", 1)
    readme = readme.replace("for v4.0", "for v4.0.1")
    (STAGING / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    parent_manifest = read_json(PARENT / "MANIFEST.json")
    manifest = parent_manifest
    manifest["datasetVersion"] = "4.0.1"
    manifest["generatedAt"] = utc_now()
    manifest["parentDataset"] = {
        "datasetVersion": "4.0",
        "path": str(PARENT),
        "manifestSha256": parent_manifest_sha,
        "immutableAndUnchanged": True,
    }
    manifest["inputs"]["parentV4_0ManifestSha256"] = parent_manifest_sha
    manifest["policy"]["thoughtPositionCorrection"] = {
        "authority": "explicit user instruction",
        "conversationId": CONVERSATION_ID,
        "operation": "move exact leading Thoughts block to assistant response end",
        "uniqueTurns": EXPECTED_UNIQUE_TURNS,
        "totalOccurrences": EXPECTED_OCCURRENCES,
        "textRewrite": False,
        "systemModified": False,
        "userModified": False,
        "greetingModified": False,
    }
    manifest["audits"]["thoughtPositionCorrectionValid"] = True
    manifest["audits"]["parentV4_0OriginalHashesUnchanged"] = True
    manifest["trainingBlockers"] = [item.replace("v4.0", "v4.0.1") for item in manifest["trainingBlockers"]]
    manifest["outputs"] = {
        name: output_file_info[name]
        for name in sorted(output_file_info)
    }

    sidecars: dict[str, dict[str, Any]] = {}
    parent_sidecars = parent_manifest["sidecars"]
    for path in sorted((STAGING / "audit").iterdir()):
        if not path.is_file():
            continue
        row_count: int | None = None
        if path.suffix == ".parquet":
            row_count = pq.read_metadata(path).num_rows
        elif path.suffix == ".jsonl":
            row_count = count_lines(path)
        info = file_info(path, row_count)
        if path.name in parent_sidecars and "rows" not in parent_sidecars[path.name]:
            info.pop("rows", None)
        sidecars[path.name] = info
    manifest["sidecars"] = sidecars
    write_json(STAGING / "MANIFEST.json", manifest)

    log("writing SHA256SUMS and performing final package verification")
    package_files = sorted(
        path for path in STAGING.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    with (STAGING / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
        for path in package_files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(STAGING).as_posix()}\n")
    final_sums = read_sha256sums(STAGING)
    verify_package_sums(STAGING, final_sums)

    # Recheck both immutable ancestors after all child writes are complete.
    verify_package_sums(PARENT, parent_sums)
    verify_v3_hashes()
    assert_equal(sha256_file(PARENT / "MANIFEST.json"), parent_manifest_sha, "parent manifest unchanged at end")

    parsed_manifest = read_json(STAGING / "MANIFEST.json")
    assert_equal(parsed_manifest["contentReady"], True, "contentReady")
    assert_equal(parsed_manifest["trainingReady"], False, "trainingReady")
    if TARGET.exists():
        raise RuntimeError(f"Target appeared during run; refusing to overwrite: {TARGET}")
    log(f"all validation passed; atomically renaming staging to {TARGET}")
    os.rename(STAGING, TARGET)
    log(f"v4.0.1 correction materialization complete: {TARGET}")


if __name__ == "__main__":
    main()
