#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import time
from collections import Counter
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = pathlib.Path(
    os.environ.get("LUSY_TRAINING_ROOT", "/home/gyr/lusychat-training")
)
RUN_ROOT = (
    ROOT
    / "judge-deepinfra-glm47-turn-v47-softjson-full-a-20260728"
)
MANIFEST_DIR = RUN_ROOT / "00_manifest"
INPUT_DIR = RUN_ROOT / "01_input"
RAW_DIR = RUN_ROOT / "02_raw"
NORMALIZED_DIR = RUN_ROOT / "03_normalized"
RETRY_DIR = RUN_ROOT / "04_retry"
REPORT_DIR = RUN_ROOT / "05_reports"

ELITE_PATH = (
    ROOT / "judge-dialogue-selection-v1/02_tiers/elite_A.parquet"
)
TURN_PATH = (
    ROOT
    / "judge-dialogue-union-50pct-v1/01_input/turn_manifest.parquet"
)
MATERIAL_PATH = (
    ROOT
    / "judge-dialogue-union-50pct-v1/01_input/dialogue_material.parquet"
)
PROMPT_SOURCE = (
    ROOT
    / "judge-deepinfra-glm47-turn-v46-promptonly-softjson-smoke-20260728"
    / "scripts/run_v46_promptonly_softjson_smoke.py"
)

MODEL = os.environ.get(
    "DEEPINFRA_JUDGE_MODEL",
    "zai-org/GLM-4.7-Flash",
)
MAX_ROWS_PER_BATCH = 2000
MAX_BYTES_PER_BATCH = 145 * 1024 * 1024
MAX_OUTPUT_TOKENS = 1536
SEED = 20260728
PROMPT_CACHE_KEY = "lusy_turn_judge_v47_softjson_full_a"
SCHEMA_VERSION = "lusy-turn-judge-v47-softjson-full-a-v1"


def load_prompt_module():
    spec = importlib.util.spec_from_file_location(
        "v46_prompt_source",
        PROMPT_SOURCE,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clean(value: Any) -> str:
    return "" if value is None else str(value)


def custom_id(dialogue_id: str, ordinal: int, native_id: str) -> str:
    dialogue_hash = hashlib.sha256(dialogue_id.encode("utf-8")).hexdigest()[:16]
    native_hash = hashlib.sha256(native_id.encode("utf-8")).hexdigest()[:8]
    return f"a_{dialogue_hash}_{ordinal:04d}_{native_hash}"


def expectation_row(
    row: dict[str, Any],
    material: dict[str, Any],
    *,
    request_id: str,
    has_greeting: bool,
) -> dict[str, Any]:
    return {
        "custom_id": request_id,
        "dialogue_id": clean(row["dialogue_id"]),
        "conversation_id": clean(row["conversation_id"]),
        "assistant_ordinal": int(row["assistant_ordinal"]),
        "native_message_id": clean(row["native_message_id"]),
        "content_sha256": clean(row["content_sha256"]),
        "is_legacy_candidate": bool(row["is_legacy_candidate"]),
        "replay_mode": clean(row["replay_mode"]),
        "memory_id": row["memory_id"],
        "memory_boundary_increase_id": row[
            "memory_boundary_increase_id"
        ],
        "language": clean(material["dominant_language"]),
        "topic": clean(material["representative_topic"]),
        "has_greeting": has_greeting,
    }


def request_row(
    prompt_module,
    candidate: dict[str, Any],
    *,
    request_id: str,
) -> dict[str, Any]:
    variant = (
        "greeting_present"
        if candidate["has_greeting"]
        else "greeting_absent"
    )
    body = prompt_module.request_body(candidate, variant)
    body["max_tokens"] = MAX_OUTPUT_TOKENS
    body["prompt_cache_key"] = PROMPT_CACHE_KEY
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


class BatchWriter:
    def __init__(self) -> None:
        self.index = -1
        self.input_handle = None
        self.expectation_handle = None
        self.input_part = None
        self.expectation_part = None
        self.input_path = None
        self.expectation_path = None
        self.input_digest = None
        self.expectation_digest = None
        self.rows = 0
        self.bytes = 0
        self.first_custom_id = ""
        self.last_custom_id = ""
        self.dialogues = set()
        self.plans = []

    def _open(self) -> None:
        self.index += 1
        stem = f"batch_{self.index:05d}"
        self.input_path = INPUT_DIR / f"{stem}.input.jsonl"
        self.expectation_path = INPUT_DIR / f"{stem}.expectations.jsonl"
        self.input_part = self.input_path.with_suffix(
            self.input_path.suffix + ".part"
        )
        self.expectation_part = self.expectation_path.with_suffix(
            self.expectation_path.suffix + ".part"
        )
        self.input_handle = self.input_part.open("wb")
        self.expectation_handle = self.expectation_part.open("wb")
        self.input_digest = hashlib.sha256()
        self.expectation_digest = hashlib.sha256()
        self.rows = 0
        self.bytes = 0
        self.first_custom_id = ""
        self.last_custom_id = ""
        self.dialogues = set()

    def should_rotate(self, next_bytes: int) -> bool:
        if self.input_handle is None:
            return False
        return (
            self.rows >= MAX_ROWS_PER_BATCH
            or self.bytes + next_bytes > MAX_BYTES_PER_BATCH
        )

    def add(
        self,
        input_bytes: bytes,
        expectation_bytes: bytes,
        *,
        request_id: str,
        dialogue_id: str,
    ) -> None:
        if self.input_handle is None:
            self._open()
        if self.should_rotate(len(input_bytes)):
            self.close()
            self._open()
        assert self.input_handle is not None
        assert self.expectation_handle is not None
        self.input_handle.write(input_bytes)
        self.expectation_handle.write(expectation_bytes)
        self.input_digest.update(input_bytes)
        self.expectation_digest.update(expectation_bytes)
        self.rows += 1
        self.bytes += len(input_bytes)
        self.first_custom_id = self.first_custom_id or request_id
        self.last_custom_id = request_id
        self.dialogues.add(dialogue_id)

    def close(self) -> None:
        if self.input_handle is None:
            return
        self.input_handle.flush()
        os.fsync(self.input_handle.fileno())
        self.input_handle.close()
        self.expectation_handle.flush()
        os.fsync(self.expectation_handle.fileno())
        self.expectation_handle.close()
        self.input_part.replace(self.input_path)
        self.expectation_part.replace(self.expectation_path)
        self.plans.append(
            {
                "batch_index": self.index,
                "batch_name": f"batch_{self.index:05d}",
                "input_path": str(self.input_path),
                "expectations_path": str(self.expectation_path),
                "requests": self.rows,
                "input_bytes": self.bytes,
                "input_sha256": self.input_digest.hexdigest(),
                "expectations_sha256": (
                    self.expectation_digest.hexdigest()
                ),
                "unique_dialogues": len(self.dialogues),
                "first_custom_id": self.first_custom_id,
                "last_custom_id": self.last_custom_id,
            }
        )
        self.input_handle = None
        self.expectation_handle = None


def main() -> None:
    started = time.time()
    complete_path = MANIFEST_DIR / "PREPARE_STATUS.json"
    if complete_path.exists():
        status = json.loads(complete_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return

    for directory in [
        MANIFEST_DIR,
        INPUT_DIR,
        RAW_DIR,
        NORMALIZED_DIR,
        RETRY_DIR,
        REPORT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    existing_inputs = list(INPUT_DIR.glob("batch_*.jsonl"))
    existing_parts = list(INPUT_DIR.glob("*.part"))
    if existing_inputs or existing_parts:
        raise RuntimeError(
            "partial production inputs already exist without a complete "
            "PREPARE_STATUS; inspect before retrying"
        )

    prompt_module = load_prompt_module()
    prompt = prompt_module.V46_PROMPT_ONLY_SOFT_JSON
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    elite_ids = pq.read_table(
        ELITE_PATH,
        columns=["dialogue_id"],
    )["dialogue_id"]
    elite_id_list = [str(value) for value in elite_ids.to_pylist()]
    turn_table = pq.read_table(
        TURN_PATH,
        columns=[
            "dialogue_id",
            "conversation_id",
            "assistant_ordinal",
            "native_message_id",
            "content_sha256",
            "is_legacy_candidate",
            "replay_mode",
            "memory_id",
            "memory_boundary_increase_id",
        ],
    )
    turn_table = turn_table.filter(
        pc.is_in(turn_table["dialogue_id"], value_set=elite_ids)
    )
    material_table = pq.read_table(
        MATERIAL_PATH,
        columns=[
            "dialogue_id",
            "conversation_id",
            "dominant_language",
            "representative_topic",
            "messages",
        ],
        filters=[("dialogue_id", "in", elite_id_list)],
    )
    material_by_id = {
        str(row["dialogue_id"]): row
        for row in material_table.to_pylist()
    }
    parsed_by_id = {
        dialogue_id: prompt_module.parse_replay(material["messages"] or [])
        for dialogue_id, material in material_by_id.items()
    }

    turn_rows = turn_table.to_pylist()
    turn_rows.sort(
        key=lambda row: (
            str(row["dialogue_id"]),
            int(row["assistant_ordinal"]),
        )
    )

    contract = {
        "schema_version": SCHEMA_VERSION,
        "created_at": started,
        "model": MODEL,
        "selection": "all assistant turns from all Elite A dialogues",
        "expected_turns": len(turn_rows),
        "expected_dialogues": len(elite_id_list),
        "request": {
            "endpoint": "/v1/chat/completions",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "seed": SEED,
            "stream": False,
            "reasoning_effort": "none",
            "prompt_cache_key": PROMPT_CACHE_KEY,
            "response_format": {"type": "json_object"},
        },
        "batching": {
            "max_rows": MAX_ROWS_PER_BATCH,
            "max_input_bytes": MAX_BYTES_PER_BATCH,
            "initial_concurrency": 2,
            "production_concurrency": 8,
        },
        "prompt": prompt,
        "prompt_sha256": prompt_sha256,
        "sources": {
            "elite_a": {
                "path": str(ELITE_PATH),
                "sha256": sha256_file(ELITE_PATH),
            },
            "turn_manifest": {
                "path": str(TURN_PATH),
                "sha256": sha256_file(TURN_PATH),
            },
            "dialogue_material": {
                "path": str(MATERIAL_PATH),
                "sha256": sha256_file(MATERIAL_PATH),
            },
            "prompt_source": {
                "path": str(PROMPT_SOURCE),
                "sha256": sha256_file(PROMPT_SOURCE),
            },
        },
        "normalization_policy": {
            "preserve_raw": True,
            "allowed": [
                "numeric_string_to_integer",
                "boolean_string_to_boolean",
                "overall_diagnosis_alias_to_overall_reason",
                "drop_extra_keys_after_recording",
            ],
            "forbidden": [
                "invent_missing_reason",
                "change_quality_score",
                "infer_missing_dimension",
                "rewrite_judge_reason",
            ],
        },
    }
    atomic_json(MANIFEST_DIR / "FROZEN_CONTRACT.json", contract)

    writer = BatchWriter()
    seen_custom_ids = set()
    seen_keys = set()
    counts = Counter()
    progress_path = MANIFEST_DIR / "PREPARE_PROGRESS.json"

    for row_index, row in enumerate(turn_rows, start=1):
        dialogue_id = clean(row["dialogue_id"])
        ordinal = int(row["assistant_ordinal"])
        native_id = clean(row["native_message_id"])
        key = (dialogue_id, ordinal)
        if key in seen_keys:
            raise RuntimeError(f"duplicate turn key: {key}")
        seen_keys.add(key)
        material = material_by_id[dialogue_id]
        parsed = parsed_by_id[dialogue_id]
        if ordinal < 1 or ordinal > len(parsed["pairs"]):
            raise RuntimeError(f"ordinal out of range: {key}")
        target = parsed["pairs"][ordinal - 1][1]
        digest = hashlib.sha256(target.encode("utf-8")).hexdigest()
        if digest != clean(row["content_sha256"]):
            raise RuntimeError(f"content hash mismatch: {key}")
        if clean(row["conversation_id"]) != clean(
            material["conversation_id"]
        ):
            raise RuntimeError(f"conversation mismatch: {key}")

        has_greeting = prompt_module.greeting_present(parsed)
        request_id = custom_id(dialogue_id, ordinal, native_id)
        if request_id in seen_custom_ids:
            raise RuntimeError(f"duplicate custom_id: {request_id}")
        seen_custom_ids.add(request_id)
        candidate = {
            **row,
            "dialogue_id": dialogue_id,
            "conversation_id": clean(row["conversation_id"]),
            "native_message_id": native_id,
            "assistant_ordinal": ordinal,
            "language": clean(material["dominant_language"]),
            "topic": clean(material["representative_topic"]),
            "parsed": parsed,
            "has_greeting": has_greeting,
        }
        request = request_row(
            prompt_module,
            candidate,
            request_id=request_id,
        )
        expectation = expectation_row(
            row,
            material,
            request_id=request_id,
            has_greeting=has_greeting,
        )
        input_bytes = (
            json.dumps(
                request,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        expectation_bytes = (
            json.dumps(
                expectation,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        writer.add(
            input_bytes,
            expectation_bytes,
            request_id=request_id,
            dialogue_id=dialogue_id,
        )
        counts["turns"] += 1
        counts[
            "greeting_present" if has_greeting else "greeting_absent"
        ] += 1
        counts[
            "legacy"
            if bool(row["is_legacy_candidate"])
            else "non_legacy"
        ] += 1

        if row_index % 10000 == 0:
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "processed_turns": row_index,
                    "expected_turns": len(turn_rows),
                    "completed_batches": len(writer.plans),
                    "current_batch_rows": writer.rows,
                    "current_batch_bytes": writer.bytes,
                    "elapsed_seconds": time.time() - started,
                },
            )

    writer.close()
    total_input_bytes = sum(plan["input_bytes"] for plan in writer.plans)
    total_requests = sum(plan["requests"] for plan in writer.plans)
    if total_requests != len(turn_rows):
        raise RuntimeError(
            f"request count mismatch: {total_requests} != {len(turn_rows)}"
        )
    if any(
        plan["input_bytes"] > MAX_BYTES_PER_BATCH
        or plan["requests"] > MAX_ROWS_PER_BATCH
        for plan in writer.plans
    ):
        raise RuntimeError("batch size limit violated")

    batch_plan = {
        "schema_version": SCHEMA_VERSION,
        "requests": total_requests,
        "batches": len(writer.plans),
        "total_input_bytes": total_input_bytes,
        "total_input_gib": total_input_bytes / 1024**3,
        "max_batch_rows": max(plan["requests"] for plan in writer.plans),
        "max_batch_bytes": max(
            plan["input_bytes"] for plan in writer.plans
        ),
        "counts": dict(counts),
        "plans": writer.plans,
    }
    atomic_json(MANIFEST_DIR / "BATCH_PLAN.json", batch_plan)
    plan_sha256 = sha256_file(MANIFEST_DIR / "BATCH_PLAN.json")
    status = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "requests": total_requests,
        "batches": len(writer.plans),
        "total_input_bytes": total_input_bytes,
        "total_input_gib": total_input_bytes / 1024**3,
        "prompt_sha256": prompt_sha256,
        "batch_plan_sha256": plan_sha256,
        "elapsed_seconds": time.time() - started,
        "completed_at": time.time(),
    }
    atomic_json(complete_path, status)
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
