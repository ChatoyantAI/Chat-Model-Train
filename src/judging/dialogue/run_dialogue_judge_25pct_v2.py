#!/usr/bin/env python3
"""Resumable 25% one-request-per-dialogue evidence Judge runner."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq

import dialogue_judge_contract_v2 as judge_v6


DEFAULT_INPUT = Path(
    "/home/gyr/lusychat-training/judge-qwen36-35b-dialogue-evidence-v2/"
    "00_frozen_input_25pct_v1"
)
DEFAULT_OUTPUT = Path(
    "/home/gyr/lusychat-training/judge-qwen36-35b-dialogue-evidence-v2/"
    "01_results_25pct"
)
DEFAULT_BASE_URL = (
    "https://867a2750e14565dc-8000."
    "us-ca-7.gpu-instance.novita.ai/v1"
)
DEFAULT_MODEL = "qwen3.6-35b-a3b-heretic-fp8"
RUNNER_VERSION = "lusy-dialogue-evidence-score-runner-v2.2.0-vllm"
JUDGE_KEY_SUFFIX = "trajectory_evidence_v2"
SEMANTIC_REPAIR_VERSION = "lusy-dialogue-semantic-repair-v1"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
GLOBAL_FATAL_STATUS = {401, 403, 404}
STOP_REQUESTED = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def repair_jsonl_tail(path: Path, recovery_path: Path) -> None:
    """Preserve and remove only an incomplete final JSONL fragment."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        size = handle.tell()
        window = min(size, 1024 * 1024)
        handle.seek(size - window)
        tail = handle.read(window)
        newline = tail.rfind(b"\n")
        truncate_at = size - window + newline + 1 if newline >= 0 else 0
        corrupt = tail[newline + 1 :] if newline >= 0 else tail
        handle.truncate(truncate_at)
    with recovery_path.open("a", encoding="utf-8") as recovery:
        recovery.write(
            compact_json(
                {
                    "created_at": utc_now(),
                    "path": str(path),
                    "discarded_bytes": len(corrupt),
                    "discarded_sha256": sha256_bytes(corrupt),
                    "discarded_base64": base64.b64encode(corrupt).decode("ascii"),
                }
            )
            + "\n"
        )


def signal_handler(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parse_jsonl_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                key = row.get("judge_key")
                if isinstance(key, str) and ":dialogue_" not in key:
                    keys.add(key)
    return keys


def acquire_lock(path: Path) -> Any:
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another runner holds {path}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
    handle.flush()
    return handle


def get_json(url: str, timeout: int = 30) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return response.status, {}
            try:
                return response.status, json.loads(raw)
            except json.JSONDecodeError:
                return response.status, {"raw_body": raw[:2000]}
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        return error.code, {"error_body": raw}
    except Exception as error:
        return 0, {"exception": repr(error)}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    api_base = args.base_url.rstrip("/")
    origin = api_base[:-3] if api_base.endswith("/v1") else api_base
    health_status, health = get_json(origin + "/health")
    models_status, models = get_json(api_base + "/models")
    props_status, props = get_json(origin + "/props")
    if health_status != 200:
        raise RuntimeError(
            f"health preflight failed status={health_status} body={health}"
        )
    if models_status != 200:
        raise RuntimeError(
            f"models preflight failed status={models_status} body={models}"
        )
    model_entries = [
        item
        for key in ("data", "models")
        for item in (models.get(key) or [])
        if isinstance(item, dict)
    ]
    model_ids = {
        str(item.get("id") or item.get("name") or item.get("model") or "")
        for item in model_entries
    }
    if args.model not in model_ids:
        raise RuntimeError(
            f"model {args.model!r} not advertised; available={sorted(model_ids)}"
        )
    selected_model = next(
        item
        for item in model_entries
        if str(item.get("id") or item.get("name") or item.get("model") or "")
        == args.model
    )
    total_slots = (
        int(props.get("total_slots") or 0) if props_status == 200 else 0
    )
    if total_slots and total_slots < args.concurrency:
        raise RuntimeError(
            f"service exposes {total_slots} slots, fewer than concurrency "
            f"{args.concurrency}"
        )
    if (
        props_status == 200
        and props.get("model_alias")
        and str(props["model_alias"]) != args.model
    ):
        raise RuntimeError(
            f"props model_alias={props.get('model_alias')!r}, expected "
            f"{args.model!r}"
        )
    n_ctx = int(
        (
            (props.get("default_generation_settings") or {}).get("n_ctx")
            if props_status == 200
            else 0
        )
        or selected_model.get("max_model_len")
        or 0
    )
    return {
        "schema_version": "lusy-judge-transport-preflight-v2",
        "checked_at": utc_now(),
        "base_url": api_base,
        "health": health,
        "model": args.model,
        "advertised_model_ids": sorted(model_ids),
        "service_implementation": str(
            selected_model.get("owned_by") or "unknown"
        ),
        "requested_concurrency": args.concurrency,
        "reported_total_slots": total_slots or None,
        "concurrency_verified_by_props": bool(total_slots),
        "n_ctx": n_ctx,
        "endpoint_slots": bool(
            props.get("endpoint_slots") if props_status == 200 else False
        ),
        "model_alias": str(
            props.get("model_alias")
            if props_status == 200 and props.get("model_alias")
            else args.model
        ),
        "props_status": props_status,
    }


@dataclass
class DialogueJob:
    sample_rank: int
    dialogue_id: str
    conversation_id: str
    character_id: str
    replay_mode: str
    language: str
    topic: str
    estimated_prompt_tokens: int
    messages: list[dict[str, Any]]
    turns: list[dict[str, Any]]


@dataclass
class PromptMetadata:
    retention: dict[str, dict[str, Any]]
    characters: dict[str, dict[str, Any]]
    localized: dict[tuple[str, str], dict[str, Any]]
    personas: dict[str, dict[str, Any]]


def load_jobs(input_dir: Path) -> list[DialogueJob]:
    turn_rows = pq.read_table(input_dir / "pilot_turn_manifest.parquet").to_pylist()
    turns_by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in turn_rows:
        turns_by_dialogue[str(row["dialogue_id"])].append(row)

    columns = [
        "dialogue_id",
        "conversation_id",
        "character_id",
        "replay_mode",
        "dominant_language",
        "representative_topic",
        "representative_target_increase_id",
        "representative_serialized_tokens",
        "messages",
        "pilot_rank",
    ]
    jobs: list[DialogueJob] = []
    for row in pq.read_table(
        input_dir / "pilot_dialogue_material.parquet",
        columns=columns,
    ).to_pylist():
        dialogue_id = str(row["dialogue_id"])
        turns = sorted(
            turns_by_dialogue.get(dialogue_id) or [],
            key=lambda value: (
                int(value["assistant_ordinal"]),
                int(value["assistant_increase_id"]),
            ),
        )
        if not turns:
            raise RuntimeError(f"{dialogue_id} has no turn manifest")
        endpoint = turns[-1]
        endpoint_ordinal = int(endpoint["assistant_ordinal"])
        if not bool(endpoint.get("is_representative_target")):
            raise RuntimeError(
                f"{dialogue_id} endpoint is not the frozen representative target"
            )
        if endpoint_ordinal != int(
            endpoint.get("visible_segment_assistant_count") or -1
        ):
            raise RuntimeError(
                f"{dialogue_id} endpoint ordinal differs from visible count"
            )
        if int(endpoint["assistant_increase_id"]) != int(
            row["representative_target_increase_id"]
        ):
            raise RuntimeError(
                f"{dialogue_id} endpoint differs from material representative"
            )
        jobs.append(
            DialogueJob(
                sample_rank=int(row["pilot_rank"]),
                dialogue_id=dialogue_id,
                conversation_id=str(row["conversation_id"]),
                character_id=str(row["character_id"]),
                replay_mode=str(row.get("replay_mode") or "unknown"),
                language=str(row.get("dominant_language") or "unknown"),
                topic=str(row.get("representative_topic") or "unknown"),
                estimated_prompt_tokens=int(
                    row.get("representative_serialized_tokens") or 0
                ),
                messages=row.get("messages") or [],
                turns=[endpoint],
            )
        )
    jobs.sort(key=lambda value: value.sample_rank)
    return jobs


def load_or_create_slot_assignment(
    output_dir: Path,
    jobs: list[DialogueJob],
    concurrency: int,
) -> tuple[dict[int, list[DialogueJob]], Path]:
    """Freeze a deterministic greedy token-balanced dialogue-to-slot mapping."""
    path = output_dir / "slot_assignment.json"
    jobs_by_id = {job.dialogue_id: job for job in jobs}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("concurrency") or 0) != concurrency:
            raise RuntimeError("slot assignment concurrency mismatch")
        raw_mapping = payload.get("dialogue_to_slot") or {}
        if set(raw_mapping) != set(jobs_by_id):
            raise RuntimeError("slot assignment dialogue set mismatch")
        mapping = {key: int(value) for key, value in raw_mapping.items()}
        if any(not 0 <= slot < concurrency for slot in mapping.values()):
            raise RuntimeError("slot assignment contains invalid slot")
    else:
        slot_loads = [0] * concurrency
        mapping: dict[str, int] = {}
        ordered = sorted(
            jobs,
            key=lambda job: (
                -job.estimated_prompt_tokens,
                hashlib.sha256(job.dialogue_id.encode("utf-8")).hexdigest(),
                job.dialogue_id,
            ),
        )
        for job in ordered:
            slot = min(range(concurrency), key=lambda value: (slot_loads[value], value))
            mapping[job.dialogue_id] = slot
            slot_loads[slot] += job.estimated_prompt_tokens
        atomic_json(
            path,
            {
                "schema_version": "lusy-dialogue-token-slot-assignment-v1",
                "created_at": utc_now(),
                "algorithm": (
                    "descending estimated prompt tokens, stable dialogue hash "
                    "tie-break, greedy assign to lowest current token load"
                ),
                "concurrency": concurrency,
                "dialogues": len(jobs),
                "requests": len(jobs),
                "estimated_prompt_tokens": sum(
                    job.estimated_prompt_tokens for job in jobs
                ),
                "slot_estimated_prompt_token_loads": slot_loads,
                "dialogue_to_slot": mapping,
            },
        )

    assigned: dict[int, list[DialogueJob]] = {
        slot: [] for slot in range(concurrency)
    }
    for dialogue_id, slot in mapping.items():
        assigned[slot].append(jobs_by_id[dialogue_id])
    for slot in assigned:
        assigned[slot].sort(key=lambda job: job.sample_rank)
    return assigned, path


def load_prompt_metadata(jobs: list[DialogueJob]) -> PromptMetadata:
    character_ids = {job.character_id for job in jobs}
    conversation_ids = {job.conversation_id for job in jobs}
    wanted_languages = {(job.character_id, job.language) for job in jobs}
    retention = {
        job.dialogue_id: {
            "dialogue_id": job.dialogue_id,
            "conversation_id": job.conversation_id,
            "character_id": job.character_id,
            "replay_mode": job.replay_mode,
            "exact_language": job.language,
        }
        for job in jobs
    }

    character_columns = [
        "character_id",
        "name",
        "intro",
        "description",
        "gender",
        "age",
        "tags",
        "greeting",
        "scenario",
        "personality",
        "qa_examples",
        "prompt",
        "style",
    ]
    characters = {}
    for row in pq.read_table(
        judge_v6.CHARACTER_PATH,
        columns=character_columns,
    ).to_pylist():
        character_id = str(row["character_id"])
        if character_id in character_ids:
            characters[character_id] = row

    description_columns = [
        "increase_id",
        "character_id",
        "language",
        "name",
        "headline",
        "intro",
        "description",
        "gender",
        "greeting",
        "greetings",
        "scenario",
        "personality",
        "qa_examples",
        "prompt",
        "example_reply",
    ]
    description_table = ds.dataset(
        str(judge_v6.DESCRIPTION_PATH),
        format="parquet",
    ).to_table(
        columns=description_columns,
        filter=ds.field("character_id").isin(sorted(character_ids)),
    )
    localized = {}
    for row in description_table.to_pylist():
        key = (
            str(row.get("character_id") or ""),
            str(row.get("language") or ""),
        )
        if key not in wanted_languages:
            continue
        old = localized.get(key)
        if old is None or int(row.get("increase_id") or -1) > int(
            old.get("increase_id") or -1
        ):
            localized[key] = row

    personas = {}
    for row in pq.read_table(judge_v6.PERSONA_PATH).to_pylist():
        conversation_id = str(row.get("conversation_id") or "")
        if conversation_id in conversation_ids:
            personas[conversation_id] = row

    missing_characters = sorted(character_ids - set(characters))
    if missing_characters:
        raise RuntimeError(
            f"{len(missing_characters)} selected character IDs lack canonical cards"
        )
    return PromptMetadata(retention, characters, localized, personas)


def build_causal_material(
    dialogue: DialogueJob,
    turn_row: dict[str, Any],
) -> dict[str, Any]:
    messages = dialogue.messages
    user_index = int(turn_row["current_user_source_index"])
    assistant_index = int(turn_row["assistant_source_index"])
    target_turn = int(turn_row["assistant_ordinal"])
    if not (0 <= user_index < assistant_index < len(messages)):
        raise ValueError("invalid causal source indices")
    if str(messages[user_index].get("role") or "") != "user":
        raise ValueError("current_user_source_index is not user")
    if str(messages[assistant_index].get("role") or "") != "assistant":
        raise ValueError("assistant_source_index is not assistant")

    systems = [
        str(message.get("content") or "").strip()
        for message in messages[:assistant_index]
        if str(message.get("role") or "") == "system"
        and str(message.get("content") or "").strip()
    ]
    memory = None
    if bool(turn_row.get("memory_wrapper_present")) and len(systems) >= 2:
        candidates = [
            value
            for value in systems
            if value.lower().startswith("long-term conversation memory:")
            or value.startswith("[MEMORY]")
            or value.startswith("[CONVERSATION MEMORY]")
        ]
        memory = candidates[-1] if candidates else systems[-1]

    prefix: list[dict[str, Any]] = []
    seen_user = False
    pair_turn = 0
    open_user_turn = None
    for index, message in enumerate(messages[:assistant_index]):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if not content or role == "system":
            continue
        if role == "assistant" and not seen_user:
            continue
        if role == "user":
            if open_user_turn is not None:
                raise ValueError("consecutive user messages in replay")
            seen_user = True
            pair_turn += 1
            open_user_turn = pair_turn
            prefix.append(
                {
                    "turn": pair_turn,
                    "role": "user",
                    "kind": "current_user" if index == user_index else "message",
                    "content": content,
                }
            )
            continue
        if role == "assistant":
            if open_user_turn is None:
                raise ValueError("assistant without preceding user")
            prefix.append(
                {
                    "turn": open_user_turn,
                    "role": "assistant",
                    "kind": "message",
                    "content": content,
                }
            )
            open_user_turn = None
    if pair_turn != target_turn or open_user_turn != target_turn:
        raise ValueError(
            f"causal ordinal mismatch built={pair_turn} manifest={target_turn}"
        )
    target = str(messages[assistant_index].get("content") or "").strip()
    if not target:
        raise ValueError("target assistant is empty")
    return {
        "memory_before_target": memory,
        "conversation_prefix": prefix,
        "target_assistant": {"turn": target_turn, "content": target},
    }


def assistant_texts_from_material(
    material: dict[str, Any],
) -> dict[int, str]:
    texts: dict[int, str] = {}
    for item in material.get("conversation_prefix") or []:
        if str(item.get("role") or "") != "assistant":
            continue
        texts[int(item["turn"])] = str(item.get("content") or "")
    target = material["target_assistant"]
    texts[int(target["turn"])] = str(target.get("content") or "")
    return texts


def validate_judgement(
    value: dict[str, Any],
    endpoint_turn: int,
    assistant_texts: dict[int, str],
) -> None:
    judge_v6.validate_judgement(
        value,
        endpoint_turn,
        assistant_texts,
    )


class RunState:
    def __init__(
        self,
        output_dir: Path,
        expected_dialogues: int,
        expected_requests: int,
    ):
        self.output_dir = output_dir
        self.valid_path = output_dir / "valid.jsonl"
        self.invalid_path = output_dir / "invalid.jsonl"
        self.deferred_path = output_dir / "deferred.jsonl"
        self.build_error_path = output_dir / "build_errors.jsonl"
        self.request_error_path = output_dir / "request_errors.jsonl"
        self.attempt_path = output_dir / "attempts.jsonl"
        recovery_path = output_dir / "recovered_jsonl_tails.jsonl"
        for path in [
            self.valid_path,
            self.invalid_path,
            self.deferred_path,
            self.build_error_path,
            self.request_error_path,
            self.attempt_path,
        ]:
            repair_jsonl_tail(path, recovery_path)
        self.progress_path = output_dir / "progress.json"
        self.completed = parse_jsonl_keys(
            [
                self.valid_path,
                self.invalid_path,
                self.build_error_path,
                self.request_error_path,
            ]
        )
        self.expected_dialogues = expected_dialogues
        self.expected_requests = expected_requests
        self.started_monotonic = time.monotonic()
        self.started_at = utc_now()
        self.lock = threading.Lock()
        self.dialogues_finished = 0
        self.attempts_this_run = 0
        self.valid_this_run = 0
        self.invalid_this_run = 0
        self.deferred_this_run = 0
        self.build_errors_this_run = 0
        self.request_errors_this_run = 0
        self.fatal_error = False
        self.score_counts: dict[str, Counter[int | str]] = {
            name: Counter() for name in judge_v6.CORE_DIMENSIONS
        }
        self.nsfw = Counter()
        self._handles = {
            "valid": self.valid_path.open("a", encoding="utf-8", buffering=1),
            "invalid": self.invalid_path.open("a", encoding="utf-8", buffering=1),
            "deferred": self.deferred_path.open("a", encoding="utf-8", buffering=1),
            "build_error": self.build_error_path.open(
                "a", encoding="utf-8", buffering=1
            ),
            "request_error": self.request_error_path.open(
                "a", encoding="utf-8", buffering=1
            ),
            "attempt": self.attempt_path.open("a", encoding="utf-8", buffering=1),
        }
        self._restore_score_statistics()
        self._write_status_locked()

    def _restore_score_statistics(self) -> None:
        if not self.valid_path.exists():
            return
        with self.valid_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                    judgement = row.get("judgement") or {}
                    for name, item in (judgement.get("dimensions") or {}).items():
                        if name in self.score_counts:
                            score = item.get("model_score")
                            self.score_counts[name][
                                score if isinstance(score, int) else "null"
                            ] += 1
                    level = (judgement.get("assistant_nsfw") or {}).get(
                        "max_level"
                    )
                    if isinstance(level, int):
                        self.nsfw[level] += 1
                except Exception:
                    continue

    def record_attempt(self, row: dict[str, Any]) -> None:
        with self.lock:
            self._handles["attempt"].write(compact_json(row) + "\n")
            self.attempts_this_run += 1

    def record(self, kind: str, row: dict[str, Any]) -> None:
        with self.lock:
            self._handles[kind].write(compact_json(row) + "\n")
            key = row.get("judge_key")
            if kind == "valid":
                self.valid_this_run += 1
                if isinstance(key, str):
                    self.completed.add(key)
                judgement = row["judgement"]
                for name, item in judgement["dimensions"].items():
                    score = item.get("model_score")
                    self.score_counts[name][
                        score if isinstance(score, int) else "null"
                    ] += 1
                self.nsfw[
                    int(judgement["assistant_nsfw"]["max_level"])
                ] += 1
            elif kind == "invalid":
                self.invalid_this_run += 1
                if isinstance(key, str):
                    self.completed.add(key)
            elif kind == "build_error":
                self.build_errors_this_run += 1
                if isinstance(key, str) and ":dialogue_" not in key:
                    self.completed.add(key)
            elif kind == "request_error":
                self.request_errors_this_run += 1
                if isinstance(key, str):
                    self.completed.add(key)
            elif kind == "deferred":
                self.deferred_this_run += 1
            self._write_status_locked()

    def record_fatal(self, row: dict[str, Any]) -> None:
        global STOP_REQUESTED
        with self.lock:
            if self.fatal_error:
                STOP_REQUESTED = True
                return
            self.fatal_error = True
            STOP_REQUESTED = True
            atomic_json(self.output_dir / "fatal_error.json", row)
            self._write_status_locked()

    def dialogue_done(self) -> None:
        with self.lock:
            self.dialogues_finished += 1
            if self.dialogues_finished % 10 == 0:
                self._write_status_locked()

    def _write_status_locked(self) -> None:
        elapsed = max(0.001, time.monotonic() - self.started_monotonic)
        terminal = len(self.completed)
        processed_this_run = (
            self.valid_this_run
            + self.invalid_this_run
            + self.build_errors_this_run
            + self.request_errors_this_run
        )
        throughput = processed_this_run * 3600 / elapsed
        remaining = max(0, self.expected_requests - terminal)
        progress = {
            "schema_version": RUNNER_VERSION,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "expected_dialogues": self.expected_dialogues,
            "expected_requests": self.expected_requests,
            "terminal_requests": terminal,
            "remaining_requests": remaining,
            "completion_fraction": terminal / max(1, self.expected_requests),
            "dialogues_finished_this_run": self.dialogues_finished,
            "valid_this_run": self.valid_this_run,
            "invalid_this_run": self.invalid_this_run,
            "deferred_this_run": self.deferred_this_run,
            "build_errors_this_run": self.build_errors_this_run,
            "request_errors_this_run": self.request_errors_this_run,
            "attempts_this_run": self.attempts_this_run,
            "throughput_dialogues_per_hour_this_run": throughput,
            "eta_hours_at_current_throughput": (
                remaining / throughput if throughput > 0 else None
            ),
            "score_distribution": {
                name: {str(key): count for key, count in sorted(values.items(), key=str)}
                for name, values in self.score_counts.items()
            },
            "nsfw_distribution": {
                str(key): value for key, value in sorted(self.nsfw.items())
            },
            "stop_requested": STOP_REQUESTED,
            "fatal_error": self.fatal_error,
        }
        atomic_json(self.progress_path, progress)

    def finish(self) -> None:
        with self.lock:
            self._write_status_locked()
            for handle in self._handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()


def post_once(
    endpoint: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw), time.perf_counter() - started
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        return error.code, {"error_body": raw}, time.perf_counter() - started
    except Exception as error:
        return 0, {"exception": repr(error)}, time.perf_counter() - started


def global_fatal_kind(status: int, error: str) -> str:
    if status in GLOBAL_FATAL_STATUS:
        return "global_non_retryable_http_status"
    lowered = error.lower()
    if status == 400 and any(
        marker in lowered
        for marker in [
            "grammar error",
            "json schema",
            "json_schema",
            "structured output",
            "response_format",
            "guided decoding",
            "unsupported schema",
            "unimplemented keys",
        ]
    ):
        return "schema_or_grammar_http_400"
    return ""


def post_with_retry(
    endpoint: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
    judge_key: str,
    phase: str,
    state: RunState,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    last = {}
    for attempt in range(max_retries + 1):
        status, response, elapsed = post_once(endpoint, payload, timeout)
        error = (
            response.get("error")
            or response.get("error_body")
            or response.get("exception")
            or ""
        )
        state.record_attempt(
            {
                "judge_key": judge_key,
                "phase": phase,
                "attempt": attempt + 1,
                "created_at": utc_now(),
                "status": status,
                "elapsed_seconds": elapsed,
                "error": str(error)[:2000],
            }
        )
        last = {
            "status": status,
            "elapsed_seconds": elapsed,
            "attempts": attempt + 1,
            "error": str(error)[:2000],
            "global_fatal_kind": global_fatal_kind(
                status,
                str(error),
            ),
        }
        if status == 200 and response.get("choices"):
            return response, last
        retryable = status == 0 or status in RETRYABLE_STATUS
        if not retryable or attempt >= max_retries:
            return None, {**last, "retryable": retryable}
        time.sleep(min(20.0, 2**attempt + random.random()))
    return None, last


def response_parts(response: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    return content, str(choice.get("finish_reason") or ""), {
        "usage": response.get("usage") or {},
        "timings": response.get("timings") or {},
        "reasoning_characters": len(reasoning),
        "reasoning_sha256": (
            sha256_bytes(reasoning.encode("utf-8")) if reasoning else ""
        ),
    }


def parse_and_validate_response(
    response: dict[str, Any],
    material: dict[str, Any],
) -> dict[str, Any]:
    content, finish_reason, response_meta = response_parts(response)
    parsed = None
    validation_error = ""
    try:
        if finish_reason == "length":
            raise ValueError("finish_reason=length")
        candidate = json.loads(content)
        validate_judgement(
            candidate,
            int(material["target_assistant"]["turn"]),
            assistant_texts_from_material(material),
        )
        parsed = candidate
    except Exception as error:
        validation_error = f"{type(error).__name__}: {error}"
    return {
        "content": content,
        "finish_reason": finish_reason,
        "response_meta": response_meta,
        "parsed": parsed,
        "validation_error": validation_error,
    }


def route_transport_failure(
    *,
    state: RunState,
    base: dict[str, Any],
    phase: str,
    transport: dict[str, Any],
    initial_invalid: dict[str, Any] | None = None,
) -> None:
    row = {
        **base,
        "created_at": utc_now(),
        "phase": phase,
        "transport": transport,
    }
    if initial_invalid is not None:
        row["initial_validation_error"] = initial_invalid.get(
            "validation_error"
        )
        row["initial_content_sha256"] = sha256_bytes(
            str(initial_invalid.get("content") or "").encode("utf-8")
        )
    fatal_kind = str(transport.get("global_fatal_kind") or "")
    if fatal_kind:
        state.record("deferred", row)
        state.record_fatal(
            {
                "schema_version": RUNNER_VERSION,
                "created_at": utc_now(),
                "reason": fatal_kind,
                "judge_key": base["judge_key"],
                "status": int(transport.get("status") or 0),
                "transport": transport,
            }
        )
    elif transport.get("retryable", True):
        state.record("deferred", row)
    else:
        state.record("request_error", row)


def evaluate_turn(
    args: argparse.Namespace,
    state: RunState,
    metadata: PromptMetadata,
    schema: dict[str, Any],
    dialogue: DialogueJob,
    turn_row: dict[str, Any],
    slot_id: int,
) -> None:
    judge_key = f"{dialogue.dialogue_id}:{JUDGE_KEY_SUFFIX}"
    if judge_key in state.completed:
        return
    base = {
        "judge_key": judge_key,
        "dialogue_id": dialogue.dialogue_id,
        "conversation_id": dialogue.conversation_id,
        "sample_rank": dialogue.sample_rank,
        "assistant_ordinal": int(turn_row["assistant_ordinal"]),
        "native_message_id": str(turn_row["native_message_id"]),
        "assistant_increase_id": int(turn_row["assistant_increase_id"]),
        "evaluation_unit": "dialogue_trajectory",
        "endpoint_rule": "frozen_representative_last_turn",
        "replay_mode": dialogue.replay_mode,
        "language": dialogue.language,
        "topic": dialogue.topic,
        "slot_id": slot_id,
    }
    try:
        material = build_causal_material(dialogue, turn_row)
        row = {
            "judge_key": judge_key,
            "language": dialogue.language,
            "causal_material": material,
        }
        markdown = judge_v6.build_markdown(
            row,
            metadata.retention,
            metadata.characters,
            metadata.localized,
            metadata.personas,
        )
        request = judge_v6.make_request(
            markdown,
            schema,
            stable_seed(judge_key),
        )
    except Exception as error:
        state.record(
            "build_error",
            {
                **base,
                "created_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return

    response, initial_transport = post_with_retry(
        args.base_url,
        request,
        args.timeout,
        args.max_retries,
        judge_key,
        "initial",
        state,
    )
    if response is None:
        route_transport_failure(
            state=state,
            base=base,
            phase="initial_transport",
            transport=initial_transport,
        )
        return

    initial = parse_and_validate_response(response, material)
    final = initial
    final_transport = initial_transport
    repair_used = False
    repair_transport: dict[str, Any] | None = None
    if initial["parsed"] is None:
        repair_used = True
        repair_request = judge_v6.make_repair_request(
            markdown,
            schema,
            stable_seed(judge_key + ":" + SEMANTIC_REPAIR_VERSION),
            str(initial["content"]),
            str(initial["validation_error"]),
            int(material["target_assistant"]["turn"]),
        )
        repair_response, repair_transport = post_with_retry(
            args.base_url,
            repair_request,
            args.timeout,
            args.max_retries,
            judge_key,
            "semantic_repair",
            state,
        )
        if repair_response is None:
            route_transport_failure(
                state=state,
                base=base,
                phase="semantic_repair_transport",
                transport=repair_transport,
                initial_invalid=initial,
            )
            return
        final = parse_and_validate_response(repair_response, material)
        final_transport = repair_transport
        if final["parsed"] is None:
            state.record(
                "invalid",
                {
                    **base,
                    "created_at": utc_now(),
                    "phase": "semantic_repair_validation",
                    "repair_attempted": True,
                    "repair_succeeded": False,
                    "initial_validation_error": initial["validation_error"],
                    "initial_finish_reason": initial["finish_reason"],
                    "initial_content_sha256": sha256_bytes(
                        str(initial["content"]).encode("utf-8")
                    ),
                    "initial_content": initial["content"],
                    "initial_transport": initial_transport,
                    "initial_response_meta": initial["response_meta"],
                    "repair_validation_error": final["validation_error"],
                    "repair_finish_reason": final["finish_reason"],
                    "repair_content_sha256": sha256_bytes(
                        str(final["content"]).encode("utf-8")
                    ),
                    "repair_content": final["content"],
                    "repair_transport": repair_transport,
                    "repair_response_meta": final["response_meta"],
                },
            )
            return

    (
        model_scores,
        arithmetic_scores,
        audited_scores,
        reason_audit,
        calibration_flags,
    ) = judge_v6.audit_score_support(
        final["parsed"],
        material["target_assistant"]["content"],
        assistant_texts_from_material(material),
        int(material["target_assistant"]["turn"]),
    )
    state.record(
        "valid",
        {
            **base,
            "created_at": utc_now(),
            "repair_used": repair_used,
            "repair_attempted": repair_used,
            "repair_succeeded": repair_used,
            "finish_reason": final["finish_reason"],
            "transport": final_transport,
            "response_meta": final["response_meta"],
            "initial_validation_error": (
                initial["validation_error"] if repair_used else ""
            ),
            "initial_content_sha256": (
                sha256_bytes(str(initial["content"]).encode("utf-8"))
                if repair_used
                else ""
            ),
            "initial_response_meta": (
                initial["response_meta"] if repair_used else {}
            ),
            "initial_transport": initial_transport,
            "repair_transport": repair_transport,
            "raw_scores": model_scores,
            "model_scores": model_scores,
            "arithmetic_scores": arithmetic_scores,
            "audited_scores": audited_scores,
            "reason_audit": reason_audit,
            "calibration_flags": calibration_flags,
            "judgement": final["parsed"],
            "content_sha256": sha256_bytes(
                str(final["content"]).encode("utf-8")
            ),
        },
    )


def worker(
    worker_id: int,
    dialogues: list[DialogueJob],
    args: argparse.Namespace,
    state: RunState,
    metadata: PromptMetadata,
    schema: dict[str, Any],
    turn_counter: list[int],
    turn_counter_lock: threading.Lock,
) -> None:
    for dialogue in dialogues:
        if STOP_REQUESTED:
            return
        turn_row = dialogue.turns[0]
        judge_key = f"{dialogue.dialogue_id}:{JUDGE_KEY_SUFFIX}"
        try:
            if judge_key in state.completed:
                continue
            with turn_counter_lock:
                if (
                    args.max_requests
                    and turn_counter[0] >= args.max_requests
                ):
                    return
                turn_counter[0] += 1
            try:
                evaluate_turn(
                    args,
                    state,
                    metadata,
                    schema,
                    dialogue,
                    turn_row,
                    worker_id,
                )
            except Exception as error:
                state.record(
                    "deferred",
                    {
                        "judge_key": judge_key,
                        "dialogue_id": dialogue.dialogue_id,
                        "conversation_id": dialogue.conversation_id,
                        "sample_rank": dialogue.sample_rank,
                        "assistant_ordinal": int(
                            turn_row["assistant_ordinal"]
                        ),
                        "created_at": utc_now(),
                        "phase": "unexpected_dialogue_worker_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
        except Exception as error:
            state.record(
                "deferred",
                {
                    "judge_key": f"{dialogue.dialogue_id}:dialogue_worker_error",
                    "dialogue_id": dialogue.dialogue_id,
                    "conversation_id": dialogue.conversation_id,
                    "sample_rank": dialogue.sample_rank,
                    "created_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        finally:
            state.dialogue_done()


def freeze_contract(
    output_dir: Path,
    input_dir: Path,
    args: argparse.Namespace,
    schema: dict[str, Any],
    slot_assignment_path: Path,
) -> None:
    runner_path = Path(__file__).resolve()
    v6_path = Path(judge_v6.__file__).resolve()
    base_builder_path = Path(judge_v6.base_v6.__file__).resolve()
    inputs = {
        "pilot_dialogues": input_dir / "pilot_dialogues.parquet",
        "pilot_turn_manifest": input_dir / "pilot_turn_manifest.parquet",
        "pilot_dialogue_material": input_dir / "pilot_dialogue_material.parquet",
    }
    checksums = {
        "runner": sha256_file(runner_path),
        "judge_v6": sha256_file(v6_path),
        "base_prompt_builder": sha256_file(base_builder_path),
        "system_prompt": sha256_bytes(
            judge_v6.SYSTEM_PROMPT.encode("utf-8")
        ),
        "schema": sha256_bytes(compact_json(schema).encode("utf-8")),
        "slot_assignment": sha256_file(slot_assignment_path),
        "canonical_characters": sha256_file(judge_v6.CHARACTER_PATH),
        "canonical_character_descriptions": sha256_file(
            judge_v6.DESCRIPTION_PATH
        ),
        "canonical_personas": sha256_file(judge_v6.PERSONA_PATH),
        **{name: sha256_file(path) for name, path in inputs.items()},
    }
    invariant = {
        "schema_version": RUNNER_VERSION,
        "model": args.model,
        "concurrency": args.concurrency,
        "worker_ids": list(range(args.concurrency)),
        "server_scheduler": "vllm",
        "request": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": 4096,
            "thinking": False,
            "server_prefix_cache": True,
            "evaluation_unit": "dialogue_trajectory",
            "one_request_per_dialogue": True,
            "endpoint_rule": (
                "maximum assistant ordinal; must equal frozen representative "
                "target and visible segment assistant count"
            ),
            "endpoint_has_extra_weight": False,
            "baseline_score": 2,
            "model_provides_final_dimension_scores": True,
            "evidence_first_scoring": True,
            "semantic_repair_attempts": 1,
            "semantic_repair_version": SEMANTIC_REPAIR_VERSION,
            "semantic_repair_seed_rule": (
                "stable_seed(judge_key + ':' + semantic_repair_version)"
            ),
        },
        "input_dir": str(input_dir),
        "dialogue_offset": args.dialogue_offset,
        "limit_dialogues": args.limit_dialogues,
        "max_requests": args.max_requests,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "checksums": checksums,
    }
    contract = {
        **invariant,
        "created_at": utc_now(),
        "initial_base_url": args.base_url,
        "contract_sha256": sha256_bytes(
            compact_json(invariant).encode("utf-8")
        ),
    }
    contract_path = output_dir / "run_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError(
                "run contract differs from existing output; use a new directory"
            )
        return
    atomic_json(contract_path, contract)
    frozen = output_dir / "frozen_contract"
    frozen.mkdir(exist_ok=True)
    shutil.copy2(runner_path, frozen / runner_path.name)
    shutil.copy2(v6_path, frozen / v6_path.name)
    shutil.copy2(base_builder_path, frozen / base_builder_path.name)
    shutil.copy2(slot_assignment_path, frozen / slot_assignment_path.name)
    (frozen / "SYSTEM_PROMPT.md").write_text(
        judge_v6.SYSTEM_PROMPT + "\n",
        encoding="utf-8",
    )
    (frozen / "judge_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dialogue-offset", type=int, default=0)
    parser.add_argument("--limit-dialogues", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 24:
        raise ValueError("concurrency must be from 1 through 24")
    if args.dialogue_offset < 0:
        raise ValueError("dialogue-offset must be non-negative")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = acquire_lock(args.output_dir / "runner.lock")
    state = None
    try:
        transport_preflight = preflight(args)
        atomic_json(
            args.output_dir / "transport_preflight_latest.json",
            transport_preflight,
        )
        judge_v6.REQUEST_MODEL = args.model
        judge_v6.REQUEST_TEMPERATURE = 0.1
        judge_v6.REQUEST_MAX_TOKENS = 4096
        judge_v6.REQUEST_THINKING = False
        schema = judge_v6.score_schema()

        jobs = load_jobs(args.input_dir)
        if args.dialogue_offset:
            jobs = jobs[args.dialogue_offset :]
        if args.limit_dialogues:
            jobs = jobs[: args.limit_dialogues]
        assigned, slot_assignment_path = load_or_create_slot_assignment(
            args.output_dir,
            jobs,
            args.concurrency,
        )
        freeze_contract(
            args.output_dir,
            args.input_dir,
            args,
            schema,
            slot_assignment_path,
        )
        metadata = load_prompt_metadata(jobs)
        expected_requests = len(jobs)
        state = RunState(args.output_dir, len(jobs), expected_requests)
        print(
            f"[{utc_now()}] model={args.model} dialogues={len(jobs)} "
            f"requests={expected_requests} concurrency={args.concurrency} "
            f"resume_terminal={len(state.completed)}",
            flush=True,
        )

        turn_counter = [0]
        turn_counter_lock = threading.Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [
                executor.submit(
                    worker,
                    worker_id,
                    assigned[worker_id],
                    args,
                    state,
                    metadata,
                    schema,
                    turn_counter,
                    turn_counter_lock,
                )
                for worker_id in range(args.concurrency)
            ]
            for future in futures:
                future.result()
    finally:
        if state is not None:
            state.finish()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()

    assert state is not None
    remaining = max(0, expected_requests - len(state.completed))
    print(
        f"[{utc_now()}] terminal={len(state.completed)} "
        f"remaining={remaining} stop_requested={STOP_REQUESTED}",
        flush=True,
    )
    if state.fatal_error:
        return 3
    return 0 if remaining == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
