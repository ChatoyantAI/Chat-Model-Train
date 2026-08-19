#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import pathlib
import time
from collections import Counter
from typing import Any

import httpx
import pyarrow.compute as pc
import pyarrow.parquet as pq


TRAINING_ROOT = pathlib.Path(
    os.environ.get("LUSY_TRAINING_ROOT", "/home/gyr/lusychat-training")
)
RUN_ROOT = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-v47-softjson-first4k-20260728"
)
ARTIFACTS = RUN_ROOT / "artifacts"
REMOTE_SOURCE_SCRIPT = (
    TRAINING_ROOT
    / "judge-deepinfra-glm47-turn-v46-promptonly-softjson-smoke-20260728"
    / "scripts/run_v46_promptonly_softjson_smoke.py"
)
LOCAL_SOURCE_SCRIPT = pathlib.Path(__file__).with_name(
    "run_v46_promptonly_softjson_smoke.py"
)
SOURCE_SCRIPT = pathlib.Path(
    os.environ.get(
        "LUSY_TURN_JUDGE_V46_SOURCE",
        LOCAL_SOURCE_SCRIPT if LOCAL_SOURCE_SCRIPT.exists() else REMOTE_SOURCE_SCRIPT,
    )
)
ELITE_PATH = (
    TRAINING_ROOT
    / "judge-dialogue-selection-v1/02_tiers/elite_A.parquet"
)
TURN_PATH = (
    TRAINING_ROOT
    / "judge-dialogue-union-50pct-v1/01_input/turn_manifest.parquet"
)
MATERIAL_PATH = (
    TRAINING_ROOT
    / "judge-dialogue-union-50pct-v1/01_input/dialogue_material.parquet"
)

TOKEN = os.environ.get("DEEPINFRA_TOKEN", "")
API_BASE = os.environ.get(
    "DEEPINFRA_OPENAI_BASE_URL",
    "https://api.deepinfra.com/v1/openai",
).rstrip("/")
MODEL = os.environ.get(
    "DEEPINFRA_JUDGE_MODEL",
    "zai-org/GLM-4.7-Flash",
)
SEED = "lusy-turn-judge-v47-softjson-first4k-20260728-v1"
SEED_INT = 20260728
BATCH_SIZE = 2000
TOTAL_REQUESTS = 4000
TERMINAL = {"completed", "failed", "expired", "cancelled"}
DIMENSIONS = [
    "character_fidelity",
    "context_coherence",
    "current_turn_fit",
    "relationship_continuity",
    "scene_progression",
    "language_roleplay_quality",
]
FLAGS = [
    "ai_identity_break",
    "character_identity_break",
    "context_break",
    "relationship_reset",
    "wrong_language",
    "format_collapse",
    "meta_refusal",
]


def load_source():
    spec = importlib.util.spec_from_file_location("v46_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load approved v46 source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V46 = load_source()
APPROVED_PROMPT = V46.V46_PROMPT_ONLY_SOFT_JSON


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def emit(stage: str, **values: Any) -> None:
    record = {"stage": stage, "time": time.time(), **values}
    with (ARTIFACTS / "events.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def stable_hash(*values: Any) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha256((SEED + "\x1e" + payload).encode()).hexdigest()


def request_body(candidate: dict[str, Any]) -> dict[str, Any]:
    variant = (
        "greeting_present"
        if candidate["has_greeting"]
        else "greeting_absent"
    )
    body = V46.request_body(candidate, variant)
    body["prompt_cache_key"] = "lusy_turn_judge_v47_softjson_first4k"
    body["max_tokens"] = 1536
    return body


def load_and_select() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    elite_array = pq.read_table(ELITE_PATH, columns=["dialogue_id"])[
        "dialogue_id"
    ]
    turn_table = pq.read_table(
        TURN_PATH,
        columns=[
            "dialogue_id",
            "conversation_id",
            "assistant_ordinal",
            "native_message_id",
            "content_sha256",
            "replay_mode",
            "memory_id",
            "memory_boundary_increase_id",
        ],
    )
    turns = turn_table.filter(
        pc.is_in(turn_table["dialogue_id"], value_set=elite_array)
    ).to_pylist()
    elite_ids = [str(value) for value in elite_array.to_pylist()]
    material_table = pq.read_table(
        MATERIAL_PATH,
        columns=[
            "dialogue_id",
            "conversation_id",
            "replay_mode",
            "dominant_language",
            "representative_topic",
            "messages",
        ],
        filters=[("dialogue_id", "in", elite_ids)],
    )
    material_by_id = {
        str(row["dialogue_id"]): row
        for row in material_table.to_pylist()
    }
    turns_by_dialogue: dict[str, list[dict[str, Any]]] = {}
    for row in turns:
        turns_by_dialogue.setdefault(str(row["dialogue_id"]), []).append(row)

    selected: list[dict[str, Any]] = []
    errors = Counter()
    dialogue_order = sorted(
        turns_by_dialogue,
        key=lambda dialogue_id: stable_hash("dialogue", dialogue_id),
    )
    selected_dialogues = 0
    for dialogue_id in dialogue_order:
        material = material_by_id.get(dialogue_id)
        if material is None:
            errors["missing_material"] += 1
            continue
        try:
            parsed = V46.parse_replay(material["messages"] or [])
        except ValueError:
            errors["parse_error"] += 1
            continue
        candidates = []
        for row in sorted(
            turns_by_dialogue[dialogue_id],
            key=lambda value: int(value["assistant_ordinal"]),
        ):
            ordinal = int(row["assistant_ordinal"])
            if ordinal < 1 or ordinal > len(parsed["pairs"]):
                errors["ordinal"] += 1
                continue
            target = parsed["pairs"][ordinal - 1][1]
            if hashlib.sha256(target.encode()).hexdigest() != str(
                row["content_sha256"]
            ):
                errors["sha256"] += 1
                continue
            candidates.append(
                {
                    **row,
                    "dialogue_id": dialogue_id,
                    "conversation_id": str(row["conversation_id"]),
                    "native_message_id": str(row["native_message_id"]),
                    "assistant_ordinal": ordinal,
                    "language": str(material["dominant_language"]),
                    "topic": str(material["representative_topic"]),
                    "parsed": parsed,
                    "has_greeting": V46.greeting_present(parsed),
                    "turn_bucket": (
                        "01_04"
                        if ordinal <= 4
                        else V46.turn_bucket(ordinal)
                    ),
                }
            )
        if not candidates:
            continue
        selected_dialogues += 1
        remaining = TOTAL_REQUESTS - len(selected)
        selected.extend(candidates[:remaining])
        if len(selected) == TOTAL_REQUESTS:
            break
    if len(selected) != TOTAL_REQUESTS:
        raise RuntimeError(f"selected {len(selected)}, expected 4000")
    return selected, {
        "elite_turns": len(turns),
        "selected_turns": len(selected),
        "selected_dialogues": selected_dialogues,
        "errors": dict(errors),
        "language": dict(Counter(row["language"] for row in selected)),
        "topic": dict(Counter(row["topic"] for row in selected)),
        "turn_bucket": dict(
            Counter(row["turn_bucket"] for row in selected)
        ),
        "greeting": dict(
            Counter(
                "present" if row["has_greeting"] else "absent"
                for row in selected
            )
        ),
    }


def prepare() -> dict[str, tuple[pathlib.Path, dict[str, Any]]]:
    status_path = ARTIFACTS / "PREPARE_STATUS.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            prepared = {}
            for name in ["batch_00000", "batch_00001"]:
                input_path = ARTIFACTS / f"{name}.input.jsonl"
                expectations = json.loads(
                    (ARTIFACTS / f"{name}.expectations.json").read_text(
                        encoding="utf-8"
                    )
                )
                prepared[name] = (input_path, expectations)
            emit("prepare_reused", requests=TOTAL_REQUESTS)
            return prepared

    selected, report = load_and_select()
    atomic_json(ARTIFACTS / "SAMPLE_REPORT.json", report)
    prepared = {}
    all_ids = set()
    for batch_index in range(2):
        name = f"batch_{batch_index:05d}"
        rows = selected[
            batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE
        ]
        batch_rows = []
        expectations = {}
        for row in rows:
            custom_id = (
                "v47_"
                + hashlib.sha256(row["dialogue_id"].encode()).hexdigest()[:14]
                + f"_{int(row['assistant_ordinal']):04d}"
            )
            if custom_id in all_ids:
                raise RuntimeError(f"duplicate custom_id {custom_id}")
            all_ids.add(custom_id)
            batch_rows.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": request_body(row),
                }
            )
            expectations[custom_id] = {
                "dialogue_id": row["dialogue_id"],
                "conversation_id": row["conversation_id"],
                "assistant_ordinal": int(row["assistant_ordinal"]),
                "native_message_id": row["native_message_id"],
                "content_sha256": str(row["content_sha256"]),
                "language": row["language"],
                "topic": row["topic"],
                "has_greeting": bool(row["has_greeting"]),
            }
        input_path = ARTIFACTS / f"{name}.input.jsonl"
        write_jsonl(input_path, batch_rows)
        atomic_json(
            ARTIFACTS / f"{name}.expectations.json",
            expectations,
        )
        prepared[name] = (input_path, expectations)
    atomic_json(
        ARTIFACTS / "FROZEN_CONTRACT.json",
        {
            "schema_version": "lusy-turn-judge-v47-softjson-first4k-v1",
            "model": MODEL,
            "prompt": APPROVED_PROMPT,
            "prompt_sha256": hashlib.sha256(
                APPROVED_PROMPT.encode()
            ).hexdigest(),
            "seed": SEED_INT,
            "response_format": {"type": "json_object"},
            "batch_size": BATCH_SIZE,
            "batches": 2,
        },
    )
    atomic_json(
        status_path,
        {
            "status": "complete",
            "requests": TOTAL_REQUESTS,
            "batches": {
                name: {
                    "requests": len(expectations),
                    "bytes": input_path.stat().st_size,
                    "sha256": hashlib.sha256(
                        input_path.read_bytes()
                    ).hexdigest(),
                }
                for name, (input_path, expectations) in prepared.items()
            },
        },
    )
    emit("prepared", sample_report=report)
    return prepared


async def upload(client: httpx.AsyncClient, path: pathlib.Path) -> dict:
    with path.open("rb") as source:
        response = await client.post(
            f"{API_BASE}/files",
            data={
                "purpose": "batch",
                "expires_after[anchor]": "created_at",
                "expires_after[seconds]": "604800",
            },
            files={"file": (path.name, source, "application/jsonl")},
            timeout=httpx.Timeout(900.0, connect=30.0),
        )
    response.raise_for_status()
    return response.json()


async def poll(
    client: httpx.AsyncClient, name: str, batch_id: str
) -> dict:
    last = None
    started = time.monotonic()
    while True:
        response = await client.get(f"{API_BASE}/batches/{batch_id}")
        response.raise_for_status()
        value = response.json()
        snapshot = {
            key: value.get(key)
            for key in [
                "id",
                "status",
                "input_file_id",
                "output_file_id",
                "error_file_id",
                "created_at",
                "in_progress_at",
                "finalizing_at",
                "completed_at",
                "request_counts",
                "usage",
                "errors",
            ]
        }
        atomic_json(ARTIFACTS / f"{name}.batch_state.json", snapshot)
        signature = (
            snapshot.get("status"),
            json.dumps(snapshot.get("request_counts"), sort_keys=True),
        )
        if signature != last:
            emit(
                "batch_progress",
                batch=name,
                status=snapshot.get("status"),
                request_counts=snapshot.get("request_counts"),
            )
            last = signature
        if snapshot.get("status") in TERMINAL:
            snapshot["observed_wall_seconds"] = time.monotonic() - started
            atomic_json(
                ARTIFACTS / f"{name}.batch_terminal.json", snapshot
            )
            return snapshot
        await asyncio.sleep(10)


async def download(
    client: httpx.AsyncClient, file_id: str, path: pathlib.Path
) -> None:
    async with client.stream(
        "GET",
        f"{API_BASE}/files/{file_id}/content",
        timeout=httpx.Timeout(900.0, connect=30.0),
    ) as response:
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as output:
            async for chunk in response.aiter_bytes():
                output.write(chunk)
        temporary.replace(path)


def validate_result(value: Any) -> list[str]:
    errors = []
    if not isinstance(value, dict):
        return ["root_not_object"]
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, dict):
        errors.append("dimensions")
    else:
        for name in DIMENSIONS:
            item = dimensions.get(name)
            if not isinstance(item, dict):
                errors.append(f"{name}:shape")
                continue
            score = item.get("score")
            reason = item.get("reason")
            if not isinstance(score, int) or isinstance(score, bool):
                errors.append(f"{name}:score_type")
            elif score < 0 or score > 4:
                errors.append(f"{name}:score_range")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"{name}:reason")
    intimacy = value.get("intimacy_quality")
    if not isinstance(intimacy, dict):
        errors.append("intimacy")
    else:
        applicable = intimacy.get("applicable")
        score = intimacy.get("score")
        if not isinstance(applicable, bool):
            errors.append("intimacy:applicable")
        if not isinstance(score, int) or isinstance(score, bool):
            errors.append("intimacy:score_type")
        elif applicable is False and score != -1:
            errors.append("intimacy:not_applicable_score")
        elif applicable is True and not 0 <= score <= 4:
            errors.append("intimacy:applicable_score")
        if not isinstance(intimacy.get("reason"), str):
            errors.append("intimacy:reason")
    repetition = value.get("semantic_repetition")
    if not isinstance(repetition, dict) or repetition.get("severity") not in {
        "none",
        "minor",
        "major",
        "critical",
    }:
        errors.append("semantic_repetition")
    flags = value.get("critical_flags")
    if not isinstance(flags, dict):
        errors.append("critical_flags")
    else:
        for key in FLAGS:
            if not isinstance(flags.get(key), bool):
                errors.append(f"critical_flags:{key}")
    for key in ["critical_reason", "nsfw_reason", "overall_reason"]:
        if not isinstance(value.get(key), str) or not value[key].strip():
            errors.append(key)
    for key in ["nsfw_level", "confidence"]:
        number = value.get(key)
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or not 0 <= number <= 4
        ):
            errors.append(key)
    return errors


def analyze(
    name: str,
    output_path: pathlib.Path,
    error_path: pathlib.Path,
    expectations: dict[str, Any],
) -> dict:
    seen = Counter()
    rows = []
    counts = Counter()
    scores = {dimension: Counter() for dimension in DIMENSIONS}
    usage = Counter()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        outer = json.loads(line)
        custom_id = str(outer.get("custom_id") or "")
        seen[custom_id] += 1
        response = outer.get("response") or {}
        body = response.get("body") or {}
        token_usage = body.get("usage") or {}
        details = (
            token_usage.get("prompt_tokens_details")
            or token_usage.get("input_tokens_details")
            or {}
        )
        usage["input_tokens"] += int(
            token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0
        )
        usage["cached_tokens"] += int(details.get("cached_tokens") or 0)
        usage["output_tokens"] += int(
            token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        )
        errors = []
        value = None
        if int(response.get("status_code") or 0) != 200:
            errors.append("provider_error")
        else:
            choices = body.get("choices") or []
            content = (
                ((choices[0].get("message") or {}).get("content"))
                if choices
                else None
            )
            try:
                value = json.loads(content)
            except Exception:
                errors.append("json_parse")
            if value is not None:
                errors.extend(validate_result(value))
        if errors:
            counts.update(errors)
        else:
            counts["valid"] += 1
            for dimension in DIMENSIONS:
                scores[dimension][
                    str(value["dimensions"][dimension]["score"])
                ] += 1
        rows.append(
            {
                "custom_id": custom_id,
                "expected": expectations.get(custom_id),
                "raw_result": value,
                "errors": sorted(set(errors)),
            }
        )
    error_ids = set()
    if error_path.exists():
        for line in error_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                error_ids.add(
                    str(json.loads(line).get("custom_id") or "")
                )
    missing = set(expectations) - set(seen) - error_ids
    duplicate = sum(max(0, count - 1) for count in seen.values())
    write_jsonl(ARTIFACTS / f"{name}.normalized.jsonl", rows)
    retry_ids = {
        row["custom_id"] for row in rows if row["errors"]
    } | missing | error_ids
    input_by_id = {
        row["custom_id"]: row
        for row in (
            json.loads(line)
            for line in (
                ARTIFACTS / f"{name}.input.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    write_jsonl(
        ARTIFACTS / f"{name}.retry_input.jsonl",
        [
            input_by_id[custom_id]
            for custom_id in sorted(retry_ids)
            if custom_id in input_by_id
        ],
    )
    report = {
        "expected": len(expectations),
        "outputs": len(rows),
        "valid": counts["valid"],
        "valid_rate": counts["valid"] / len(expectations),
        "missing": len(missing),
        "duplicate": duplicate,
        "error_file": len(error_ids),
        "retry": len(retry_ids),
        "error_counts": {
            key: value for key, value in counts.items() if key != "valid"
        },
        "score_distributions": {
            key: dict(value) for key, value in scores.items()
        },
        "usage": {
            **usage,
            "cache_fraction": (
                usage["cached_tokens"] / usage["input_tokens"]
                if usage["input_tokens"]
                else 0
            ),
        },
    }
    atomic_json(ARTIFACTS / f"{name}.analysis.json", report)
    return report


async def run_one(
    client: httpx.AsyncClient,
    name: str,
    input_path: pathlib.Path,
    expectations: dict[str, Any],
) -> dict:
    analysis_path = ARTIFACTS / f"{name}.analysis.json"
    terminal_path = ARTIFACTS / f"{name}.batch_terminal.json"
    if analysis_path.exists() and terminal_path.exists():
        return {
            "terminal": json.loads(terminal_path.read_text()),
            "analysis": json.loads(analysis_path.read_text()),
        }
    created_path = ARTIFACTS / f"{name}.batch_created.json"
    if created_path.exists():
        created = json.loads(created_path.read_text())
        batch_id = str(created["id"])
        input_file_id = str(created["input_file_id"])
        emit("batch_resuming", batch=name, batch_id=batch_id)
    else:
        uploaded = await upload(client, input_path)
        input_file_id = str(uploaded["id"])
        atomic_json(ARTIFACTS / f"{name}.upload.json", uploaded)
        response = await client.post(
            f"{API_BASE}/batches",
            json={
                "input_file_id": input_file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        response.raise_for_status()
        created = response.json()
        batch_id = str(created["id"])
        atomic_json(created_path, created)
        emit("batch_created", batch=name, batch_id=batch_id, requests=2000)
    terminal = await poll(client, name, batch_id)
    output_path = ARTIFACTS / f"{name}.output.jsonl"
    error_path = ARTIFACTS / f"{name}.errors.jsonl"
    if terminal.get("output_file_id"):
        await download(
            client, str(terminal["output_file_id"]), output_path
        )
    else:
        output_path.write_text("", encoding="utf-8")
    if terminal.get("error_file_id"):
        await download(
            client, str(terminal["error_file_id"]), error_path
        )
    else:
        error_path.write_text("", encoding="utf-8")
    analysis = analyze(name, output_path, error_path, expectations)
    cleanup = []
    for file_id in [
        input_file_id,
        str(terminal.get("output_file_id") or ""),
        str(terminal.get("error_file_id") or ""),
    ]:
        if not file_id:
            continue
        try:
            response = await client.delete(f"{API_BASE}/files/{file_id}")
            cleanup.append(
                {
                    "id": file_id,
                    "status_code": response.status_code,
                    "deleted": response.status_code < 300
                    or response.status_code == 404,
                }
            )
        except Exception as error:
            cleanup.append(
                {"id": file_id, "deleted": False, "error": str(error)[:200]}
            )
    atomic_json(ARTIFACTS / f"{name}.remote_cleanup.json", cleanup)
    emit(
        "batch_complete",
        batch=name,
        valid=analysis["valid"],
        retry=analysis["retry"],
    )
    return {"terminal": terminal, "analysis": analysis}


async def main() -> None:
    started = time.monotonic()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    prepared = prepare()
    if not TOKEN:
        raise RuntimeError("DEEPINFRA_TOKEN missing")
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
        "User-Agent": "lusychat-turn-judge-v47-first4k/1.0",
    }
    async with httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(300.0, connect=30.0, pool=300.0),
        limits=httpx.Limits(
            max_connections=8,
            max_keepalive_connections=8,
            keepalive_expiry=120,
        ),
    ) as client:
        names = ["batch_00000", "batch_00001"]
        values = await asyncio.gather(
            *[
                run_one(client, name, *prepared[name])
                for name in names
            ]
        )
    final = {
        "schema_version": "lusy-turn-judge-v47-softjson-first4k-v1",
        "model": MODEL,
        "requests": TOTAL_REQUESTS,
        "results": dict(zip(names, values)),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(ARTIFACTS / "FINAL_REPORT.json", final)
    emit("complete", wall_seconds=final["wall_seconds"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        atomic_json(
            ARTIFACTS / "FATAL_ERROR.json",
            {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "time": time.time(),
            },
        )
        raise
