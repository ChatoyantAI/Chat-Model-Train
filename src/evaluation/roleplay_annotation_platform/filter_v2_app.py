"""Standalone V2 single-sample filtering service.

This app deliberately uses its own cookie, result database and result directory.
It reads the shared sample database but never writes to the V1 annotation store.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import sqlite3
import time
import zlib
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .llm_judge import (
    backend_model_id,
    build_sample_packet,
    extract_judgment,
    judge_model_gateway,
    load_judge_config,
    provider_preferences_for_model,
    score_logprobs_enabled_for_model,
)
from .single_sample_filter_v2 import (
    PROTOCOL_VERSION,
    SCORE_TOP_LOGPROBS,
    attach_score_distributions,
    build_gate_messages,
    build_gate_schema,
    build_score_messages,
    build_score_schema,
    compact_score_logprobs,
    decode_score_output,
    deficient_item_ids,
    item_map,
    load_filter_rubric_v2,
    not_applicable_item_ids,
    rubric_items,
    summarize_scores,
    validate_gate_output,
    validate_score_output,
)
from .model_catalog import OpenRouterModelCatalog


MODULE_DIR = Path(__file__).resolve().parent
STATIC_DIR = MODULE_DIR / "static_filter_v2"
RUBRIC_PATH = Path(
    os.environ.get(
        "ROLEPLAY_FILTER_V2_RUBRIC_JSON",
        MODULE_DIR / "single_sample_filter_rubric_v2.json",
    )
)
SAMPLE_DB = Path(
    os.environ.get(
        "ROLEPLAY_FILTER_V2_SAMPLE_DB", MODULE_DIR / "data" / "samples.sqlite3"
    )
)
RESULT_DB = Path(
    os.environ.get(
        "ROLEPLAY_FILTER_V2_RESULTS_DB",
        MODULE_DIR / "results_v2" / "filter_reviews.sqlite3",
    )
)
RESULT_DIR = Path(
    os.environ.get(
        "ROLEPLAY_FILTER_V2_RESULTS_DIR", MODULE_DIR / "results_v2" / "by_user"
    )
)
RUBRIC_DOC = Path(
    os.environ.get(
        "ROLEPLAY_FILTER_V2_RUBRIC_DOC",
        MODULE_DIR.parents[2]
        / "docs"
        / "evaluation"
        / "ROLEPLAY_SINGLE_SAMPLE_FILTER_RUBRIC_V2.md",
    )
)
COOKIE_NAME = "roleplay_filter_v2_session"
USERNAME_RE = re.compile(r"^[\w\-]{1,40}$", re.UNICODE)
APP_STATE: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def result_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(RESULT_DB, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@contextmanager
def sample_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(
        f"file:{SAMPLE_DB.as_posix()}?mode=ro", uri=True, timeout=30
    )
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def initialize_results() -> None:
    RESULT_DB.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with result_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES users(username),
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS filter_reviews (
                username TEXT NOT NULL REFERENCES users(username),
                seq INTEGER NOT NULL,
                evaluation_id TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('draft', 'submitted')),
                gates_json TEXT NOT NULL,
                items_json TEXT NOT NULL,
                notes TEXT NOT NULL,
                decision TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(username, seq, protocol_version)
            );
            CREATE INDEX IF NOT EXISTS idx_filter_reviews_user_status
                ON filter_reviews(username, protocol_version, status, seq);
            CREATE TABLE IF NOT EXISTS llm_filter_runs (
                run_id TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES users(username),
                seq INTEGER NOT NULL,
                evaluation_id TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                phase TEXT NOT NULL CHECK(phase IN ('primary', 'verification')),
                parent_run_id TEXT,
                model_requested TEXT NOT NULL,
                model_resolved TEXT,
                provider TEXT,
                status TEXT NOT NULL CHECK(status IN ('completed', 'invalid', 'failed')),
                output_json TEXT,
                decision_json TEXT,
                raw_response_json TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_llm_filter_runs_user_created
                ON llm_filter_runs(username, created_at DESC);
            """
        )


def load_total_samples() -> int:
    if not SAMPLE_DB.is_file():
        raise RuntimeError(f"sample database not found: {SAMPLE_DB}")
    with sample_connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM samples").fetchone()
    total = int(row["count"] or 0)
    if total <= 0:
        raise RuntimeError("sample database contains no samples")
    return total


@asynccontextmanager
async def lifespan(_: FastAPI):
    APP_STATE["rubric"] = load_filter_rubric_v2(RUBRIC_PATH)
    APP_STATE["total"] = load_total_samples()
    initialize_results()
    base_config = load_judge_config()
    catalog = OpenRouterModelCatalog(base_config)
    await catalog.refresh()
    APP_STATE["llm_catalog"] = catalog
    APP_STATE["llm_config"] = replace(
        base_config,
        models=catalog.model_ids,
        reasoning_models=catalog.reasoning_model_ids,
    )
    yield


app = FastAPI(
    title="Roleplay Single-Sample Filter V2",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; base-uri 'none'; connect-src 'self'; "
        "font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
        "img-src 'self' data:; object-src 'none'; script-src 'self'; "
        "style-src 'self'"
    )
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), payment=()"
    )
    return response


def require_username(request: Request) -> str:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="请先输入用户名")
    now = utc_now()
    with result_connection() as connection:
        row = connection.execute(
            "SELECT username FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=401, detail="登录状态已失效")
        username = str(row["username"])
        connection.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE token = ?", (now, token)
        )
        connection.execute(
            "UPDATE users SET last_seen_at = ? WHERE username = ?", (now, username)
        )
    return username


def sample_row(seq: int) -> sqlite3.Row:
    if seq < 1 or seq > int(APP_STATE["total"]):
        raise HTTPException(status_code=404, detail="样本序号不存在")
    with sample_connection() as connection:
        row = connection.execute(
            "SELECT seq, evaluation_id, trigger_type, payload_zlib FROM samples WHERE seq = ?",
            (seq,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="样本不存在")
    return row


def decode_sample(row: sqlite3.Row) -> dict[str, Any]:
    record = json.loads(zlib.decompress(row["payload_zlib"]).decode("utf-8"))
    messages: list[dict[str, str]] = []
    greeting_seen = False
    for message in record["input_messages"]:
        role = str(message.get("role", "unknown"))
        display_role = role
        if role == "assistant" and not greeting_seen:
            display_role = "greeting"
            greeting_seen = True
        messages.append(
            {
                "role": role,
                "display_role": display_role,
                "content": str(message.get("content", "")),
            }
        )
    output = record["output_message"]
    return {
        "seq": int(row["seq"]),
        "evaluation_id": str(row["evaluation_id"]),
        "trigger_type": str(row["trigger_type"]),
        "messages": messages,
        "candidate": {"role": "assistant", "content": str(output.get("content", ""))},
    }


def existing_review(username: str, seq: int) -> dict[str, Any] | None:
    with result_connection() as connection:
        row = connection.execute(
            "SELECT status, gates_json, items_json, notes, decision, updated_at "
            "FROM filter_reviews WHERE username = ? AND seq = ? AND protocol_version = ?",
            (username, seq, PROTOCOL_VERSION),
        ).fetchone()
    if row is None:
        return None
    return {
        "status": str(row["status"]),
        "gates": json.loads(str(row["gates_json"])),
        "items": json.loads(str(row["items_json"])),
        "notes": str(row["notes"]),
        "decision": str(row["decision"]),
        "updated_at": str(row["updated_at"]),
    }


def progress_for(username: str) -> dict[str, int]:
    with result_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS saved, "
            "SUM(CASE WHEN status = 'submitted' THEN 1 ELSE 0 END) AS submitted "
            "FROM filter_reviews WHERE username = ? AND protocol_version = ?",
            (username, PROTOCOL_VERSION),
        ).fetchone()
    return {
        "saved": int(row["saved"] or 0),
        "submitted": int(row["submitted"] or 0),
        "total": int(APP_STATE["total"]),
    }


def next_unsubmitted(username: str, after: int = 0) -> int | None:
    total = int(APP_STATE["total"])
    with result_connection() as connection:
        rows = connection.execute(
            "SELECT seq FROM filter_reviews WHERE username = ? "
            "AND protocol_version = ? AND status = 'submitted'",
            (username, PROTOCOL_VERSION),
        ).fetchall()
    submitted = {int(row["seq"]) for row in rows}
    if len(submitted) >= total:
        return None
    for seq in range(after + 1, total + 1):
        if seq not in submitted:
            return seq
    for seq in range(1, after + 1):
        if seq not in submitted:
            return seq
    return None


def normalize_human_review(
    gates_raw: Any, items_raw: Any, *, submitting: bool
) -> tuple[dict[str, bool], dict[str, str], str]:
    rubric = APP_STATE["rubric"]
    gate_ids = [gate["id"] for gate in rubric["gates"]]
    item_by_id = item_map(rubric)
    if not isinstance(gates_raw, dict):
        raise HTTPException(status_code=400, detail="gates 必须是对象")
    if not isinstance(items_raw, dict):
        raise HTTPException(status_code=400, detail="items 必须是对象")
    if set(gates_raw) - set(gate_ids):
        raise HTTPException(status_code=400, detail="包含未知 Gate")
    if set(items_raw) - set(item_by_id):
        raise HTTPException(status_code=400, detail="包含未知原子项")

    gates: dict[str, bool] = {}
    for gate_id, value in gates_raw.items():
        if type(value) is not bool:
            raise HTTPException(status_code=400, detail=f"{gate_id} 必须为布尔值")
        gates[gate_id] = value
    if submitting and set(gates) != set(gate_ids):
        raise HTTPException(status_code=400, detail="请完成全部 Gate 判断")

    items: dict[str, str] = {}
    for item_id, value in items_raw.items():
        legal = {str(candidate) for candidate in item_by_id[item_id]["scale"]}
        if item_by_id[item_id].get("allows_na"):
            legal.add("N/A")
        value = str(value)
        if value not in legal:
            raise HTTPException(
                status_code=400,
                detail=f"{item_id} 分数无效；允许值为 {sorted(legal)}",
            )
        items[item_id] = value

    failed_gates = sorted(gate_id for gate_id, value in gates.items() if value)
    if submitting and not failed_gates and set(items) != set(item_by_id):
        missing = sorted(set(item_by_id) - set(items))
        raise HTTPException(status_code=400, detail=f"请完成全部原子项：{missing}")
    if failed_gates:
        decision = "needs_gate_verification"
    elif any(value.isdigit() and int(value) < 2 for value in items.values()):
        decision = "needs_item_verification"
    elif submitting:
        decision = "keep_candidate"
    else:
        decision = "draft"
    return gates, items, decision


def write_snapshot(
    *,
    username: str,
    seq: int,
    evaluation_id: str,
    status: str,
    gates: dict[str, bool],
    items: dict[str, str],
    notes: str,
    decision: str,
    updated_at: str,
) -> None:
    user_dir = RESULT_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "username": username,
        "seq": seq,
        "evaluation_id": evaluation_id,
        "status": status,
        "gates": gates,
        "items": items,
        "score_summary": summarize_scores(
            {
                item_id: {
                    "value": value,
                    "code": None,
                    "distribution_complete": False,
                }
                for item_id, value in items.items()
            },
            APP_STATE["rubric"],
        ),
        "notes": notes,
        "decision": decision,
        "updated_at": updated_at,
    }
    target = user_dir / f"{seq:06d}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "cost_usd": max(0.0, float(usage.get("cost") or 0)),
    }


def _combined_usage(*payloads: dict[str, Any]) -> dict[str, Any]:
    result = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    for payload in payloads:
        usage = _usage(payload)
        for key in result:
            result[key] += usage[key]
    result["cost_usd"] = round(float(result["cost_usd"]), 10)
    return result


def _primary_decision(
    gates: dict[str, bool],
    scores: dict[str, dict[str, Any]],
    score_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    gate_targets = sorted(gate_id for gate_id, value in gates.items() if value)
    item_targets = deficient_item_ids(scores)
    return {
        "decision": "keep" if not gate_targets and not item_targets else "review",
        "verification_targets": {"gates": gate_targets, "items": item_targets},
        "quality_score": score_summary.get("quality_score") if score_summary else None,
        "scoring_policy": (
            "review when P(score<2 | applicable)>=0.5; discrete score<2 is fallback"
        ),
    }


def _catalog_model(model: str) -> dict[str, Any] | None:
    snapshot = APP_STATE["llm_catalog"].public_snapshot()
    return next(
        (entry for entry in snapshot.get("models", []) if entry.get("id") == model),
        None,
    )


def _score_logprobs_available(model: str) -> bool:
    entry = _catalog_model(model)
    return bool(
        entry
        and entry.get("reports_token_logprobs") is True
        and score_logprobs_enabled_for_model(APP_STATE["llm_config"], model)
    )


async def _call_openrouter(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    schema_name: str,
    temperature: float,
    request_logprobs: bool = False,
) -> tuple[dict[str, Any], str, dict[str, Any], int]:
    config = APP_STATE["llm_config"]
    if not config.enabled or not config.api_key:
        raise HTTPException(status_code=503, detail="服务器尚未配置 OpenRouter Key")
    if model not in config.models:
        raise HTTPException(status_code=400, detail="模型不在服务器允许列表中")
    if judge_model_gateway(model) != "openrouter":
        raise HTTPException(status_code=400, detail="V2 当前只开放 OpenRouter 模型")
    provider_preferences = provider_preferences_for_model(config, model)
    ignored = list(provider_preferences.get("ignore") or [])
    for unavailable in ("coreweave", "coreweave/fp4", "coreweave/fp8"):
        if unavailable not in ignored:
            ignored.append(unavailable)
    provider_preferences["ignore"] = ignored
    if model == "google/gemma-4-31b-it":
        # Parasail advertises the required combination of structured outputs,
        # logprobs and the full context window. Novita's shared pool was
        # repeatedly rate-limited during the live V2 smoke test.
        provider_preferences["order"] = [
            "parasail/fp8",
            "venice/bf16",
            "novita/bf16",
        ]
    body: dict[str, Any] = {
        "model": backend_model_id(model),
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        "provider": provider_preferences,
    }
    if request_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = SCORE_TOP_LOGPROBS
        body["top_p"] = 1
    if model in config.reasoning_models:
        body["reasoning"] = {"effort": config.reasoning_effort, "exclude": True}
    started = time.perf_counter()
    response: httpx.Response | None = None
    last_transport_error: httpx.TransportError | None = None
    attempts = max(1, int(config.rate_limit_max_attempts))
    async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
        for attempt in range(attempts):
            try:
                response = await client.post(
                    config.openrouter_url,
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                last_transport_error = None
            except httpx.TransportError as error:
                last_transport_error = error
                response = None
            retryable = response is None or response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }
            if response is not None and response.status_code == 403:
                try:
                    error_payload = response.json()
                    metadata = (error_payload.get("error") or {}).get("metadata") or {}
                    previous = metadata.get("previous_errors") or []
                    retryable = any(
                        isinstance(record, dict) and int(record.get("code") or 0) == 429
                        for record in previous
                    )
                except (AttributeError, TypeError, ValueError):
                    retryable = False
            if not retryable or attempt + 1 >= attempts:
                break
            retry_after = (
                response.headers.get("Retry-After") if response is not None else None
            )
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            if delay <= 0:
                delay = min(
                    float(config.rate_limit_backoff_max_seconds),
                    float(config.rate_limit_backoff_seconds) * (2**attempt),
                )
            await asyncio.sleep(max(0.0, delay))
    if response is None:
        error_name = (
            type(last_transport_error).__name__ if last_transport_error else "unknown"
        )
        raise HTTPException(
            status_code=502, detail=f"OpenRouter 网络错误：{error_name}"
        ) from None
    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter 返回非 JSON（HTTP {response.status_code}）",
        ) from None
    if response.status_code >= 400:
        detail = payload.get("error", payload) if isinstance(payload, dict) else payload
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter HTTP {response.status_code}：{str(detail)[:500]}",
        )
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="OpenRouter 返回结构无效")
    try:
        output, raw_content = extract_judgment(payload)
    except Exception as error:
        raise HTTPException(
            status_code=422, detail=f"模型输出不是有效 JSON：{str(error)[:500]}"
        ) from None
    return output, raw_content, payload, latency_ms


def _combined_payload(
    *,
    model: str,
    gate_payload: dict[str, Any] | None,
    score_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    payloads = [
        payload for payload in ([gate_payload] if gate_payload else []) + score_payloads
    ]
    usage = _combined_usage(*payloads)
    resolved = next(
        (
            payload.get("model")
            for payload in reversed(payloads)
            if payload.get("model")
        ),
        backend_model_id(model),
    )
    provider = next(
        (
            payload.get("provider")
            for payload in reversed(payloads)
            if payload.get("provider")
        ),
        None,
    )
    return {
        "model": resolved,
        "provider": provider,
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cost": usage["cost_usd"],
            "completion_tokens_details": {
                "reasoning_tokens": usage["reasoning_tokens"]
            },
        },
        "phases": {
            "gate": gate_payload,
            "score_groups": score_payloads,
        },
    }


def _score_batches(
    rubric: dict[str, Any], item_ids: list[str] | None = None
) -> list[tuple[str, list[str]]]:
    selected = set(item_ids) if item_ids is not None else None
    batches: list[tuple[str, list[str]]] = []
    for dimension in rubric["dimensions"]:
        candidates = [
            item
            for item in dimension["items"]
            if selected is None or str(item["id"]) in selected
        ]
        current_ids: list[str] = []
        current_chars = 0
        batch_number = 1
        for item in candidates:
            item_chars = len(str(item.get("reference_markdown") or ""))
            if current_ids and current_chars + item_chars > 6000:
                batches.append((f"{dimension['id']}_{batch_number}", current_ids))
                batch_number += 1
                current_ids = []
                current_chars = 0
            current_ids.append(str(item["id"]))
            current_chars += item_chars
        if current_ids:
            batches.append((f"{dimension['id']}_{batch_number}", current_ids))
    return batches


async def _score_candidate(
    *,
    model: str,
    packet: dict[str, Any],
    item_ids: list[str] | None = None,
    independent: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]], int, bool]:
    rubric = APP_STATE["rubric"]
    batches = _score_batches(rubric, item_ids)
    logprobs_requested = _score_logprobs_available(model)
    # Two parallel groups keep interactive latency reasonable without bursting
    # five long requests into one upstream provider at the same instant.
    semaphore = asyncio.Semaphore(
        max(1, min(2, APP_STATE["llm_config"].max_concurrency))
    )

    async def run_batch(dimension_id: str, ids: list[str]):
        async with semaphore:
            output, raw_content, payload, latency_ms = await _call_openrouter(
                model=model,
                messages=build_score_messages(
                    rubric, packet, item_ids=ids, independent=independent
                ),
                schema=build_score_schema(rubric, ids),
                schema_name=f"roleplay_filter_{dimension_id}_scores_v2",
                temperature=1,
                request_logprobs=logprobs_requested,
            )
        validate_score_output(output, rubric, ids)
        scores = decode_score_output(output, rubric, ids)
        usable = (
            attach_score_distributions(scores, payload, raw_content, rubric)
            if logprobs_requested
            else 0
        )
        if not logprobs_requested:
            for record in scores.values():
                record["distribution_method"] = "route_does_not_report_logprobs"
        compact_score_logprobs(payload, usable_distribution_count=usable)
        payload["score_group"] = dimension_id
        payload["score_item_ids"] = ids
        return scores, payload, latency_ms

    try:
        results = await asyncio.gather(
            *(run_batch(dimension_id, ids) for dimension_id, ids in batches)
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail=f"V2 评分输出校验失败：{error}"
        ) from None

    scores: dict[str, dict[str, Any]] = {}
    payloads: list[dict[str, Any]] = []
    latencies: list[int] = []
    for batch_scores, payload, latency_ms in results:
        scores.update(batch_scores)
        payloads.append(payload)
        latencies.append(latency_ms)
    summary = summarize_scores(scores, rubric)
    summary["score_request_count"] = len(batches)
    summary["score_request_groups"] = [dimension_id for dimension_id, _ in batches]
    return scores, summary, payloads, max(latencies, default=0), logprobs_requested


def _store_llm_run(
    *,
    username: str,
    seq: int,
    evaluation_id: str,
    phase: str,
    model: str,
    parent_run_id: str | None,
    output: dict[str, Any],
    decision: dict[str, Any],
    payload: dict[str, Any],
    latency_ms: int,
) -> str:
    run_id = f"fv2_{secrets.token_urlsafe(18)}"
    usage = _usage(payload)
    choice = ((payload.get("choices") or [{}])[0]) if isinstance(payload, dict) else {}
    model_resolved = str(payload.get("model", "")) or None
    provider = str(payload.get("provider", "")) or None
    with result_connection() as connection:
        connection.execute(
            """
            INSERT INTO llm_filter_runs(
                run_id, username, seq, evaluation_id, protocol_version, phase,
                parent_run_id, model_requested, model_resolved, provider, status,
                output_json, decision_json, raw_response_json, prompt_tokens,
                completion_tokens, reasoning_tokens, cost_usd, latency_ms,
                error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                run_id,
                username,
                seq,
                evaluation_id,
                PROTOCOL_VERSION,
                phase,
                parent_run_id,
                model,
                model_resolved,
                provider,
                json.dumps(output, ensure_ascii=False, separators=(",", ":")),
                json.dumps(decision, ensure_ascii=False, separators=(",", ":")),
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage["reasoning_tokens"],
                usage["cost_usd"],
                latency_ms,
                utc_now(),
            ),
        )
    return run_id


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/rubric")
def rubric_document() -> PlainTextResponse:
    if not RUBRIC_DOC.is_file():
        raise HTTPException(status_code=404, detail="V2 文档不存在")
    return PlainTextResponse(
        RUBRIC_DOC.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8"
    )


@app.get("/health")
def health() -> dict[str, Any]:
    config = APP_STATE.get("llm_config")
    return {
        "status": "ok",
        "service": "roleplay-single-sample-filter-v2",
        "protocol_version": PROTOCOL_VERSION,
        "samples": int(APP_STATE.get("total", 0)),
        "llm_enabled": bool(config and config.enabled),
        "result_db": "isolated",
    }


@app.post("/api/login")
async def login(request: Request) -> JSONResponse:
    body = await request.json()
    username = str(body.get("username", "")).strip()
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(
            status_code=400, detail="用户名须为 1–40 个中文、字母、数字、下划线或连字符"
        )
    now = utc_now()
    token = secrets.token_urlsafe(32)
    with result_connection() as connection:
        connection.execute(
            "INSERT INTO users(username, created_at, last_seen_at) VALUES (?, ?, ?) "
            "ON CONFLICT(username) DO UPDATE SET last_seen_at = excluded.last_seen_at",
            (username, now, now),
        )
        connection.execute(
            "INSERT INTO sessions(token, username, created_at, last_seen_at) VALUES (?, ?, ?, ?)",
            (token, username, now, now),
        )
    response = JSONResponse(
        {
            "username": username,
            "progress": progress_for(username),
            "next_seq": next_unsubmitted(username),
        }
    )
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        secure=False,
    )
    return response


@app.post("/api/logout")
def logout(request: Request) -> JSONResponse:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        with result_connection() as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME)
    return response


@app.get("/api/me")
def me(request: Request) -> dict[str, Any]:
    username = require_username(request)
    return {
        "username": username,
        "progress": progress_for(username),
        "next_seq": next_unsubmitted(username),
    }


@app.get("/api/rubric")
def rubric(request: Request) -> dict[str, Any]:
    require_username(request)
    return APP_STATE["rubric"]


@app.get("/api/sample/{seq}")
def sample(seq: int, request: Request) -> dict[str, Any]:
    username = require_username(request)
    payload = decode_sample(sample_row(seq))
    payload["total"] = int(APP_STATE["total"])
    payload["review"] = existing_review(username, seq)
    payload["progress"] = progress_for(username)
    return payload


@app.get("/api/prompt-preview/{seq}")
def prompt_preview(seq: int, request: Request) -> dict[str, Any]:
    require_username(request)
    decoded = decode_sample(sample_row(seq))
    packet = build_sample_packet(decoded)
    score_requests = [
        {
            "group": dimension_id,
            "item_ids": ids,
            "messages": build_score_messages(APP_STATE["rubric"], packet, item_ids=ids),
            "schema": build_score_schema(APP_STATE["rubric"], ids),
            "temperature": 1,
            "logprobs": True,
            "top_logprobs": SCORE_TOP_LOGPROBS,
        }
        for dimension_id, ids in _score_batches(APP_STATE["rubric"])
    ]
    return {
        "gate": {
            "messages": build_gate_messages(APP_STATE["rubric"], packet),
            "schema": build_gate_schema(APP_STATE["rubric"]),
            "temperature": 0,
            "logprobs": False,
        },
        "score_requests": score_requests,
        "sample_packet": packet,
    }


@app.get("/api/llm/config")
def llm_config(request: Request) -> dict[str, Any]:
    require_username(request)
    config = APP_STATE["llm_config"]
    return {
        "enabled": bool(config.enabled),
        "models": [
            model
            for model in config.models
            if judge_model_gateway(model) == "openrouter"
        ],
        "server_side_key": bool(config.api_key),
        "gate_temperature": 0,
        "score_temperature": 1,
        "score_top_logprobs": SCORE_TOP_LOGPROBS,
        "score_protocol": "single-character score codes; expected value when all legal logits are returned",
        "catalog": APP_STATE["llm_catalog"].public_snapshot(),
    }


@app.post("/api/review")
async def save_review(request: Request) -> dict[str, Any]:
    username = require_username(request)
    body = await request.json()
    if body.get("protocol_version") != PROTOCOL_VERSION:
        raise HTTPException(status_code=409, detail="V2 规则已经变化，请刷新页面")
    try:
        seq = int(body.get("seq"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="样本序号无效") from None
    status = str(body.get("status", "draft"))
    if status not in {"draft", "submitted"}:
        raise HTTPException(status_code=400, detail="状态无效")
    gates, items, decision = normalize_human_review(
        body.get("gates", {}), body.get("items", {}), submitting=status == "submitted"
    )
    notes = str(body.get("notes", "")).strip()
    if len(notes) > 10000:
        raise HTTPException(status_code=400, detail="备注过长")
    row = sample_row(seq)
    evaluation_id = str(row["evaluation_id"])
    now = utc_now()
    with result_connection() as connection:
        existing = connection.execute(
            "SELECT created_at FROM filter_reviews WHERE username = ? AND seq = ? AND protocol_version = ?",
            (username, seq, PROTOCOL_VERSION),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        connection.execute(
            """
            INSERT INTO filter_reviews(
                username, seq, evaluation_id, protocol_version, status, gates_json,
                items_json, notes, decision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, seq, protocol_version) DO UPDATE SET
                evaluation_id=excluded.evaluation_id, status=excluded.status,
                gates_json=excluded.gates_json, items_json=excluded.items_json,
                notes=excluded.notes, decision=excluded.decision,
                updated_at=excluded.updated_at
            """,
            (
                username,
                seq,
                evaluation_id,
                PROTOCOL_VERSION,
                status,
                json.dumps(gates, ensure_ascii=False, separators=(",", ":")),
                json.dumps(items, ensure_ascii=False, separators=(",", ":")),
                notes,
                decision,
                created_at,
                now,
            ),
        )
    write_snapshot(
        username=username,
        seq=seq,
        evaluation_id=evaluation_id,
        status=status,
        gates=gates,
        items=items,
        notes=notes,
        decision=decision,
        updated_at=now,
    )
    return {
        "ok": True,
        "status": status,
        "decision": decision,
        "progress": progress_for(username),
        "next_seq": next_unsubmitted(username, after=seq)
        if status == "submitted"
        else seq,
        "updated_at": now,
    }


@app.post("/api/llm/primary")
async def llm_primary(request: Request) -> dict[str, Any]:
    username = require_username(request)
    body = await request.json()
    try:
        seq = int(body.get("seq"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="样本序号无效") from None
    model = str(body.get("model", "")).strip()
    decoded = decode_sample(sample_row(seq))
    packet = build_sample_packet(decoded)
    gate_output, _, gate_payload, gate_latency_ms = await _call_openrouter(
        model=model,
        messages=build_gate_messages(APP_STATE["rubric"], packet),
        schema=build_gate_schema(APP_STATE["rubric"]),
        schema_name="roleplay_single_filter_gate_v2",
        temperature=0,
    )
    try:
        validate_gate_output(gate_output, APP_STATE["rubric"])
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail=f"V2 Gate 输出校验失败：{error}"
        ) from None

    gates = gate_output["gates"]
    score_payloads: list[dict[str, Any]] = []
    scores: dict[str, dict[str, Any]] = {}
    score_summary: dict[str, Any] | None = None
    score_latency_ms = 0
    logprobs_requested = False
    if not any(gates.values()):
        (
            scores,
            score_summary,
            score_payloads,
            score_latency_ms,
            logprobs_requested,
        ) = await _score_candidate(
            model=model,
            packet=packet,
        )

    output = {
        "gates": gates,
        "scores": scores,
        "score_summary": score_summary,
    }
    decision = _primary_decision(gates, scores, score_summary)
    payload = _combined_payload(
        model=model, gate_payload=gate_payload, score_payloads=score_payloads
    )
    latency_ms = gate_latency_ms + score_latency_ms
    run_id = _store_llm_run(
        username=username,
        seq=seq,
        evaluation_id=decoded["evaluation_id"],
        phase="primary",
        model=model,
        parent_run_id=None,
        output=output,
        decision=decision,
        payload=payload,
        latency_ms=latency_ms,
    )
    return {
        "run_id": run_id,
        "model": model,
        "model_resolved": payload.get("model"),
        "provider": payload.get("provider"),
        "output": output,
        "decision": decision,
        "usage": _usage(payload),
        "latency_ms": latency_ms,
        "logprobs_requested": logprobs_requested,
    }


@app.post("/api/llm/verify")
async def llm_verify(request: Request) -> dict[str, Any]:
    username = require_username(request)
    body = await request.json()
    parent_run_id = str(body.get("run_id", "")).strip()
    model = str(body.get("model", "")).strip()
    with result_connection() as connection:
        parent = connection.execute(
            "SELECT * FROM llm_filter_runs WHERE run_id = ? AND username = ? AND phase = 'primary'",
            (parent_run_id, username),
        ).fetchone()
    if parent is None:
        raise HTTPException(status_code=404, detail="主审核结果不存在")
    if str(parent["model_requested"]).split("/", 1)[0] == model.split("/", 1)[0]:
        raise HTTPException(status_code=400, detail="独立复核必须选择不同模型家族")
    decision = json.loads(str(parent["decision_json"]))
    if decision.get("decision") != "review":
        raise HTTPException(status_code=400, detail="该主审核结果不需要复核")
    targets = decision["verification_targets"]
    gate_ids = list(targets.get("gates") or [])
    item_ids = list(targets.get("items") or [])
    seq = int(parent["seq"])
    decoded = decode_sample(sample_row(seq))
    packet = build_sample_packet(decoded)
    gate_payload: dict[str, Any] | None = None
    score_payloads: list[dict[str, Any]] = []
    verified_gates: dict[str, bool] = {}
    verified_scores: dict[str, dict[str, Any]] = {}
    score_summary: dict[str, Any] | None = None
    gate_latency_ms = 0
    score_latency_ms = 0
    logprobs_requested = False
    if gate_ids:
        gate_output, _, gate_payload, gate_latency_ms = await _call_openrouter(
            model=model,
            messages=build_gate_messages(
                APP_STATE["rubric"], packet, gate_ids=gate_ids, independent=True
            ),
            schema=build_gate_schema(APP_STATE["rubric"], gate_ids),
            schema_name="roleplay_single_filter_verify_gate_v2",
            temperature=0,
        )
        try:
            validate_gate_output(gate_output, APP_STATE["rubric"], gate_ids)
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail=f"V2 Gate 复核输出校验失败：{error}"
            ) from None
        verified_gates = gate_output["gates"]
    if item_ids:
        (
            verified_scores,
            score_summary,
            score_payloads,
            score_latency_ms,
            logprobs_requested,
        ) = await _score_candidate(
            model=model,
            packet=packet,
            item_ids=item_ids,
            independent=True,
        )

    output = {
        "gates": verified_gates,
        "scores": verified_scores,
        "score_summary": score_summary,
    }
    confirmed_gates = sorted(
        gate_id for gate_id, value in verified_gates.items() if value
    )
    confirmed_items = deficient_item_ids(verified_scores)
    applicability_conflicts = not_applicable_item_ids(verified_scores)
    if confirmed_gates:
        final = "reject"
    elif confirmed_items:
        final = "exclude"
    elif applicability_conflicts:
        final = "review"
    else:
        final = "keep"
    final_decision = {
        "decision": final,
        "confirmed_gates": confirmed_gates,
        "confirmed_items": confirmed_items,
        "applicability_conflicts": applicability_conflicts,
        "quality_score": score_summary.get("quality_score") if score_summary else None,
    }
    payload = _combined_payload(
        model=model, gate_payload=gate_payload, score_payloads=score_payloads
    )
    latency_ms = gate_latency_ms + score_latency_ms
    run_id = _store_llm_run(
        username=username,
        seq=seq,
        evaluation_id=str(parent["evaluation_id"]),
        phase="verification",
        model=model,
        parent_run_id=parent_run_id,
        output=output,
        decision=final_decision,
        payload=payload,
        latency_ms=latency_ms,
    )
    return {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "model": model,
        "model_resolved": payload.get("model"),
        "provider": payload.get("provider"),
        "output": output,
        "decision": final_decision,
        "usage": _usage(payload),
        "latency_ms": latency_ms,
        "logprobs_requested": logprobs_requested,
    }
