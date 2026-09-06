"""Isolated conversation filtering workbench and persistent, user-started queue."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import sqlite3
import time
from urllib.parse import urlsplit
import zlib

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .conversation_filter import (
    VERSION,
    DEEPINFRA_MODEL,
    DEEPINFRA_URL,
    build_request,
    judge_packet,
    output_schema,
    rubric_document,
    rubric_payload,
    screening,
    validate_output,
)
from .conversation_platform_context import clean_role_profile
from .conversation_ranking import rank_records, csv_text
from .llm_judge import extract_judgment, load_judge_config
from .model_catalog import OpenRouterModelCatalog

MODULE = Path(__file__).resolve().parent
SAMPLE_DB = Path(
    os.environ.get(
        "ROLEPLAY_CONVERSATION_DATA_DB", MODULE / "data/conversations.sqlite3"
    )
)
RESULT_DB = Path(
    os.environ.get(
        "ROLEPLAY_CONVERSATION_RESULT_DB", MODULE / "results_conversation/jobs.sqlite3"
    )
)
STATIC = MODULE / "static_conversation"
COOKIE = "roleplay_conversation_session"
STATE = {}
MAX_CONCURRENCY = 100


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def results():
    connection = sqlite3.connect(RESULT_DB, timeout=30, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


@contextmanager
def data():
    connection = sqlite3.connect(
        f"file:{SAMPLE_DB.as_posix()}?mode=ro", uri=True, timeout=30
    )
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


def initialize():
    RESULT_DB.parent.mkdir(parents=True, exist_ok=True)
    with results() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY,username TEXT NOT NULL,expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(
         id TEXT PRIMARY KEY, username TEXT NOT NULL, nonce TEXT NOT NULL, model TEXT NOT NULL,
         status TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL, scope_json TEXT NOT NULL,
         template_json TEXT NOT NULL, rubric_hash TEXT NOT NULL, data_hash TEXT NOT NULL,
         total INTEGER NOT NULL, budget REAL NOT NULL, spent REAL NOT NULL DEFAULT 0,
         unknown_cost_attempts INTEGER NOT NULL DEFAULT 0, concurrency INTEGER NOT NULL,
         message TEXT NOT NULL DEFAULT '', UNIQUE(username,nonce));
        CREATE INDEX IF NOT EXISTS idx_dialogue_jobs_user ON jobs(username,created DESC);
        CREATE TABLE IF NOT EXISTS job_items(
         job_id TEXT NOT NULL REFERENCES jobs(id),seq INTEGER NOT NULL,position INTEGER NOT NULL,
         status TEXT NOT NULL DEFAULT 'queued',attempts INTEGER NOT NULL DEFAULT 0,
         remaining INTEGER NOT NULL DEFAULT 2,ready_at REAL NOT NULL DEFAULT 0,
         output_json TEXT,error TEXT NOT NULL DEFAULT '',min_score INTEGER,na_count INTEGER,
         major_issues INTEGER NOT NULL DEFAULT 0,needs_review INTEGER NOT NULL DEFAULT 0,
         PRIMARY KEY(job_id,seq),UNIQUE(job_id,position));
        CREATE INDEX IF NOT EXISTS idx_dialogue_items_claim ON job_items(status,ready_at,job_id,position);
        CREATE INDEX IF NOT EXISTS idx_dialogue_items_job_claim ON job_items(job_id,status,ready_at,position);
        CREATE TABLE IF NOT EXISTS attempts(
         id INTEGER PRIMARY KEY,job_id TEXT NOT NULL,seq INTEGER NOT NULL,attempt INTEGER NOT NULL,
         status TEXT NOT NULL,created TEXT NOT NULL,request_hash TEXT NOT NULL,
         request_json TEXT NOT NULL,raw_json TEXT NOT NULL,error TEXT NOT NULL,
         prompt_tokens INTEGER,completion_tokens INTEGER,cost REAL,latency_ms INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_dialogue_attempts_item ON attempts(job_id,seq,id DESC);
        """)
        if "max_attempts" not in {r[1] for r in c.execute("PRAGMA table_info(jobs)")}:
            c.execute(
                "ALTER TABLE jobs ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 2"
            )
        if "dispatch_after" not in {r[1] for r in c.execute("PRAGMA table_info(jobs)")}:
            c.execute(
                "ALTER TABLE jobs ADD COLUMN dispatch_after REAL NOT NULL DEFAULT 0"
            )
        # An interrupted HTTP request may already have been billed. Never blindly
        # replay it on process restart; make recovery an explicit user action.
        c.execute(
            "UPDATE job_items SET status='interrupted',error='服务重启时请求仍在途；请先查看记录后重试' WHERE status='running'"
        )
        c.execute(
            "UPDATE jobs SET status='paused',message='服务已重启，请检查后继续；中断项需手动重试' WHERE status IN ('running','queued')"
        )


def manifest():
    if not SAMPLE_DB.exists():
        return {"ready": False, "report": {}, "languages": {}}
    with data() as c:
        row = c.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
    return json.loads(row[0]) if row else {"ready": False}


def dataset_hash():
    info = STATE["manifest"]
    return info.get("dataset_sha256") or info["source_sha256"]


def load_conversation(seq: int):
    with data() as c:
        row = c.execute("SELECT * FROM conversations WHERE seq=?", (seq,)).fetchone()
    if row is None:
        raise HTTPException(404, "没有这段对话")
    packet = json.loads(zlib.decompress(row["payload_zlib"]))
    packet["role_profile"] = clean_role_profile(packet.get("role_profile", {}))
    packet.update(seq=seq, context_hash=row["context_hash"])
    return packet


def username(request):
    token = request.cookies.get(COOKIE)
    with results() as c:
        row = c.execute(
            "SELECT username FROM sessions WHERE token=? AND expires>?",
            (token, time.time()),
        ).fetchone()
    if row is None:
        raise HTTPException(401, "请先输入用户名")
    return row[0]


def owned_job(job_id, user):
    with results() as c:
        row = c.execute(
            "SELECT * FROM jobs WHERE id=? AND username=?", (job_id, user)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "任务不存在或不属于当前用户名")
    return dict(row)


def redact(value):
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    for key in (
        STATE["config"].api_key,
        STATE["config"].tatu_api_key,
        STATE.get("deepinfra_key", ""),
    ):
        if key:
            serialized = serialized.replace(key, "<SERVER_SECRET_REDACTED>")
    serialized = re.sub(
        r"sk-or-v1-[A-Za-z0-9_-]+", "<SERVER_SECRET_REDACTED>", serialized
    )
    return json.loads(serialized)


def model_available(model):
    if model == DEEPINFRA_MODEL:
        return bool(STATE.get("deepinfra_key"))
    return (
        not model.startswith(("tatu::", "deepinfra::"))
        and model in STATE["config"].models
        and bool(STATE["config"].api_key)
    )


def model_transport(model):
    # Routing is keyed by the saved model ID, so existing OpenRouter jobs never
    # silently switch provider when a new default or a new key is configured.
    if model == DEEPINFRA_MODEL:
        return DEEPINFRA_URL, STATE.get("deepinfra_key", ""), "deepinfra"
    if model.startswith(("deepinfra::", "tatu::")):
        raise ValueError("不支持的请求渠道")
    return STATE["config"].openrouter_url, STATE["config"].api_key, "openrouter"


def public_models():
    snapshot = dict(STATE["catalog"].public_snapshot())
    entries = (
        [dict(entry) for entry in snapshot["models"]] if STATE["config"].api_key else []
    )
    if STATE.get("deepinfra_key"):
        entries.insert(
            0,
            {
                "id": DEEPINFRA_MODEL,
                "name": "GLM 5.3 Flash",
                "label": "GLM 5.3 Flash · DeepInfra 直连",
                "gateway": "deepinfra",
                "backend_model": "zai-org/GLM-5.3-Flash",
                "reports_cost_usd": True,
                "cost_basis": "usage.estimated_cost",
                "reports_token_logprobs": False,
            },
        )
    snapshot["models"] = entries
    snapshot["default_model"] = (
        DEEPINFRA_MODEL
        if STATE.get("deepinfra_key")
        else next(
            (m["id"] for m in entries if m["id"] == "z-ai/glm-5.3-flash"),
            entries[0]["id"] if entries else "",
        )
    )
    return snapshot


def dataset_where(filters):
    clauses, params = [], []
    for field, column in (
        ("language", "language"),
        ("character", "character_id"),
        ("user", "user_key"),
    ):
        value = str(filters.get(field, "")).strip()
        if value:
            clauses.append(f"c.{column}=?")
            params.append(value)
    query = str(filters.get("q", "")).strip()
    if query:
        clauses.append(
            "(c.character_name LIKE ? ESCAPE '\\' OR c.conversation_id=? OR CAST(c.seq AS TEXT)=?)"
        )
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.extend(["%" + escaped + "%", query, query])
    return clauses, params


def select_scope(scope):
    clauses, params = dataset_where(scope)
    mode = scope.get("mode", "random")
    if mode == "current":
        clauses.append("c.seq=?")
        params.append(int(scope.get("seq", 0)))
    elif mode == "range":
        start, end = int(scope.get("start", 1)), int(scope.get("end", 100))
        if start < 1 or end < start:
            raise ValueError("对话序号范围不正确")
        clauses.append("c.seq BETWEEN ? AND ?")
        params.extend([start, end])
    elif mode not in ("random", "balanced", "all"):
        raise ValueError("不支持的范围类型")
    query = "SELECT c.seq,c.character_id,c.user_key,c.candidate_count,c.text_chars FROM conversations c"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    with data() as c:
        rows = [dict(r) for r in c.execute(query + " ORDER BY c.seq", params)]
    rng = random.Random(int(scope.get("seed", 20260906)))
    if mode == "balanced":
        from collections import Counter

        frequency = Counter(r["character_id"] for r in rows)
        rows.sort(
            key=lambda r: (
                -math.log(max(rng.random(), 1e-15))
                * math.sqrt(frequency[r["character_id"]])
            )
        )
    elif mode == "random":
        rng.shuffle(rows)
    target = (
        int(scope.get("count", 100)) if mode in ("random", "balanced") else len(rows)
    )
    if target < 1 or target > 20000:
        raise ValueError("每次任务请选择 1～20,000 段对话")
    role_cap, user_cap = (
        int(scope.get("character_cap", 0)),
        int(scope.get("user_cap", 0)),
    )
    if role_cap < 0 or user_cap < 0:
        raise ValueError("数量上限不能为负数；0 表示不限")
    from collections import Counter

    rc, uc, selected = Counter(), Counter(), []
    for row in rows:
        if (role_cap and rc[row["character_id"]] >= role_cap) or (
            user_cap and uc[row["user_key"]] >= user_cap
        ):
            continue
        selected.append(row)
        rc[row["character_id"]] += 1
        uc[row["user_key"]] += 1
        if len(selected) >= target:
            break
    return selected, {
        "conversations": len(selected),
        "candidate_replies": sum(r["candidate_count"] for r in selected),
        "characters": len(rc),
        "users": len(uc),
        "requested": target,
        "shortfall": target - len(selected),
        "text_chars": sum(r["text_chars"] for r in selected),
        "seed": int(scope.get("seed", 20260906)),
    }


def retry_delay(response, attempt):
    if response is not None:
        value = response.headers.get("Retry-After")
        if value:
            try:
                delay = float(value)
            except ValueError:
                try:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    delay = 0
            if delay > 0:
                return min(delay, 3600)
    if response is not None and response.status_code == 429:
        return min(300, 30 * 2 ** min(attempt - 1, 4)) + random.random()
    return min(60, 5 * 2 ** min(attempt - 1, 5)) + random.random()


class Runtime:
    def __init__(self):
        self.tasks = []
        self.client = httpx.AsyncClient(
            timeout=STATE["config"].request_timeout_seconds,
            limits=httpx.Limits(
                max_connections=MAX_CONCURRENCY,
                max_keepalive_connections=MAX_CONCURRENCY,
            ),
        )

    async def start(self):
        self.tasks = [
            asyncio.create_task(self.worker()) for _ in range(MAX_CONCURRENCY)
        ]

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.client.aclose()

    def claim(self):
        with results() as c:
            c.execute("BEGIN IMMEDIATE")
            jobs = c.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY created"
            ).fetchall()
            for job in jobs:
                pending = c.execute(
                    "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status IN ('queued','running')",
                    (job["id"],),
                ).fetchone()[0]
                if not pending:
                    errors = c.execute(
                        "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status!='completed'",
                        (job["id"],),
                    ).fetchone()[0]
                    c.execute(
                        "UPDATE jobs SET status=?,updated=? WHERE id=?",
                        (
                            "completed_errors" if errors else "completed",
                            now(),
                            job["id"],
                        ),
                    )
                    continue
                if job["data_hash"] != dataset_hash():
                    c.execute(
                        "UPDATE jobs SET status='paused',message='上下文拼接版本已更新，请新建任务，避免混用旧请求' WHERE id=?",
                        (job["id"],),
                    )
                    continue
                if job["spent"] >= job["budget"]:
                    c.execute(
                        "UPDATE jobs SET status='paused',message='已达到费用停止阈值；在途请求仍会保存' WHERE id=?",
                        (job["id"],),
                    )
                    continue
                if job["dispatch_after"] > time.time():
                    continue
                running = c.execute(
                    "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status='running'",
                    (job["id"],),
                ).fetchone()[0]
                if running >= job["concurrency"]:
                    continue
                item = c.execute(
                    "SELECT * FROM job_items WHERE job_id=? AND status='queued' AND ready_at<=? ORDER BY position LIMIT 1",
                    (job["id"], time.time()),
                ).fetchone()
                if item:
                    c.execute(
                        "UPDATE job_items SET status='running',attempts=attempts+1,remaining=remaining-1 WHERE job_id=? AND seq=?",
                        (job["id"], item["seq"]),
                    )
                    c.execute(
                        "UPDATE jobs SET status='running',updated=? WHERE id=?",
                        (now(), job["id"]),
                    )
                    return dict(job), {
                        **dict(item),
                        "attempts": item["attempts"] + 1,
                        "remaining": item["remaining"] - 1,
                    }
        return None

    async def worker(self):
        while True:
            claim = None
            try:
                claim = self.claim()
                if not claim:
                    await asyncio.sleep(1)
                    continue
                await self.evaluate(*claim)
            except asyncio.CancelledError:
                raise
            except Exception:
                # No exception / configuration repr in logs: may contain secrets.
                if claim:
                    job, item = claim
                    with results() as c:
                        c.execute(
                            "UPDATE job_items SET status='interrupted',error='后台处理意外中断；请检查请求记录后手动重试' WHERE job_id=? AND seq=? AND status='running'",
                            (job["id"], item["seq"]),
                        )
                        c.execute(
                            "UPDATE jobs SET status='paused',message='后台处理意外中断，已暂停派发；未自动重复请求' WHERE id=? AND status!='cancelled'",
                            (job["id"],),
                        )
                await asyncio.sleep(2)

    async def evaluate(self, job, item):
        started = time.monotonic()
        packet = load_conversation(item["seq"])
        body = json.loads(job["template_json"])
        body["messages"].append(
            {
                "role": "user",
                "content": "请评价以下对话的候选回复组：\n"
                + json.dumps(judge_packet(packet), ensure_ascii=False),
            }
        )
        request_json = json.dumps(body, ensure_ascii=False)
        request_hash = hashlib.sha256(request_json.encode()).hexdigest()
        output = None
        error = ""
        payload = {}
        raw = {}
        response = None
        status, retryable, pause = "failed", False, False
        endpoint, api_key, gateway = model_transport(job["model"])
        try:
            if not api_key:
                raise ValueError("所选渠道未配置 Key")
            response = await self.client.post(
                endpoint,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                json=body,
            )
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if response.status_code >= 400:
                info = payload.get("error", {}) if isinstance(payload, dict) else {}
                error = (
                    f"HTTP {response.status_code}："
                    + str(
                        info.get("message", "请求商返回错误")
                        if isinstance(info, dict)
                        else info
                    )[:1200]
                )
                retryable = response.status_code in (408, 429, 500, 502, 503, 504)
                pause = response.status_code in (401, 402, 403)
                raw = {"http_status": response.status_code, "error": info}
            else:
                choice = (payload.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                raw = {
                    "content": message.get("content"),
                    "refusal": message.get("refusal"),
                    "finish_reason": choice.get("finish_reason"),
                    "model": payload.get("model"),
                    "provider": "DeepInfra (direct)"
                    if gateway == "deepinfra"
                    else payload.get("provider"),
                    "id": payload.get("id"),
                    "usage": payload.get("usage"),
                }
                try:
                    parsed, _ = extract_judgment(payload, allow_json_fence=True)
                    raw["parsed_json"] = parsed
                    output = validate_output(parsed, packet)
                    status = "completed"
                except (ValueError, TypeError, KeyError) as exc:
                    status, error, retryable = "invalid", str(exc), True
        except httpx.HTTPError as exc:
            error = f"网络请求失败：{type(exc).__name__}；服务端是否计费未知"
            retryable = True
        except Exception as exc:
            error = f"响应处理失败：{type(exc).__name__}"
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        cost_field = "estimated_cost" if gateway == "deepinfra" else "cost"
        raw_cost = usage.get(cost_field)
        raw.update(gateway=gateway, endpoint=endpoint, cost_basis="usage." + cost_field)
        try:
            cost = float(raw_cost) if raw_cost is not None else None
            if cost is not None and (not math.isfinite(cost) or cost < 0):
                cost = None
        except (ValueError, TypeError):
            cost = None
        success_http = response is not None and 200 <= response.status_code < 300
        if success_http and cost is None:
            pause = True
        raw, error = redact(raw), redact(error)
        if output is not None:
            output = redact(output)
        rescreen = screening(output) if output else None
        min_score = (
            min(
                (
                    r["value"]
                    for r in output["scores"].values()
                    if type(r["value"]) is int
                ),
                default=None,
            )
            if output
            else None
        )
        retry_at = (
            time.time() + retry_delay(response, item["attempts"]) if retryable else 0
        )
        with results() as c:
            current = c.execute(
                "SELECT status FROM jobs WHERE id=?", (job["id"],)
            ).fetchone()[0]
            next_status = (
                "queued"
                if retryable
                and item["remaining"] > 0
                and current != "cancelled"
                and not pause
                else status
            )
            c.execute(
                "INSERT INTO attempts(job_id,seq,attempt,status,created,request_hash,request_json,raw_json,error,prompt_tokens,completion_tokens,cost,latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job["id"],
                    item["seq"],
                    item["attempts"],
                    status,
                    now(),
                    request_hash,
                    request_json,
                    json.dumps(raw, ensure_ascii=False),
                    error,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    cost,
                    int((time.monotonic() - started) * 1000),
                ),
            )
            c.execute(
                "UPDATE job_items SET status=?,ready_at=?,output_json=?,error=?,min_score=?,na_count=?,major_issues=?,needs_review=? WHERE job_id=? AND seq=?",
                (
                    next_status,
                    retry_at if next_status == "queued" else 0,
                    json.dumps(output, ensure_ascii=False) if output else None,
                    error,
                    min_score,
                    len(rescreen["not_applicable"]) if rescreen else None,
                    int(rescreen["major_issues"]) if rescreen else 0,
                    int(
                        bool(
                            output
                            and (
                                output.get("warnings") or output.get("data_limitations")
                            )
                        )
                    ),
                    job["id"],
                    item["seq"],
                ),
            )
            if response is not None and response.status_code == 429:
                c.execute(
                    "UPDATE jobs SET dispatch_after=MAX(dispatch_after,?) WHERE id=?",
                    (retry_at, job["id"]),
                )
            unknown = int(cost is None and (success_http or response is None))
            c.execute(
                "UPDATE jobs SET spent=spent+?,unknown_cost_attempts=unknown_cost_attempts+?,updated=? WHERE id=?",
                (cost or 0, unknown, now(), job["id"]),
            )
            if pause and current != "cancelled":
                c.execute(
                    "UPDATE jobs SET status='paused',message=? WHERE id=?",
                    (
                        "渠道未返回费用，已暂停后续派发；结果仍保存，请查看后决定是否继续"
                        if success_http and cost is None
                        else error,
                        job["id"],
                    ),
                )


@asynccontextmanager
async def lifespan(app):
    initialize()
    STATE["manifest"] = manifest()
    config = load_judge_config()
    STATE["deepinfra_key"] = os.environ.get("DEEPINFRA_API_KEY", "").strip()
    # Reuse the established model catalog and route policy, but not V1 queues/DB.
    catalog = OpenRouterModelCatalog(replace(config, tatu_api_key=""))
    await catalog.refresh()
    STATE["catalog"] = catalog
    STATE["config"] = replace(
        config, models=catalog.model_ids, reasoning_models=catalog.reasoning_model_ids
    )
    runtime = Runtime()
    STATE["runtime"] = runtime
    await runtime.start()
    yield
    await runtime.close()


app = FastAPI(
    title="对话过滤评分",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")


@app.middleware("http")
async def security(request: Request, call_next):
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if request.headers.get("x-requested-with") != "conversation-filter" or (
            origin and urlsplit(origin).netloc != request.headers.get("host")
        ):
            return JSONResponse({"detail": "请从本页面发起操作"}, status_code=403)
    response = await call_next(request)
    response.headers.update(
        {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        }
    )
    return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "data_ready": STATE.get("manifest", {}).get("ready", False),
    }


@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    user = str(body.get("username", "")).strip()
    if not re.fullmatch(r"[\w\-]{1,40}", user):
        raise HTTPException(400, "用户名使用 1～40 位中文、字母、数字、下划线或短横线")
    token = secrets.token_urlsafe(32)
    with results() as c:
        c.execute(
            "INSERT INTO sessions VALUES (?,?,?)",
            (token, user, time.time() + 30 * 86400),
        )
    response = JSONResponse({"username": user})
    response.set_cookie(
        COOKIE, token, httponly=True, samesite="lax", max_age=30 * 86400
    )
    return response


@app.post("/api/logout")
def logout(request: Request):
    with results() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (request.cookies.get(COOKIE),))
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE)
    return response


@app.get("/api/me")
def me(request: Request):
    user = username(request)
    info = STATE["manifest"]
    return {
        "username": user,
        "version": VERSION,
        "data": {
            k: info.get(k) for k in ("ready", "report", "languages", "source_sha256")
        },
        "api_configured": bool(STATE["config"].api_key or STATE.get("deepinfra_key")),
        "default_model": public_models()["default_model"],
        "max_concurrency": MAX_CONCURRENCY,
    }


@app.get("/api/rubric")
def rubric(request: Request):
    username(request)
    return rubric_payload()


@app.get("/rubric.md")
def doc(request: Request):
    username(request)
    return PlainTextResponse(rubric_document())


@app.get("/api/models")
def models(request: Request):
    username(request)
    return public_models()


@app.post("/api/models/refresh")
async def refresh_models(request: Request):
    username(request)
    catalog = STATE["catalog"]
    result = await catalog.refresh()
    STATE["config"] = replace(
        STATE["config"],
        models=catalog.model_ids,
        reasoning_models=catalog.reasoning_model_ids,
    )
    return public_models()


@app.get("/api/conversations")
def list_conversations(
    request: Request,
    page: int = 1,
    limit: int = 30,
    job_id: str = "",
    result_status: str = "",
    minimum: int = 2,
    exclude_major: bool = True,
):
    user = username(request)
    if not STATE["manifest"].get("ready"):
        return {"rows": [], "total": 0, "page": 1, "ready": False}
    limit = min(100, max(1, limit))
    page = max(1, page)
    minimum = max(0, min(3, minimum))
    clauses, params = dataset_where(request.query_params)
    fields = "c.seq,c.conversation_id,c.character_id,c.character_name,c.user_key,c.language,c.languages_json,c.candidate_count,c.turn_count,c.text_chars"
    join = ""
    if job_id:
        owned_job(job_id, user)
        join = " JOIN results_db.job_items i ON c.seq=i.seq AND i.job_id=?"
        params.insert(0, job_id)
        fields += ",i.status,i.min_score,i.na_count,i.major_issues,i.needs_review,i.attempts,i.error"
        if result_status == "eligible":
            clauses.extend(
                ["i.status='completed'", "i.min_score>=?", "i.needs_review=0"]
            )
            params.append(minimum)
            if exclude_major:
                clauses.append("i.major_issues=0")
        elif result_status == "review":
            clauses.append(
                "i.status='completed' AND (i.min_score<? OR i.needs_review=1"
                + (" OR i.major_issues=1" if exclude_major else "")
                + ")"
            )
            params.append(minimum)
        elif result_status == "errors":
            clauses.append("i.status IN ('failed','invalid','interrupted')")
        elif result_status in ("completed", "running", "queued", "cancelled"):
            clauses.append("i.status=?")
            params.append(result_status)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with data() as c:
        if job_id:
            c.execute(
                "ATTACH DATABASE ? AS results_db",
                (f"file:{RESULT_DB.as_posix()}?mode=ro",),
            )
        total = c.execute(
            "SELECT COUNT(*) FROM conversations c" + join + where, params
        ).fetchone()[0]
        rows = [
            dict(r)
            for r in c.execute(
                "SELECT "
                + fields
                + " FROM conversations c"
                + join
                + where
                + " ORDER BY c.seq LIMIT ? OFFSET ?",
                [*params, limit, (page - 1) * limit],
            )
        ]
    return {"rows": rows, "total": total, "page": page, "limit": limit, "ready": True}


@app.get("/api/conversations/{seq}")
def conversation(seq: int, request: Request):
    username(request)
    return load_conversation(seq)


@app.get("/api/request-preview/{seq}")
def preview(seq: int, request: Request, model: str = "", job_id: str = ""):
    user = username(request)
    packet = load_conversation(seq)
    if job_id:
        job = owned_job(job_id, user)
        with results() as c:
            member = c.execute(
                "SELECT 1 FROM job_items WHERE job_id=? AND seq=?", (job_id, seq)
            ).fetchone()
            attempt = c.execute(
                "SELECT request_json,request_hash,raw_json FROM attempts WHERE job_id=? AND seq=? ORDER BY id DESC LIMIT 1",
                (job_id, seq),
            ).fetchone()
        if not member:
            raise HTTPException(404, "该对话不在此任务中")
        if attempt:
            sent = json.loads(attempt[0])
            endpoint, _, gateway = model_transport(job["model"])
            sanitized = json.loads(attempt[2]).get("request_display_sanitized", False)
            return {
                "body": sent,
                "endpoint": endpoint,
                "gateway": gateway,
                "request_hash": attempt[1],
                "source": "历史请求（已按要求移除数据库 prompt，未重新发送；旧评分保留）"
                if sanitized
                else "实际已发送请求",
                "schema": sent["response_format"]["json_schema"]["schema"],
                "authorization": "仅服务器持有，不返回",
            }
        body = json.loads(job["template_json"])
        body["messages"].append(
            {
                "role": "user",
                "content": "请评价以下对话的候选回复组：\n"
                + json.dumps(judge_packet(packet), ensure_ascii=False),
            }
        )
    else:
        if not model_available(model):
            raise HTTPException(400, "请先选择已配置渠道的可用模型")
        body, _ = build_request(packet, model, STATE["config"])
    endpoint, _, gateway = model_transport(job["model"] if job_id else model)
    return {
        "body": body,
        "endpoint": endpoint,
        "gateway": gateway,
        "schema": body["response_format"]["json_schema"]["schema"],
        "source": "与执行共用的真实请求构造",
        "authorization": "仅服务器持有，不返回",
        "rubric_hash": rubric_payload()["sha256"],
    }


@app.post("/api/selection-preview")
async def selection_preview(request: Request):
    username(request)
    if not STATE["manifest"].get("ready"):
        raise HTTPException(503, "对话索引尚未就绪")
    try:
        selected, summary = select_scope(await request.json())
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from None
    return {
        "summary": summary,
        "first_sequences": [r["seq"] for r in selected[:20]],
        "paid_request": False,
    }


@app.post("/api/jobs")
async def create_job(request: Request):
    user = username(request)
    body = await request.json()
    if not body.get("acknowledge_snapshot"):
        raise HTTPException(400, "请确认使用离线上下文快照进行对话级评分")
    if not STATE["manifest"].get("ready"):
        raise HTTPException(503, "数据或服务器模型配置尚未就绪")
    model = str(body.get("model", ""))
    if not model_available(model):
        raise HTTPException(400, "模型不在当前可用目录中，或其渠道尚未配置 Key")
    try:
        budget = float(body.get("budget", 5))
        concurrency = int(body.get("concurrency", 10))
        max_attempts = int(body.get("max_attempts", 2))
        if (
            not math.isfinite(budget)
            or not 0 < budget <= 10000
            or not 1 <= concurrency <= MAX_CONCURRENCY
        ):
            raise ValueError(
                f"费用停止阈值应为 0～10,000 美元，并发应为 1～{MAX_CONCURRENCY}"
            )
        if not 1 <= max_attempts <= 6:
            raise ValueError("每段自动尝试上限为 1～6 次（含首次）")
        if type(body.get("start_paused", False)) is not bool:
            raise ValueError("start_paused 必须为布尔值")
        scope = body.get("scope", {})
        selected, summary = select_scope(scope)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from None
    if not selected:
        raise HTTPException(400, "该范围没有可选对话")
    nonce = str(body.get("nonce", ""))[:120]
    if not nonce:
        raise HTTPException(400, "缺少防重复提交标识，请重新点击启动")
    packet = load_conversation(selected[0]["seq"])
    template, _ = build_request(packet, model, STATE["config"])
    template["messages"] = template["messages"][:1]
    job_id = secrets.token_hex(12)
    with results() as c:
        existing = c.execute(
            "SELECT id FROM jobs WHERE username=? AND nonce=?", (user, nonce)
        ).fetchone()
        if existing:
            return {"job_id": existing[0], "duplicate_prevented": True}
        prepared = body.get("start_paused", False)
        c.execute(
            "INSERT INTO jobs(id,username,nonce,model,status,created,updated,scope_json,template_json,rubric_hash,data_hash,total,budget,concurrency,max_attempts,message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                user,
                nonce,
                model,
                "paused" if prepared else "queued",
                now(),
                now(),
                json.dumps(scope, ensure_ascii=False),
                json.dumps(template, ensure_ascii=False),
                rubric_payload()["sha256"],
                dataset_hash(),
                len(selected),
                budget,
                concurrency,
                max_attempts,
                "已准备，尚未启动；点击继续才会发送付费请求" if prepared else "",
            ),
        )
        c.executemany(
            "INSERT INTO job_items(job_id,seq,position,remaining) VALUES (?,?,?,?)",
            (
                (job_id, row["seq"], position, max_attempts)
                for position, row in enumerate(selected, 1)
            ),
        )
    return {"job_id": job_id, "summary": summary, "prepared": prepared}


@app.get("/api/jobs")
def jobs(request: Request):
    user = username(request)
    with results() as c:
        rows = c.execute(
            "SELECT id,model,status,total,spent,budget,concurrency,created,message,unknown_cost_attempts FROM jobs WHERE username=? ORDER BY created DESC LIMIT 100",
            (user,),
        ).fetchall()
    return {"jobs": [dict(r) for r in rows]}


@app.get("/api/jobs/{job_id}")
def job_detail(
    job_id: str, request: Request, minimum: int = 2, exclude_major: bool = True
):
    job = owned_job(job_id, username(request))
    with results() as c:
        counts = dict(
            c.execute(
                "SELECT status,COUNT(*) FROM job_items WHERE job_id=? GROUP BY status",
                (job_id,),
            )
        )
        eligible = c.execute(
            "SELECT COUNT(*) FROM job_items WHERE job_id=? AND status='completed' AND min_score>=? AND needs_review=0"
            + (" AND major_issues=0" if exclude_major else ""),
            (job_id, max(0, min(3, minimum))),
        ).fetchone()[0]
        attempt_counts = dict(
            c.execute(
                "SELECT status,COUNT(*) FROM attempts WHERE job_id=? GROUP BY status",
                (job_id,),
            )
        )
        rate_limited = c.execute(
            "SELECT COUNT(*) FROM attempts WHERE job_id=? AND error LIKE 'HTTP 429%'",
            (job_id,),
        ).fetchone()[0]
    return {
        k: job[k]
        for k in (
            "id",
            "model",
            "status",
            "created",
            "updated",
            "total",
            "spent",
            "budget",
            "unknown_cost_attempts",
            "concurrency",
            "max_attempts",
            "message",
            "rubric_hash",
            "data_hash",
        )
    } | {
        "counts": counts,
        "eligible": eligible,
        "attempt_counts": attempt_counts,
        "rate_limited_attempts": rate_limited,
        "rate_limit_wait_seconds": max(
            0, math.ceil(job["dispatch_after"] - time.time())
        ),
        "scope": json.loads(job["scope_json"]),
        "rubric_version": VERSION,
        "budget_note": "停止阈值不是硬账单上限：已在途请求仍可能产生费用；未知费用单独列出。",
    }


@app.post("/api/jobs/{job_id}/control")
async def control(job_id: str, request: Request):
    job = owned_job(job_id, username(request))
    body = await request.json()
    action = body.get("action")
    with results() as c:
        if action == "pause":
            if job["status"] in ("running", "queued"):
                c.execute(
                    "UPDATE jobs SET status='paused',message='已暂停派发；在途请求会继续保存' WHERE id=?",
                    (job_id,),
                )
        elif action in ("resume", "retry"):
            if (
                job["rubric_hash"] != rubric_payload()["sha256"]
                or job["data_hash"] != dataset_hash()
            ):
                raise HTTPException(409, "任务规则或数据版本不同，请新建任务，避免混用")
            try:
                budget = float(body.get("budget", job["budget"]))
            except (ValueError, TypeError):
                raise HTTPException(400, "费用阈值无效") from None
            if not math.isfinite(budget) or not job["spent"] < budget <= 10000:
                raise HTTPException(400, "继续前请把费用停止阈值设置为大于已知累计费用")
            if action == "retry":
                c.execute(
                    "UPDATE job_items SET status='queued',remaining=?,ready_at=0,error='' WHERE job_id=? AND status IN ('failed','invalid','interrupted')",
                    (job["max_attempts"], job_id),
                )
            elif job["status"] == "cancelled":
                raise HTTPException(400, "已取消的任务不能恢复等待项，请按需要新建任务")
            c.execute(
                "UPDATE jobs SET status='queued',budget=?,message='' WHERE id=?",
                (budget, job_id),
            )
        elif action == "cancel":
            c.execute(
                "UPDATE job_items SET status='cancelled' WHERE job_id=? AND status='queued'",
                (job_id,),
            )
            c.execute(
                "UPDATE jobs SET status='cancelled',message='等待项已取消；在途请求仍会保存' WHERE id=?",
                (job_id,),
            )
        else:
            raise HTTPException(400, "未知操作")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/ranking")
def ranking(
    job_id: str, request: Request, page: int = 1, limit: int = 50, format: str = "json"
):
    job = owned_job(job_id, username(request))
    if format not in ("json", "full", "csv"):
        raise HTTPException(400, "不支持的排名格式")
    with results() as c:
        c.execute(
            "ATTACH DATABASE ? AS source_db", (f"file:{SAMPLE_DB.as_posix()}?mode=ro",)
        )
        records = c.execute(
            """SELECT i.seq,i.status,i.output_json,i.error,
            d.conversation_id,d.character_name,d.language,d.candidate_count
            FROM job_items i JOIN source_db.conversations d ON d.seq=i.seq
            WHERE i.job_id=? ORDER BY i.position""",
            (job_id,),
        ).fetchall()
    report = rank_records(records)
    report["job_id"] = job_id
    report["job_status"] = job["status"]
    report["rubric_hash"] = job["rubric_hash"]
    report["data_hash"] = job["data_hash"]
    if format == "csv":
        return PlainTextResponse(
            csv_text(report),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="conversation-ranking-{job_id}.csv"'
            },
        )
    if format == "full":
        return report
    page, limit = max(1, page), min(100, max(1, limit))
    report["rows"] = report["rows"][(page - 1) * limit : page * limit]
    report["excluded"] = report["excluded"][:20]
    return {**report, "page": page, "limit": limit}


@app.get("/api/jobs/{job_id}/result/{seq}")
def result(
    job_id: str,
    seq: int,
    request: Request,
    minimum: int = 2,
    exclude_major: bool = True,
):
    owned_job(job_id, username(request))
    with results() as c:
        row = c.execute(
            "SELECT * FROM job_items WHERE job_id=? AND seq=?", (job_id, seq)
        ).fetchone()
        attempts = [
            dict(r)
            for r in c.execute(
                "SELECT id,attempt,status,created,request_hash,raw_json,error,prompt_tokens,completion_tokens,cost,latency_ms FROM attempts WHERE job_id=? AND seq=? ORDER BY id",
                (job_id, seq),
            )
        ]
    if row is None:
        raise HTTPException(404, "该对话不在任务中")
    output = json.loads(row["output_json"]) if row["output_json"] else None
    for attempt in attempts:
        attempt["raw"] = json.loads(attempt.pop("raw_json"))
    return {
        "seq": seq,
        "status": row["status"],
        "error": row["error"],
        "output": output,
        "screening": screening(output, max(0, min(3, minimum)), exclude_major)
        if output
        else None,
        "attempts": attempts,
    }


@app.get("/api/jobs/{job_id}/export")
def export(
    job_id: str,
    request: Request,
    minimum: int = 2,
    exclude_major: bool = True,
    eligible_only: bool = False,
):
    job = owned_job(job_id, username(request))
    minimum = max(0, min(3, minimum))

    def generate():
        with results() as c:
            for row in c.execute(
                "SELECT seq,status,output_json,error FROM job_items WHERE job_id=? ORDER BY position",
                (job_id,),
            ):
                output = json.loads(row["output_json"]) if row["output_json"] else None
                selected = screening(output, minimum, exclude_major) if output else None
                if eligible_only and (not selected or selected["label"] != "eligible"):
                    continue
                packet = load_conversation(row["seq"])
                record = {
                    "job_id": job_id,
                    "username": job["username"],
                    "seq": row["seq"],
                    "conversation_id": packet["conversation_id"],
                    "context_hash": packet["context_hash"],
                    "rubric_hash": job["rubric_hash"],
                    "model": job["model"],
                    "status": row["status"],
                    "error": row["error"],
                    "result": output,
                    "screening": selected,
                    "candidate_message_ids": [
                        m["candidate_message_id"]
                        for m in packet["messages"]
                        if m["scope"] == "candidate"
                    ],
                    "note": "对话级结果；候选 message_id 仅作关联，不表示每轮单独审核通过。",
                }
                yield json.dumps(record, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="conversation-{job_id}.jsonl"'
        },
    )
