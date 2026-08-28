from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import zlib
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


MODULE_DIR = Path(__file__).resolve().parent
STATIC_DIR = MODULE_DIR / "static"
RUBRIC_PATH = Path(os.environ.get("ROLEPLAY_RUBRIC_JSON", MODULE_DIR / "rubric.json"))
SAMPLE_DB = Path(os.environ.get("ROLEPLAY_SAMPLE_DB", MODULE_DIR / "data" / "samples.sqlite3"))
RESULT_DB = Path(
    os.environ.get("ROLEPLAY_RESULTS_DB", MODULE_DIR / "results" / "annotations.sqlite3")
)
RESULT_DIR = Path(
    os.environ.get("ROLEPLAY_RESULTS_DIR", MODULE_DIR / "results" / "by_user")
)
COOKIE_NAME = "roleplay_annotation_session"
USERNAME_RE = re.compile(r"^[\w\-]{1,40}$", re.UNICODE)
BOOLEAN_GATES = {"G-D1", "G-D2", "G-D3", "G-P1", "G-P2", "G-P3"}
ENUM_GATES = {
    "G-S1": {"yes", "no", "unclear"},
    "G-S2": {"pass", "fail", "N/A"},
    "G-S3": {"yes", "no", "N/A"},
}
APP_STATE: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rubric() -> dict[str, Any]:
    with RUBRIC_PATH.open("r", encoding="utf-8") as handle:
        rubric = json.load(handle)
    items = [item for dimension in rubric["dimensions"] for item in dimension["items"]]
    ids = [item["id"] for item in items]
    if len(ids) != 25 or len(set(ids)) != 25:
        raise RuntimeError("rubric must contain 25 unique atomic item IDs")
    rubric["item_ids"] = ids
    return rubric


def result_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(RESULT_DB, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def sample_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(SAMPLE_DB, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


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
            CREATE TABLE IF NOT EXISTS annotations (
                username TEXT NOT NULL REFERENCES users(username),
                seq INTEGER NOT NULL,
                evaluation_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('draft', 'submitted')),
                gates_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                gate_notes TEXT NOT NULL,
                overall_notes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (username, seq)
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_user_status
                ON annotations(username, status, seq);
            """
        )


def load_sample_metadata() -> dict[str, str]:
    if not SAMPLE_DB.is_file():
        raise RuntimeError(f"sample database not found: {SAMPLE_DB}")
    with sample_connection() as connection:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    metadata = {row["key"]: row["value"] for row in rows}
    if int(metadata.get("sample_count", "0")) <= 0:
        raise RuntimeError("sample database contains no samples")
    return metadata


@asynccontextmanager
async def lifespan(_: FastAPI):
    APP_STATE["rubric"] = load_rubric()
    APP_STATE["sample_metadata"] = load_sample_metadata()
    APP_STATE["total"] = int(APP_STATE["sample_metadata"]["sample_count"])
    initialize_results()
    yield


app = FastAPI(
    title="Roleplay Blind Annotation",
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
    if seq < 1 or seq > APP_STATE["total"]:
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
        "trigger_type": str(row["trigger_type"]),
        "messages": messages,
        "candidate": {
            "role": "assistant",
            "content": str(output.get("content", "")),
        },
    }


def progress_for(username: str) -> dict[str, int]:
    with result_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS saved, "
            "SUM(CASE WHEN status = 'submitted' THEN 1 ELSE 0 END) AS submitted "
            "FROM annotations WHERE username = ?",
            (username,),
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
            "SELECT seq FROM annotations WHERE username = ? AND status = 'submitted'",
            (username,),
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


def existing_annotation(username: str, seq: int) -> dict[str, Any] | None:
    with result_connection() as connection:
        row = connection.execute(
            "SELECT status, gates_json, scores_json, gate_notes, overall_notes, updated_at "
            "FROM annotations WHERE username = ? AND seq = ?",
            (username, seq),
        ).fetchone()
    if row is None:
        return None
    return {
        "status": row["status"],
        "gates": json.loads(row["gates_json"]),
        "scores": json.loads(row["scores_json"]),
        "gate_notes": row["gate_notes"],
        "overall_notes": row["overall_notes"],
        "updated_at": row["updated_at"],
    }


def normalize_scores(raw: Any, submitting: bool) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="scores 必须是对象")
    expected = set(APP_STATE["rubric"]["item_ids"])
    unknown = set(raw) - expected
    if unknown:
        raise HTTPException(status_code=400, detail=f"未知评分项：{sorted(unknown)}")
    if submitting and set(raw) != expected:
        missing = sorted(expected - set(raw))
        raise HTTPException(status_code=400, detail=f"还有未评分项：{missing}")

    normalized: dict[str, dict[str, Any]] = {}
    for item_id, value in raw.items():
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"{item_id} 格式错误")
        score = value.get("value")
        if score == "N/A":
            normalized_score: int | str = "N/A"
        elif isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 3:
            normalized_score = score
        else:
            raise HTTPException(status_code=400, detail=f"{item_id} 分数无效")
        evidence = str(value.get("evidence", "")).strip()
        if len(evidence) > 4000:
            raise HTTPException(status_code=400, detail=f"{item_id} 证据过长")
        if submitting and normalized_score in {0, 1, 3} and not evidence:
            raise HTTPException(status_code=400, detail=f"{item_id} 的该分数需要填写证据")
        normalized[item_id] = {"value": normalized_score, "evidence": evidence}
    return normalized


def normalize_gates(raw: Any, submitting: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="gates 必须是对象")
    expected = BOOLEAN_GATES | set(ENUM_GATES)
    unknown = set(raw) - expected
    if unknown:
        raise HTTPException(status_code=400, detail=f"未知 Gate：{sorted(unknown)}")
    if submitting and set(raw) != expected:
        raise HTTPException(status_code=400, detail="请完成全部 Gate 判断")
    normalized: dict[str, Any] = {}
    for gate_id, value in raw.items():
        if gate_id in BOOLEAN_GATES:
            if not isinstance(value, bool):
                raise HTTPException(status_code=400, detail=f"{gate_id} 必须为是或否")
            normalized[gate_id] = value
        elif value not in ENUM_GATES[gate_id]:
            raise HTTPException(status_code=400, detail=f"{gate_id} 取值无效")
        else:
            normalized[gate_id] = value
    return normalized


def write_snapshot(
    username: str,
    seq: int,
    evaluation_id: str,
    status: str,
    gates: dict[str, Any],
    scores: dict[str, Any],
    gate_notes: str,
    overall_notes: str,
    created_at: str,
    updated_at: str,
) -> None:
    user_dir = RESULT_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "schema_version": 1,
        "rubric_version": APP_STATE["rubric"]["version"],
        "username": username,
        "sequence_number": seq,
        "evaluation_id": evaluation_id,
        "status": status,
        "gates": gates,
        "scores": scores,
        "gate_notes": gate_notes,
        "overall_notes": overall_notes,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    target = user_dir / f"{seq:06d}.json"
    temporary = user_dir / f".{seq:06d}.{secrets.token_hex(6)}.tmp"
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, target)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "samples": APP_STATE.get("total", 0),
        "rubric_version": APP_STATE.get("rubric", {}).get("version"),
    }


@app.post("/api/login")
async def login(request: Request) -> JSONResponse:
    body = await request.json()
    username = str(body.get("username", "")).strip()
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(
            status_code=400,
            detail="用户名须为 1–40 个中文、字母、数字、下划线或连字符",
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
    public_rubric = dict(APP_STATE["rubric"])
    public_rubric.pop("item_ids", None)
    return public_rubric


@app.get("/api/sample/{seq}")
def sample(seq: int, request: Request) -> dict[str, Any]:
    username = require_username(request)
    row = sample_row(seq)
    payload = decode_sample(row)
    payload["total"] = APP_STATE["total"]
    payload["annotation"] = existing_annotation(username, seq)
    payload["progress"] = progress_for(username)
    return payload


@app.post("/api/annotation")
async def save_annotation(request: Request) -> dict[str, Any]:
    username = require_username(request)
    body = await request.json()
    try:
        seq = int(body.get("seq"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="样本序号无效") from None
    status = str(body.get("status", "draft"))
    if status not in {"draft", "submitted"}:
        raise HTTPException(status_code=400, detail="状态无效")
    submitting = status == "submitted"
    gates = normalize_gates(body.get("gates", {}), submitting)
    scores = normalize_scores(body.get("scores", {}), submitting)
    gate_notes = str(body.get("gate_notes", "")).strip()
    overall_notes = str(body.get("overall_notes", "")).strip()
    if len(gate_notes) > 10000 or len(overall_notes) > 10000:
        raise HTTPException(status_code=400, detail="备注过长")

    row = sample_row(seq)
    evaluation_id = str(row["evaluation_id"])
    now = utc_now()
    with result_connection() as connection:
        existing = connection.execute(
            "SELECT created_at FROM annotations WHERE username = ? AND seq = ?",
            (username, seq),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        connection.execute(
            """
            INSERT INTO annotations(
                username, seq, evaluation_id, status, gates_json, scores_json,
                gate_notes, overall_notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, seq) DO UPDATE SET
                evaluation_id = excluded.evaluation_id,
                status = excluded.status,
                gates_json = excluded.gates_json,
                scores_json = excluded.scores_json,
                gate_notes = excluded.gate_notes,
                overall_notes = excluded.overall_notes,
                updated_at = excluded.updated_at
            """,
            (
                username,
                seq,
                evaluation_id,
                status,
                json.dumps(gates, ensure_ascii=False, separators=(",", ":")),
                json.dumps(scores, ensure_ascii=False, separators=(",", ":")),
                gate_notes,
                overall_notes,
                created_at,
                now,
            ),
        )

    write_snapshot(
        username,
        seq,
        evaluation_id,
        status,
        gates,
        scores,
        gate_notes,
        overall_notes,
        created_at,
        now,
    )
    return {
        "ok": True,
        "status": status,
        "progress": progress_for(username),
        "next_seq": next_unsubmitted(username, after=seq) if submitting else seq,
        "updated_at": now,
    }
