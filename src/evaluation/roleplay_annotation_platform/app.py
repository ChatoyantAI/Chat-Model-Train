from __future__ import annotations

import csv
import hashlib
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
LABELS_CSV = Path(
    os.environ.get("ROLEPLAY_LABELS_CSV", MODULE_DIR / "data" / "evaluation_labels.csv")
)
COOKIE_NAME = "roleplay_annotation_session"
USERNAME_RE = re.compile(r"^[\w\-]{1,40}$", re.UNICODE)
APP_STATE: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_rubric() -> dict[str, Any]:
    with RUBRIC_PATH.open("r", encoding="utf-8") as handle:
        rubric = json.load(handle)
    items = [item for dimension in rubric["dimensions"] for item in dimension["items"]]
    ids = [item["id"] for item in items]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("rubric must contain unique atomic item IDs")
    valid_scores: dict[str, list[int | str]] = {}
    for item in items:
        scale = item.get("scale")
        if not isinstance(scale, list) or not scale:
            raise RuntimeError(f"rubric item {item['id']} has no score scale")
        if any(score not in {"0", "1", "2", "3"} for score in scale):
            raise RuntimeError(f"rubric item {item['id']} has an invalid score scale")
        if set(item.get("anchors", {})) != {*scale, "N/A"}:
            raise RuntimeError(f"rubric item {item['id']} has incomplete score anchors")
        valid_scores[item["id"]] = [*[int(score) for score in scale], "N/A"]

    gate_ids: list[str] = []
    boolean_gate_ids: list[str] = []
    enum_gate_values: dict[str, list[str]] = {}
    for group in rubric.get("gates", []):
        for gate in group.get("items", []):
            gate_id = str(gate["id"])
            gate_ids.append(gate_id)
            if gate.get("type") == "boolean":
                boolean_gate_ids.append(gate_id)
            elif gate.get("type") == "enum":
                enum_gate_values[gate_id] = [
                    str(option["value"]) for option in gate.get("options", [])
                ]
            else:
                raise RuntimeError(f"rubric gate {gate_id} has an invalid type")
    if not gate_ids or len(gate_ids) != len(set(gate_ids)):
        raise RuntimeError("rubric must contain unique Gate IDs")
    rubric["item_ids"] = ids
    rubric["valid_scores"] = valid_scores
    rubric["gate_ids"] = gate_ids
    rubric["boolean_gate_ids"] = boolean_gate_ids
    rubric["enum_gate_values"] = enum_gate_values
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
                rubric_version TEXT NOT NULL,
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
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(annotations)").fetchall()
        }
        if "rubric_version" not in columns:
            connection.execute(
                "ALTER TABLE annotations ADD COLUMN rubric_version TEXT NOT NULL "
                "DEFAULT 'legacy'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_user_rubric_status "
            "ON annotations(username, rubric_version, status, seq)"
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


def load_battle_pairs() -> list[dict[str, Any]]:
    if not LABELS_CSV.is_file():
        return []

    required = {
        "evaluation_id",
        "pair_group_id",
        "model_name",
        "human_selected",
        "source_completed_at",
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    label_ids: set[str] = set()
    with LABELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"battle labels missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            evaluation_id = str(row.get("evaluation_id", "")).strip()
            pair_group_id = str(row.get("pair_group_id", "")).strip()
            selected_raw = str(row.get("human_selected", "")).strip().lower()
            if not evaluation_id or not pair_group_id:
                raise RuntimeError(f"battle labels line {line_number} has an empty ID")
            if evaluation_id in label_ids:
                raise RuntimeError(f"duplicate battle evaluation_id: {evaluation_id}")
            if selected_raw in {"1", "true", "yes"}:
                selected = True
            elif selected_raw in {"0", "false", "no"}:
                selected = False
            else:
                raise RuntimeError(
                    f"battle labels line {line_number} has invalid human_selected"
                )
            label_ids.add(evaluation_id)
            groups.setdefault(pair_group_id, []).append(
                {
                    "evaluation_id": evaluation_id,
                    "model_name": str(row.get("model_name", "")).strip() or "未知模型",
                    "selected": selected,
                    "completed_at": str(row.get("source_completed_at", "")).strip(),
                }
            )

    with sample_connection() as connection:
        sample_ids = {
            str(row["evaluation_id"])
            for row in connection.execute("SELECT evaluation_id FROM samples")
        }
    missing_samples = label_ids - sample_ids
    if missing_samples:
        example = sorted(missing_samples)[:3]
        raise RuntimeError(f"battle labels reference missing samples: {example}")

    pairs: list[dict[str, Any]] = []
    for pair_group_id, entries in groups.items():
        if len(entries) != 2 or sum(bool(entry["selected"]) for entry in entries) != 1:
            raise RuntimeError(
                f"battle pair {pair_group_id} must contain two candidates and one winner"
            )
        entries.sort(
            key=lambda entry: hashlib.sha256(
                f"{pair_group_id}:{entry['evaluation_id']}".encode("utf-8")
            ).hexdigest()
        )
        pairs.append(
            {
                "entries": entries,
                "completed_at": entries[0]["completed_at"],
            }
        )
    return pairs


@asynccontextmanager
async def lifespan(_: FastAPI):
    APP_STATE["rubric"] = load_rubric()
    APP_STATE["sample_metadata"] = load_sample_metadata()
    APP_STATE["total"] = int(APP_STATE["sample_metadata"]["sample_count"])
    APP_STATE["battle_pairs"] = load_battle_pairs()
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


def sample_row_by_evaluation_id(evaluation_id: str) -> sqlite3.Row:
    with sample_connection() as connection:
        row = connection.execute(
            "SELECT seq, evaluation_id, trigger_type, payload_zlib "
            "FROM samples WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Battle 候选不存在")
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
            "FROM annotations WHERE username = ? AND rubric_version = ?",
            (username, APP_STATE["rubric"]["version"]),
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
            "SELECT seq FROM annotations "
            "WHERE username = ? AND rubric_version = ? AND status = 'submitted'",
            (username, APP_STATE["rubric"]["version"]),
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
            "FROM annotations WHERE username = ? AND seq = ? AND rubric_version = ?",
            (username, seq, APP_STATE["rubric"]["version"]),
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
        if score not in APP_STATE["rubric"]["valid_scores"][item_id] or isinstance(
            score, bool
        ):
            raise HTTPException(status_code=400, detail=f"{item_id} 分数无效")
        normalized_score: int | str = score
        evidence = str(value.get("evidence", "")).strip()
        if len(evidence) > 4000:
            raise HTTPException(status_code=400, detail=f"{item_id} 证据过长")
        normalized[item_id] = {"value": normalized_score, "evidence": evidence}
    return normalized


def normalize_gates(raw: Any, submitting: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="gates 必须是对象")
    expected = set(APP_STATE["rubric"]["gate_ids"])
    boolean_gates = set(APP_STATE["rubric"]["boolean_gate_ids"])
    enum_gates = {
        gate_id: set(values)
        for gate_id, values in APP_STATE["rubric"]["enum_gate_values"].items()
    }
    unknown = set(raw) - expected
    if unknown:
        raise HTTPException(status_code=400, detail=f"未知 Gate：{sorted(unknown)}")
    if submitting and set(raw) != expected:
        raise HTTPException(status_code=400, detail="请完成全部 Gate 判断")
    normalized: dict[str, Any] = {}
    for gate_id, value in raw.items():
        if gate_id in boolean_gates:
            if not isinstance(value, bool):
                raise HTTPException(status_code=400, detail=f"{gate_id} 必须为是或否")
            normalized[gate_id] = value
        elif value not in enum_gates[gate_id]:
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


@app.get("/battle")
def battle_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "battle.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "samples": APP_STATE.get("total", 0),
        "battle_pairs": len(APP_STATE.get("battle_pairs", [])),
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


@app.get("/api/battle/{pair_seq}")
def battle(pair_seq: int, request: Request) -> dict[str, Any]:
    require_username(request)
    pairs = APP_STATE.get("battle_pairs", [])
    if not pairs:
        raise HTTPException(status_code=503, detail="Battle 回放数据尚未配置")
    if pair_seq < 1 or pair_seq > len(pairs):
        raise HTTPException(status_code=404, detail="Battle 序号不存在")

    pair = pairs[pair_seq - 1]
    decoded = [
        decode_sample(sample_row_by_evaluation_id(str(entry["evaluation_id"])))
        for entry in pair["entries"]
    ]
    context_match = (
        decoded[0]["messages"] == decoded[1]["messages"]
        and decoded[0]["trigger_type"] == decoded[1]["trigger_type"]
    )
    candidates = []
    online_selected = ""
    for index, (entry, payload) in enumerate(zip(pair["entries"], decoded, strict=True)):
        label = "A" if index == 0 else "B"
        selected = bool(entry["selected"])
        if selected:
            online_selected = label
        candidates.append(
            {
                "label": label,
                "content": payload["candidate"]["content"],
                "online_selected": selected,
            }
        )
    return {
        "pair_seq": pair_seq,
        "total": len(pairs),
        "trigger_type": decoded[0]["trigger_type"],
        "messages": decoded[0]["messages"],
        "context_match": context_match,
        "candidates": candidates,
        "online_selected": online_selected,
        "source_completed_at": pair["completed_at"],
    }


@app.get("/api/battle/{pair_seq}/models")
def battle_models(pair_seq: int, request: Request) -> dict[str, Any]:
    require_username(request)
    pairs = APP_STATE.get("battle_pairs", [])
    if not pairs:
        raise HTTPException(status_code=503, detail="Battle 回放数据尚未配置")
    if pair_seq < 1 or pair_seq > len(pairs):
        raise HTTPException(status_code=404, detail="Battle 序号不存在")
    entries = pairs[pair_seq - 1]["entries"]
    return {
        "pair_seq": pair_seq,
        "models": [
            {
                "label": "A" if index == 0 else "B",
                "model_name": str(entry["model_name"]),
            }
            for index, entry in enumerate(entries)
        ],
    }


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
                username, seq, evaluation_id, rubric_version, status, gates_json, scores_json,
                gate_notes, overall_notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, seq) DO UPDATE SET
                evaluation_id = excluded.evaluation_id,
                rubric_version = excluded.rubric_version,
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
                APP_STATE["rubric"]["version"],
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
