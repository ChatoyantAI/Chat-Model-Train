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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .battle_weight_fit import (
    BATTLE_WEIGHT_FIT_VERSION,
    fit_battle_weights,
    score_battle_pair,
)
from .llm_judge import (
    GATE_DISCARD_CONDITIONS,
    build_gate_request,
    build_gate_schema,
    build_score_request,
    build_score_schema,
    canonical_json,
    judge_model_gateway,
    judge_request_url,
    load_judge_config,
    public_config,
)
from .llm_runtime import (
    MAX_COMPARISON_MODELS,
    MAX_COMPARISON_SAMPLES,
    MIN_COMPARISON_MODELS,
    LLMJudgeRuntime,
)
from .model_catalog import OpenRouterModelCatalog
from .rubric_migrations import (
    LEGACY_NARRATIVE_RUBRIC_VERSIONS,
    PASSTHROUGH_RUBRIC_VERSIONS,
    merge_legacy_narrative_scores,
)


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
LEGACY_SAFETY_GATE_IDS = frozenset({"G-S", "G-S1", "G-S2", "G-S3"})


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
        allows_na = bool(item.get("allows_na", True))
        expected_anchors = {*scale, *({"N/A"} if allows_na else set())}
        if set(item.get("anchors", {})) != expected_anchors:
            raise RuntimeError(f"rubric item {item['id']} has incomplete score anchors")
        valid_scores[item["id"]] = [
            *[int(score) for score in scale],
            *(["N/A"] if allows_na else []),
        ]

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


def remove_legacy_safety_gates(gates: dict[str, Any]) -> dict[str, Any]:
    """Drop retired safety fields while preserving every active Gate answer."""
    return {
        gate_id: value
        for gate_id, value in gates.items()
        if gate_id not in LEGACY_SAFETY_GATE_IDS
    }


def migrate_annotations_to_current_rubric() -> int:
    """Upgrade supported human annotations while preserving completed progress."""
    migrated_rows: list[dict[str, Any]] = []
    current_version = str(APP_STATE["rubric"]["version"])
    legacy_versions = LEGACY_NARRATIVE_RUBRIC_VERSIONS
    # V0.9+ changes the meaning and shape of H/L scores, so older answers must
    # remain attached to their original rubric instead of being carried forward.
    compatible_versions = (
        legacy_versions | PASSTHROUGH_RUBRIC_VERSIONS
        if current_version in {"roleplay-rubric-v0.7", "roleplay-rubric-v0.8"}
        else frozenset()
    )
    with result_connection() as connection:
        rows = connection.execute(
            "SELECT username, seq, evaluation_id, rubric_version, status, gates_json, scores_json, "
            "gate_notes, overall_notes, created_at, updated_at FROM annotations"
        ).fetchall()
        for row in rows:
            row_version = str(row["rubric_version"])
            if row_version not in {*compatible_versions, current_version}:
                continue
            try:
                gates = json.loads(str(row["gates_json"]))
                scores = json.loads(str(row["scores_json"]))
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(gates, dict) or not isinstance(scores, dict):
                continue
            migrated_gates = remove_legacy_safety_gates(gates)
            migrated_scores = (
                merge_legacy_narrative_scores(scores)
                if row_version in legacy_versions
                else scores
            )
            if (
                row_version == current_version
                and migrated_gates == gates
                and migrated_scores == scores
            ):
                continue
            connection.execute(
                "UPDATE annotations SET rubric_version = ?, gates_json = ?, scores_json = ? "
                "WHERE username = ? AND seq = ? AND rubric_version = ?",
                (
                    current_version,
                    json.dumps(
                        migrated_gates,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        migrated_scores,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    str(row["username"]),
                    int(row["seq"]),
                    row_version,
                ),
            )
            migrated_rows.append(
                {
                    **dict(row),
                    "gates": migrated_gates,
                    "scores": migrated_scores,
                }
            )

    for row in migrated_rows:
        write_snapshot(
            str(row["username"]),
            int(row["seq"]),
            str(row["evaluation_id"]),
            str(row["status"]),
            row["gates"],
            row["scores"],
            str(row["gate_notes"]),
            str(row["overall_notes"]),
            str(row["created_at"]),
            str(row["updated_at"]),
        )
    return len(migrated_rows)


def ensure_versioned_annotation_primary_key(connection: sqlite3.Connection) -> None:
    """Keep one human annotation per sample *and* rubric version."""
    table_info = connection.execute("PRAGMA table_info(annotations)").fetchall()
    primary_key_columns = [
        str(row["name"])
        for row in sorted(table_info, key=lambda row: int(row["pk"]))
        if int(row["pk"]) > 0
    ]
    if primary_key_columns == ["username", "seq", "rubric_version"]:
        return

    connection.execute("DROP TABLE IF EXISTS annotations_versioned")
    connection.execute(
        """
        CREATE TABLE annotations_versioned (
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
            PRIMARY KEY (username, seq, rubric_version)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO annotations_versioned(
            username, seq, evaluation_id, rubric_version, status, gates_json,
            scores_json, gate_notes, overall_notes, created_at, updated_at
        )
        SELECT username, seq, evaluation_id, rubric_version, status, gates_json,
               scores_json, gate_notes, overall_notes, created_at, updated_at
        FROM annotations
        """
    )
    connection.execute("DROP TABLE annotations")
    connection.execute("ALTER TABLE annotations_versioned RENAME TO annotations")


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
                PRIMARY KEY (username, seq, rubric_version)
            );
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
        ensure_versioned_annotation_primary_key(connection)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_user_status "
            "ON annotations(username, status, seq)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotations_user_rubric_status "
            "ON annotations(username, rubric_version, status, seq)"
        )
    migrate_annotations_to_current_rubric()


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
        sample_sequences = {
            str(row["evaluation_id"]): int(row["seq"])
            for row in connection.execute("SELECT seq, evaluation_id FROM samples")
        }
    missing_samples = label_ids - set(sample_sequences)
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
        for entry in entries:
            entry["seq"] = sample_sequences[str(entry["evaluation_id"])]
        pairs.append(
            {
                "pair_group_id": pair_group_id,
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
    base_config = load_judge_config()
    catalog = OpenRouterModelCatalog(base_config)
    await catalog.refresh()
    APP_STATE["llm_catalog"] = catalog
    available_models = catalog.model_ids
    APP_STATE["llm_available_models"] = available_models
    APP_STATE["llm_config"] = replace(
        base_config,
        models=available_models,
        reasoning_models=catalog.reasoning_model_ids,
    )
    runtime = LLMJudgeRuntime(
        config=APP_STATE["llm_config"],
        rubric=APP_STATE["rubric"],
        connection_factory=result_connection,
        sample_loader=load_llm_sample,
        utc_now=utc_now,
    )
    APP_STATE["llm_runtime"] = runtime
    runtime.initialize()
    active_models = runtime.active_job_models()
    runtime_config = replace(
        APP_STATE["llm_config"],
        models=tuple(dict.fromkeys((*available_models, *active_models))),
    )
    APP_STATE["llm_config"] = runtime_config
    runtime.update_model_catalog(
        config=runtime_config,
        available_models=available_models,
    )
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


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


def load_llm_sample(seq: int) -> tuple[str, dict[str, Any]]:
    row = sample_row(seq)
    return str(row["evaluation_id"]), decode_sample(row)


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


def llm_runtime() -> LLMJudgeRuntime:
    runtime = APP_STATE.get("llm_runtime")
    if not isinstance(runtime, LLMJudgeRuntime):
        raise HTTPException(status_code=503, detail="LLM Judge 尚未初始化")
    return runtime


def _body_int(body: dict[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    try:
        value = int(body.get(key))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{key} 必须是整数") from None
    if value < minimum or value > maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{key} 必须在 {minimum}–{maximum} 之间",
        )
    return value


def _job_sequences(body: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    total = int(APP_STATE["total"])
    scope_type = str(body.get("scope_type", "current"))
    if scope_type == "current":
        seq = _body_int(body, "seq", minimum=1, maximum=total)
        return [seq], {"type": "current", "seq": seq}
    if scope_type == "range":
        start = _body_int(body, "start_seq", minimum=1, maximum=total)
        end = _body_int(body, "end_seq", minimum=1, maximum=total)
        if end < start:
            raise HTTPException(status_code=400, detail="结束序号不能小于开始序号")
        return list(range(start, end + 1)), {
            "type": "range",
            "start_seq": start,
            "end_seq": end,
        }
    if scope_type == "random":
        count = _body_int(body, "random_count", minimum=1, maximum=total)
        sequences = secrets.SystemRandom().sample(range(1, total + 1), count)
        return sequences, {"type": "random", "count": count}
    if scope_type == "all":
        return list(range(1, total + 1)), {"type": "all"}
    raise HTTPException(status_code=400, detail="未知样本范围")


def _battle_job_sequences(body: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    pairs = APP_STATE.get("battle_pairs", [])
    total = len(pairs)
    if not total:
        raise HTTPException(status_code=503, detail="Battle 回放数据尚未配置")
    scope_type = str(body.get("scope_type", "current"))
    if scope_type == "current":
        pair_seq = _body_int(body, "pair_seq", minimum=1, maximum=total)
        pair_sequences = [pair_seq]
        selection = {"selection": "current", "pair_seq": pair_seq}
    elif scope_type == "range":
        start = _body_int(body, "start_pair", minimum=1, maximum=total)
        end = _body_int(body, "end_pair", minimum=1, maximum=total)
        if end < start:
            raise HTTPException(status_code=400, detail="结束组号不能小于开始组号")
        pair_sequences = list(range(start, end + 1))
        selection = {
            "selection": "range",
            "start_pair": start,
            "end_pair": end,
        }
    elif scope_type == "random":
        count = _body_int(body, "random_count", minimum=1, maximum=total)
        pair_sequences = secrets.SystemRandom().sample(range(1, total + 1), count)
        selection = {"selection": "random", "count": count}
    elif scope_type == "all":
        pair_sequences = list(range(1, total + 1))
        selection = {"selection": "all"}
    else:
        raise HTTPException(status_code=400, detail="未知 Battle 范围")

    max_pairs = max(1, int(APP_STATE["llm_config"].max_job_samples) // 2)
    if len(pair_sequences) > max_pairs:
        raise HTTPException(
            status_code=400,
            detail=f"单个 Battle 任务最多允许 {max_pairs} 组",
        )
    sample_sequences = [
        int(entry["seq"])
        for pair_seq in pair_sequences
        for entry in pairs[pair_seq - 1]["entries"]
    ]
    if len(sample_sequences) != len(set(sample_sequences)):
        raise HTTPException(status_code=500, detail="Battle 配对包含重复候选样本")
    return sample_sequences, {
        "type": "battle",
        **selection,
        "pair_sequences": pair_sequences,
        "pair_count": len(pair_sequences),
    }


def _battle_payload(pair_seq: int) -> dict[str, Any]:
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
                "seq": int(entry["seq"]),
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


def _battle_job(username: str, job_id: str) -> dict[str, Any]:
    job = llm_runtime().get_job(username, job_id)
    if job.get("scope", {}).get("type") != "battle":
        raise LookupError("LLM Judge Battle 任务不存在")
    return job


def _battle_weight_fit_pairs(username: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    items = {
        int(item["seq"]): item
        for item in llm_runtime().list_weight_fit_items(username, str(job["job_id"]))
    }
    result: list[dict[str, Any]] = []
    for pair_seq in map(int, job["scope"]["pair_sequences"]):
        pair = APP_STATE["battle_pairs"][pair_seq - 1]
        left_source, right_source = pair["entries"]
        left = items.get(int(left_source["seq"]), {})
        right = items.get(int(right_source["seq"]), {})
        result.append(
            {
                "pair_seq": pair_seq,
                "fold_key": str(pair.get("pair_group_id", pair_seq)),
                "online_selected": "A" if left_source["selected"] else "B",
                "left_status": str(left.get("status", "missing")),
                "right_status": str(right.get("status", "missing")),
                "left_scores": left.get("scores"),
                "right_scores": right.get("scores"),
            }
        )
    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/battle")
def battle_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "battle.html")


@app.get("/llm-judge")
def llm_judge_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "llm_judge.html")


@app.get("/llm-compare")
def llm_compare_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "llm_compare.html")


@app.get("/llm-battle")
def llm_battle_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "llm_battle.html")


@app.get("/health")
def health() -> dict[str, Any]:
    config = APP_STATE.get("llm_config")
    return {
        "status": "ok",
        "samples": APP_STATE.get("total", 0),
        "battle_pairs": len(APP_STATE.get("battle_pairs", [])),
        "rubric_version": APP_STATE.get("rubric", {}).get("version"),
        "llm_judge_enabled": bool(config and config.enabled),
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
    payload = _battle_payload(pair_seq)
    for candidate in payload["candidates"]:
        candidate.pop("seq", None)
    return payload


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


@app.get("/api/llm-judge/config")
def llm_judge_config(request: Request) -> dict[str, Any]:
    require_username(request)
    payload = public_config(APP_STATE["llm_config"])
    catalog_payload = APP_STATE["llm_catalog"].public_snapshot()
    payload["models"] = catalog_payload.pop("models")
    payload["model_catalog"] = catalog_payload
    payload.update(
        {
            "rubric_version": APP_STATE["rubric"]["version"],
            "rubric_sha256": APP_STATE["rubric"]["source_sha256"],
            "sample_count": int(APP_STATE["total"]),
            "comparison_limits": {
                "min_models": MIN_COMPARISON_MODELS,
                "max_models": MAX_COMPARISON_MODELS,
                "max_samples": MAX_COMPARISON_SAMPLES,
            },
            "battle_limits": {
                "pair_count": len(APP_STATE.get("battle_pairs", [])),
                "max_pairs_per_job": max(1, int(APP_STATE["llm_config"].max_job_samples) // 2),
            },
        }
    )
    return payload


@app.post("/api/llm-judge/models/refresh")
async def refresh_llm_judge_models(request: Request) -> dict[str, Any]:
    require_username(request)
    catalog: OpenRouterModelCatalog = APP_STATE["llm_catalog"]
    payload = await catalog.refresh()
    available_models = catalog.model_ids
    runtime = llm_runtime()
    active_models = runtime.active_job_models()
    config = replace(
        APP_STATE["llm_config"],
        models=tuple(dict.fromkeys((*available_models, *active_models))),
        reasoning_models=catalog.reasoning_model_ids,
    )
    APP_STATE["llm_available_models"] = available_models
    APP_STATE["llm_config"] = config
    runtime.update_model_catalog(
        config=config,
        available_models=available_models,
    )
    return payload


@app.get("/api/llm-judge/schema")
def llm_judge_schema(request: Request) -> dict[str, Any]:
    require_username(request)
    gate_schema = build_gate_schema(APP_STATE["rubric"])
    score_schema = build_score_schema(APP_STATE["rubric"])
    schema = {
        "phase_1_gate": gate_schema,
        "phase_2_dimensions": score_schema,
    }
    schema_sha256 = hashlib.sha256(canonical_json(schema).encode("utf-8")).hexdigest()
    return {
        "rubric_version": APP_STATE["rubric"]["version"],
        "rubric_sha256": APP_STATE["rubric"]["source_sha256"],
        "schema_sha256": schema_sha256,
        "schema": schema,
        "pipeline": [
            {
                "phase": "gate",
                "always_runs": True,
                "on_failure": "discard_without_dimension_scoring",
            },
            {
                "phase": "dimensions",
                "runs_only_if": "no discard Gate is triggered",
            },
        ],
        "discard_conditions": list(GATE_DISCARD_CONDITIONS),
    }


@app.get("/api/llm-judge/request-preview")
def llm_judge_request_preview(
    request: Request,
    seq: int = 1,
    model: str | None = None,
) -> dict[str, Any]:
    """Build the preview through the exact same function used by queue workers."""
    require_username(request)
    config = APP_STATE["llm_config"]
    available_models = APP_STATE["llm_available_models"]
    selected_model = (model or (available_models[0] if available_models else "")).strip()
    if not selected_model:
        raise HTTPException(status_code=503, detail="服务器尚未配置可用 Judge 模型")
    if selected_model not in available_models:
        raise HTTPException(
            status_code=400,
            detail="所选模型不在当前 Judge 兼容模型目录中",
        )
    sample = decode_sample(sample_row(seq))
    try:
        gate_body, sample_packet, gate_hash = build_gate_request(
            selected_model, APP_STATE["rubric"], sample, config
        )
        score_body, _, score_hash = build_score_request(
            selected_model, APP_STATE["rubric"], sample, config
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    gateway = judge_model_gateway(selected_model)
    headers = {
        "Authorization": "Bearer <SERVER_ONLY_REDACTED>",
        "Content-Type": "application/json",
    }
    if gateway == "openrouter":
        headers.update(
            {
                "HTTP-Referer": "http://roleplay-annotation.local/llm-judge",
                "X-Title": "Roleplay LLM Judge",
            }
        )
    return {
        "seq": seq,
        "method": "POST",
        "gateway": gateway,
        "endpoint": judge_request_url(config, selected_model),
        "headers": headers,
        "model": selected_model,
        "request_hashes": {
            "gate": gate_hash,
            "dimensions": score_hash,
        },
        "gate_policy": {
            "on_any_match": "discard_without_dimension_scoring",
            "discard_conditions": list(GATE_DISCARD_CONDITIONS),
            "continues_without_discard": [],
        },
        "prompt_version": public_config(config)["prompt_version"],
        "schema_version": public_config(config)["schema_version"],
        "sample_packet": sample_packet,
        "phases": [
            {
                "phase": "gate",
                "condition": "always",
                "request_body": gate_body,
            },
            {
                "phase": "dimensions",
                "condition": "only_if_gate_passes",
                "request_body": score_body,
            },
        ],
        "redactions": [
            "OpenRouter API Key" if gateway == "openrouter" else "TatuCloud API Key"
        ],
        "excluded_sample_fields": [
            "evaluation_id",
            "source model identity",
            "human vote",
            "battle winner",
        ],
    }


@app.post("/api/llm-judge/jobs")
async def create_llm_judge_job(request: Request) -> dict[str, Any]:
    username = require_username(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
    model = str(body.get("model", "")).strip()
    sequences, scope = _job_sequences(body)
    try:
        budget_limit_usd = float(body.get("budget_limit_usd", 5.0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="预算必须是数字") from None
    try:
        return llm_runtime().create_job(
            username=username,
            model=model,
            sequences=sequences,
            scope=scope,
            budget_limit_usd=budget_limit_usd,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.get("/api/llm-judge/jobs")
def list_llm_judge_jobs(request: Request, limit: int = 30) -> dict[str, Any]:
    username = require_username(request)
    return {
        "jobs": llm_runtime().list_jobs(
            username,
            limit=limit,
            exclude_scope_type="battle",
        )
    }


@app.post("/api/llm-judge/battles")
async def create_llm_judge_battle(request: Request) -> dict[str, Any]:
    username = require_username(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
    model = str(body.get("model", "")).strip()
    sequences, scope = _battle_job_sequences(body)
    try:
        budget_limit_usd = float(body.get("budget_limit_usd", 5.0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="预算必须是数字") from None
    try:
        return llm_runtime().create_job(
            username=username,
            model=model,
            sequences=sequences,
            scope=scope,
            budget_limit_usd=budget_limit_usd,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.get("/api/llm-judge/battles")
def list_llm_judge_battles(request: Request, limit: int = 30) -> dict[str, Any]:
    username = require_username(request)
    jobs = llm_runtime().list_jobs(
        username,
        limit=max(1, min(int(limit), 50)),
        scope_type="battle",
    )
    return {"battles": jobs}


@app.get("/api/llm-judge/battles/{job_id}")
def get_llm_judge_battle(job_id: str, request: Request) -> dict[str, Any]:
    username = require_username(request)
    try:
        return _battle_job(username, job_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@app.get("/api/llm-judge/battles/{job_id}/weight-fit")
def get_llm_judge_battle_weight_fit(
    job_id: str, request: Request
) -> dict[str, Any]:
    username = require_username(request)
    try:
        _battle_job(username, job_id)
        return {"fit": llm_runtime().get_battle_weight_fit(username, job_id)}
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@app.post("/api/llm-judge/battles/{job_id}/weight-fit")
def fit_llm_judge_battle_weights(job_id: str, request: Request) -> dict[str, Any]:
    username = require_username(request)
    try:
        job = _battle_job(username, job_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    if job["rubric_sha256"] != APP_STATE["rubric"]["source_sha256"]:
        raise HTTPException(
            status_code=409,
            detail="该 Battle 使用的不是当前评分规则，不能拟合当前原子项权重",
        )
    if job["counts"]["pending"] or job["counts"]["running"]:
        raise HTTPException(
            status_code=409,
            detail="请等待 Battle 候选全部结束后再拟合权重",
        )
    pairs = _battle_weight_fit_pairs(username, job)
    try:
        report = fit_battle_weights(APP_STATE["rubric"], pairs)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    report.update(
        {
            "job_id": job_id,
            "judge_model": job["model"],
            "rubric_version": job["rubric_version"],
            "rubric_sha256": job["rubric_sha256"],
        }
    )
    llm_runtime().save_battle_weight_fit(
        username,
        job_id,
        fit_version=BATTLE_WEIGHT_FIT_VERSION,
        report=report,
    )
    return {"fit": report}


@app.get("/api/llm-judge/battles/{job_id}/pairs")
def list_llm_judge_battle_pairs(
    job_id: str,
    request: Request,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    username = require_username(request)
    try:
        job = _battle_job(username, job_id)
        pair_sequences = [int(value) for value in job["scope"]["pair_sequences"]]
        safe_offset = max(0, int(offset))
        safe_limit = max(1, min(int(limit), 250))
        selected = pair_sequences[safe_offset : safe_offset + safe_limit]
        item_page = llm_runtime().list_items(
            username,
            job_id,
            offset=safe_offset * 2,
            limit=max(1, len(selected) * 2),
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    items = item_page["items"]
    rows = []
    for index, pair_seq in enumerate(selected):
        candidate_items = items[index * 2 : index * 2 + 2]
        rows.append(
            {
                "position": safe_offset + index + 1,
                "pair_seq": pair_seq,
                "candidates": [
                    {"label": "A" if candidate_index == 0 else "B", **item}
                    for candidate_index, item in enumerate(candidate_items)
                ],
            }
        )
    return {
        "job_id": job_id,
        "offset": safe_offset,
        "limit": safe_limit,
        "total": len(pair_sequences),
        "pairs": rows,
    }


@app.get("/api/llm-judge/battles/{job_id}/pairs/{pair_seq}")
def get_llm_judge_battle_pair(
    job_id: str, pair_seq: int, request: Request
) -> dict[str, Any]:
    username = require_username(request)
    try:
        job = _battle_job(username, job_id)
        pair_sequences = [int(value) for value in job["scope"]["pair_sequences"]]
        pair_index = pair_sequences.index(int(pair_seq))
        items = llm_runtime().list_items(
            username, job_id, offset=pair_index * 2, limit=2
        )["items"]
    except ValueError:
        raise HTTPException(status_code=404, detail="该任务不包含这组 Battle") from None
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    payload = _battle_payload(pair_seq)
    results = []
    for index, item in enumerate(items):
        judgment = None
        if item.get("run_id"):
            judgment = llm_runtime().get_judgment(
                username, job_id, str(item["run_id"])
            )
        results.append(
            {
                "label": "A" if index == 0 else "B",
                "item": item,
                "judgment": judgment,
            }
        )
    learned_comparison = None
    fit = llm_runtime().get_battle_weight_fit(username, job_id)
    if fit is not None:
        left_result, right_result = results
        left_judgment = left_result.get("judgment") or {}
        right_judgment = right_result.get("judgment") or {}
        left_scores = (left_judgment.get("judgment") or {}).get("scores")
        right_scores = (right_judgment.get("judgment") or {}).get("scores")
        try:
            learned_comparison = score_battle_pair(
                APP_STATE["rubric"],
                {
                    "left_status": str(left_result["item"].get("status", "missing")),
                    "right_status": str(right_result["item"].get("status", "missing")),
                    "left_scores": left_scores,
                    "right_scores": right_scores,
                },
                fit["parameters"],
            )
        except ValueError as error:
            learned_comparison = {
                "comparable": False,
                "pair_kind": "parameter_error",
                "reason": str(error),
            }
    return {
        "job": job,
        "battle": payload,
        "results": results,
        "learned_comparison": learned_comparison,
    }


@app.post("/api/llm-judge/comparisons")
async def create_llm_judge_comparison(request: Request) -> dict[str, Any]:
    username = require_username(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="请求体不是合法 JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
    raw_models = body.get("models")
    if not isinstance(raw_models, list):
        raise HTTPException(status_code=400, detail="models 必须是模型 ID 数组")
    models = [str(model).strip() for model in raw_models if str(model).strip()]
    sequences, scope = _job_sequences(body)
    try:
        budget_per_model_usd = float(body.get("budget_per_model_usd", 5.0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="每模型预算必须是数字") from None
    try:
        return llm_runtime().create_comparison(
            username=username,
            models=models,
            sequences=sequences,
            scope=scope,
            budget_per_model_usd=budget_per_model_usd,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.get("/api/llm-judge/comparisons")
def list_llm_judge_comparisons(
    request: Request, limit: int = 30
) -> dict[str, Any]:
    username = require_username(request)
    return {"comparisons": llm_runtime().list_comparisons(username, limit=limit)}


@app.get("/api/llm-judge/comparisons/{comparison_id}")
def get_llm_judge_comparison(
    comparison_id: str, request: Request
) -> dict[str, Any]:
    username = require_username(request)
    try:
        return llm_runtime().get_comparison(username, comparison_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@app.get("/api/llm-judge/comparisons/{comparison_id}/samples/{seq}")
def get_llm_judge_comparison_sample(
    comparison_id: str, seq: int, request: Request
) -> dict[str, Any]:
    username = require_username(request)
    try:
        return llm_runtime().get_comparison_sample(username, comparison_id, seq)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@app.post("/api/llm-judge/comparisons/{comparison_id}/{action}")
def control_llm_judge_comparison(
    comparison_id: str, action: str, request: Request
) -> dict[str, Any]:
    username = require_username(request)
    if action not in {"pause", "resume", "cancel"}:
        raise HTTPException(status_code=404, detail="未知比较任务操作")
    try:
        return llm_runtime().control_comparison(username, comparison_id, action)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.get("/api/llm-judge/jobs/{job_id}")
def get_llm_judge_job(job_id: str, request: Request) -> dict[str, Any]:
    username = require_username(request)
    try:
        return llm_runtime().get_job(username, job_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@app.get("/api/llm-judge/jobs/{job_id}/items")
def list_llm_judge_items(
    job_id: str,
    request: Request,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    username = require_username(request)
    try:
        return llm_runtime().list_items(
            username, job_id, offset=offset, limit=limit
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@app.get("/api/llm-judge/jobs/{job_id}/judgments/{run_id}")
def get_llm_judgment(job_id: str, run_id: str, request: Request) -> dict[str, Any]:
    username = require_username(request)
    try:
        return llm_runtime().get_judgment(username, job_id, run_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None


@app.post("/api/llm-judge/jobs/{job_id}/{action}")
def control_llm_judge_job(
    job_id: str, action: str, request: Request
) -> dict[str, Any]:
    username = require_username(request)
    runtime = llm_runtime()
    try:
        if action == "pause":
            return runtime.pause_job(username, job_id)
        if action == "resume":
            return runtime.resume_job(username, job_id)
        if action == "cancel":
            return runtime.cancel_job(username, job_id)
        raise HTTPException(status_code=404, detail="未知任务操作")
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@app.post("/api/annotation")
async def save_annotation(request: Request) -> dict[str, Any]:
    username = require_username(request)
    body = await request.json()
    current_rubric = APP_STATE["rubric"]
    if (
        body.get("rubric_version") != current_rubric["version"]
        or body.get("rubric_sha256") != current_rubric["source_sha256"]
    ):
        raise HTTPException(
            status_code=409,
            detail="评分标准已经更新，请完整刷新页面后再继续标注",
        )
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
            "SELECT created_at FROM annotations "
            "WHERE username = ? AND seq = ? AND rubric_version = ?",
            (username, seq, APP_STATE["rubric"]["version"]),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        connection.execute(
            """
            INSERT INTO annotations(
                username, seq, evaluation_id, rubric_version, status, gates_json, scores_json,
                gate_notes, overall_notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username, seq, rubric_version) DO UPDATE SET
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
