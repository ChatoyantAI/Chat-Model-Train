from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import sqlite3
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

import httpx

from .llm_judge import (
    JUDGMENT_SCHEMA_VERSION,
    PROMPT_VERSION,
    JudgeConfig,
    JudgeValidationError,
    build_gate_request,
    build_score_request,
    canonical_json,
    attach_score_logprob_confidence,
    backend_model_id,
    compact_openrouter_logprobs,
    gate_failure_ids,
    judge_model_gateway,
    judge_request_url,
    parse_and_validate_gate_response,
    parse_and_validate_score_response,
    provider_label_for_model,
)
from .rubric_migrations import scores_for_current_rubric


JOB_STATUSES = {
    "queued",
    "running",
    "paused",
    "completed",
    "completed_with_errors",
    "cancelled",
    "budget_exhausted",
}
ACTIVE_JOB_STATUSES = {"queued", "running", "paused"}
ITEM_STATUSES = {
    "pending",
    "running",
    "completed",
    "discarded",
    "invalid",
    "failed",
    "cancelled",
}
FINAL_ITEM_STATUSES = {"completed", "discarded", "invalid", "failed", "cancelled"}
MAX_ATTEMPTS = 3
MIN_COMPARISON_MODELS = 2
MAX_COMPARISON_MODELS = 20
MAX_COMPARISON_SAMPLES = 500


class JudgePhaseHTTPError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        payload: dict[str, Any],
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.payload = payload
        self.retry_after_seconds = retry_after_seconds


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return max(0.0, min(seconds, 600.0))


def _rate_limit_retry_delay(
    *,
    job_id: str,
    seq: int,
    attempt_number: int,
    config: JudgeConfig,
    retry_after_seconds: float | None,
) -> float:
    exponential = config.rate_limit_backoff_seconds * (3 ** max(0, attempt_number - 1))
    base_delay = min(exponential, config.rate_limit_backoff_max_seconds)
    seed = hashlib.sha256(f"{job_id}:{seq}:{attempt_number}".encode("utf-8")).digest()
    fraction = int.from_bytes(seed[:8], "big") / float(2**64 - 1)
    jittered = base_delay * (0.8 + 0.4 * fraction)
    return max(jittered, retry_after_seconds or 0.0)


def _public_model_outputs(raw_response_json: str | None) -> dict[str, Any]:
    if not raw_response_json:
        return {}
    try:
        payloads = json.loads(raw_response_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(payloads, dict):
        return {}

    outputs: dict[str, Any] = {}
    for phase, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(choice, dict):
            choice = {}
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {}
        content = message.get("content")
        parsed_json = None
        if isinstance(content, str):
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                pass
        elif isinstance(content, (dict, list)):
            parsed_json = content
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        completion_details = usage.get("completion_tokens_details")
        if not isinstance(completion_details, dict):
            completion_details = {}
        reasoning = message.get("reasoning")
        outputs[str(phase)] = {
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "content_type": "null" if content is None else type(content).__name__,
            "content": content,
            "parsed_json": parsed_json,
            "refusal": message.get("refusal"),
            "logprobs_summary": choice.get("logprobs_summary"),
            "reasoning_tokens": _as_int(completion_details.get("reasoning_tokens")),
            "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        }
    return outputs


class LLMJudgeRuntime:
    def __init__(
        self,
        *,
        config: JudgeConfig,
        rubric: dict[str, Any],
        connection_factory: Callable[[], sqlite3.Connection],
        sample_loader: Callable[[int], tuple[str, dict[str, Any]]],
        utc_now: Callable[[], str],
    ) -> None:
        self.config = config
        self.rubric = rubric
        self.connection_factory = connection_factory
        self.sample_loader = sample_loader
        self.utc_now = utc_now
        self._workers: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        self._wake = asyncio.Event()
        self._retry_sleep = asyncio.sleep
        self.available_models = set(config.models)
        self.gate_ids = tuple(
            str(gate["id"]) for group in rubric["gates"] for gate in group["items"]
        )

    def update_model_catalog(
        self,
        *,
        config: JudgeConfig,
        available_models: tuple[str, ...],
    ) -> None:
        self.config = config
        self.available_models = set(available_models)

    def active_job_models(self) -> tuple[str, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT model FROM llm_jobs "
                "WHERE status IN ('queued', 'running', 'paused') "
                "AND rubric_version = ? AND rubric_sha256 = ?",
                (self.rubric["version"], self.rubric["source_sha256"]),
            ).fetchall()
        return tuple(str(row["model"]) for row in rows)

    @contextmanager
    def _connection(self):
        connection = self.connection_factory()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS llm_jobs (
                    job_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL REFERENCES users(username),
                    model TEXT NOT NULL,
                    rubric_version TEXT NOT NULL,
                    rubric_sha256 TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    budget_limit_usd REAL NOT NULL,
                    status TEXT NOT NULL,
                    total_count INTEGER NOT NULL,
                    actual_cost_usd REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_llm_jobs_user_created
                    ON llm_jobs(username, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_llm_jobs_status_created
                    ON llm_jobs(status, created_at);

                CREATE TABLE IF NOT EXISTS llm_job_items (
                    job_id TEXT NOT NULL REFERENCES llm_jobs(job_id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    locked_at TEXT,
                    run_id TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, seq),
                    UNIQUE(job_id, position)
                );
                CREATE INDEX IF NOT EXISTS idx_llm_items_claim
                    ON llm_job_items(status, job_id, position);

                CREATE TABLE IF NOT EXISTS llm_judgments (
                    run_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES llm_jobs(job_id),
                    seq INTEGER NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    request_hash TEXT,
                    model_requested TEXT NOT NULL,
                    model_resolved TEXT,
                    provider TEXT,
                    rubric_version TEXT NOT NULL,
                    rubric_sha256 TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    gates_json TEXT,
                    scores_json TEXT,
                    raw_response_json TEXT,
                    raw_content TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_llm_judgments_job_seq
                    ON llm_judgments(job_id, seq, attempt_number);
                CREATE INDEX IF NOT EXISTS idx_llm_judgments_eval_model
                    ON llm_judgments(evaluation_id, model_requested, created_at);

                CREATE TABLE IF NOT EXISTS llm_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL REFERENCES users(username),
                    rubric_version TEXT NOT NULL,
                    rubric_sha256 TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    scope_json TEXT NOT NULL,
                    models_json TEXT NOT NULL,
                    budget_per_model_usd REAL NOT NULL,
                    total_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_llm_comparisons_user_created
                    ON llm_comparisons(username, created_at DESC);

                CREATE TABLE IF NOT EXISTS llm_comparison_jobs (
                    comparison_id TEXT NOT NULL
                        REFERENCES llm_comparisons(comparison_id) ON DELETE CASCADE,
                    model_position INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    job_id TEXT NOT NULL UNIQUE REFERENCES llm_jobs(job_id),
                    PRIMARY KEY(comparison_id, model_position),
                    UNIQUE(comparison_id, model)
                );
                CREATE INDEX IF NOT EXISTS idx_llm_comparison_jobs_job
                    ON llm_comparison_jobs(job_id);

                CREATE TABLE IF NOT EXISTS llm_battle_weight_fits (
                    job_id TEXT PRIMARY KEY
                        REFERENCES llm_jobs(job_id) ON DELETE CASCADE,
                    fit_version TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            now = self.utc_now()
            current_version = str(self.rubric["version"])
            current_sha256 = str(self.rubric["source_sha256"])
            retired_job_filter = (
                "job_id IN (SELECT job_id FROM llm_jobs "
                "WHERE status IN ('queued', 'running', 'paused') "
                "AND NOT (rubric_version = ? AND rubric_sha256 = ?))"
            )
            connection.execute(
                "UPDATE llm_job_items SET status = 'cancelled', locked_at = NULL, "
                "error_code = 'rubric_retired', "
                "error_message = '任务使用的评分规则已更新，未完成样本已取消', "
                f"updated_at = ? WHERE status IN ('pending', 'running') AND {retired_job_filter}",
                (now, current_version, current_sha256),
            )
            connection.execute(
                "UPDATE llm_jobs SET status = 'cancelled', updated_at = ?, finished_at = ? "
                "WHERE status IN ('queued', 'running', 'paused') "
                "AND NOT (rubric_version = ? AND rubric_sha256 = ?)",
                (now, now, current_version, current_sha256),
            )
            connection.execute(
                "UPDATE llm_job_items SET status = 'pending', locked_at = NULL, "
                "run_id = NULL, updated_at = ? WHERE status = 'running' "
                "AND job_id IN (SELECT job_id FROM llm_jobs "
                "WHERE rubric_version = ? AND rubric_sha256 = ?)",
                (now, current_version, current_sha256),
            )
            connection.execute(
                "UPDATE llm_jobs SET status = 'queued', updated_at = ? "
                "WHERE status = 'running' AND rubric_version = ? AND rubric_sha256 = ?",
                (now, current_version, current_sha256),
            )

    async def start(self) -> None:
        if not self.config.enabled or self._workers:
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.request_timeout_seconds),
            limits=httpx.Limits(
                max_connections=self.config.max_concurrency,
                max_keepalive_connections=self.config.max_concurrency,
            ),
        )
        self._workers = [
            asyncio.create_task(self._worker_loop(index), name=f"llm-judge-{index}")
            for index in range(self.config.max_concurrency)
        ]
        self._wake.set()

    async def stop(self) -> None:
        workers, self._workers = self._workers, []
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def create_job(
        self,
        *,
        username: str,
        model: str,
        sequences: Iterable[int],
        scope: dict[str, Any],
        budget_limit_usd: float,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            raise ValueError(self.config.disabled_reason)
        if model not in self.available_models:
            raise ValueError("所选模型不在当前 Judge 兼容模型目录中")
        gateway = judge_model_gateway(model)
        if gateway == "openrouter" and not self.config.api_key:
            raise ValueError("服务器尚未配置 OPENROUTER_API_KEY")
        if gateway == "tatu" and not self.config.tatu_api_key:
            raise ValueError("服务器尚未配置 TATUCLOUD_API_KEY")
        if budget_limit_usd <= 0 or budget_limit_usd > self.config.max_job_budget_usd:
            raise ValueError(
                f"预算必须大于 0 且不超过 ${self.config.max_job_budget_usd:g}"
            )

        ordered_sequences = list(dict.fromkeys(int(seq) for seq in sequences))
        if not ordered_sequences:
            raise ValueError("批量任务至少需要一个样本")
        if len(ordered_sequences) > self.config.max_job_samples:
            raise ValueError(f"单任务最多允许 {self.config.max_job_samples} 个样本")

        loaded: list[tuple[int, str]] = []
        for seq in ordered_sequences:
            evaluation_id, _ = self.sample_loader(seq)
            loaded.append((seq, evaluation_id))

        job_id = f"job_{secrets.token_urlsafe(18)}"
        now = self.utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO llm_jobs(
                    job_id, username, model, rubric_version, rubric_sha256,
                    prompt_version, schema_version, scope_json, budget_limit_usd,
                    status, total_count, actual_cost_usd, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, 0, ?, ?)
                """,
                (
                    job_id,
                    username,
                    model,
                    self.rubric["version"],
                    self.rubric["source_sha256"],
                    PROMPT_VERSION,
                    JUDGMENT_SCHEMA_VERSION,
                    canonical_json(scope),
                    budget_limit_usd,
                    len(loaded),
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO llm_job_items(
                    job_id, position, seq, evaluation_id, status,
                    attempt_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)
                """,
                [
                    (job_id, position, seq, evaluation_id, now, now)
                    for position, (seq, evaluation_id) in enumerate(loaded, start=1)
                ],
            )
        self._wake.set()
        return self.get_job(username, job_id)

    def create_comparison(
        self,
        *,
        username: str,
        models: Iterable[str],
        sequences: Iterable[int],
        scope: dict[str, Any],
        budget_per_model_usd: float,
    ) -> dict[str, Any]:
        if not self.config.enabled:
            raise ValueError(self.config.disabled_reason)
        ordered_models = list(dict.fromkeys(str(model).strip() for model in models))
        if not MIN_COMPARISON_MODELS <= len(ordered_models) <= MAX_COMPARISON_MODELS:
            raise ValueError(
                f"一次比较必须选择 {MIN_COMPARISON_MODELS}–{MAX_COMPARISON_MODELS} 个不同模型"
            )
        unavailable = [
            model for model in ordered_models if model not in self.available_models
        ]
        if unavailable:
            raise ValueError(f"模型不在当前兼容目录中：{'、'.join(unavailable)}")
        if (
            budget_per_model_usd <= 0
            or budget_per_model_usd > self.config.max_job_budget_usd
        ):
            raise ValueError(
                f"每模型预算必须大于 0 且不超过 ${self.config.max_job_budget_usd:g}"
            )

        ordered_sequences = list(dict.fromkeys(int(seq) for seq in sequences))
        if not ordered_sequences:
            raise ValueError("比较任务至少需要一个样本")
        if len(ordered_sequences) > MAX_COMPARISON_SAMPLES:
            raise ValueError(f"一次比较最多允许 {MAX_COMPARISON_SAMPLES} 个样本")
        for seq in ordered_sequences:
            self.sample_loader(seq)

        comparison_id = f"cmp_{secrets.token_urlsafe(18)}"
        now = self.utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO llm_comparisons(
                    comparison_id, username, rubric_version, rubric_sha256,
                    prompt_version, schema_version, scope_json, models_json,
                    budget_per_model_usd, total_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    comparison_id,
                    username,
                    self.rubric["version"],
                    self.rubric["source_sha256"],
                    PROMPT_VERSION,
                    JUDGMENT_SCHEMA_VERSION,
                    canonical_json(scope),
                    canonical_json(ordered_models),
                    budget_per_model_usd,
                    len(ordered_sequences),
                    now,
                    now,
                ),
            )

        created_job_ids: list[str] = []
        try:
            for position, model in enumerate(ordered_models, start=1):
                job = self.create_job(
                    username=username,
                    model=model,
                    sequences=ordered_sequences,
                    scope={**scope, "comparison_id": comparison_id},
                    budget_limit_usd=budget_per_model_usd,
                )
                created_job_ids.append(str(job["job_id"]))
                with self._connection() as connection:
                    connection.execute(
                        """
                        INSERT INTO llm_comparison_jobs(
                            comparison_id, model_position, model, job_id
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (comparison_id, position, model, job["job_id"]),
                    )
        except Exception:
            with self._connection() as connection:
                connection.execute(
                    "DELETE FROM llm_comparisons WHERE comparison_id = ?",
                    (comparison_id,),
                )
                for job_id in created_job_ids:
                    connection.execute(
                        "DELETE FROM llm_job_items WHERE job_id = ?", (job_id,)
                    )
                    connection.execute(
                        "DELETE FROM llm_jobs WHERE job_id = ?", (job_id,)
                    )
            raise
        return self.get_comparison(username, comparison_id)

    @staticmethod
    def _comparison_status(jobs: list[dict[str, Any]]) -> str:
        statuses = {str(job["status"]) for job in jobs}
        if "running" in statuses:
            return "running"
        if "queued" in statuses:
            return "queued"
        if "paused" in statuses:
            return "paused"
        if statuses and statuses <= {"completed"}:
            return "completed"
        if "budget_exhausted" in statuses:
            return "budget_exhausted"
        if "cancelled" in statuses and statuses <= {"completed", "cancelled"}:
            return "cancelled"
        return "completed_with_errors"

    def _comparison_row(
        self, username: str, comparison_id: str
    ) -> tuple[sqlite3.Row, list[sqlite3.Row]]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM llm_comparisons WHERE comparison_id = ? AND username = ?",
                (comparison_id, username),
            ).fetchone()
            if row is None:
                raise LookupError("多模型比较任务不存在")
            job_rows = connection.execute(
                """
                SELECT model_position, model, job_id
                FROM llm_comparison_jobs WHERE comparison_id = ?
                ORDER BY model_position
                """,
                (comparison_id,),
            ).fetchall()
        return row, job_rows

    def _comparison_item_rows(
        self, comparison_id: str, *, seq: int | None = None
    ) -> list[sqlite3.Row]:
        where = "AND item.seq = ?" if seq is not None else ""
        parameters: tuple[Any, ...] = (
            (comparison_id, int(seq)) if seq is not None else (comparison_id,)
        )
        with self._connection() as connection:
            return connection.execute(
                f"""
                SELECT mapping.model_position, mapping.model, mapping.job_id,
                       item.position AS sample_position, item.seq, item.status,
                       item.run_id, item.error_code, item.error_message,
                       judgment.model_resolved, judgment.provider,
                       judgment.gates_json, judgment.scores_json,
                       judgment.prompt_tokens, judgment.completion_tokens,
                       judgment.reasoning_tokens, judgment.cost_usd,
                       judgment.latency_ms
                FROM llm_comparison_jobs mapping
                JOIN llm_job_items item ON item.job_id = mapping.job_id
                LEFT JOIN llm_judgments judgment ON judgment.run_id = item.run_id
                WHERE mapping.comparison_id = ? {where}
                ORDER BY item.position, mapping.model_position
                """,
                parameters,
            ).fetchall()

    @staticmethod
    def _record_value(record: dict[str, Any] | None) -> Any:
        return record.get("value") if isinstance(record, dict) else None

    def _comparison_metrics(
        self,
        models: list[str],
        rows: list[sqlite3.Row],
        *,
        rubric_version: str | None = None,
        rubric_sha256: str | None = None,
    ) -> dict[str, Any]:
        model_count = len(models)
        sample_records: dict[int, dict[str, Any]] = {}
        for row in rows:
            seq = int(row["seq"])
            sample = sample_records.setdefault(
                seq,
                {
                    "seq": seq,
                    "position": int(row["sample_position"]),
                    "models": {},
                },
            )
            raw_scores = (
                json.loads(str(row["scores_json"])) if row["scores_json"] else None
            )
            sample["models"][str(row["model"])] = {
                "status": str(row["status"]),
                "gates": json.loads(str(row["gates_json"]))
                if row["gates_json"]
                else None,
                "scores": scores_for_current_rubric(
                    raw_scores,
                    rubric_version or str(self.rubric["version"]),
                ),
            }

        current_rubric_version = str(self.rubric["version"])
        current_rubric_sha256 = str(self.rubric["source_sha256"])
        source_rubric_version = rubric_version or current_rubric_version
        source_rubric_sha256 = (
            current_rubric_sha256 if rubric_version is None else rubric_sha256
        )
        rubric_metadata_is_current = (
            source_rubric_version == current_rubric_version
            and source_rubric_sha256 == current_rubric_sha256
        )
        item_metadata = [
            {
                "dimension": str(dimension["name"]),
                "id": str(item["id"]),
                "name": str(item["name"]),
                "scale": [str(value) for value in item["scale"]],
                "allows_na": bool(item.get("allows_na")),
            }
            for dimension in self.rubric["dimensions"]
            for item in dimension["items"]
        ]

        def rate(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator, 6) if denominator else None

        def score_value(value: Any) -> Any:
            """Canonicalize numeric score representations while preserving N/A."""
            if value is None:
                return None
            if value == "N/A":
                return "N/A"
            if isinstance(value, bool):
                return value
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                return value
            return str(numeric) if str(value).strip() == str(numeric) else value

        def numeric_score(value: Any) -> int | None:
            if value is None or value == "N/A" or isinstance(value, bool):
                return None
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                return None
            return numeric if str(value).strip() == str(numeric) else None

        def allowed_score_value(value: Any, allowed_values: set[str]) -> Any:
            normalized = score_value(value)
            if not isinstance(normalized, str):
                return None
            return normalized if normalized in allowed_values else None

        def allowed_gate_value(value: Any) -> bool | None:
            # Current Gate items are boolean error triggers. `type(value) is bool`
            # deliberately rejects 0/1, which compare equal to False/True in
            # Python but are not valid Gate judgments.
            return value if type(value) is bool else None

        def allowed_logprob_confidence(value: Any) -> float | None:
            try:
                probability = float(value)
            except (TypeError, ValueError):
                return None
            return probability if 0.0 <= probability <= 1.0 else None

        def new_metric_counts() -> dict[str, int]:
            return {
                "comparable": 0,
                "agreed": 0,
                "possible_units": 0,
                "full_units": 0,
                "partial_units": 0,
                "insufficient_units": 0,
                "possible_ratings": 0,
                "available_ratings": 0,
                "na_ratings": 0,
                "full_coverage_agreed": 0,
                "full_coverage_within_one": 0,
                "pairwise_possible": 0,
                "pairwise_comparable": 0,
                "pairwise_exact": 0,
                "pairwise_adjacent": 0,
                "pairwise_severe": 0,
                "pairwise_numeric_far": 0,
                "pairwise_na_mismatch": 0,
                "logprob_confidence_ratings": 0,
                "logprob_distance_comparable": 0,
                "logprob_distance_raw_sum": 0.0,
                "logprob_distance_adjusted_sum": 0.0,
            }

        def observe_metric(
            counts: dict[str, int],
            values: list[Any],
            *,
            ordinal: bool,
            logprob_confidences: list[Any] | None = None,
            max_numeric_distance: float = 1.0,
        ) -> None:
            normalized = [score_value(value) if ordinal else value for value in values]
            normalized_logprob_confidences = (
                [allowed_logprob_confidence(value) for value in logprob_confidences]
                if ordinal and logprob_confidences is not None
                else [None] * len(normalized)
            )
            available = [value for value in normalized if value is not None]
            available_count = len(available)
            counts["possible_units"] += 1
            counts["possible_ratings"] += model_count
            counts["available_ratings"] += available_count
            counts["na_ratings"] += sum(value == "N/A" for value in available)

            if ordinal:
                for value, logprob_confidence in zip(
                    normalized,
                    normalized_logprob_confidences,
                ):
                    if value is None:
                        continue
                    if logprob_confidence is not None:
                        counts["logprob_confidence_ratings"] += 1

            if available_count == model_count:
                counts["full_units"] += 1
                if len(set(available)) == 1:
                    counts["full_coverage_agreed"] += 1
                    if ordinal:
                        counts["full_coverage_within_one"] += 1
                elif ordinal and "N/A" not in available:
                    numeric_values = [numeric_score(value) for value in available]
                    if all(value is not None for value in numeric_values):
                        numeric_range = [
                            value for value in numeric_values if value is not None
                        ]
                        if max(numeric_range) - min(numeric_range) <= 1:
                            counts["full_coverage_within_one"] += 1
            elif available_count >= 2:
                counts["partial_units"] += 1
            else:
                counts["insufficient_units"] += 1

            if available_count >= 2:
                counts["comparable"] += 1
                if len(set(available)) == 1:
                    counts["agreed"] += 1

            counts["pairwise_possible"] += model_count * (model_count - 1) // 2
            for left_index, left_value in enumerate(normalized):
                for right_index in range(left_index + 1, len(normalized)):
                    right_value = normalized[right_index]
                    if left_value is None or right_value is None:
                        continue
                    counts["pairwise_comparable"] += 1
                    if (
                        ordinal
                        and normalized_logprob_confidences[left_index] is not None
                        and normalized_logprob_confidences[right_index] is not None
                    ):
                        if left_value == right_value:
                            logprob_raw_distance = 0.0
                        elif left_value == "N/A" or right_value == "N/A":
                            logprob_raw_distance = 1.0
                        else:
                            left_numeric = numeric_score(left_value)
                            right_numeric = numeric_score(right_value)
                            if left_numeric is None or right_numeric is None:
                                logprob_raw_distance = 1.0
                            else:
                                logprob_raw_distance = min(
                                    1.0,
                                    abs(left_numeric - right_numeric)
                                    / max(1.0, max_numeric_distance),
                                )
                        logprob_factor = (
                            normalized_logprob_confidences[left_index]
                            * normalized_logprob_confidences[right_index]
                        ) ** 0.5
                        counts["logprob_distance_comparable"] += 1
                        counts["logprob_distance_raw_sum"] += logprob_raw_distance
                        counts["logprob_distance_adjusted_sum"] += (
                            logprob_raw_distance * logprob_factor
                        )
                    if left_value == right_value:
                        counts["pairwise_exact"] += 1
                        if ordinal:
                            # Exact N/A is also within one category, so adjacent is
                            # a useful superset of exact agreement.
                            counts["pairwise_adjacent"] += 1
                        continue
                    if not ordinal:
                        continue
                    if left_value == "N/A" or right_value == "N/A":
                        counts["pairwise_na_mismatch"] += 1
                        counts["pairwise_severe"] += 1
                        continue
                    left_numeric = numeric_score(left_value)
                    right_numeric = numeric_score(right_value)
                    if left_numeric is None or right_numeric is None:
                        # Valid judgments cannot reach this branch. Treating an
                        # unknown unequal category as severe avoids presenting it
                        # as adjacent agreement if historical data is malformed.
                        counts["pairwise_severe"] += 1
                        continue
                    if abs(left_numeric - right_numeric) <= 1:
                        counts["pairwise_adjacent"] += 1
                    else:
                        counts["pairwise_numeric_far"] += 1
                        counts["pairwise_severe"] += 1

        def metric_payload(counts: dict[str, int], *, ordinal: bool) -> dict[str, Any]:
            comparable = counts["comparable"]
            agreed = counts["agreed"]
            full_comparable = counts["full_units"]
            full_agreed = counts["full_coverage_agreed"]
            full_within_one = counts["full_coverage_within_one"]
            available_ratings = counts["available_ratings"]
            possible_ratings = counts["possible_ratings"]
            pairwise_comparable = counts["pairwise_comparable"]
            pairwise: dict[str, Any] = {
                "possible": counts["pairwise_possible"],
                "comparable": pairwise_comparable,
                "exact": counts["pairwise_exact"],
                "coverage_rate": rate(pairwise_comparable, counts["pairwise_possible"]),
                "exact_rate": rate(counts["pairwise_exact"], pairwise_comparable),
            }
            if ordinal:
                pairwise.update(
                    {
                        "adjacent": counts["pairwise_adjacent"],
                        "severe": counts["pairwise_severe"],
                        "numeric_far": counts["pairwise_numeric_far"],
                        "na_mismatch": counts["pairwise_na_mismatch"],
                        "adjacent_rate": rate(
                            counts["pairwise_adjacent"], pairwise_comparable
                        ),
                        "severe_rate": rate(
                            counts["pairwise_severe"], pairwise_comparable
                        ),
                    }
                )
            payload: dict[str, Any] = {
                # Legacy available-case fields: a unit is comparable when at
                # least two models produced a value, and agreement is evaluated
                # only among those available values.
                "comparable": comparable,
                "agreed": agreed,
                "disagreed": comparable - agreed,
                "agreement_rate": rate(agreed, comparable),
                # Strict exact agreement only uses units where every requested
                # model supplied a value. N/A is a real value in both views.
                "full_coverage_comparable": full_comparable,
                "full_coverage_agreed": full_agreed,
                "full_coverage_disagreed": full_comparable - full_agreed,
                "full_coverage_agreement_rate": rate(full_agreed, full_comparable),
                "coverage": {
                    "possible_ratings": possible_ratings,
                    "available_ratings": available_ratings,
                    "missing_ratings": possible_ratings - available_ratings,
                    "rate": rate(available_ratings, possible_ratings),
                    "full_units": counts["full_units"],
                    "partial_units": counts["partial_units"],
                    "insufficient_units": counts["insufficient_units"],
                    "na_ratings": counts["na_ratings"],
                    "na_rate": rate(counts["na_ratings"], available_ratings),
                },
                "pairwise": pairwise,
            }
            if ordinal:
                logprob_distance_comparable = counts["logprob_distance_comparable"]
                logprob_raw_distance_mean = (
                    counts["logprob_distance_raw_sum"] / logprob_distance_comparable
                    if logprob_distance_comparable
                    else None
                )
                logprob_adjusted_distance_mean = (
                    counts["logprob_distance_adjusted_sum"]
                    / logprob_distance_comparable
                    if logprob_distance_comparable
                    else None
                )
                payload.update(
                    {
                        "full_coverage_within_one": full_within_one,
                        "full_coverage_within_one_rate": rate(
                            full_within_one, full_comparable
                        ),
                        "confidence": {
                            "logprob": {
                                "recorded_ratings": counts[
                                    "logprob_confidence_ratings"
                                ],
                                "coverage_rate": rate(
                                    counts["logprob_confidence_ratings"],
                                    available_ratings,
                                ),
                                "distance": {
                                    "unit": "normalized_0_to_1",
                                    "comparable": logprob_distance_comparable,
                                    "raw_mean": (
                                        round(logprob_raw_distance_mean, 6)
                                        if logprob_raw_distance_mean is not None
                                        else None
                                    ),
                                    "adjusted_mean": (
                                        round(
                                            logprob_adjusted_distance_mean,
                                            6,
                                        )
                                        if logprob_adjusted_distance_mean is not None
                                        else None
                                    ),
                                    "reduction": (
                                        round(
                                            logprob_raw_distance_mean
                                            - logprob_adjusted_distance_mean,
                                            6,
                                        )
                                        if logprob_raw_distance_mean is not None
                                        and logprob_adjusted_distance_mean is not None
                                        else None
                                    ),
                                },
                            },
                        },
                    }
                )
            return payload

        atomic_totals = new_metric_counts()
        dimension_totals = {
            str(dimension["name"]): new_metric_counts()
            for dimension in self.rubric["dimensions"]
        }
        atomic_item_totals = {
            str(item["id"]): new_metric_counts() for item in item_metadata
        }
        gate_totals = {gate_id: new_metric_counts() for gate_id in self.gate_ids}
        all_gate_totals = new_metric_counts()

        def new_pair_counts(*, ordinal: bool) -> dict[str, int]:
            counts = {"possible": 0, "comparable": 0, "agreed": 0, "exact": 0}
            if ordinal:
                counts.update(
                    {"adjacent": 0, "severe": 0, "numeric_far": 0, "na_mismatch": 0}
                )
            return counts

        atomic_pairwise: dict[tuple[str, str], dict[str, int]] = {}
        gate_pairwise: dict[tuple[str, str], dict[str, int]] = {}
        for left_index, left in enumerate(models):
            for right in models[left_index + 1 :]:
                atomic_pairwise[(left, right)] = new_pair_counts(ordinal=True)
                gate_pairwise[(left, right)] = new_pair_counts(ordinal=False)

        def observe_model_pairs(
            counts_by_pair: dict[tuple[str, str], dict[str, int]],
            values: list[Any],
            *,
            ordinal: bool,
        ) -> None:
            normalized = [score_value(value) if ordinal else value for value in values]
            for left_index, left in enumerate(models):
                left_value = normalized[left_index]
                for right_index, right in enumerate(
                    models[left_index + 1 :], left_index + 1
                ):
                    right_value = normalized[right_index]
                    counts = counts_by_pair[(left, right)]
                    counts["possible"] += 1
                    if left_value is None or right_value is None:
                        continue
                    counts["comparable"] += 1
                    if left_value == right_value:
                        counts["agreed"] += 1
                        counts["exact"] += 1
                        if ordinal:
                            counts["adjacent"] += 1
                        continue
                    if not ordinal:
                        continue
                    if left_value == "N/A" or right_value == "N/A":
                        counts["na_mismatch"] += 1
                        counts["severe"] += 1
                        continue
                    left_numeric = numeric_score(left_value)
                    right_numeric = numeric_score(right_value)
                    if left_numeric is None or right_numeric is None:
                        counts["severe"] += 1
                    elif abs(left_numeric - right_numeric) <= 1:
                        counts["adjacent"] += 1
                    else:
                        counts["numeric_far"] += 1
                        counts["severe"] += 1

        sample_summaries: list[dict[str, Any]] = []
        total_model_sample_slots = 0
        total_atomic_participants = 0
        total_gate_participants = 0
        for sample in sorted(
            sample_records.values(), key=lambda value: value["position"]
        ):
            records = sample["models"]
            sample_atomic = new_metric_counts()
            sample_gates = new_metric_counts()
            for gate_id in self.gate_ids:
                values = [
                    allowed_gate_value(
                        self._record_value(
                            (records.get(model) or {}).get("gates", {}).get(gate_id)
                        )
                        if isinstance((records.get(model) or {}).get("gates"), dict)
                        else None
                    )
                    for model in models
                ]
                observe_metric(sample_gates, values, ordinal=False)
                observe_metric(all_gate_totals, values, ordinal=False)
                observe_metric(gate_totals[gate_id], values, ordinal=False)
                observe_model_pairs(gate_pairwise, values, ordinal=False)

            for metadata in item_metadata:
                dimension_name = str(metadata["dimension"])
                item_id = str(metadata["id"])
                if rubric_metadata_is_current:
                    allowed_values = set(metadata["scale"])
                    if metadata["allows_na"]:
                        allowed_values.add("N/A")
                else:
                    # The database does not contain a full historical rubric
                    # snapshot. Preserve ordinary historical 0–3/N/A scores
                    # instead of applying today's scale (for example, an item
                    # that used to allow N/A), while still excluding malformed
                    # values and schema control tokens such as SKIP.
                    allowed_values = {"0", "1", "2", "3", "N/A"}
                score_records = []
                for model in models:
                    model_scores = (records.get(model) or {}).get("scores")
                    score_records.append(
                        model_scores.get(item_id)
                        if isinstance(model_scores, dict)
                        else None
                    )
                values = [
                    allowed_score_value(
                        self._record_value(record),
                        allowed_values,
                    )
                    for record in score_records
                ]
                logprob_confidences = [
                    allowed_logprob_confidence(record.get("logprob_confidence"))
                    if isinstance(record, dict)
                    else None
                    for record in score_records
                ]
                numeric_scale = [
                    int(value)
                    for value in metadata["scale"]
                    if str(value).lstrip("-").isdigit()
                ]
                max_numeric_distance = (
                    max(numeric_scale) - min(numeric_scale)
                    if len(numeric_scale) >= 2
                    else 1
                )
                observe_metric(
                    sample_atomic,
                    values,
                    ordinal=True,
                    logprob_confidences=logprob_confidences,
                    max_numeric_distance=max_numeric_distance,
                )
                observe_metric(
                    atomic_totals,
                    values,
                    ordinal=True,
                    logprob_confidences=logprob_confidences,
                    max_numeric_distance=max_numeric_distance,
                )
                observe_metric(
                    dimension_totals[dimension_name],
                    values,
                    ordinal=True,
                    logprob_confidences=logprob_confidences,
                    max_numeric_distance=max_numeric_distance,
                )
                observe_metric(
                    atomic_item_totals[item_id],
                    values,
                    ordinal=True,
                    logprob_confidences=logprob_confidences,
                    max_numeric_distance=max_numeric_distance,
                )
                observe_model_pairs(atomic_pairwise, values, ordinal=True)

            ready_count = sum(
                1
                for model in models
                if (records.get(model) or {}).get("status") in FINAL_ITEM_STATUSES
            )
            atomic_participant_count = sum(
                isinstance((records.get(model) or {}).get("scores"), dict)
                for model in models
            )
            gate_participant_count = sum(
                isinstance((records.get(model) or {}).get("gates"), dict)
                for model in models
            )
            total_model_sample_slots += model_count
            total_atomic_participants += atomic_participant_count
            total_gate_participants += gate_participant_count
            sample_atomic_payload = metric_payload(sample_atomic, ordinal=True)
            sample_gate_payload = metric_payload(sample_gates, ordinal=False)
            sample_summaries.append(
                {
                    "position": sample["position"],
                    "seq": sample["seq"],
                    "ready_model_count": ready_count,
                    "model_count": model_count,
                    "atomic_participant_count": atomic_participant_count,
                    "gate_participant_count": gate_participant_count,
                    "atomic_comparable": sample_atomic_payload["comparable"],
                    "atomic_agreed": sample_atomic_payload["agreed"],
                    "atomic_disagreed": sample_atomic_payload["disagreed"],
                    "atomic_full_coverage_comparable": sample_atomic_payload[
                        "full_coverage_comparable"
                    ],
                    "atomic_full_coverage_agreed": sample_atomic_payload[
                        "full_coverage_agreed"
                    ],
                    "atomic_expected_ratings": sample_atomic_payload["coverage"][
                        "possible_ratings"
                    ],
                    "atomic_available_ratings": sample_atomic_payload["coverage"][
                        "available_ratings"
                    ],
                    "atomic_missing_ratings": sample_atomic_payload["coverage"][
                        "missing_ratings"
                    ],
                    "gate_comparable": sample_gate_payload["comparable"],
                    "gate_agreed": sample_gate_payload["agreed"],
                    "gate_disagreed": sample_gate_payload["disagreed"],
                    "gate_full_coverage_comparable": sample_gate_payload[
                        "full_coverage_comparable"
                    ],
                    "gate_full_coverage_agreed": sample_gate_payload[
                        "full_coverage_agreed"
                    ],
                    "gate_expected_ratings": sample_gate_payload["coverage"][
                        "possible_ratings"
                    ],
                    "gate_available_ratings": sample_gate_payload["coverage"][
                        "available_ratings"
                    ],
                    "gate_missing_ratings": sample_gate_payload["coverage"][
                        "missing_ratings"
                    ],
                }
            )

        def model_pair_rows(
            counts_by_pair: dict[tuple[str, str], dict[str, int]],
            *,
            ordinal: bool,
        ) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for (left, right), counts in counts_by_pair.items():
                comparable = counts["comparable"]
                row: dict[str, Any] = {
                    "left": left,
                    "right": right,
                    **counts,
                    "disagreed": comparable - counts["exact"],
                    "coverage_rate": rate(comparable, counts["possible"]),
                    "agreement_rate": rate(counts["exact"], comparable),
                    "exact_rate": rate(counts["exact"], comparable),
                }
                if ordinal:
                    row["adjacent_rate"] = rate(counts["adjacent"], comparable)
                    row["severe_rate"] = rate(counts["severe"], comparable)
                rows.append(row)
            return rows

        gate_names = {
            str(gate["id"]): str(gate["question"])
            for group in self.rubric["gates"]
            for gate in group["items"]
        }
        atomic_pairwise_rows = model_pair_rows(atomic_pairwise, ordinal=True)
        gate_pairwise_rows = model_pair_rows(gate_pairwise, ordinal=False)
        atomic_payload = metric_payload(atomic_totals, ordinal=True)
        atomic_payload.update(
            {
                "model_sample_slots": total_model_sample_slots,
                "participating_model_samples": total_atomic_participants,
                "excluded_model_samples": total_model_sample_slots
                - total_atomic_participants,
            }
        )
        gate_payload = metric_payload(all_gate_totals, ordinal=False)
        gate_payload.update(
            {
                "model_sample_slots": total_model_sample_slots,
                "participating_model_samples": total_gate_participants,
                "excluded_model_samples": total_model_sample_slots
                - total_gate_participants,
            }
        )

        return {
            "rubric_metadata": {
                "comparison_version": source_rubric_version,
                "comparison_sha256": source_rubric_sha256,
                "loaded_version": current_rubric_version,
                "loaded_sha256": current_rubric_sha256,
                "status": (
                    "current"
                    if rubric_metadata_is_current
                    else "historical_snapshot_unavailable"
                ),
            },
            "atomic": atomic_payload,
            "gates": gate_payload,
            "dimensions": [
                {
                    "name": name,
                    **metric_payload(counts, ordinal=True),
                }
                for name, counts in dimension_totals.items()
            ],
            "atomic_items": [
                {
                    **metadata,
                    **metric_payload(
                        atomic_item_totals[str(metadata["id"])], ordinal=True
                    ),
                }
                for metadata in item_metadata
            ],
            "gate_items": [
                {
                    "id": gate_id,
                    "name": gate_names.get(gate_id, gate_id),
                    **metric_payload(counts, ordinal=False),
                }
                for gate_id, counts in gate_totals.items()
            ],
            "atomic_pairwise": atomic_pairwise_rows,
            "gate_pairwise": gate_pairwise_rows,
            # Kept for older clients; this has always represented atomic scores only.
            "pairwise": atomic_pairwise_rows,
            "samples": sample_summaries,
        }

    def get_comparison(
        self,
        username: str,
        comparison_id: str,
        *,
        include_metrics: bool = True,
    ) -> dict[str, Any]:
        row, mapping_rows = self._comparison_row(username, comparison_id)
        jobs = [
            self.get_job(username, str(mapping["job_id"])) for mapping in mapping_rows
        ]
        models = [str(mapping["model"]) for mapping in mapping_rows]
        payload: dict[str, Any] = {
            "comparison_id": str(row["comparison_id"]),
            "rubric_version": str(row["rubric_version"]),
            "rubric_sha256": str(row["rubric_sha256"]),
            "prompt_version": str(row["prompt_version"]),
            "schema_version": str(row["schema_version"]),
            "scope": json.loads(str(row["scope_json"])),
            "models": models,
            "budget_per_model_usd": float(row["budget_per_model_usd"]),
            "total_budget_limit_usd": float(row["budget_per_model_usd"]) * len(models),
            "actual_cost_usd": sum(float(job["actual_cost_usd"]) for job in jobs),
            "total_count": int(row["total_count"]),
            "status": self._comparison_status(jobs),
            "jobs": jobs,
            "created_at": str(row["created_at"]),
            "updated_at": max(
                (str(job["updated_at"]) for job in jobs), default=str(row["updated_at"])
            ),
        }
        if include_metrics:
            payload["metrics"] = self._comparison_metrics(
                models,
                self._comparison_item_rows(comparison_id),
                rubric_version=str(row["rubric_version"]),
                rubric_sha256=str(row["rubric_sha256"]),
            )
        return payload

    def list_comparisons(
        self, username: str, *, limit: int = 30
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 100))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT comparison_id FROM llm_comparisons WHERE username = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (username, safe_limit),
            ).fetchall()
        return [
            self.get_comparison(
                username, str(row["comparison_id"]), include_metrics=False
            )
            for row in rows
        ]

    def get_comparison_sample(
        self, username: str, comparison_id: str, seq: int
    ) -> dict[str, Any]:
        comparison = self.get_comparison(username, comparison_id, include_metrics=False)
        rows = self._comparison_item_rows(comparison_id, seq=seq)
        if not rows:
            raise LookupError("比较任务中不存在该样本")
        by_model: dict[str, dict[str, Any]] = {}
        for row in rows:
            gates = json.loads(str(row["gates_json"])) if row["gates_json"] else None
            raw_scores = (
                json.loads(str(row["scores_json"])) if row["scores_json"] else None
            )
            scores = scores_for_current_rubric(
                raw_scores,
                str(comparison["rubric_version"]),
            )
            by_model[str(row["model"])] = {
                "model": str(row["model"]),
                "job_id": str(row["job_id"]),
                "status": str(row["status"]),
                "run_id": row["run_id"],
                "model_resolved": row["model_resolved"],
                "provider": row["provider"],
                "gates": gates,
                "scores": scores,
                "error_code": row["error_code"],
                "error_message": row["error_message"],
                "usage": {
                    "prompt_tokens": _as_int(row["prompt_tokens"]),
                    "completion_tokens": _as_int(row["completion_tokens"]),
                    "reasoning_tokens": _as_int(row["reasoning_tokens"]),
                    "cost_usd": _as_float(row["cost_usd"]),
                    "latency_ms": _as_int(row["latency_ms"]),
                },
            }

        models = comparison["models"]

        def criterion_row(
            criterion_id: str,
            *,
            name: str,
            dimension: str,
            record_key: str,
            scale: list[str] | None = None,
            allows_na: bool = False,
        ) -> dict[str, Any]:
            cells: dict[str, Any] = {}
            values: list[Any] = []
            for model in models:
                records = by_model.get(model, {}).get(record_key)
                record = (
                    records.get(criterion_id) if isinstance(records, dict) else None
                )
                cells[model] = record
                values.append(self._record_value(record))
            available_values = [value for value in values if value is not None]
            participant_count = len(available_values)
            excluded_model_count = len(values) - participant_count
            comparable = participant_count >= 2
            agreement = comparable and len(set(available_values)) == 1
            return {
                "id": criterion_id,
                "name": name,
                "dimension": dimension,
                "comparable": comparable,
                "agreement": agreement,
                "has_difference": comparable and not agreement,
                "participant_count": participant_count,
                "excluded_model_count": excluded_model_count,
                "models": cells,
                "scale": scale,
                "allows_na": allows_na,
            }

        gate_names = {
            str(gate["id"]): str(gate["question"])
            for group in self.rubric["gates"]
            for gate in group["items"]
        }
        gates = [
            criterion_row(
                gate_id,
                name=gate_names[gate_id],
                dimension="Gate",
                record_key="gates",
            )
            for gate_id in self.gate_ids
        ]
        scores = [
            criterion_row(
                str(item["id"]),
                name=str(item["name"]),
                dimension=str(dimension["name"]),
                record_key="scores",
                scale=[str(value) for value in item["scale"]],
                allows_na=bool(item.get("allows_na")),
            )
            for dimension in self.rubric["dimensions"]
            for item in dimension["items"]
        ]
        _, sample = self.sample_loader(int(seq))
        return {
            "comparison_id": comparison_id,
            "seq": int(seq),
            "models": models,
            "model_results": [by_model[model] for model in models],
            "gates": gates,
            "scores": scores,
            "sample": sample,
        }

    def control_comparison(
        self, username: str, comparison_id: str, action: str
    ) -> dict[str, Any]:
        comparison = self.get_comparison(username, comparison_id, include_metrics=False)
        for job in comparison["jobs"]:
            status = str(job["status"])
            if action == "pause" and status in {"queued", "running"}:
                self.pause_job(username, str(job["job_id"]))
            elif action == "resume" and status == "paused":
                self.resume_job(username, str(job["job_id"]))
            elif action == "cancel" and status in ACTIVE_JOB_STATUSES:
                self.cancel_job(username, str(job["job_id"]))
        return self.get_comparison(username, comparison_id)

    def list_jobs(
        self,
        username: str,
        limit: int = 30,
        *,
        scope_type: str | None = None,
        exclude_scope_type: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 100))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT job_id, scope_json FROM llm_jobs WHERE username = ? "
                "ORDER BY created_at DESC",
                (username,),
            ).fetchall()
        if scope_type is not None:
            rows = [
                row
                for row in rows
                if json.loads(str(row["scope_json"])).get("type") == scope_type
            ]
        if exclude_scope_type is not None:
            rows = [
                row
                for row in rows
                if json.loads(str(row["scope_json"])).get("type") != exclude_scope_type
            ]
        return [self.get_job(username, str(row["job_id"])) for row in rows[:safe_limit]]

    def get_job(self, username: str, job_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM llm_jobs WHERE job_id = ? AND username = ?",
                (job_id, username),
            ).fetchone()
            if row is None:
                raise LookupError("LLM Judge 任务不存在")
            counts = {
                str(item["status"]): int(item["count"])
                for item in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM llm_job_items "
                    "WHERE job_id = ? GROUP BY status",
                    (job_id,),
                ).fetchall()
            }
        completed = sum(counts.get(status, 0) for status in FINAL_ITEM_STATUSES)
        return {
            "job_id": str(row["job_id"]),
            "model": str(row["model"]),
            "rubric_version": str(row["rubric_version"]),
            "rubric_sha256": str(row["rubric_sha256"]),
            "prompt_version": str(row["prompt_version"]),
            "schema_version": str(row["schema_version"]),
            "scope": json.loads(str(row["scope_json"])),
            "budget_limit_usd": float(row["budget_limit_usd"]),
            "actual_cost_usd": float(row["actual_cost_usd"]),
            "status": str(row["status"]),
            "total_count": int(row["total_count"]),
            "finished_count": completed,
            "counts": {status: counts.get(status, 0) for status in ITEM_STATUSES},
            "created_at": str(row["created_at"]),
            "started_at": row["started_at"],
            "updated_at": str(row["updated_at"]),
            "finished_at": row["finished_at"],
        }

    def list_items(
        self,
        username: str,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        self.get_job(username, job_id)
        safe_offset = max(0, int(offset))
        safe_limit = max(1, min(int(limit), 500))
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT position, seq, status, attempt_count, run_id,
                       error_code, error_message, updated_at
                FROM llm_job_items WHERE job_id = ?
                ORDER BY position LIMIT ? OFFSET ?
                """,
                (job_id, safe_limit, safe_offset),
            ).fetchall()
        return {
            "job_id": job_id,
            "offset": safe_offset,
            "limit": safe_limit,
            "items": [
                {
                    "position": int(row["position"]),
                    "seq": int(row["seq"]),
                    "status": str(row["status"]),
                    "attempt_count": int(row["attempt_count"]),
                    "run_id": row["run_id"],
                    "error_code": row["error_code"],
                    "error_message": row["error_message"],
                    "updated_at": str(row["updated_at"]),
                }
                for row in rows
            ],
        }

    def get_judgment(self, username: str, job_id: str, run_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT j.*, job.rubric_version AS source_rubric_version
                FROM llm_judgments j
                JOIN llm_jobs job ON job.job_id = j.job_id
                WHERE j.run_id = ? AND j.job_id = ? AND job.username = ?
                """,
                (run_id, job_id, username),
            ).fetchone()
        if row is None:
            raise LookupError("LLM Judge 结果不存在")
        gates = json.loads(str(row["gates_json"])) if row["gates_json"] else None
        raw_scores = json.loads(str(row["scores_json"])) if row["scores_json"] else None
        scores = scores_for_current_rubric(
            raw_scores,
            str(row["source_rubric_version"]),
        )
        failed_gates = gate_failure_ids(gates) if gates is not None else []
        raw_output = _public_model_outputs(row["raw_response_json"])
        error_message = row["error_message"]
        if error_message == "模型结构化输出不是字符串":
            exhausted = next(
                (
                    output
                    for output in raw_output.values()
                    if output.get("finish_reason") == "length"
                    and output.get("content") is None
                ),
                None,
            )
            if exhausted is not None:
                error_message = (
                    "模型达到输出长度上限但没有生成最终 JSON"
                    "（finish_reason=length，"
                    f"reasoning_tokens={exhausted['reasoning_tokens']}）"
                )
        return {
            "run_id": str(row["run_id"]),
            "job_id": str(row["job_id"]),
            "seq": int(row["seq"]),
            "attempt_number": int(row["attempt_number"]),
            "request_hash": row["request_hash"],
            "model_requested": str(row["model_requested"]),
            "model_resolved": row["model_resolved"],
            "provider": row["provider"],
            "status": str(row["status"]),
            "rubric_version": str(row["source_rubric_version"]),
            "display_rubric_version": str(self.rubric["version"]),
            "judgment": (
                {
                    "gate_decision": "discard" if failed_gates else "pass",
                    "failed_gate_ids": failed_gates,
                    "gates": gates,
                    "scores": scores,
                }
                if gates is not None
                else None
            ),
            "usage": {
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "reasoning_tokens": int(row["reasoning_tokens"]),
                "cached_tokens": int(row["cached_tokens"]),
                "cost_usd": float(row["cost_usd"]),
                "latency_ms": int(row["latency_ms"]),
            },
            "error_code": row["error_code"],
            "error_message": error_message,
            "raw_output": raw_output,
            "created_at": str(row["created_at"]),
        }

    def list_weight_fit_items(self, username: str, job_id: str) -> list[dict[str, Any]]:
        """Return compact, current-rubric job outputs for Battle weight fitting."""
        self.get_job(username, job_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT item.position, item.seq, item.status, item.run_id,
                       judgment.gates_json, judgment.scores_json,
                       job.rubric_version AS source_rubric_version
                FROM llm_job_items AS item
                JOIN llm_jobs AS job ON job.job_id = item.job_id
                LEFT JOIN llm_judgments AS judgment ON judgment.run_id = item.run_id
                WHERE item.job_id = ? AND job.username = ?
                ORDER BY item.position
                """,
                (job_id, username),
            ).fetchall()
        result = []
        for row in rows:
            raw_scores = (
                json.loads(str(row["scores_json"])) if row["scores_json"] else None
            )
            result.append(
                {
                    "position": int(row["position"]),
                    "seq": int(row["seq"]),
                    "status": str(row["status"]),
                    "run_id": row["run_id"],
                    "gates": (
                        json.loads(str(row["gates_json"]))
                        if row["gates_json"]
                        else None
                    ),
                    "scores": scores_for_current_rubric(
                        raw_scores,
                        str(row["source_rubric_version"]),
                    ),
                }
            )
        return result

    def get_battle_weight_fit(
        self, username: str, job_id: str
    ) -> dict[str, Any] | None:
        self.get_job(username, job_id)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT fit.report_json
                FROM llm_battle_weight_fits AS fit
                JOIN llm_jobs AS job ON job.job_id = fit.job_id
                WHERE fit.job_id = ? AND job.username = ?
                """,
                (job_id, username),
            ).fetchone()
        return json.loads(str(row["report_json"])) if row is not None else None

    def save_battle_weight_fit(
        self,
        username: str,
        job_id: str,
        *,
        fit_version: str,
        report: dict[str, Any],
    ) -> None:
        self.get_job(username, job_id)
        now = self.utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO llm_battle_weight_fits(
                    job_id, fit_version, report_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    fit_version = excluded.fit_version,
                    report_json = excluded.report_json,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    fit_version,
                    json.dumps(report, ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                ),
            )

    def pause_job(self, username: str, job_id: str) -> dict[str, Any]:
        return self._change_job_status(username, job_id, "pause")

    def resume_job(self, username: str, job_id: str) -> dict[str, Any]:
        result = self._change_job_status(username, job_id, "resume")
        self._wake.set()
        return result

    def cancel_job(self, username: str, job_id: str) -> dict[str, Any]:
        return self._change_job_status(username, job_id, "cancel")

    def _change_job_status(
        self, username: str, job_id: str, action: str
    ) -> dict[str, Any]:
        now = self.utc_now()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM llm_jobs WHERE job_id = ? AND username = ?",
                (job_id, username),
            ).fetchone()
            if row is None:
                raise LookupError("LLM Judge 任务不存在")
            current = str(row["status"])
            if action == "pause":
                if current not in {"queued", "running"}:
                    raise ValueError("只有等待中或运行中的任务可以暂停")
                connection.execute(
                    "UPDATE llm_jobs SET status = 'paused', updated_at = ? WHERE job_id = ?",
                    (now, job_id),
                )
            elif action == "resume":
                if current != "paused":
                    raise ValueError("只有已暂停任务可以继续")
                connection.execute(
                    "UPDATE llm_jobs SET status = 'queued', updated_at = ? WHERE job_id = ?",
                    (now, job_id),
                )
            elif action == "cancel":
                if current not in ACTIVE_JOB_STATUSES:
                    raise ValueError("当前任务不能取消")
                connection.execute(
                    "UPDATE llm_jobs SET status = 'cancelled', updated_at = ?, "
                    "finished_at = ? WHERE job_id = ?",
                    (now, now, job_id),
                )
                connection.execute(
                    "UPDATE llm_job_items SET status = 'cancelled', updated_at = ? "
                    "WHERE job_id = ? AND status = 'pending'",
                    (now, job_id),
                )
            else:
                raise ValueError("未知任务操作")
        return self.get_job(username, job_id)

    async def _worker_loop(self, _: int) -> None:
        while True:
            item = self._claim_item()
            if item is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=2.0)
                except TimeoutError:
                    pass
                continue
            await self._execute_item(item)

    def _claim_item(self) -> dict[str, Any] | None:
        while True:
            now = self.utc_now()
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT item.job_id, item.seq, item.evaluation_id,
                           item.attempt_count, job.model,
                           job.actual_cost_usd, job.budget_limit_usd
                    FROM llm_job_items item
                    JOIN llm_jobs job ON job.job_id = item.job_id
                    WHERE item.status = 'pending'
                      AND job.status IN ('queued', 'running')
                      AND job.rubric_version = ?
                      AND job.rubric_sha256 = ?
                    ORDER BY job.created_at, item.position
                    LIMIT 1
                    """,
                    (self.rubric["version"], self.rubric["source_sha256"]),
                ).fetchone()
                if row is None:
                    return None
                job_id = str(row["job_id"])
                if float(row["actual_cost_usd"]) >= float(row["budget_limit_usd"]):
                    connection.execute(
                        "UPDATE llm_jobs SET status = 'budget_exhausted', updated_at = ?, "
                        "finished_at = ? WHERE job_id = ?",
                        (now, now, job_id),
                    )
                    connection.execute(
                        "UPDATE llm_job_items SET status = 'cancelled', updated_at = ? "
                        "WHERE job_id = ? AND status = 'pending'",
                        (now, job_id),
                    )
                    continue
                attempt_number = int(row["attempt_count"]) + 1
                connection.execute(
                    """
                    UPDATE llm_job_items
                    SET status = 'running', attempt_count = ?, locked_at = ?,
                        error_code = NULL, error_message = NULL, updated_at = ?
                    WHERE job_id = ? AND seq = ? AND status = 'pending'
                    """,
                    (attempt_number, now, now, job_id, int(row["seq"])),
                )
                connection.execute(
                    """
                    UPDATE llm_jobs
                    SET status = 'running', started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (now, now, job_id),
                )
                return {
                    "job_id": job_id,
                    "seq": int(row["seq"]),
                    "evaluation_id": str(row["evaluation_id"]),
                    "attempt_number": attempt_number,
                    "model": str(row["model"]),
                }

    async def _post_phase(
        self, body: dict[str, Any], *, requested_model: str
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("LLM Judge HTTP 客户端尚未启动")
        gateway = judge_model_gateway(requested_model)
        api_key = self.config.tatu_api_key if gateway == "tatu" else self.config.api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if gateway == "openrouter":
            headers.update(
                {
                    "HTTP-Referer": "http://roleplay-annotation.local/llm-judge",
                    "X-Title": "Roleplay LLM Judge",
                }
            )
        if gateway == "tatu":
            return await self._post_tatu_stream(
                body,
                requested_model=requested_model,
                headers=headers,
            )
        response = await self._client.post(
            judge_request_url(self.config, requested_model),
            headers=headers,
            json=body,
            timeout=(
                self.config.tatu_request_timeout_seconds
                if gateway == "tatu"
                else self.config.request_timeout_seconds
            ),
        )
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {
                "http_status": response.status_code,
                "body": response.text[:4000],
            }
        if not isinstance(payload, dict):
            payload = {
                "http_status": response.status_code,
                "unexpected_payload": payload,
            }
        if response.status_code >= 400:
            detail = payload.get("error", payload)
            raise JudgePhaseHTTPError(
                code=f"{gateway}_http_{response.status_code}",
                message=self._redact(canonical_json(detail)),
                retryable=response.status_code == 429 or response.status_code >= 500,
                payload=payload,
                retry_after_seconds=_retry_after_seconds(
                    response.headers.get("Retry-After")
                ),
            )
        return payload

    async def _post_tatu_stream(
        self,
        body: dict[str, Any],
        *,
        requested_model: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("LLM Judge HTTP 客户端尚未启动")
        streamed_body = dict(body)
        streamed_body["stream"] = True
        response_model: str | None = None
        response_provider: str | None = provider_label_for_model(requested_model)
        response_id: str | None = None
        finish_reason: Any = None
        native_finish_reason: Any = None
        refusal: Any = None
        usage: dict[str, Any] = {}
        content_parts: list[str] = []
        received_event = False

        async with self._client.stream(
            "POST",
            judge_request_url(self.config, requested_model),
            headers=headers,
            json=streamed_body,
            timeout=self.config.tatu_request_timeout_seconds,
        ) as response:
            if response.status_code >= 400:
                raw_body = await response.aread()
                try:
                    payload = json.loads(raw_body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = {
                        "http_status": response.status_code,
                        "body": raw_body.decode("utf-8", errors="replace")[:4000],
                    }
                if not isinstance(payload, dict):
                    payload = {
                        "http_status": response.status_code,
                        "unexpected_payload": payload,
                    }
                detail = payload.get("error", payload)
                raise JudgePhaseHTTPError(
                    code=f"tatu_http_{response.status_code}",
                    message=self._redact(canonical_json(detail)),
                    retryable=(
                        response.status_code == 429 or response.status_code >= 500
                    ),
                    payload=payload,
                    retry_after_seconds=_retry_after_seconds(
                        response.headers.get("Retry-After")
                    ),
                )

            async for line in response.aiter_lines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("event:", ":")):
                    continue
                if stripped.startswith("data:"):
                    stripped = stripped[5:].strip()
                if stripped == "[DONE]":
                    break
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise JudgePhaseHTTPError(
                        code="tatu_stream_invalid",
                        message=(
                            "TatuCloud 流式响应包含无法解析的 JSON 事件："
                            f"第 {error.lineno} 行第 {error.colno} 列"
                        ),
                        retryable=True,
                        payload={"stream_line": stripped[:2000]},
                    ) from error
                if not isinstance(event, dict):
                    continue
                received_event = True
                response_id = str(event.get("id", "")) or response_id
                response_model = str(event.get("model", "")) or response_model
                response_provider = str(event.get("provider", "")) or response_provider
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage.update(event_usage)
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
                if choice.get("native_finish_reason") is not None:
                    native_finish_reason = choice.get("native_finish_reason")
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    delta = choice.get("message")
                if not isinstance(delta, dict):
                    continue
                if delta.get("refusal") is not None:
                    refusal = delta.get("refusal")
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
                elif isinstance(piece, list):
                    for block in piece:
                        if isinstance(block, dict) and isinstance(
                            block.get("text"), str
                        ):
                            content_parts.append(str(block["text"]))

        if not received_event:
            raise JudgePhaseHTTPError(
                code="tatu_stream_empty",
                message="TatuCloud 流式响应未返回任何 JSON 事件",
                retryable=True,
                payload={},
            )
        return {
            "id": response_id,
            "model": response_model or backend_model_id(requested_model),
            "provider": response_provider,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "".join(content_parts) if content_parts else None,
                        "refusal": refusal,
                    },
                    "finish_reason": finish_reason,
                    "native_finish_reason": native_finish_reason,
                }
            ],
            "usage": usage,
        }

    async def _execute_item(self, item: dict[str, Any]) -> None:
        run_id = f"run_{secrets.token_urlsafe(18)}"
        started = time.perf_counter()
        request_hash: str | None = None
        raw_payload: dict[str, Any] = {}
        raw_content_by_phase: dict[str, str] = {}
        judgment: dict[str, Any] | None = None
        status = "failed"
        error_code: str | None = None
        error_message: str | None = None
        retryable = False
        retry_after_seconds: float | None = None
        model_resolved: str | None = None
        provider: str | None = None
        usage: dict[str, Any] = {}
        score_logprob_confidence_count = 0
        requested_model = str(item["model"])
        gateway = judge_model_gateway(requested_model)
        provider = provider_label_for_model(requested_model)

        try:
            evaluation_id, sample = self.sample_loader(int(item["seq"]))
            if evaluation_id != item["evaluation_id"]:
                raise RuntimeError("样本 evaluation_id 与任务记录不一致")
            gate_body, sample_packet, gate_hash = build_gate_request(
                str(item["model"]), self.rubric, sample, self.config
            )
            try:
                raw_payload["gate"] = await self._post_phase(
                    gate_body, requested_model=requested_model
                )
            except JudgePhaseHTTPError as error:
                raw_payload["gate"] = error.payload
                raise
            gates, failed_gates, gate_content = parse_and_validate_gate_response(
                raw_payload["gate"],
                self.rubric,
                sample_packet,
                allow_json_fence=gateway == "tatu",
            )
            raw_content_by_phase["gate"] = gate_content
            judgment = {
                "gates": gates,
                "scores": None,
                "failed_gate_ids": failed_gates,
            }
            request_hashes = {"gate": gate_hash}
            if failed_gates:
                status = "discarded"
            else:
                score_body, _, score_hash = build_score_request(
                    str(item["model"]),
                    self.rubric,
                    sample,
                    self.config,
                    validated_gates=gates,
                )
                try:
                    raw_payload["dimensions"] = await self._post_phase(
                        score_body, requested_model=requested_model
                    )
                except JudgePhaseHTTPError as error:
                    raw_payload["dimensions"] = error.payload
                    raise
                scores, score_content = parse_and_validate_score_response(
                    raw_payload["dimensions"],
                    self.rubric,
                    sample_packet,
                    allow_json_fence=gateway == "tatu",
                )
                if gateway == "openrouter":
                    score_logprob_confidence_count = attach_score_logprob_confidence(
                        scores,
                        raw_payload["dimensions"],
                        score_content,
                        self.rubric,
                    )
                raw_content_by_phase["dimensions"] = score_content
                judgment["scores"] = scores
                request_hashes["dimensions"] = score_hash
                status = "completed"
            request_hash = hashlib.sha256(
                canonical_json(request_hashes).encode("utf-8")
            ).hexdigest()
            final_payload = (
                raw_payload.get("dimensions") or raw_payload.get("gate") or {}
            )
            model_resolved = str(final_payload.get("model", "")) or None
            provider = (
                str(final_payload.get("provider", ""))
                or provider_label_for_model(requested_model)
                or None
            )
        except JudgePhaseHTTPError as error:
            error_code = error.code
            error_message = str(error)
            retryable = error.retryable
            retry_after_seconds = error.retry_after_seconds
        except JudgeValidationError as error:
            status = "invalid"
            error_code = "invalid_judgment"
            error_message = self._redact(str(error))
            retryable = True
        except httpx.TransportError as error:
            error_code = f"{gateway}_network_error"
            error_message = self._redact(str(error) or type(error).__name__)
            retryable = True
        except Exception as error:  # keep a malformed sample from stopping the queue
            error_code = "judge_internal_error"
            error_message = self._redact(f"{type(error).__name__}: {error}")

        dimensions_payload = raw_payload.get("dimensions")
        if gateway == "openrouter" and isinstance(dimensions_payload, dict):
            compact_openrouter_logprobs(
                dimensions_payload,
                score_confidence_count=score_logprob_confidence_count,
            )

        final_payload = raw_payload.get("dimensions") or raw_payload.get("gate") or {}
        if isinstance(final_payload, dict):
            model_resolved = model_resolved or (
                str(final_payload.get("model", "")) or None
            )
            provider = provider or (
                str(final_payload.get("provider", ""))
                or provider_label_for_model(requested_model)
                or None
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        phase_usages = [
            payload.get("usage") or {}
            for payload in raw_payload.values()
            if isinstance(payload, dict)
        ]
        usage = {
            "prompt_tokens": sum(
                _as_int(value.get("prompt_tokens")) for value in phase_usages
            ),
            "completion_tokens": sum(
                _as_int(value.get("completion_tokens")) for value in phase_usages
            ),
            "reasoning_tokens": sum(
                _as_int(
                    value.get("reasoning_tokens")
                    or (value.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                )
                for value in phase_usages
            ),
            "cached_tokens": sum(
                _as_int(
                    value.get("cached_tokens")
                    or (value.get("prompt_tokens_details") or {}).get("cached_tokens")
                )
                for value in phase_usages
            ),
            "cost": sum(
                max(0.0, _as_float(value.get("cost"))) for value in phase_usages
            ),
        }
        max_attempts = (
            self.config.rate_limit_max_attempts
            if error_code is not None and error_code.endswith("_http_429")
            else MAX_ATTEMPTS
        )
        retryable = retryable and int(item["attempt_number"]) < max_attempts
        if retryable and error_code is not None and error_code.endswith("_http_429"):
            delay_seconds = _rate_limit_retry_delay(
                job_id=str(item["job_id"]),
                seq=int(item["seq"]),
                attempt_number=int(item["attempt_number"]),
                config=self.config,
                retry_after_seconds=retry_after_seconds,
            )
            await self._retry_sleep(delay_seconds)
        elif (
            retryable
            and error_code is not None
            and error_code.endswith("_network_error")
        ):
            delay_seconds = _rate_limit_retry_delay(
                job_id=str(item["job_id"]),
                seq=int(item["seq"]),
                attempt_number=int(item["attempt_number"]),
                config=self.config,
                retry_after_seconds=None,
            )
            await self._retry_sleep(delay_seconds)
        self._finish_attempt(
            item=item,
            run_id=run_id,
            request_hash=request_hash,
            model_resolved=model_resolved,
            provider=provider,
            status=status,
            judgment=judgment,
            raw_payload=raw_payload or None,
            raw_content=(
                canonical_json(raw_content_by_phase) if raw_content_by_phase else None
            ),
            usage=usage,
            latency_ms=latency_ms,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
        )
        self._wake.set()

    def _finish_attempt(
        self,
        *,
        item: dict[str, Any],
        run_id: str,
        request_hash: str | None,
        model_resolved: str | None,
        provider: str | None,
        status: str,
        judgment: dict[str, Any] | None,
        raw_payload: dict[str, Any] | None,
        raw_content: str | None,
        usage: dict[str, Any],
        latency_ms: int,
        error_code: str | None,
        error_message: str | None,
        retryable: bool,
    ) -> None:
        details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        prompt_tokens = _as_int(usage.get("prompt_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens"))
        reasoning_tokens = _as_int(
            usage.get("reasoning_tokens") or details.get("reasoning_tokens")
        )
        cached_tokens = _as_int(
            usage.get("cached_tokens") or prompt_details.get("cached_tokens")
        )
        cost = max(0.0, _as_float(usage.get("cost")))
        now = self.utc_now()
        gates_json = (
            canonical_json(judgment["gates"])
            if judgment is not None and judgment.get("gates") is not None
            else None
        )
        scores_json = (
            canonical_json(judgment["scores"])
            if judgment is not None and judgment.get("scores") is not None
            else None
        )
        raw_response_json = (
            canonical_json(raw_payload) if raw_payload is not None else None
        )

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO llm_judgments(
                    run_id, job_id, seq, evaluation_id, attempt_number,
                    request_hash, model_requested, model_resolved, provider,
                    rubric_version, rubric_sha256, prompt_version, schema_version,
                    status, gates_json, scores_json, raw_response_json, raw_content,
                    prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens,
                    cost_usd, latency_ms, error_code, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    item["job_id"],
                    item["seq"],
                    item["evaluation_id"],
                    item["attempt_number"],
                    request_hash,
                    item["model"],
                    model_resolved,
                    provider,
                    self.rubric["version"],
                    self.rubric["source_sha256"],
                    PROMPT_VERSION,
                    JUDGMENT_SCHEMA_VERSION,
                    status,
                    gates_json,
                    scores_json,
                    raw_response_json,
                    raw_content,
                    prompt_tokens,
                    completion_tokens,
                    reasoning_tokens,
                    cached_tokens,
                    cost,
                    latency_ms,
                    error_code,
                    error_message,
                    now,
                ),
            )
            final_item_status = "pending" if retryable else status
            connection.execute(
                """
                UPDATE llm_job_items
                SET status = ?, locked_at = NULL, run_id = ?, error_code = ?,
                    error_message = ?, updated_at = ?
                WHERE job_id = ? AND seq = ?
                """,
                (
                    final_item_status,
                    None if retryable else run_id,
                    error_code,
                    error_message,
                    now,
                    item["job_id"],
                    item["seq"],
                ),
            )
            connection.execute(
                "UPDATE llm_jobs SET actual_cost_usd = actual_cost_usd + ?, "
                "updated_at = ? WHERE job_id = ?",
                (cost, now, item["job_id"]),
            )
            self._finalize_job_if_needed(connection, str(item["job_id"]), now)

    def _finalize_job_if_needed(
        self, connection: sqlite3.Connection, job_id: str, now: str
    ) -> None:
        job = connection.execute(
            "SELECT status, actual_cost_usd, budget_limit_usd FROM llm_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if job is None:
            return
        status = str(job["status"])
        if status in {"cancelled", "budget_exhausted"}:
            return
        if float(job["actual_cost_usd"]) >= float(job["budget_limit_usd"]):
            connection.execute(
                "UPDATE llm_job_items SET status = 'cancelled', updated_at = ? "
                "WHERE job_id = ? AND status = 'pending'",
                (now, job_id),
            )
            connection.execute(
                "UPDATE llm_jobs SET status = 'budget_exhausted', updated_at = ?, "
                "finished_at = ? WHERE job_id = ?",
                (now, now, job_id),
            )
            return
        if status == "paused":
            return
        remaining = connection.execute(
            "SELECT COUNT(*) AS count FROM llm_job_items "
            "WHERE job_id = ? AND status IN ('pending', 'running')",
            (job_id,),
        ).fetchone()
        if int(remaining["count"]) > 0:
            return
        errors = connection.execute(
            "SELECT COUNT(*) AS count FROM llm_job_items "
            "WHERE job_id = ? AND status IN ('invalid', 'failed')",
            (job_id,),
        ).fetchone()
        final_status = "completed_with_errors" if int(errors["count"]) else "completed"
        connection.execute(
            "UPDATE llm_jobs SET status = ?, updated_at = ?, finished_at = ? "
            "WHERE job_id = ?",
            (final_status, now, now, job_id),
        )

    def _redact(self, message: str) -> str:
        cleaned = message
        for secret in (self.config.api_key, self.config.tatu_api_key):
            if secret:
                cleaned = cleaned.replace(secret, "[REDACTED]")
        return cleaned[:2000]
