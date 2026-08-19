from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx

from .candidate_clients import CandidateCompletion, chat_completion
from .config import PROJECT_ROOT, load_openrouter_key, load_secret_value
from .dataset import load_seeds, preflight, sha256_file, staged_cohorts
from .ledger import append_jsonl, utc_now, write_new_json
from .openrouter import CompletionRecord, OpenRouterClient
from .prompts import (
    all_prompt_hashes,
    bridge_messages,
    canonical_json,
    normalize_bridge_text,
    prompt_hash,
)
from .roleplay_prompts import build_deepseek_system, build_gemma_system, normalized_character_reference
from .workflow import blind_judge_pair, blind_judge_trajectories


T = TypeVar("T")
class PilotRunner:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        run_dir: Path,
        seeds: list[dict[str, Any]],
        concurrency: int,
        max_turns: int,
        target_depths: dict[str, int] | None = None,
        trajectory_checkpoints: dict[str, list[int]] | None = None,
        run_kind: str = "pilot20",
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.seeds = seeds
        self.concurrency = concurrency
        self.max_turns = max_turns
        self.target_depths = target_depths or {
            seed["sample_id"]: max_turns for seed in seeds
        }
        self.trajectory_checkpoints = trajectory_checkpoints or {
            seed["sample_id"]: [
                min(max_turns, len(seed["ordered_user_messages"]))
            ]
            for seed in seeds
        }
        self.run_kind = run_kind
        self.conversation_gate = asyncio.Semaphore(concurrency)
        self.gate = asyncio.Semaphore(concurrency)
        self.glm_concurrency = int(config["policy"].get("glm_max_concurrency", 8))
        self.glm_gate = asyncio.Semaphore(self.glm_concurrency)
        self.deepseek_concurrency = int(config["policy"].get("deepseek_max_concurrency", 8))
        self.deepseek_gate = asyncio.Semaphore(self.deepseek_concurrency)
        self.bridge_concurrency = int(config["policy"].get("bridge_max_concurrency", concurrency))
        self.bridge_gate = asyncio.Semaphore(self.bridge_concurrency)
        self.lock = asyncio.Lock()
        self.active_external = 0
        self.peak_external = 0
        self.active_conversations = 0
        self.peak_conversations = 0
        self.active_glm = 0
        self.peak_glm = 0
        self.active_deepseek = 0
        self.peak_deepseek = 0
        self.active_bridge = 0
        self.peak_bridge = 0
        self.completed = 0
        self.failed = 0
        self.started = time.monotonic()
        self.calls = Counter()
        self.retries = Counter()
        self.providers: dict[str, Counter[str]] = defaultdict(Counter)
        self.bridge_outputs: Counter[str] = Counter()
        self.first_turn_outcomes = Counter()
        self.trajectory_outcomes = Counter()
        self.trajectory_outcomes_by_checkpoint: dict[str, Counter[str]] = defaultdict(Counter)
        self.completed_trajectory_checkpoints = Counter()
        self.costs = Counter()
        self.tokens_prompt = Counter()
        self.tokens_completion = Counter()
        self.stop_reasons = Counter()
        self.conversation_seconds: list[float] = []
        self.glm_base: OpenRouterClient | None = None
        self.http_client: httpx.AsyncClient | None = None
        self.gemma_key = load_secret_value(
            config["candidates"]["gemma"]["secret_file"],
            config["candidates"]["gemma"]["api_key_env"],
        )
        self.openrouter_key = load_openrouter_key(config)

    async def gated(self, factory: Callable[[], Awaitable[T]]) -> T:
        async with self.gate:
            async with self.lock:
                self.active_external += 1
                self.peak_external = max(self.peak_external, self.active_external)
            try:
                return await factory()
            finally:
                async with self.lock:
                    self.active_external -= 1

    async def complete_json(self, **kwargs: Any) -> CompletionRecord:
        assert self.glm_base is not None
        async with self.glm_gate:
            async with self.lock:
                self.active_glm += 1
                self.peak_glm = max(self.peak_glm, self.active_glm)
            try:
                return await self.gated(lambda: self.glm_base.complete_json(**kwargs))
            finally:
                async with self.lock:
                    self.active_glm -= 1

    async def _record_candidate(
        self,
        *,
        sample_id: str,
        branch: str,
        turn: int,
        messages: list[dict[str, str]],
        record: CandidateCompletion,
    ) -> None:
        payload = {
            "at": utc_now(),
            "kind": "candidate",
            "sample_id": sample_id,
            "branch": branch,
            "turn": turn,
            "content": record.text,
            "content_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
            "prompt_sha256": prompt_hash(messages),
            "response_id": record.response_id,
            "model": record.model,
            "provider": record.provider,
            "usage": record.usage,
            "latency_seconds": record.latency_seconds,
            "attempt_count": record.attempt_count,
        }
        async with self.lock:
            append_jsonl(self.run_dir / "responses.jsonl", payload)
            self._add_usage("candidate:" + branch, record.usage, record.attempt_count, record.provider)

    async def _record_glm(
        self,
        *,
        sample_id: str,
        kind: str,
        turn: int,
        record: CompletionRecord,
        branch: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "at": utc_now(),
            "kind": kind,
            "sample_id": sample_id,
            "branch": branch,
            "turn": turn,
            "result": record.parsed.model_dump(),
            "response_id": record.response_id,
            "model": record.model,
            "provider": record.provider,
            "usage": record.usage,
            "latency_seconds": record.latency_seconds,
            "attempt_count": record.attempt_count,
            "prompt_sha256": record.prompt_sha256,
        }
        if extra:
            payload.update(extra)
        async with self.lock:
            append_jsonl(self.run_dir / "responses.jsonl", payload)
            self._add_usage("glm:" + kind, record.usage, record.attempt_count, record.provider)

    async def _record_bridge(
        self,
        *,
        sample_id: str,
        branch: str,
        turn: int,
        messages: list[dict[str, str]],
        original_next_user: str,
        record: CandidateCompletion,
    ) -> None:
        content = record.text.strip()
        payload = {
            "at": utc_now(),
            "kind": "bridge",
            "sample_id": sample_id,
            "branch": branch,
            "turn": turn,
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "original_next_user_sha256": hashlib.sha256(
                original_next_user.encode("utf-8")
            ).hexdigest(),
            "prompt_sha256": prompt_hash(messages),
            "response_id": record.response_id,
            "model": record.model,
            "provider": record.provider,
            "usage": record.usage,
            "latency_seconds": record.latency_seconds,
            "attempt_count": record.attempt_count,
        }
        async with self.lock:
            append_jsonl(self.run_dir / "responses.jsonl", payload)
            self.bridge_outputs[branch] += 1
            self._add_usage("bridge:luna", record.usage, record.attempt_count, record.provider)

    def _add_usage(self, kind: str, usage: dict[str, Any], attempts: int, provider: str | None) -> None:
        self.calls[kind] += 1
        self.retries[kind] += max(0, attempts - 1)
        self.providers[kind][provider or "self_hosted"] += 1
        self.tokens_prompt[kind] += int(usage.get("prompt_tokens") or 0)
        self.tokens_completion[kind] += int(usage.get("completion_tokens") or 0)
        self.costs[kind] += float(usage.get("cost") or 0.0)

    async def _candidate(
        self,
        *,
        sample_id: str,
        branch: str,
        turn: int,
        messages: list[dict[str, str]],
    ) -> CandidateCompletion:
        assert self.http_client is not None
        candidate = self.config["candidates"]["gemma" if branch == "gemma" else "deepseek_v4_flash"]
        key = self.gemma_key if branch == "gemma" else self.openrouter_key
        async def call() -> CandidateCompletion:
            return await self.gated(lambda: chat_completion(
                url=candidate["base_url"],
                api_key=key,
                model=candidate["model"],
                messages=messages,
                temperature=candidate["temperature"],
                max_tokens=candidate["max_tokens"],
                top_p=candidate.get("top_p"),
                top_k=candidate.get("top_k"),
                repetition_penalty=candidate.get("repetition_penalty"),
                http_client=self.http_client,
            ))
        if branch == "deepseek":
            async with self.deepseek_gate:
                async with self.lock:
                    self.active_deepseek += 1
                    self.peak_deepseek = max(self.peak_deepseek, self.active_deepseek)
                try:
                    record = await call()
                finally:
                    async with self.lock:
                        self.active_deepseek -= 1
        else:
            record = await call()
        await self._record_candidate(
            sample_id=sample_id, branch=branch, turn=turn, messages=messages, record=record
        )
        return record

    async def _bridge(
        self,
        *,
        sample_id: str,
        branch: str,
        turn: int,
        history: list[dict[str, str]],
        candidate_reply: str,
        original_next_user: str,
    ) -> CandidateCompletion:
        assert self.http_client is not None
        bridge = self.config["bridge"]
        messages = bridge_messages(
            history=history,
            candidate_reply=candidate_reply,
            original_next_user=original_next_user,
        )
        async with self.bridge_gate:
            async with self.lock:
                self.active_bridge += 1
                self.peak_bridge = max(self.peak_bridge, self.active_bridge)
            try:
                record = await self.gated(lambda: chat_completion(
                    url=bridge["base_url"],
                    api_key=self.openrouter_key,
                    model=bridge["model"],
                    messages=messages,
                    temperature=bridge.get("temperature"),
                    max_tokens=bridge["max_tokens"],
                    provider=bridge.get("provider"),
                    reasoning_effort=bridge.get("reasoning_effort"),
                    request_id=f"{self.run_kind}:{sample_id}:bridge:{branch}:{turn}",
                    timeout_seconds=bridge.get("timeout_seconds", 240),
                    max_retries=bridge.get("max_retries", 2),
                    http_client=self.http_client,
                ))
                record = replace(record, text=normalize_bridge_text(record.text))
            finally:
                async with self.lock:
                    self.active_bridge -= 1
        await self._record_bridge(
            sample_id=sample_id,
            branch=branch,
            turn=turn,
            messages=messages,
            original_next_user=original_next_user,
            record=record,
        )
        return record

    async def _record_pair_judgment(
        self,
        *,
        sample_id: str,
        turn: int,
        prefix: str,
        judgment: dict[str, Any],
        checkpoint: int | None = None,
    ) -> None:
        extra = {"checkpoint": checkpoint} if checkpoint is not None else None
        await self._record_glm(
            sample_id=sample_id,
            kind=prefix + "_order_1",
            turn=turn,
            record=judgment["first"],
            extra={"assignment": judgment["first_assignment"], **(extra or {})},
        )
        await self._record_glm(
            sample_id=sample_id,
            kind=prefix + "_order_2",
            turn=turn,
            record=judgment["second"],
            extra={"assignment": judgment["second_assignment"], **(extra or {})},
        )
        async with self.lock:
            append_jsonl(self.run_dir / "events.jsonl", {
                "at": utc_now(), "event": prefix + "_aggregate", "sample_id": sample_id,
                "turn": turn, **({"checkpoint": checkpoint} if checkpoint is not None else {}),
                **judgment["aggregate"],
            })
            target = self.first_turn_outcomes if prefix == "first_turn_judge" else self.trajectory_outcomes
            outcome = judgment["aggregate"]["outcome"]
            winner = judgment["aggregate"].get("winner") or "none"
            target[f"{outcome}:{winner}"] += 1
            if prefix == "trajectory_judge" and checkpoint is not None:
                self.trajectory_outcomes_by_checkpoint[str(checkpoint)][f"{outcome}:{winner}"] += 1
                self.completed_trajectory_checkpoints[str(checkpoint)] += 1

    async def run_conversation(self, source_index: int, seed: dict[str, Any]) -> None:
        async with self.conversation_gate:
            async with self.lock:
                self.active_conversations += 1
                self.peak_conversations = max(
                    self.peak_conversations, self.active_conversations
                )
            try:
                await self._run_conversation(source_index, seed)
            finally:
                async with self.lock:
                    self.active_conversations -= 1

    async def _run_conversation(self, source_index: int, seed: dict[str, Any]) -> None:
        began = time.monotonic()
        sample_id = seed["sample_id"]
        try:
            role_card = seed["character"]["role_card"]
            reference = normalized_character_reference(role_card)
            greeting = seed["greeting"]
            target_depth = self.target_depths[sample_id]
            user_messages = seed["ordered_user_messages"][:target_depth]
            if not user_messages:
                raise RuntimeError("conversation has no user messages")
            gemma_system = build_gemma_system(role_card)
            deepseek_system = build_deepseek_system(role_card)
            history_g: list[dict[str, str]] = [{"role": "assistant", "content": greeting}]
            history_d: list[dict[str, str]] = [{"role": "assistant", "content": greeting}]
            current_g = user_messages[0]
            current_d = user_messages[0]
            completed_turns = 0
            stop_reason = "completed_available_turns"
            random_base = int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16)
            checkpoints = set(self.trajectory_checkpoints[sample_id])
            judged_checkpoints: list[int] = []

            for turn in range(len(user_messages)):
                messages_g = [{"role": "system", "content": gemma_system}, *history_g, {"role": "user", "content": current_g}]
                messages_d = [{"role": "system", "content": deepseek_system}, *history_d, {"role": "user", "content": current_d}]
                reply_g, reply_d = await asyncio.gather(
                    self._candidate(sample_id=sample_id, branch="gemma", turn=turn, messages=messages_g),
                    self._candidate(sample_id=sample_id, branch="deepseek", turn=turn, messages=messages_d),
                )

                history_before_g = list(history_g) + [{"role": "user", "content": current_g}]
                history_before_d = list(history_d) + [{"role": "user", "content": current_d}]
                history_g.extend([{"role": "user", "content": current_g}, {"role": "assistant", "content": reply_g.text}])
                history_d.extend([{"role": "user", "content": current_d}, {"role": "assistant", "content": reply_d.text}])
                completed_turns += 1

                judge_task: asyncio.Task[dict[str, Any]] | None = None
                if turn == 0:
                    judge_task = asyncio.create_task(blind_judge_pair(
                        self,
                        request_id=f"{self.run_kind}:{sample_id}:first",
                        random_seed=random_base ^ self.config["benchmark"]["random_seed"],
                        character_reference=reference,
                        history=[{"role": "assistant", "content": greeting}],
                        current_user=user_messages[0],
                        candidate_name_1="gemma",
                        candidate_response_1=reply_g.text,
                        candidate_name_2="deepseek",
                        candidate_response_2=reply_d.text,
                    ))

                trajectory_task: asyncio.Task[dict[str, Any]] | None = None
                if completed_turns in checkpoints:
                    trajectory_task = asyncio.create_task(blind_judge_trajectories(
                        self,
                        request_id=(
                            f"{self.run_kind}:{sample_id}:trajectory:{completed_turns}"
                        ),
                        random_seed=(
                            random_base
                            ^ self.config["benchmark"]["random_seed"]
                            ^ 0xA5A5
                            ^ completed_turns
                        ),
                        character_reference=reference,
                        greeting=greeting,
                        candidate_name_1="gemma",
                        trajectory_1=history_g[1:],
                        candidate_name_2="deepseek",
                        trajectory_2=history_d[1:],
                        checkpoint=completed_turns,
                    ))

                if turn + 1 >= len(user_messages):
                    pending_judges = [
                        task for task in (judge_task, trajectory_task) if task is not None
                    ]
                    judge_results = await asyncio.gather(*pending_judges)
                    result_index = 0
                    if judge_task is not None:
                        judgment = judge_results[result_index]
                        result_index += 1
                        await self._record_pair_judgment(
                            sample_id=sample_id, turn=turn, prefix="first_turn_judge", judgment=judgment
                        )
                    if trajectory_task is not None:
                        trajectory = judge_results[result_index]
                        await self._record_pair_judgment(
                            sample_id=sample_id,
                            turn=turn,
                            prefix="trajectory_judge",
                            judgment=trajectory,
                            checkpoint=completed_turns,
                        )
                        judged_checkpoints.append(completed_turns)
                    break

                original_next = user_messages[turn + 1]
                bridge_task = asyncio.gather(
                    self._bridge(
                        sample_id=sample_id, branch="gemma", turn=turn,
                        history=history_before_g,
                        candidate_reply=reply_g.text, original_next_user=original_next,
                    ),
                    self._bridge(
                        sample_id=sample_id, branch="deepseek", turn=turn,
                        history=history_before_d,
                        candidate_reply=reply_d.text, original_next_user=original_next,
                    ),
                )
                pending: list[Awaitable[Any]] = [bridge_task]
                if judge_task is not None:
                    pending.append(judge_task)
                if trajectory_task is not None:
                    pending.append(trajectory_task)
                results = await asyncio.gather(*pending)
                bridges = results[0]
                result_index = 1
                if judge_task is not None:
                    judgment = results[result_index]
                    result_index += 1
                    await self._record_pair_judgment(
                        sample_id=sample_id, turn=turn, prefix="first_turn_judge", judgment=judgment
                    )
                if trajectory_task is not None:
                    trajectory = results[result_index]
                    await self._record_pair_judgment(
                        sample_id=sample_id,
                        turn=turn,
                        prefix="trajectory_judge",
                        judgment=trajectory,
                        checkpoint=completed_turns,
                    )
                    judged_checkpoints.append(completed_turns)
                bridge_g, bridge_d = bridges
                current_g = bridge_g.text.strip()
                current_d = bridge_d.text.strip()

            if sorted(judged_checkpoints) != sorted(checkpoints):
                raise RuntimeError(
                    f"trajectory checkpoint mismatch: expected={sorted(checkpoints)} "
                    f"actual={sorted(judged_checkpoints)}"
                )

            elapsed = time.monotonic() - began
            async with self.lock:
                self.completed += 1
                self.conversation_seconds.append(elapsed)
                self.stop_reasons[stop_reason] += 1
                append_jsonl(self.run_dir / "events.jsonl", {
                    "at": utc_now(), "event": "conversation_completed", "sample_id": sample_id,
                    "source_index": source_index, "turns": completed_turns,
                    "target_depth": target_depth,
                    "trajectory_checkpoints": judged_checkpoints,
                    "stop_reason": stop_reason, "elapsed_seconds": elapsed,
                })
                self._write_progress_unlocked()
                print(
                    f"PROGRESS completed={self.completed} failed={self.failed} total={len(self.seeds)} "
                    f"elapsed={time.monotonic() - self.started:.1f}s cost=${sum(self.costs.values()):.4f}",
                    flush=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - began
            async with self.lock:
                self.failed += 1
                self.conversation_seconds.append(elapsed)
                self.stop_reasons["error"] += 1
                append_jsonl(self.run_dir / "errors.jsonl", {
                    "at": utc_now(), "sample_id": sample_id, "source_index": source_index,
                    "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": elapsed,
                })
                self._write_progress_unlocked()
                print(
                    f"PROGRESS completed={self.completed} failed={self.failed} total={len(self.seeds)} "
                    f"elapsed={time.monotonic() - self.started:.1f}s cost=${sum(self.costs.values()):.4f}",
                    flush=True,
                )

    def _progress_value(self, done: bool = False) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        return {
            "updated_at": utc_now(),
            "done": done,
            "total_conversations": len(self.seeds),
            "completed_conversations": self.completed,
            "failed_conversations": self.failed,
            "active_conversations": self.active_conversations,
            "peak_conversations": self.peak_conversations,
            "conversation_concurrency_limit": self.concurrency,
            "active_external": self.active_external,
            "peak_external": self.peak_external,
            "active_glm": self.active_glm,
            "peak_glm": self.peak_glm,
            "glm_concurrency_limit": self.glm_concurrency,
            "active_deepseek": self.active_deepseek,
            "peak_deepseek": self.peak_deepseek,
            "deepseek_concurrency_limit": self.deepseek_concurrency,
            "active_bridge": self.active_bridge,
            "peak_bridge": self.peak_bridge,
            "bridge_concurrency_limit": self.bridge_concurrency,
            "elapsed_seconds": elapsed,
            "calls": dict(self.calls),
            "retries": dict(self.retries),
            "cost_usd": dict(self.costs),
            "cost_usd_total": sum(self.costs.values()),
            "first_turn_outcomes": dict(self.first_turn_outcomes),
            "trajectory_outcomes": dict(self.trajectory_outcomes),
            "trajectory_outcomes_by_checkpoint": {
                key: dict(value)
                for key, value in sorted(self.trajectory_outcomes_by_checkpoint.items())
            },
            "completed_trajectory_checkpoints": dict(
                sorted(self.completed_trajectory_checkpoints.items())
            ),
            "bridge_outputs": dict(self.bridge_outputs),
            "stop_reasons": dict(self.stop_reasons),
        }

    def _write_progress_unlocked(self, done: bool = False) -> None:
        path = self.run_dir / "progress.json"
        temp = self.run_dir / ".progress.json.tmp"
        temp.write_text(json.dumps(self._progress_value(done), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temp, 0o600)
        os.replace(temp, path)

    async def run(self) -> dict[str, Any]:
        timeout = httpx.Timeout(300, connect=20)
        limits = httpx.Limits(max_connections=max(40, self.concurrency * 2), max_keepalive_connections=max(20, self.concurrency))
        self.http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
        try:
            async with OpenRouterClient(self.config) as glm:
                self.glm_base = glm
                self._write_progress_unlocked()
                await asyncio.gather(*(self.run_conversation(index, seed) for index, seed in enumerate(self.seeds)))
        finally:
            await self.http_client.aclose()
        async with self.lock:
            self._write_progress_unlocked(done=True)
            summary = self._progress_value(done=True)
            summary.update({
                "schema_version": "1.0",
                "run_dir": str(self.run_dir),
                "concurrency": self.concurrency,
                "conversation_concurrency": self.concurrency,
                "glm_concurrency": self.glm_concurrency,
                "deepseek_concurrency": self.deepseek_concurrency,
                "bridge_concurrency": self.bridge_concurrency,
                "max_turns": self.max_turns,
                "run_kind": self.run_kind,
                "providers": {key: dict(value) for key, value in self.providers.items()},
                "prompt_tokens": dict(self.tokens_prompt),
                "completion_tokens": dict(self.tokens_completion),
                "mean_conversation_seconds": (
                    sum(self.conversation_seconds) / len(self.conversation_seconds)
                    if self.conversation_seconds else None
                ),
            })
            write_new_json(self.run_dir / "summary.json", summary)
            append_jsonl(self.run_dir / "events.jsonl", {
                "at": utc_now(), "event": self.run_kind + "_completed", "completed": self.completed,
                "failed": self.failed, "elapsed_seconds": summary["elapsed_seconds"],
                "cost_usd_total": summary["cost_usd_total"],
            })
        return summary


async def run_pilot20(
    config: dict[str, Any],
    *,
    count: int = 20,
    concurrency: int = 20,
    max_turns: int = 10,
    run_name: str | None = None,
) -> dict[str, Any]:
    data_dir = Path(config["benchmark"]["data_dir"])
    seeds = load_seeds(data_dir / "rollout-seeds.jsonl.gz")[:count]
    if len(seeds) != count:
        raise RuntimeError(f"requested {count} seeds but found {len(seeds)}")
    name = run_name or ("pilot20-c20-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = PROJECT_ROOT / "runs" / name
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    sample_ids = [seed["sample_id"] for seed in seeds]
    contract = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "kind": "fixed-first20-concurrent-calibration",
        "formal_benchmark": False,
        "count": count,
        "concurrency": concurrency,
        "glm_concurrency": int(config["policy"].get("glm_max_concurrency", 8)),
        "deepseek_concurrency": int(config["policy"].get("deepseek_max_concurrency", 8)),
        "bridge_concurrency": int(config["policy"].get("bridge_max_concurrency", concurrency)),
        "max_turns": max_turns,
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "input_sha256": {
            "rollout-seeds.jsonl.gz": sha256_file(data_dir / "rollout-seeds.jsonl.gz"),
            "dialogue-selection.json.gz": sha256_file(data_dir / "dialogue-selection.json.gz"),
        },
        "models": {
            "gemma": config["candidates"]["gemma"]["model"],
            "deepseek": config["candidates"]["deepseek_v4_flash"]["model"],
            "glm": config["openrouter"]["model"],
            "bridge": config["bridge"]["model"],
        },
        "candidate_generation": {
            name: {
                key: candidate[key]
                for key in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")
                if key in candidate
            }
            for name, candidate in config["candidates"].items()
        },
        "glm_provider": config["openrouter"]["provider"],
        "bridge_provider": config["bridge"].get("provider"),
        "prompt_hashes": all_prompt_hashes(),
        "bridge_policy": {
            "output": "plain text containing only the next user message",
            "historical_next_user": "reference input",
            "verifier": False,
            "actions_or_contract": False,
        },
    }
    write_new_json(run_dir / "run_contract.json", contract)
    append_jsonl(run_dir / "events.jsonl", {"at": utc_now(), "event": "pilot_started"})
    runner = PilotRunner(
        config=config, run_dir=run_dir, seeds=seeds, concurrency=concurrency, max_turns=max_turns
    )
    return await runner.run()


def build_formal_plan(
    seeds: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, list[int]]]:
    cohort30, cohort60 = staged_cohorts(seeds)
    target_depths: dict[str, int] = {}
    checkpoints: dict[str, list[int]] = {}
    for seed in seeds:
        sample_id = seed["sample_id"]
        target = 60 if sample_id in cohort60 else 30 if sample_id in cohort30 else 10
        actual = min(target, len(seed["ordered_user_messages"]))
        if actual < 1:
            raise RuntimeError(f"conversation has no user messages: {sample_id}")
        sample_checkpoints = {min(10, actual)}
        if target >= 30:
            sample_checkpoints.add(min(30, actual))
        if target >= 60:
            sample_checkpoints.add(min(60, actual))
        target_depths[sample_id] = target
        checkpoints[sample_id] = sorted(sample_checkpoints)
    return target_depths, checkpoints


async def run_formal500(
    config: dict[str, Any],
    *,
    concurrency: int = 20,
    run_name: str | None = None,
) -> dict[str, Any]:
    if not config["policy"].get("formal_run_enabled"):
        raise RuntimeError("formal benchmark is disabled")
    report, staged_plan = preflight(config)
    if not report["ready"]:
        raise RuntimeError("benchmark preflight failed")

    data_dir = Path(config["benchmark"]["data_dir"])
    seeds = load_seeds(data_dir / "rollout-seeds.jsonl.gz")
    expected_count = int(config["benchmark"]["expected_seed_count"])
    if len(seeds) != expected_count:
        raise RuntimeError(f"expected {expected_count} seeds but found {len(seeds)}")
    target_depths, checkpoints = build_formal_plan(seeds)
    planned_turns = sum(
        min(target_depths[seed["sample_id"]], len(seed["ordered_user_messages"]))
        for seed in seeds
    )
    expected_turns = int(config["benchmark"]["expected_staged_anchor_count"])
    if planned_turns != expected_turns or len(staged_plan) != expected_turns:
        raise RuntimeError(
            f"formal staged plan mismatch: planned={planned_turns} "
            f"preflight={len(staged_plan)} expected={expected_turns}"
        )
    checkpoint_count = sum(len(value) for value in checkpoints.values())
    expected_checkpoint_count = sum(config["benchmark"]["checkpoint_cohorts"])
    if checkpoint_count != expected_checkpoint_count:
        raise RuntimeError(
            f"formal checkpoint count mismatch: {checkpoint_count} != "
            f"{expected_checkpoint_count}"
        )

    name = run_name or (
        "formal500-c20-v4-gemma200-250-staged10-30-60-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir = PROJECT_ROOT / "runs" / name
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    sample_ids = [seed["sample_id"] for seed in seeds]
    target_depth_distribution = Counter(target_depths.values())
    actual_depth_distribution = Counter(
        min(target_depths[seed["sample_id"]], len(seed["ordered_user_messages"]))
        for seed in seeds
    )
    checkpoint_distribution = Counter(
        checkpoint for values in checkpoints.values() for checkpoint in values
    )
    contract = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "kind": "fixed-500-formal-staged-10-30-60",
        "formal_benchmark": True,
        "count": len(seeds),
        "concurrency": concurrency,
        "glm_concurrency": int(config["policy"].get("glm_max_concurrency", 8)),
        "deepseek_concurrency": int(
            config["policy"].get("deepseek_max_concurrency", 8)
        ),
        "bridge_concurrency": int(
            config["policy"].get("bridge_max_concurrency", concurrency)
        ),
        "planned_turns_per_branch": planned_turns,
        "planned_bridge_calls": 2 * (planned_turns - len(seeds)),
        "planned_first_turn_judge_calls": 2 * len(seeds),
        "planned_trajectory_checkpoints": checkpoint_count,
        "planned_trajectory_judge_calls": 2 * checkpoint_count,
        "target_depth_distribution": {
            str(key): value for key, value in sorted(target_depth_distribution.items())
        },
        "actual_depth_distribution": {
            str(key): value for key, value in sorted(actual_depth_distribution.items())
        },
        "checkpoint_distribution": {
            str(key): value for key, value in sorted(checkpoint_distribution.items())
        },
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "input_sha256": {
            "rollout-seeds.jsonl.gz": sha256_file(data_dir / "rollout-seeds.jsonl.gz"),
            "dialogue-selection.json.gz": sha256_file(data_dir / "dialogue-selection.json.gz"),
            "selection-manifest.json": sha256_file(data_dir / "selection-manifest.json"),
            "SHA256SUMS": sha256_file(data_dir / "SHA256SUMS"),
        },
        "config_sha256": hashlib.sha256(
            Path(config["_config_path"]).read_bytes()
        ).hexdigest(),
        "models": {
            "gemma": config["candidates"]["gemma"]["model"],
            "deepseek": config["candidates"]["deepseek_v4_flash"]["model"],
            "glm": config["openrouter"]["model"],
            "bridge": config["bridge"]["model"],
        },
        "candidate_generation": {
            candidate_name: {
                key: candidate[key]
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "max_tokens",
                )
                if key in candidate
            }
            for candidate_name, candidate in config["candidates"].items()
        },
        "glm_provider": config["openrouter"]["provider"],
        "bridge_provider": config["bridge"].get("provider"),
        "prompt_hashes": all_prompt_hashes(),
        "bridge_policy": {
            "output": "plain text containing only the next user message",
            "historical_next_user": "reference input",
            "verifier": False,
            "actions_or_contract": False,
        },
    }
    write_new_json(run_dir / "run_contract.json", contract)
    append_jsonl(run_dir / "events.jsonl", {
        "at": utc_now(),
        "event": "formal500_started",
        "planned_turns_per_branch": planned_turns,
        "planned_trajectory_checkpoints": checkpoint_count,
    })
    runner = PilotRunner(
        config=config,
        run_dir=run_dir,
        seeds=seeds,
        concurrency=concurrency,
        max_turns=max(target_depths.values()),
        target_depths=target_depths,
        trajectory_checkpoints=checkpoints,
        run_kind="formal500",
    )
    return await runner.run()


async def run_replay_completed(
    config: dict[str, Any],
    *,
    source_run: str,
    count: int,
    concurrency: int = 8,
    run_name: str | None = None,
) -> dict[str, Any]:
    if count < 1:
        raise RuntimeError("replay count must be positive")
    if Path(source_run).name != source_run or any(x in source_run for x in ("/", "\\")):
        raise RuntimeError("source_run must be a run directory name, not a path")

    source_dir = PROJECT_ROOT / "runs" / source_run
    source_events_path = source_dir / "events.jsonl"
    source_responses_path = source_dir / "responses.jsonl"
    source_contract_path = source_dir / "run_contract.json"
    if not all(path.is_file() for path in (
        source_events_path, source_responses_path, source_contract_path
    )):
        raise RuntimeError(f"source run is incomplete or missing: {source_run}")

    completed_events: list[dict[str, Any]] = []
    for line in source_events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "conversation_completed":
            completed_events.append(event)
    if len(completed_events) < count:
        raise RuntimeError(
            f"source run has only {len(completed_events)} completed conversations; "
            f"requested {count}"
        )

    selected_events = completed_events[:count]
    selected_ids = [event["sample_id"] for event in selected_events]
    if len(set(selected_ids)) != count:
        raise RuntimeError("source completed-conversation selection contains duplicate sample IDs")

    data_dir = Path(config["benchmark"]["data_dir"])
    all_seeds = load_seeds(data_dir / "rollout-seeds.jsonl.gz")
    seed_by_id = {seed["sample_id"]: seed for seed in all_seeds}
    missing = sorted(set(selected_ids) - set(seed_by_id))
    if missing:
        raise RuntimeError(f"selected sample IDs missing from immutable seeds: {missing[:5]}")

    selected_event_by_id = {event["sample_id"]: event for event in selected_events}
    seeds = [seed for seed in all_seeds if seed["sample_id"] in selected_event_by_id]
    target_depths: dict[str, int] = {}
    checkpoints: dict[str, list[int]] = {}
    for seed in seeds:
        sample_id = seed["sample_id"]
        event = selected_event_by_id[sample_id]
        turns = int(event["turns"])
        sample_checkpoints = [int(value) for value in event["trajectory_checkpoints"]]
        if turns < 1 or turns > len(seed["ordered_user_messages"]):
            raise RuntimeError(
                f"invalid replay depth for {sample_id}: {turns} / "
                f"{len(seed['ordered_user_messages'])}"
            )
        if not sample_checkpoints or sample_checkpoints != sorted(set(sample_checkpoints)):
            raise RuntimeError(f"invalid replay checkpoints for {sample_id}")
        if sample_checkpoints[-1] > turns:
            raise RuntimeError(f"checkpoint exceeds replay depth for {sample_id}")
        target_depths[sample_id] = turns
        checkpoints[sample_id] = sample_checkpoints

    planned_turns = sum(target_depths.values())
    checkpoint_count = sum(len(value) for value in checkpoints.values())
    selection_lines = [
        canonical_json({
            "sample_id": sample_id,
            "turns": target_depths[sample_id],
            "checkpoints": checkpoints[sample_id],
        })
        for sample_id in [seed["sample_id"] for seed in seeds]
    ]
    name = run_name or (
        f"replay{count}-c{concurrency}-bf16-vs-fp4-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir = PROJECT_ROOT / "runs" / name
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    sample_ids = [seed["sample_id"] for seed in seeds]
    contract = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "kind": "matched-replay-of-completed-source-conversations",
        "formal_benchmark": False,
        "matched_comparison": True,
        "source_run": source_run,
        "source_completed_conversations_available": len(completed_events),
        "count": count,
        "concurrency": concurrency,
        "gemma_deployment_precision": "BF16",
        "gemma_deployment_precision_provenance": "user-reported endpoint deployment",
        "planned_turns_per_branch": planned_turns,
        "planned_bridge_calls": 2 * (planned_turns - count),
        "planned_first_turn_judge_calls": 2 * count,
        "planned_trajectory_checkpoints": checkpoint_count,
        "planned_trajectory_judge_calls": 2 * checkpoint_count,
        "actual_depth_distribution": {
            str(key): value for key, value in sorted(Counter(target_depths.values()).items())
        },
        "checkpoint_distribution": {
            str(key): value
            for key, value in sorted(Counter(
                checkpoint for values in checkpoints.values() for checkpoint in values
            ).items())
        },
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "source_selection_sha256": hashlib.sha256(
            "\n".join(selection_lines).encode()
        ).hexdigest(),
        "source_artifact_sha256": {
            "run_contract.json": sha256_file(source_contract_path),
            "events.jsonl": sha256_file(source_events_path),
            "responses.jsonl": sha256_file(source_responses_path),
        },
        "input_sha256": {
            "rollout-seeds.jsonl.gz": sha256_file(data_dir / "rollout-seeds.jsonl.gz"),
            "dialogue-selection.json.gz": sha256_file(data_dir / "dialogue-selection.json.gz"),
            "selection-manifest.json": sha256_file(data_dir / "selection-manifest.json"),
            "SHA256SUMS": sha256_file(data_dir / "SHA256SUMS"),
        },
        "config_sha256": hashlib.sha256(
            Path(config["_config_path"]).read_bytes()
        ).hexdigest(),
        "models": {
            "gemma": config["candidates"]["gemma"]["model"],
            "deepseek": config["candidates"]["deepseek_v4_flash"]["model"],
            "glm": config["openrouter"]["model"],
            "bridge": config["bridge"]["model"],
        },
        "candidate_generation": {
            candidate_name: {
                key: candidate[key]
                for key in (
                    "temperature", "top_p", "top_k", "repetition_penalty", "max_tokens"
                )
                if key in candidate
            }
            for candidate_name, candidate in config["candidates"].items()
        },
        "glm_provider": config["openrouter"]["provider"],
        "bridge_provider": config["bridge"].get("provider"),
        "prompt_hashes": all_prompt_hashes(),
        "bridge_policy": {
            "output": "plain text containing only the next user message",
            "historical_next_user": "reference input",
            "verifier": False,
            "actions_or_contract": False,
        },
    }
    write_new_json(run_dir / "run_contract.json", contract)
    append_jsonl(run_dir / "events.jsonl", {
        "at": utc_now(),
        "event": "matched_replay_started",
        "source_run": source_run,
        "count": count,
        "planned_turns_per_branch": planned_turns,
        "planned_trajectory_checkpoints": checkpoint_count,
    })
    runner = PilotRunner(
        config=config,
        run_dir=run_dir,
        seeds=seeds,
        concurrency=concurrency,
        max_turns=max(target_depths.values()),
        target_depths=target_depths,
        trajectory_checkpoints=checkpoints,
        run_kind=f"replay{count}_bf16",
    )
    return await runner.run()
