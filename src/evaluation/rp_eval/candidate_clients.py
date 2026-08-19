from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx


class CandidateTransportError(RuntimeError):
    pass


class RetryableCandidateTransportError(CandidateTransportError):
    def __init__(self, message: str, wait_seconds: float | None = None) -> None:
        super().__init__(message)
        self.wait_seconds = wait_seconds


@dataclass(frozen=True)
class CandidateCompletion:
    text: str
    response_id: str | None
    model: str | None
    provider: str | None
    usage: dict[str, Any]
    latency_seconds: float
    attempt_count: int


async def chat_completion(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    provider: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    request_id: str | None = None,
    timeout_seconds: float = 240,
    max_retries: int = 5,
    http_client: httpx.AsyncClient | None = None,
) -> CandidateCompletion:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if provider:
        payload["provider"] = provider
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if request_id:
        payload["user"] = request_id
    started = time.monotonic()
    errors: list[str] = []
    timeout = httpx.Timeout(timeout_seconds, connect=20)
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=timeout)
    try:
        for attempt in range(1, max_retries + 2):
            retry_wait: float | None = None
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    header = response.headers.get("Retry-After")
                    try:
                        retry_wait = float(header) if header else None
                    except ValueError:
                        retry_wait = None
                    raise RetryableCandidateTransportError(
                        f"retryable HTTP {response.status_code}: {response.text[:500]}",
                        wait_seconds=retry_wait,
                    )
                if response.status_code >= 400:
                    raise CandidateTransportError(f"HTTP {response.status_code}: {response.text[:1000]}")
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise CandidateTransportError("missing choices")
                text = choices[0].get("message", {}).get("content") or ""
                if not text.strip():
                    raise CandidateTransportError("empty assistant content")
                return CandidateCompletion(
                    text=text,
                    response_id=body.get("id"),
                    model=body.get("model"),
                    provider=body.get("provider"),
                    usage=body.get("usage") or {},
                    latency_seconds=time.monotonic() - started,
                    attempt_count=attempt,
                )
            except (httpx.HTTPError, CandidateTransportError, ValueError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt > max_retries:
                    break
                if isinstance(exc, RetryableCandidateTransportError) and exc.wait_seconds is not None:
                    retry_wait = exc.wait_seconds
                delay = retry_wait if retry_wait is not None else min(30.0, (2 ** attempt) + random.random() * 2)
                await asyncio.sleep(max(1.0, min(60.0, delay)))
    finally:
        if owns_client:
            await client.aclose()
    raise CandidateTransportError("; ".join(errors))
