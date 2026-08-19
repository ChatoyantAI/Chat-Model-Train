from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from .config import load_openrouter_key
from .json_tools import extract_json_object
from .prompts import prompt_hash


T = TypeVar("T", bound=BaseModel)


class OpenRouterError(RuntimeError):
    pass


class RetryableOpenRouterError(OpenRouterError):
    def __init__(self, message: str, wait_seconds: float | None = None) -> None:
        super().__init__(message)
        self.wait_seconds = wait_seconds


@dataclass(frozen=True)
class CompletionRecord:
    parsed: BaseModel
    raw_text: str
    response_id: str | None
    model: str | None
    provider: str | None
    usage: dict[str, Any]
    latency_seconds: float
    prompt_sha256: str
    attempt_count: int


class OpenRouterClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.section = config["openrouter"]
        self.api_key = load_openrouter_key(config)
        timeout = httpx.Timeout(
            self.section["timeout_seconds"],
            connect=self.section["connect_timeout_seconds"],
        )
        self.client = httpx.AsyncClient(base_url=self.section["base_url"], timeout=timeout)

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.section["http_referer"],
            "X-Title": self.section["x_title"],
        }

    async def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        schema: type[T],
        role: str,
        request_id: str,
    ) -> CompletionRecord:
        role_config = self.config["glm_roles"][role]
        payload: dict[str, Any] = {
            "model": self.section["model"],
            "messages": messages,
            "temperature": role_config["temperature"],
            "max_tokens": role_config["max_tokens"],
            "stream": False,
            "provider": self.section["provider"],
            "reasoning": {"effort": role_config["reasoning_effort"], "exclude": True},
            "user": request_id,
        }
        started = time.monotonic()
        errors: list[str] = []
        max_attempts = int(self.section["max_retries"]) + 1
        for attempt in range(1, max_attempts + 1):
            retry_wait: float | None = None
            try:
                response = await self.client.post("/chat/completions", headers=self._headers(), json=payload)
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    header = response.headers.get("Retry-After")
                    try:
                        retry_wait = float(header) if header else None
                    except ValueError:
                        retry_wait = None
                    raise RetryableOpenRouterError(
                        f"retryable HTTP {response.status_code}: {response.text[:500]}",
                        wait_seconds=retry_wait,
                    )
                if response.status_code >= 400:
                    raise OpenRouterError(f"HTTP {response.status_code}: {response.text[:1000]}")
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise OpenRouterError(f"missing choices: {json.dumps(body, ensure_ascii=False)[:1000]}")
                raw_text = choices[0].get("message", {}).get("content") or ""
                parsed = schema.model_validate(extract_json_object(raw_text))
                return CompletionRecord(
                    parsed=parsed,
                    raw_text=raw_text,
                    response_id=body.get("id"),
                    model=body.get("model"),
                    provider=body.get("provider"),
                    usage=body.get("usage") or {},
                    latency_seconds=time.monotonic() - started,
                    prompt_sha256=prompt_hash(messages),
                    attempt_count=attempt,
                )
            except (httpx.HTTPError, ValueError, OpenRouterError) as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt >= max_attempts:
                    break
                if isinstance(exc, RetryableOpenRouterError) and exc.wait_seconds is not None:
                    retry_wait = exc.wait_seconds
                delay = retry_wait if retry_wait is not None else min(30.0, (2 ** attempt) + random.random() * 2)
                await asyncio.sleep(max(1.0, min(60.0, delay)))
        raise OpenRouterError("; ".join(errors))
