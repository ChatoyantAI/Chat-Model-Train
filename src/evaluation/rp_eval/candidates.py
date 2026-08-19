from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CandidateRequest:
    sample_id: str
    character_role_card: dict[str, Any]
    greeting: str
    history: list[dict[str, str]]
    current_user: str


@dataclass(frozen=True)
class CandidateResponse:
    text: str
    model: str
    provider: str | None
    usage: dict[str, Any]
    raw_metadata: dict[str, Any]


class CandidateAdapter(Protocol):
    name: str

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        ...


class PendingAdapter:
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    async def generate(self, request: CandidateRequest) -> CandidateResponse:
        raise RuntimeError(f"candidate adapter {self.name} is not ready: {self.reason}")
