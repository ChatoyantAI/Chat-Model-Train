from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DimensionPair(StrictModel):
    A: int = Field(ge=1, le=5)
    B: int = Field(ge=1, le=5)


class JudgeScores(StrictModel):
    character_fidelity: DimensionPair
    context_continuity: DimensionPair
    response_relevance: DimensionPair
    narrative_progression: DimensionPair
    emotional_fit: DimensionPair
    user_agency: DimensionPair
    style_naturalness: DimensionPair
    non_repetition: DimensionPair
    memory_consistency: DimensionPair
    future_compatibility: DimensionPair


class JudgeResult(StrictModel):
    winner: Literal["A", "B", "TIE", "BOTH_BAD"]
    confidence: float = Field(ge=0.0, le=1.0)
    scores: JudgeScores
    failure_tags_A: list[str]
    failure_tags_B: list[str]
    short_reason: str


class SmokeResult(StrictModel):
    status: Literal["ok"]
    nonce: str
