from __future__ import annotations

import hashlib
import json
from typing import Any


BRIDGE_SYSTEM = """Write exactly one short, single-line message (maximum 120 characters) that the human user would say next after the candidate assistant reply, using the historical next user message only as a hint; output nothing else."""

JUDGE_SYSTEM = """You are an impartial blind judge comparing two candidate assistant replies in a role-play benchmark.
Judge only the visible normalized character reference, prior conversation, and current user message. The labels A and B are randomized.
Never guess model identity. Do not use verbosity as a proxy for quality. Respect the user's language and the intended adult or non-adult context.
The pairwise winner is primary; dimension scores are diagnostic and must not be mechanically summed to choose a winner.
Penalize character drift, contradictions, repetition, parroting, controlling the user's actions/thoughts/dialogue, irrelevant refusal, moralizing, meta-talk, language mismatch, stagnation, and unsupported facts.
Return one JSON object only. Keep short_reason concise and do not reveal chain-of-thought."""

TRAJECTORY_JUDGE_SYSTEM = """You are an impartial blind judge comparing two complete role-play trajectories produced from the same initial conversation seed.
Trajectory A and Trajectory B may contain different later user messages because each branch minimally adapted the same historical user intent to preserve continuity. Do not require later user messages to be textually identical.
Evaluate the overall user experience across the full trajectory: long-term character consistency, memory and factual continuity, natural narrative progression, emotional development, preservation of user agency, repetition and looping, irrelevant refusal or moralizing, language and style consistency, and whether the conversation remains easy to continue naturally.
Judge assistant behavior, not the elegance of the adapted user messages. You are not given Bridge statistics and must not guess model identity. The labels A and B are randomized.
The pairwise winner is primary; dimension scores are diagnostic and must not be mechanically summed to choose a winner.
Return one JSON object only. Keep short_reason concise and do not reveal chain-of-thought."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()


def normalize_bridge_text(text: str) -> str:
    return " ".join(text.split())


def bridge_messages(
    *,
    history: list[dict[str, str]],
    candidate_reply: str,
    original_next_user: str,
) -> list[dict[str, str]]:
    payload = {
        "visible_history": history,
        "candidate_assistant_reply": candidate_reply,
        "historical_next_user_message": original_next_user,
    }
    return [{"role": "system", "content": BRIDGE_SYSTEM}, {"role": "user", "content": canonical_json(payload)}]


def judge_messages(
    *,
    character_reference: dict[str, Any],
    history: list[dict[str, str]],
    current_user: str,
    response_a: str,
    response_b: str,
    trajectory: bool = False,
) -> list[dict[str, str]]:
    payload = {
        "evaluation_type": "trajectory_checkpoint" if trajectory else "same_prompt_single_turn",
        "character_reference": character_reference,
        "visible_history": history,
        "current_user_message": current_user,
        "response_A": response_a,
        "response_B": response_b,
        "score_scale": "integer 1..5, where 5 is best",
        "required_dimensions": [
            "character_fidelity", "context_continuity", "response_relevance",
            "narrative_progression", "emotional_fit", "user_agency",
            "style_naturalness", "non_repetition", "memory_consistency",
            "future_compatibility"
        ],
        "required_json_fields": {
            "winner": "A | B | TIE | BOTH_BAD",
            "confidence": "0..1",
            "scores": "object containing every required dimension with integer A and B",
            "failure_tags_A": "array of short uppercase tags",
            "failure_tags_B": "array of short uppercase tags",
            "short_reason": "brief comparative explanation"
        }
    }
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": canonical_json(payload)}]


def trajectory_judge_messages(
    *,
    character_reference: dict[str, Any],
    greeting: str,
    trajectory_a: list[dict[str, str]],
    trajectory_b: list[dict[str, str]],
    checkpoint: int,
) -> list[dict[str, str]]:
    payload = {
        "evaluation_type": "trajectory_checkpoint",
        "checkpoint_user_turn": checkpoint,
        "shared_initial_seed": {
            "character_reference": character_reference,
            "greeting": greeting,
        },
        "trajectory_A": trajectory_a,
        "trajectory_B": trajectory_b,
        "score_scale": "integer 1..5, where 5 is best",
        "required_dimensions": [
            "character_fidelity", "context_continuity", "response_relevance",
            "narrative_progression", "emotional_fit", "user_agency",
            "style_naturalness", "non_repetition", "memory_consistency",
            "future_compatibility"
        ],
        "required_json_fields": {
            "winner": "A | B | TIE | BOTH_BAD",
            "confidence": "0..1",
            "scores": "object containing every required dimension with integer A and B",
            "failure_tags_A": "array of short uppercase tags",
            "failure_tags_B": "array of short uppercase tags",
            "short_reason": "brief comparative explanation"
        },
    }
    return [
        {"role": "system", "content": TRAJECTORY_JUDGE_SYSTEM},
        {"role": "user", "content": canonical_json(payload)},
    ]


def all_prompt_hashes() -> dict[str, str]:
    return {
        "bridge_system": hashlib.sha256(BRIDGE_SYSTEM.encode()).hexdigest(),
        "judge_system": hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest(),
        "trajectory_judge_system": hashlib.sha256(TRAJECTORY_JUDGE_SYSTEM.encode()).hexdigest(),
    }
