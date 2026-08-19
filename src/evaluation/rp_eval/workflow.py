from __future__ import annotations

import random
from typing import Any

from .openrouter import CompletionRecord, OpenRouterClient
from .prompts import judge_messages, trajectory_judge_messages
from .schemas import JudgeResult
from .scoring import aggregate_position_swap


async def blind_judge_pair(
    client: OpenRouterClient,
    *,
    request_id: str,
    random_seed: int,
    character_reference: dict[str, Any],
    history: list[dict[str, str]],
    current_user: str,
    candidate_name_1: str,
    candidate_response_1: str,
    candidate_name_2: str,
    candidate_response_2: str,
    trajectory: bool = False,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    items = [(candidate_name_1, candidate_response_1), (candidate_name_2, candidate_response_2)]
    rng.shuffle(items)
    first_assignment = {"A": items[0][0], "B": items[1][0]}
    first_messages = judge_messages(
        character_reference=character_reference,
        history=history,
        current_user=current_user,
        response_a=items[0][1],
        response_b=items[1][1],
        trajectory=trajectory,
    )
    first = await client.complete_json(messages=first_messages, schema=JudgeResult, role="judge", request_id=request_id + ":order1")

    second_assignment = {"A": items[1][0], "B": items[0][0]}
    second_messages = judge_messages(
        character_reference=character_reference,
        history=history,
        current_user=current_user,
        response_a=items[1][1],
        response_b=items[0][1],
        trajectory=trajectory,
    )
    second = await client.complete_json(messages=second_messages, schema=JudgeResult, role="judge", request_id=request_id + ":order2")
    aggregate = aggregate_position_swap(
        first.parsed.model_dump(), second.parsed.model_dump(), first_assignment, second_assignment
    )
    return {
        "first": first,
        "second": second,
        "first_assignment": first_assignment,
        "second_assignment": second_assignment,
        "aggregate": aggregate,
    }


async def blind_judge_trajectories(
    client: OpenRouterClient,
    *,
    request_id: str,
    random_seed: int,
    character_reference: dict[str, Any],
    greeting: str,
    candidate_name_1: str,
    trajectory_1: list[dict[str, str]],
    candidate_name_2: str,
    trajectory_2: list[dict[str, str]],
    checkpoint: int,
) -> dict[str, Any]:
    rng = random.Random(random_seed)
    items = [(candidate_name_1, trajectory_1), (candidate_name_2, trajectory_2)]
    rng.shuffle(items)
    first_assignment = {"A": items[0][0], "B": items[1][0]}
    first_messages = trajectory_judge_messages(
        character_reference=character_reference,
        greeting=greeting,
        trajectory_a=items[0][1],
        trajectory_b=items[1][1],
        checkpoint=checkpoint,
    )
    first = await client.complete_json(
        messages=first_messages, schema=JudgeResult, role="judge", request_id=request_id + ":order1"
    )
    second_assignment = {"A": items[1][0], "B": items[0][0]}
    second_messages = trajectory_judge_messages(
        character_reference=character_reference,
        greeting=greeting,
        trajectory_a=items[1][1],
        trajectory_b=items[0][1],
        checkpoint=checkpoint,
    )
    second = await client.complete_json(
        messages=second_messages, schema=JudgeResult, role="judge", request_id=request_id + ":order2"
    )
    aggregate = aggregate_position_swap(
        first.parsed.model_dump(), second.parsed.model_dump(), first_assignment, second_assignment
    )
    return {
        "first": first,
        "second": second,
        "first_assignment": first_assignment,
        "second_assignment": second_assignment,
        "aggregate": aggregate,
    }
