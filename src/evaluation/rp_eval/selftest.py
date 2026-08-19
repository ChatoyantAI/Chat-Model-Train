from __future__ import annotations

from .json_tools import extract_json_object
from .prompts import bridge_messages, normalize_bridge_text
from .schemas import JudgeResult
from .scoring import aggregate_position_swap


def run() -> int:
    from .cli import _seed_is_nsfw

    assert _seed_is_nsfw({"metadata": {"nsfw": "sfw"}}) is False
    assert _seed_is_nsfw({"metadata": {"nsfw": "nsfw"}}) is True
    assert extract_json_object('```json\n{"x":1}\n```') == {"x": 1}
    bridge = bridge_messages(
        history=[{"role": "user", "content": "hi"}],
        candidate_reply="hello",
        original_next_user="how are you?",
    )
    assert len(bridge) == 2
    assert "exactly one short, single-line message" in bridge[0]["content"]
    assert normalize_bridge_text(" hello\n\nworld ") == "hello world"
    dimensions = {
        key: {"A": 4, "B": 3}
        for key in [
            "character_fidelity", "context_continuity", "response_relevance", "narrative_progression",
            "emotional_fit", "user_agency", "style_naturalness", "non_repetition",
            "memory_consistency", "future_compatibility"
        ]
    }
    one = JudgeResult.model_validate({
        "winner": "A", "confidence": 0.9, "scores": dimensions,
        "failure_tags_A": [], "failure_tags_B": ["REPETITION"], "short_reason": "A is better"
    }).model_dump()
    two = dict(one)
    two["winner"] = "B"
    result = aggregate_position_swap(one, two, {"A": "gemma", "B": "deepseek"}, {"A": "deepseek", "B": "gemma"})
    assert result == {
        "outcome": "strict_win", "winner": "gemma",
        "first_unblinded": "gemma", "second_unblinded": "gemma"
    }
    print("SELFTEST_OK")
    return 0
