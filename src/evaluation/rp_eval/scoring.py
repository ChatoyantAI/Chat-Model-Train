from __future__ import annotations

from typing import Any


def unblind_winner(winner: str, assignment: dict[str, str]) -> str:
    if winner in {"A", "B"}:
        return assignment[winner]
    return winner.lower()


def aggregate_position_swap(
    first: dict[str, Any],
    second: dict[str, Any],
    first_assignment: dict[str, str],
    second_assignment: dict[str, str],
) -> dict[str, Any]:
    one = unblind_winner(first["winner"], first_assignment)
    two = unblind_winner(second["winner"], second_assignment)
    candidates = set(first_assignment.values())
    if one == two and one in candidates:
        outcome = "strict_win"
        winner = one
    elif one == two == "tie":
        outcome, winner = "tie", None
    elif one == two == "both_bad":
        outcome, winner = "both_bad", None
    elif one in candidates and two == "tie":
        outcome, winner = "weak_win", one
    elif two in candidates and one == "tie":
        outcome, winner = "weak_win", two
    else:
        outcome, winner = "uncertain", None
    return {"outcome": outcome, "winner": winner, "first_unblinded": one, "second_unblinded": two}
