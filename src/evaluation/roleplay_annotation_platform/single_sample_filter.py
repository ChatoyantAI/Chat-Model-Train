"""Deterministic filtering for one role-play candidate.

The filter deliberately does not learn from Battle winners.  It treats score 2
as the rubric's minimum complete/usable level, score 3 as optional excellence,
and 0/1 as an observed deficiency that cannot be averaged away by unrelated
strengths.  A negative decision from one judge is routed to targeted
verification; automatic exclusion requires independent confirmation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from statistics import mean, median
from typing import Any, Iterable


FILTER_VERSION = "roleplay-single-sample-filter-v1"
COMPLETE_SCORE = 2
DEFAULT_CONFIRMATIONS = 2


def _item_metadata(
    rubric: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    items: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = {}
    for dimension in rubric.get("dimensions", []):
        group_id = str(dimension.get("id") or dimension.get("name") or "unknown")
        group_items: list[str] = []
        for item in dimension.get("items", []):
            item_id = str(item["id"])
            scale = [int(value) for value in item.get("scale", [])]
            if not scale or max(scale) < COMPLETE_SCORE:
                raise ValueError(f"rubric item {item_id} has no complete score")
            items[item_id] = {
                "id": item_id,
                "name": str(item.get("name") or item_id),
                "group_id": group_id,
                "max_score": max(scale),
                "starred": bool(item.get("starred")),
                "allows_na": bool(item.get("allows_na", True)),
            }
            group_items.append(item_id)
        groups[group_id] = group_items
    if not items:
        raise ValueError("rubric has no atomic items")
    return items, groups


def _numeric_score(record: Any, allowed_max: int) -> int | None:
    if not isinstance(record, dict):
        raise ValueError("score record must be an object")
    value = record.get("value")
    if value == "N/A":
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a score")
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid score value: {value!r}") from exc
    if score < 0 or score > allowed_max:
        raise ValueError(f"score {score} is outside 0..{allowed_max}")
    return score


def _gate_triggers(judgment: dict[str, Any]) -> set[str]:
    explicit = judgment.get("failed_gate_ids")
    if isinstance(explicit, list):
        return {str(value) for value in explicit if str(value)}
    triggers: set[str] = set()
    gates = judgment.get("gates")
    if isinstance(gates, dict):
        for gate_id, record in gates.items():
            if isinstance(record, dict) and record.get("value") is True:
                triggers.add(str(gate_id))
    if judgment.get("status") == "discarded" and not triggers:
        triggers.add("unknown_gate")
    return triggers


def _lower_mode(values: list[int]) -> int:
    counts = Counter(values)
    frequency = max(counts.values())
    return min(value for value, count in counts.items() if count == frequency)


def filter_single_sample(
    rubric: dict[str, Any],
    judgments: Iterable[dict[str, Any]],
    *,
    required_confirmations: int = DEFAULT_CONFIRMATIONS,
    expected_judges: int | None = None,
) -> dict[str, Any]:
    """Return a filtering decision for one candidate.

    Accepted judgment shape::

        {
          "judge_id": "model family or stable judge id",
          "status": "completed|discarded|invalid|failed",
          "gates": {...},
          "failed_gate_ids": [...],
          "scores": {"C1": {"value": "2"}, ...}
        }

    One clean completed judgment may produce ``keep``.  Any negative signal from
    only one judge produces ``review`` and targeted verification fields.  Gate
    rejection or score-based exclusion requires ``required_confirmations``
    independent judges.
    """

    if required_confirmations < 1:
        raise ValueError("required_confirmations must be positive")
    item_meta, groups = _item_metadata(rubric)
    judgments = list(judgments)

    judge_ids: list[str] = []
    for index, judgment in enumerate(judgments):
        judge_id = str(
            judgment.get("judge_id") or judgment.get("model") or f"judge-{index + 1}"
        )
        if judge_id in judge_ids:
            raise ValueError(f"duplicate judge_id: {judge_id}")
        judge_ids.append(judge_id)

    valid = [
        judgment
        for judgment in judgments
        if judgment.get("status") in {"completed", "discarded"}
    ]
    completed = [
        judgment for judgment in valid if judgment.get("status") == "completed"
    ]
    unavailable = [
        str(judgment.get("judge_id") or judgment.get("model") or "unknown")
        for judgment in judgments
        if judgment.get("status") not in {"completed", "discarded"}
    ]

    if not valid:
        return {
            "filter_version": FILTER_VERSION,
            "decision": "retry",
            "recommended_action": "retry_judge",
            "assurance": "none",
            "reason_codes": ["no_valid_judgment"],
            "verification_targets": {"gates": [], "items": []},
            "unavailable_judges": unavailable,
            "items": {},
            "groups": {},
            "metrics": {},
        }

    expected = expected_judges if expected_judges is not None else len(judgments)
    missing_judge_count = max(0, expected - len(valid))

    gate_votes: dict[str, int] = defaultdict(int)
    for judgment in valid:
        for gate_id in _gate_triggers(judgment):
            gate_votes[gate_id] += 1
    confirmed_gates = sorted(
        gate_id
        for gate_id, votes in gate_votes.items()
        if votes >= required_confirmations
    )
    unconfirmed_gates = sorted(
        gate_id
        for gate_id, votes in gate_votes.items()
        if votes < required_confirmations
    )

    item_rows: dict[str, dict[str, Any]] = {}
    confirmed_deficiencies: list[str] = []
    unconfirmed_deficiencies: list[str] = []
    applicability_disagreements: list[str] = []
    score_disagreements: list[str] = []
    invalid_score_records: list[str] = []

    for item_id, metadata in item_meta.items():
        numeric: list[int] = []
        na_count = 0
        missing_count = 0
        invalid_count = 0
        for judgment in completed:
            record = (judgment.get("scores") or {}).get(item_id)
            if record is None:
                missing_count += 1
                continue
            try:
                score = _numeric_score(record, int(metadata["max_score"]))
            except ValueError:
                invalid_count += 1
                invalid_score_records.append(
                    f"{judgment.get('judge_id') or judgment.get('model') or 'unknown'}:{item_id}"
                )
                continue
            if score is None:
                na_count += 1
            else:
                numeric.append(score)

        if numeric and na_count:
            applicability_disagreements.append(item_id)
        deficient_votes = sum(score < COMPLETE_SCORE for score in numeric)
        complete_votes = sum(score >= COMPLETE_SCORE for score in numeric)
        confirmed_deficient = deficient_votes >= required_confirmations
        if confirmed_deficient:
            confirmed_deficiencies.append(item_id)
        elif deficient_votes:
            unconfirmed_deficiencies.append(item_id)

        score_range = max(numeric) - min(numeric) if numeric else None
        if score_range is not None and score_range >= 2:
            score_disagreements.append(item_id)
        confirmed_three = (
            metadata["max_score"] >= 3
            and sum(score == 3 for score in numeric) >= required_confirmations
            and all(score >= COMPLETE_SCORE for score in numeric)
        )
        display_score = _lower_mode(numeric) if numeric else None
        if display_score == 3 and not confirmed_three:
            display_score = 2
        if confirmed_three:
            display_score = 3

        item_rows[item_id] = {
            "name": metadata["name"],
            "group_id": metadata["group_id"],
            "applicable": bool(numeric),
            "scores": numeric,
            "na_votes": na_count,
            "missing_votes": missing_count,
            "invalid_votes": invalid_count,
            "deficient_votes": deficient_votes,
            "complete_votes": complete_votes,
            "confirmed_deficiency": confirmed_deficient,
            "display_score": display_score,
            "median_score": median(numeric) if numeric else None,
            "score_range": score_range,
            "confirmed_three": confirmed_three,
        }

    applicable_rows = [row for row in item_rows.values() if row["applicable"]]
    complete_item_count = sum(row["deficient_votes"] == 0 for row in applicable_rows)
    completion_rate = (
        complete_item_count / len(applicable_rows) if applicable_rows else None
    )
    confirmed_three_items = sorted(
        item_id for item_id, row in item_rows.items() if row["confirmed_three"]
    )
    proposed_three_items = sorted(
        item_id
        for item_id, row in item_rows.items()
        if any(score == 3 for score in row["scores"]) and not row["confirmed_three"]
    )

    group_rows: dict[str, dict[str, Any]] = {}
    normalized_group_values: list[float] = []
    for group_id, item_ids in groups.items():
        normalized = [
            item_rows[item_id]["median_score"] / item_meta[item_id]["max_score"]
            for item_id in item_ids
            if item_rows[item_id]["median_score"] is not None
        ]
        group_score = mean(normalized) if normalized else None
        if group_score is not None:
            normalized_group_values.append(group_score)
        group_rows[group_id] = {
            "applicable_item_count": len(normalized),
            "raw_normalized_score": group_score,
        }
    raw_quality_index = (
        100 * mean(normalized_group_values) if normalized_group_values else None
    )

    reason_codes: list[str] = []
    verification_gates = list(unconfirmed_gates)
    verification_items = sorted(
        set(
            unconfirmed_deficiencies + applicability_disagreements + score_disagreements
        )
    )
    if confirmed_gates:
        decision = "reject"
        action = "drop_from_training"
        reason_codes.append("confirmed_hard_gate")
    elif confirmed_deficiencies:
        decision = "exclude"
        action = "drop_or_repair"
        reason_codes.append("confirmed_incomplete_item")
    elif (
        unconfirmed_gates
        or unconfirmed_deficiencies
        or applicability_disagreements
        or score_disagreements
        or invalid_score_records
        or missing_judge_count
        or unavailable
        or not completed
    ):
        decision = "review"
        action = "run_targeted_verifier"
        if unconfirmed_gates:
            reason_codes.append("unconfirmed_hard_gate")
        if unconfirmed_deficiencies:
            reason_codes.append("unconfirmed_incomplete_item")
        if applicability_disagreements:
            reason_codes.append("applicability_disagreement")
        if score_disagreements:
            reason_codes.append("score_disagreement")
        if invalid_score_records:
            reason_codes.append("invalid_score_record")
        if missing_judge_count or unavailable:
            reason_codes.append("judge_result_missing")
        if not completed:
            reason_codes.append("no_completed_score")
    elif not applicable_rows:
        decision = "review"
        action = "run_targeted_verifier"
        reason_codes.append("no_applicable_scores")
    else:
        decision = "keep"
        action = "keep_for_training"
        reason_codes.append("all_applicable_items_complete")

    if confirmed_three_items:
        quality_tier = "confirmed_standout"
    elif proposed_three_items:
        quality_tier = "standout_candidate"
    else:
        quality_tier = "complete_baseline" if decision == "keep" else "not_established"

    assurance = (
        "multi_judge" if len(valid) >= required_confirmations else "single_judge"
    )
    return {
        "filter_version": FILTER_VERSION,
        "decision": decision,
        "recommended_action": action,
        "assurance": assurance,
        "quality_tier": quality_tier,
        "reason_codes": reason_codes,
        "verification_targets": {
            "gates": verification_gates,
            "items": verification_items,
        },
        "confirmed_gates": confirmed_gates,
        "confirmed_deficiencies": confirmed_deficiencies,
        "unconfirmed_deficiencies": unconfirmed_deficiencies,
        "applicability_disagreements": applicability_disagreements,
        "score_disagreements": score_disagreements,
        "unavailable_judges": unavailable,
        "items": item_rows,
        "groups": group_rows,
        "metrics": {
            "valid_judges": len(valid),
            "completed_judges": len(completed),
            "expected_judges": expected,
            "applicable_item_count": len(applicable_rows),
            "complete_item_count": complete_item_count,
            "completion_rate": completion_rate,
            "raw_quality_index": raw_quality_index,
            "confirmed_three_count": len(confirmed_three_items),
            "confirmed_three_items": confirmed_three_items,
            "proposed_three_items": proposed_three_items,
        },
    }


def resolve_targeted_verification(
    primary_result: dict[str, Any],
    verifier: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a primary ``review`` using an independent targeted verifier.

    ``verifier`` may contain ``items`` records with ``applicable``, ``complete``
    and ``clearly_failed`` booleans, plus optional ``gates`` booleans where true
    means the suspected Gate error is independently confirmed.
    """

    result = deepcopy(primary_result)
    if result.get("decision") != "review":
        raise ValueError("targeted verification only applies to review results")

    requested = result.get("verification_targets") or {}
    requested_items = set(requested.get("items") or [])
    requested_gates = set(requested.get("gates") or [])
    verifier_items = verifier.get("items") if isinstance(verifier, dict) else None
    verifier_gates = verifier.get("gates") if isinstance(verifier, dict) else None
    verifier_items = verifier_items if isinstance(verifier_items, dict) else {}
    verifier_gates = verifier_gates if isinstance(verifier_gates, dict) else {}

    unresolved_items: list[str] = []
    cleared_items: list[str] = []
    confirmed_items: list[str] = list(result.get("confirmed_deficiencies") or [])
    applicability_disagreements: list[str] = list(
        result.get("applicability_disagreements") or []
    )
    for item_id in sorted(requested_items):
        verdict = verifier_items.get(item_id)
        if not isinstance(verdict, dict):
            unresolved_items.append(item_id)
            continue
        fields = (
            verdict.get("applicable"),
            verdict.get("complete"),
            verdict.get("clearly_failed"),
        )
        if any(type(value) is not bool for value in fields):
            unresolved_items.append(item_id)
            continue
        applicable, complete, clearly_failed = fields
        if (complete and clearly_failed) or (
            not applicable and (complete or clearly_failed)
        ):
            unresolved_items.append(item_id)
            continue
        if not applicable:
            applicability_disagreements.append(item_id)
            unresolved_items.append(item_id)
        elif complete:
            cleared_items.append(item_id)
        else:
            confirmed_items.append(item_id)
            row = result.get("items", {}).get(item_id)
            if isinstance(row, dict):
                row["verifier_clearly_failed"] = clearly_failed
                row["confirmed_deficiency"] = True

    unresolved_gates: list[str] = []
    cleared_gates: list[str] = []
    confirmed_gates: list[str] = list(result.get("confirmed_gates") or [])
    for gate_id in sorted(requested_gates):
        verdict = verifier_gates.get(gate_id)
        if type(verdict) is not bool:
            unresolved_gates.append(gate_id)
        elif verdict:
            confirmed_gates.append(gate_id)
        else:
            cleared_gates.append(gate_id)

    confirmed_items = sorted(set(confirmed_items))
    confirmed_gates = sorted(set(confirmed_gates))
    applicability_disagreements = sorted(set(applicability_disagreements))
    unresolved_items = sorted(set(unresolved_items))
    unresolved_gates = sorted(set(unresolved_gates))

    if confirmed_gates:
        decision = "reject"
        action = "drop_from_training"
        reason_codes = ["confirmed_hard_gate"]
    elif confirmed_items:
        decision = "exclude"
        action = "drop_or_repair"
        reason_codes = ["confirmed_incomplete_item"]
    elif unresolved_items or unresolved_gates:
        decision = "review"
        action = "human_or_third_judge_review"
        reason_codes = ["targeted_verification_unresolved"]
        if applicability_disagreements:
            reason_codes.append("applicability_disagreement")
    else:
        decision = "keep"
        action = "keep_for_training"
        reason_codes = ["all_primary_concerns_cleared"]

    result.update(
        {
            "decision": decision,
            "recommended_action": action,
            "assurance": "targeted_multi_judge",
            "reason_codes": reason_codes,
            "confirmed_gates": confirmed_gates,
            "confirmed_deficiencies": confirmed_items,
            "applicability_disagreements": applicability_disagreements,
            "verification_targets": {
                "gates": unresolved_gates,
                "items": unresolved_items,
            },
            "targeted_verification": {
                "cleared_gates": sorted(cleared_gates),
                "cleared_items": sorted(cleared_items),
                "confirmed_gates": confirmed_gates,
                "confirmed_items": confirmed_items,
            },
        }
    )
    return result
