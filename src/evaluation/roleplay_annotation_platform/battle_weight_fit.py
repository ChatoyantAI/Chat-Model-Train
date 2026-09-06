"""Learn and apply interpretable roleplay Battle score weights.

The model is deliberately symmetric: swapping candidates A and B negates the
logit and therefore changes ``P(A wins)`` into ``1 - P(A wins)``. Numeric atom
scores supply direction, one-sided N/A values reduce comparable coverage
without favoring either side, and Gate has an independent coefficient for vote
prediction. Gate remains a hard constraint in the data-filtering pipeline.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


BATTLE_WEIGHT_FIT_VERSION = "battle-weight-fit-v2"
DEFAULT_MAX_STEPS = 1400
DEFAULT_LEARNING_RATE = 0.035
DEFAULT_WEIGHT_REGULARIZATION = 0.025
DEFAULT_SCALE_REGULARIZATION = 0.002


@dataclass(frozen=True)
class FitDataset:
    labels: np.ndarray
    deltas: np.ndarray
    active: np.ndarray
    gates: np.ndarray
    folds: np.ndarray
    pair_kinds: tuple[str, ...]
    item_ids: tuple[str, ...]
    item_dimensions: tuple[str, ...]
    item_maxima: np.ndarray
    prior_weights: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ModelParameters:
    item_weights: np.ndarray
    temperature: float
    gate_coefficient: float


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=float)
    positive = array >= 0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    negative_exp = np.exp(array[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return float(result) if result.ndim == 0 else result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _rubric_metadata(
    rubric: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    dimensions = list(rubric.get("dimensions", []))
    items: list[dict[str, Any]] = []
    prior: list[float] = []
    dimension_count = max(1, len(dimensions))
    for dimension in dimensions:
        dimension_items = list(dimension.get("items", []))
        if not dimension_items:
            continue
        for item in dimension_items:
            scale = [int(value) for value in item.get("scale", [])]
            if not scale or max(scale) <= 0:
                raise ValueError(f"评分项 {item.get('id', '?')} 没有有效数字量表")
            items.append(
                {
                    "id": str(item["id"]),
                    "name": str(item.get("name", item["id"])),
                    "dimension_id": str(dimension["id"]),
                    "dimension_name": str(dimension.get("name", dimension["id"])),
                    "maximum": max(scale),
                }
            )
            prior.append(1.0 / dimension_count / len(dimension_items))
    if not items:
        raise ValueError("评分规则中没有可拟合的原子项")
    prior_array = np.asarray(prior, dtype=float)
    prior_array /= np.sum(prior_array)
    return items, prior_array


def build_fit_dataset(
    rubric: dict[str, Any],
    pairs: list[dict[str, Any]],
    *,
    fold_count: int = 5,
) -> tuple[FitDataset, list[dict[str, Any]]]:
    items, prior = _rubric_metadata(rubric)
    item_ids = tuple(item["id"] for item in items)
    item_dimensions = tuple(item["dimension_id"] for item in items)
    maxima = np.asarray([item["maximum"] for item in items], dtype=float)
    safe_folds = max(2, int(fold_count))

    labels: list[float] = []
    deltas: list[list[float]] = []
    active: list[list[float]] = []
    gates: list[float] = []
    folds: list[int] = []
    pair_kinds: list[str] = []
    status_counts: dict[str, int] = {}
    both_na_count = 0
    na_mismatch_count = 0
    numeric_count = 0
    pairs_with_na_mismatch = 0

    for pair in pairs:
        left_status = str(pair.get("left_status", ""))
        right_status = str(pair.get("right_status", ""))
        status_key = f"{left_status}/{right_status}"
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        gate = 0.0
        pair_kind = ""
        if left_status == "completed" and right_status == "completed":
            pair_kind = "scored"
        elif left_status == "completed" and right_status == "discarded":
            pair_kind = "gate"
            gate = 1.0
        elif left_status == "discarded" and right_status == "completed":
            pair_kind = "gate"
            gate = -1.0
        else:
            continue

        left_scores = pair.get("left_scores") if pair_kind == "scored" else {}
        right_scores = pair.get("right_scores") if pair_kind == "scored" else {}
        if pair_kind == "scored" and (
            not isinstance(left_scores, dict) or not isinstance(right_scores, dict)
        ):
            continue

        row_delta: list[float] = []
        row_active: list[float] = []
        pair_has_mismatch = False
        for item, maximum in zip(items, maxima, strict=True):
            left_record = (
                left_scores.get(item["id"]) if isinstance(left_scores, dict) else None
            )
            right_record = (
                right_scores.get(item["id"]) if isinstance(right_scores, dict) else None
            )
            left_value = (
                left_record.get("value") if isinstance(left_record, dict) else None
            )
            right_value = (
                right_record.get("value") if isinstance(right_record, dict) else None
            )
            left_na = left_value == "N/A"
            right_na = right_value == "N/A"
            if pair_kind == "gate":
                row_delta.append(0.0)
                row_active.append(0.0)
                continue
            if left_na and right_na:
                both_na_count += 1
                row_delta.append(0.0)
                row_active.append(0.0)
                continue
            row_active.append(1.0)
            if left_na != right_na:
                na_mismatch_count += 1
                pair_has_mismatch = True
                row_delta.append(0.0)
                continue
            try:
                numeric_left = int(str(left_value))
                numeric_right = int(str(right_value))
            except (TypeError, ValueError):
                # Missing legacy fields behave like a one-sided applicability conflict:
                # no direction, but the unavailable item still lowers pair reliability.
                na_mismatch_count += 1
                pair_has_mismatch = True
                row_delta.append(0.0)
                continue
            numeric_count += 1
            row_delta.append((numeric_left - numeric_right) / maximum)
        if pair_has_mismatch:
            pairs_with_na_mismatch += 1
        if pair_kind == "scored" and not any(row_active):
            continue

        selected = str(pair.get("online_selected", "")).upper()
        if selected not in {"A", "B"}:
            continue
        fold_key = str(pair.get("fold_key") or pair.get("pair_seq") or len(labels))
        fold = (
            int.from_bytes(hashlib.sha256(fold_key.encode()).digest()[:8], "big")
            % safe_folds
        )
        labels.append(1.0 if selected == "A" else 0.0)
        deltas.append(row_delta)
        active.append(row_active)
        gates.append(gate)
        folds.append(fold)
        pair_kinds.append(pair_kind)

    if len(labels) < 20 or len(set(labels)) < 2:
        raise ValueError("至少需要 20 组且同时包含 A、B 胜出的可用 Battle")
    dataset = FitDataset(
        labels=np.asarray(labels, dtype=float),
        deltas=np.asarray(deltas, dtype=float),
        active=np.asarray(active, dtype=float),
        gates=np.asarray(gates, dtype=float),
        folds=np.asarray(folds, dtype=int),
        pair_kinds=tuple(pair_kinds),
        item_ids=item_ids,
        item_dimensions=item_dimensions,
        item_maxima=maxima,
        prior_weights=prior,
        diagnostics={
            "input_pairs": len(pairs),
            "usable_pairs": len(labels),
            "scored_pairs": pair_kinds.count("scored"),
            "one_sided_gate_pairs": pair_kinds.count("gate"),
            "excluded_pairs": len(pairs) - len(labels),
            "status_pairs": status_counts,
            "numeric_item_comparisons": numeric_count,
            "both_na_items": both_na_count,
            "one_sided_na_items": na_mismatch_count,
            "pairs_with_one_sided_na": pairs_with_na_mismatch,
        },
    )
    return dataset, items


def _decode_parameters(raw: np.ndarray, item_count: int) -> ModelParameters:
    weights = _softmax(raw[:item_count])
    temperature = math.exp(float(np.clip(raw[item_count], -5.0, 6.0)))
    gate = math.exp(float(np.clip(raw[item_count + 1], -5.0, 6.0)))
    return ModelParameters(weights, temperature, gate)


def _model_outputs(
    dataset: FitDataset,
    parameters: ModelParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator = dataset.deltas @ parameters.item_weights
    denominator = dataset.active @ parameters.item_weights
    differences = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    logits = (
        parameters.temperature * differences
        + parameters.gate_coefficient * dataset.gates
    )
    probabilities = np.asarray(_sigmoid(logits), dtype=float)
    return differences, logits, probabilities


def _objective_and_gradient(
    raw: np.ndarray,
    dataset: FitDataset,
    indices: np.ndarray,
    *,
    weight_regularization: float,
    scale_regularization: float,
) -> tuple[float, np.ndarray]:
    subset = FitDataset(
        labels=dataset.labels[indices],
        deltas=dataset.deltas[indices],
        active=dataset.active[indices],
        gates=dataset.gates[indices],
        folds=dataset.folds[indices],
        pair_kinds=tuple(dataset.pair_kinds[int(index)] for index in indices),
        item_ids=dataset.item_ids,
        item_dimensions=dataset.item_dimensions,
        item_maxima=dataset.item_maxima,
        prior_weights=dataset.prior_weights,
        diagnostics={},
    )
    item_count = len(dataset.item_ids)
    parameters = _decode_parameters(raw, item_count)
    differences, logits, probabilities = _model_outputs(subset, parameters)
    labels = subset.labels
    data_loss = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
    weights = parameters.item_weights
    prior = dataset.prior_weights
    kl = float(np.sum(weights * np.log(np.maximum(weights, 1e-15) / prior)))
    target_temperature = math.log(5.0)
    scale_penalty = (raw[item_count] - target_temperature) ** 2 + raw[
        item_count + 1
    ] ** 2
    objective = (
        data_loss + weight_regularization * kl + scale_regularization * scale_penalty
    )

    errors = (probabilities - labels) / len(labels)
    denominator = subset.active @ weights
    numerator_terms = subset.deltas
    numerator = numerator_terms @ weights
    safe = denominator > 1e-12
    dloss_ddifference = errors * parameters.temperature
    derivative_weights = np.zeros_like(numerator_terms)
    derivative_weights[safe] = (
        numerator_terms[safe] * denominator[safe, None]
        - numerator[safe, None] * subset.active[safe]
    ) / np.square(denominator[safe, None])
    gradient_weights = np.sum(dloss_ddifference[:, None] * derivative_weights, axis=0)
    gradient_weights += weight_regularization * (
        np.log(np.maximum(weights, 1e-15) / prior) + 1.0
    )
    gradient_theta = weights * (
        gradient_weights - float(np.dot(gradient_weights, weights))
    )

    gradient_temperature_raw = float(
        np.sum(errors * parameters.temperature * differences)
        + 2.0 * scale_regularization * (raw[item_count] - target_temperature)
    )
    gradient_gate_raw = float(
        np.sum(errors * parameters.gate_coefficient * subset.gates)
        + 2.0 * scale_regularization * raw[item_count + 1]
    )
    gradient = np.concatenate(
        [
            gradient_theta,
            np.asarray(
                [
                    gradient_temperature_raw,
                    gradient_gate_raw,
                ]
            ),
        ]
    )
    return objective, gradient


def _fit_parameters(
    dataset: FitDataset,
    indices: np.ndarray,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_regularization: float = DEFAULT_WEIGHT_REGULARIZATION,
    scale_regularization: float = DEFAULT_SCALE_REGULARIZATION,
) -> tuple[ModelParameters, dict[str, Any]]:
    item_count = len(dataset.item_ids)
    best_overall: tuple[float, np.ndarray, int] | None = None
    for start_index in range(3):
        raw = np.concatenate(
            [
                np.log(dataset.prior_weights),
                np.asarray(
                    [
                        math.log(5.0),
                        0.0,
                    ]
                ),
            ]
        )
        if start_index:
            rng = np.random.default_rng(7103 + start_index)
            raw[:item_count] += rng.normal(0.0, 0.04, item_count)
        first_moment = np.zeros_like(raw)
        second_moment = np.zeros_like(raw)
        best = (float("inf"), raw.copy(), 0)
        stale_steps = 0
        for step in range(1, max_steps + 1):
            objective, gradient = _objective_and_gradient(
                raw,
                dataset,
                indices,
                weight_regularization=weight_regularization,
                scale_regularization=scale_regularization,
            )
            if objective + 1e-9 < best[0]:
                best = (objective, raw.copy(), step)
                stale_steps = 0
            else:
                stale_steps += 1
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > 10.0:
                gradient *= 10.0 / gradient_norm
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
            corrected_first = first_moment / (1.0 - 0.9**step)
            corrected_second = second_moment / (1.0 - 0.999**step)
            raw -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            raw[:item_count] -= np.mean(raw[:item_count])
            raw[item_count:] = np.clip(raw[item_count:], -7.0, 7.0)
            if stale_steps >= 220:
                break
        if best_overall is None or best[0] < best_overall[0]:
            best_overall = best
    assert best_overall is not None
    return _decode_parameters(best_overall[1], item_count), {
        "objective": round(best_overall[0], 8),
        "steps": best_overall[2],
        "restarts": 3,
        "regularization": {
            "weights_to_equal_dimension_prior": weight_regularization,
            "temperature_and_gate": scale_regularization,
        },
    }


def _auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if not positive or not negative:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    pair_kinds: tuple[str, ...],
) -> dict[str, Any]:
    probabilities = np.asarray(_sigmoid(logits), dtype=float)
    predictions = logits > 0
    ties = logits == 0
    correct = predictions == labels.astype(bool)
    non_tie_count = int(np.sum(~ties))
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    result: dict[str, Any] = {
        "count": len(labels),
        "auc": round(_auc(labels, logits) or 0.0, 6),
        "direction_accuracy_excluding_ties": round(
            float(np.mean(correct[~ties])) if non_tie_count else 0.0, 6
        ),
        "direction_accuracy_ties_half": round(
            float((np.sum(correct & ~ties) + 0.5 * np.sum(ties)) / len(labels)), 6
        ),
        "log_loss": round(
            float(
                -np.mean(
                    labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
                )
            ),
            6,
        ),
        "brier_score": round(float(np.mean(np.square(probabilities - labels))), 6),
        "ties": int(np.sum(ties)),
    }
    kinds = np.asarray(pair_kinds)
    for kind in ("scored", "gate"):
        mask = kinds == kind
        if np.any(mask):
            result[f"{kind}_count"] = int(np.sum(mask))
            result[f"{kind}_auc"] = round(_auc(labels[mask], logits[mask]) or 0.0, 6)
            result[f"{kind}_accuracy"] = round(float(np.mean(correct[mask])), 6)
    return result


def _parameter_report(
    parameters: ModelParameters,
    items: list[dict[str, Any]],
    fold_parameters: list[ModelParameters] | None = None,
) -> dict[str, Any]:
    dimension_weights: dict[str, float] = {}
    dimension_names: dict[str, str] = {}
    for item, weight in zip(items, parameters.item_weights, strict=True):
        dimension_id = item["dimension_id"]
        dimension_weights[dimension_id] = dimension_weights.get(
            dimension_id, 0.0
        ) + float(weight)
        dimension_names[dimension_id] = item["dimension_name"]
    item_rows = []
    for index, (item, weight) in enumerate(
        zip(items, parameters.item_weights, strict=True)
    ):
        dimension_weight = dimension_weights[item["dimension_id"]]
        row = {
            "id": item["id"],
            "name": item["name"],
            "dimension_id": item["dimension_id"],
            "dimension_name": item["dimension_name"],
            "overall_weight": round(float(weight), 8),
            "within_dimension_weight": round(float(weight) / dimension_weight, 8),
        }
        if fold_parameters:
            fold_values = np.asarray(
                [entry.item_weights[index] for entry in fold_parameters]
            )
            row["cross_fold_mean"] = round(float(np.mean(fold_values)), 8)
            row["cross_fold_std"] = round(float(np.std(fold_values)), 8)
        item_rows.append(row)
    dimension_rows = []
    for dimension_id, weight in dimension_weights.items():
        row = {
            "id": dimension_id,
            "name": dimension_names[dimension_id],
            "weight": round(weight, 8),
        }
        if fold_parameters:
            fold_values = np.asarray(
                [
                    sum(
                        float(candidate.item_weights[index])
                        for index, item in enumerate(items)
                        if item["dimension_id"] == dimension_id
                    )
                    for candidate in fold_parameters
                ]
            )
            row["cross_fold_mean"] = round(float(np.mean(fold_values)), 8)
            row["cross_fold_std"] = round(float(np.std(fold_values)), 8)
        dimension_rows.append(row)
    dimension_rows.sort(key=lambda row: row["weight"], reverse=True)
    item_rows.sort(key=lambda row: row["overall_weight"], reverse=True)
    return {
        "dimensions": dimension_rows,
        "items": item_rows,
        "temperature": round(parameters.temperature, 8),
        "gate_coefficient": round(parameters.gate_coefficient, 8),
        "gate_implied_probability": round(
            float(_sigmoid(parameters.gate_coefficient)), 8
        ),
    }


def _score_signal_report(dataset: FitDataset) -> dict[str, Any]:
    """Describe whether each atom actually separates the voted candidates.

    This is deliberately descriptive rather than another fitted coefficient.
    Sparse hard-check items can be useful filters even when they almost never
    rank two valid candidates; the report makes that distinction visible.
    """
    kinds = np.asarray(dataset.pair_kinds)
    scored_mask = kinds == "scored"
    labels = dataset.labels[scored_mask]
    deltas = dataset.deltas[scored_mask]
    active = dataset.active[scored_mask]
    rows = []
    for index, item_id in enumerate(dataset.item_ids):
        available = active[:, index] > 0
        different = available & ~np.isclose(deltas[:, index], 0.0, atol=1e-12)
        different_count = int(np.sum(different))
        direction_accuracy = (
            float(
                np.mean(
                    (deltas[different, index] > 0) == labels[different].astype(bool)
                )
            )
            if different_count
            else None
        )
        available_count = int(np.sum(available))
        item_auc = (
            _auc(labels[available], deltas[available, index])
            if available_count
            else None
        )
        rows.append(
            {
                "id": item_id,
                "dimension_id": dataset.item_dimensions[index],
                "available_pairs": available_count,
                "different_pairs": different_count,
                "different_rate": round(
                    different_count / len(labels) if len(labels) else 0.0, 6
                ),
                "direction_accuracy_when_different": (
                    round(direction_accuracy, 6)
                    if direction_accuracy is not None
                    else None
                ),
                "auc": round(float(item_auc), 6) if item_auc is not None else None,
                "signal_role": (
                    "ranking_signal"
                    if different_count >= max(20, math.ceil(len(labels) * 0.1))
                    else "sparse_check"
                ),
            }
        )

    denominator = np.sum(active, axis=1)
    equal_differences = np.divide(
        np.sum(deltas, axis=1),
        denominator,
        out=np.zeros(len(labels), dtype=float),
        where=denominator > 1e-12,
    )
    return {
        "scored_pairs": len(labels),
        "equal_weight_pair_ties": int(
            np.sum(np.isclose(equal_differences, 0.0, atol=1e-12))
        ),
        "equal_weight_mean_absolute_difference": round(
            float(np.mean(np.abs(equal_differences))) if len(labels) else 0.0,
            8,
        ),
        "items": rows,
    }


def score_battle_pair(
    rubric: dict[str, Any],
    pair: dict[str, Any],
    parameter_report: dict[str, Any],
) -> dict[str, Any]:
    """Apply a saved fit report to one A/B Battle pair.

    The input uses the same ``left_status``/``right_status`` and score record
    shape as :func:`build_fit_dataset`.  This public function is intentionally
    separate from training so a saved report can be audited and reproduced
    without fitting it again.
    """

    items, _ = _rubric_metadata(rubric)
    raw_item_parameters = parameter_report.get("items", [])
    item_weights = {
        str(row.get("id")): float(row.get("overall_weight", 0.0))
        for row in raw_item_parameters
        if isinstance(row, dict) and row.get("id") is not None
    }
    missing_weights = [item["id"] for item in items if item["id"] not in item_weights]
    if missing_weights:
        raise ValueError(f"拟合参数缺少原子项权重：{', '.join(missing_weights)}")
    weight_total = sum(max(0.0, item_weights[item["id"]]) for item in items)
    if weight_total <= 0.0:
        raise ValueError("拟合参数的原子项权重总和必须大于 0")
    item_weights = {
        item_id: max(0.0, weight) / weight_total
        for item_id, weight in item_weights.items()
    }
    try:
        temperature = max(0.0, float(parameter_report.get("temperature", 1.0)))
        gate_coefficient = max(
            0.0, float(parameter_report.get("gate_coefficient", 1.0))
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "拟合参数的 temperature 或 gate_coefficient 不是数字"
        ) from error

    left_status = str(pair.get("left_status", ""))
    right_status = str(pair.get("right_status", ""))
    common = {
        "fit_version": BATTLE_WEIGHT_FIT_VERSION,
        "left_status": left_status,
        "right_status": right_status,
    }
    if left_status == "completed" and right_status == "discarded":
        logit = gate_coefficient
        return {
            **common,
            "comparable": True,
            "pair_kind": "gate",
            "weighted_score_difference": 0.0,
            "logit": round(logit, 8),
            "a_win_probability": round(float(_sigmoid(logit)), 8),
            "predicted_winner": "A",
            "filter_winner": "A",
            "gate_direction": 1,
        }
    if left_status == "discarded" and right_status == "completed":
        logit = -gate_coefficient
        return {
            **common,
            "comparable": True,
            "pair_kind": "gate",
            "weighted_score_difference": 0.0,
            "logit": round(logit, 8),
            "a_win_probability": round(float(_sigmoid(logit)), 8),
            "predicted_winner": "B",
            "filter_winner": "B",
            "gate_direction": -1,
        }
    if left_status == "discarded" and right_status == "discarded":
        return {
            **common,
            "comparable": False,
            "pair_kind": "both_discarded",
            "reason": "A、B 都被 Gate 淘汰，不能用原子评分比较",
            "filter_winner": None,
        }
    if left_status != "completed" or right_status != "completed":
        return {
            **common,
            "comparable": False,
            "pair_kind": "technical_failure",
            "reason": "A、B 至少一侧没有可用的完成结果",
            "filter_winner": None,
        }

    left_scores = pair.get("left_scores")
    right_scores = pair.get("right_scores")
    if not isinstance(left_scores, dict) or not isinstance(right_scores, dict):
        return {
            **common,
            "comparable": False,
            "pair_kind": "technical_failure",
            "reason": "完成结果缺少原子评分",
            "filter_winner": None,
        }

    numerator = 0.0
    denominator = 0.0
    numeric_weight = 0.0
    comparable_items = 0
    both_na_items = 0
    one_sided_na_items = 0
    dimension_contributions: dict[str, float] = {}
    dimension_names: dict[str, str] = {}
    for item in items:
        item_id = item["id"]
        weight = item_weights[item_id]
        left_record = left_scores.get(item_id)
        right_record = right_scores.get(item_id)
        left_value = left_record.get("value") if isinstance(left_record, dict) else None
        right_value = (
            right_record.get("value") if isinstance(right_record, dict) else None
        )
        left_na = left_value == "N/A"
        right_na = right_value == "N/A"
        if left_na and right_na:
            both_na_items += 1
            continue

        # One-sided N/A and malformed legacy records retain denominator weight
        # but provide no direction.  This is the same shrink-to-zero rule used
        # during fitting.
        denominator += weight
        if left_na != right_na:
            one_sided_na_items += 1
            continue
        try:
            numeric_left = int(str(left_value))
            numeric_right = int(str(right_value))
        except (TypeError, ValueError):
            one_sided_na_items += 1
            continue

        comparable_items += 1
        numeric_weight += weight
        normalized_difference = (numeric_left - numeric_right) / item["maximum"]
        contribution = weight * normalized_difference
        numerator += contribution
        dimension_id = item["dimension_id"]
        dimension_names[dimension_id] = item["dimension_name"]
        dimension_contributions[dimension_id] = (
            dimension_contributions.get(dimension_id, 0.0) + contribution
        )

    if denominator <= 1e-12:
        return {
            **common,
            "comparable": False,
            "pair_kind": "scored",
            "reason": "所有原子项在 A、B 两侧都是 N/A",
            "filter_winner": None,
            "comparable_items": 0,
            "both_na_items": both_na_items,
            "one_sided_na_items": one_sided_na_items,
        }

    difference = numerator / denominator
    logit = temperature * difference
    probability = float(_sigmoid(logit))
    if logit > 1e-12:
        winner = "A"
    elif logit < -1e-12:
        winner = "B"
    else:
        winner = "tie"
    return {
        **common,
        "comparable": True,
        "pair_kind": "scored",
        "weighted_score_difference": round(difference, 8),
        "logit": round(logit, 8),
        "a_win_probability": round(probability, 8),
        "predicted_winner": winner,
        "filter_winner": winner if winner in {"A", "B"} else None,
        "gate_direction": 0,
        "comparable_weight_coverage": round(numeric_weight / denominator, 8),
        "comparable_items": comparable_items,
        "both_na_items": both_na_items,
        "one_sided_na_items": one_sided_na_items,
        "dimension_contributions": [
            {
                "id": dimension_id,
                "name": dimension_names[dimension_id],
                "score_difference_contribution": round(contribution / denominator, 8),
            }
            for dimension_id, contribution in sorted(
                dimension_contributions.items(),
                key=lambda entry: abs(entry[1]),
                reverse=True,
            )
        ],
    }


def fit_battle_weights(
    rubric: dict[str, Any],
    pairs: list[dict[str, Any]],
    *,
    fold_count: int = 5,
) -> dict[str, Any]:
    dataset, items = build_fit_dataset(rubric, pairs, fold_count=fold_count)
    all_indices = np.arange(len(dataset.labels))
    final_parameters, optimizer = _fit_parameters(dataset, all_indices)
    differences, logits, _ = _model_outputs(dataset, final_parameters)

    fold_parameters: list[ModelParameters] = []
    out_of_fold_logits = np.zeros(len(dataset.labels), dtype=float)
    seen = np.zeros(len(dataset.labels), dtype=bool)
    fold_reports = []
    for fold in sorted(set(int(value) for value in dataset.folds)):
        training = np.flatnonzero(dataset.folds != fold)
        testing = np.flatnonzero(dataset.folds == fold)
        if (
            not len(training)
            or not len(testing)
            or len(set(dataset.labels[training])) < 2
        ):
            continue
        parameters, fold_optimizer = _fit_parameters(dataset, training)
        fold_parameters.append(parameters)
        _, fold_logits, _ = _model_outputs(dataset, parameters)
        out_of_fold_logits[testing] = fold_logits[testing]
        seen[testing] = True
        fold_reports.append(
            {
                "fold": fold + 1,
                "training_count": len(training),
                "testing_count": len(testing),
                "metrics": _metrics(
                    dataset.labels[testing],
                    fold_logits[testing],
                    pair_kinds=tuple(dataset.pair_kinds[index] for index in testing),
                ),
                "optimizer": fold_optimizer,
            }
        )
    if not np.all(seen):
        raise ValueError("交叉验证折叠无法覆盖全部可用 Battle")

    baseline_parameters = ModelParameters(
        item_weights=dataset.prior_weights,
        temperature=1.0,
        gate_coefficient=1.0,
    )
    baseline_differences, _, _ = _model_outputs(dataset, baseline_parameters)
    baseline_logits = baseline_differences + dataset.gates
    warnings: list[str] = []
    cross_validation = _metrics(
        dataset.labels,
        out_of_fold_logits,
        pair_kinds=dataset.pair_kinds,
    )
    baseline = _metrics(
        dataset.labels,
        baseline_logits,
        pair_kinds=dataset.pair_kinds,
    )
    if cross_validation["auc"] <= baseline["auc"]:
        warnings.append(
            "交叉验证 AUC 没有超过等维度、等原子项基线，当前权重只适合诊断，不应直接定版。"
        )
    scored_auc = float(cross_validation.get("scored_auc", 0.0))
    scored_accuracy = float(cross_validation.get("scored_accuracy", 0.0))
    if scored_auc < 0.6:
        warnings.append(
            "双方均通过 Gate 时，原子评分预测线上选择的五折 AUC 低于 0.60；"
            "这组权重只能用于发现量表问题，不能作为数据筛选或模型排名依据。"
        )
    if scored_accuracy < 0.6:
        warnings.append(
            "双方均通过 Gate 时，折外方向命中率低于 60%；当前绝对分尚不能稳定还原用户二选一偏好。"
        )
    if fold_parameters:
        dimension_fold_std = []
        for dimension_id in sorted(set(dataset.item_dimensions)):
            indices = [
                index
                for index, value in enumerate(dataset.item_dimensions)
                if value == dimension_id
            ]
            values = [
                float(np.sum(entry.item_weights[indices])) for entry in fold_parameters
            ]
            dimension_fold_std.append(float(np.std(values)))
        if max(dimension_fold_std, default=0.0) > 0.08:
            warnings.append(
                "不同交叉验证折学到的维度权重波动较大，说明样本量或评分信号仍不足。"
            )
    recommendation_status = (
        "candidate"
        if scored_auc >= 0.65 and scored_accuracy >= 0.62
        else "diagnostic_only"
    )
    return {
        "fit_version": BATTLE_WEIGHT_FIT_VERSION,
        "method": {
            "target": "原始线上 A/B 选择",
            "model": "受约束的成对 Logistic 模型",
            "score_formula": "P(A胜)=sigmoid(temperature×加权评分差+gate_coefficient×Gate方向)",
            "weight_constraints": "维度与原子项有效权重非负且总和为1",
            "na_policy": "双方N/A移除；单边N/A不给方向并收缩总体分差",
            "gate_policy": "筛选时硬淘汰；拟合时用独立Gate系数解释线上选择",
            "fold_policy": f"按 Battle 稳定标识散列为 {fold_count} 折",
            "logprobs_used": False,
        },
        "data": dataset.diagnostics,
        "metrics": {
            "equal_weight_baseline": baseline,
            "training": _metrics(
                dataset.labels,
                logits,
                pair_kinds=dataset.pair_kinds,
            ),
            "cross_validation": cross_validation,
            "folds": fold_reports,
        },
        "recommendation": {
            "status": recommendation_status,
            "meaning": (
                "折外指标达到候选门槛，仍需独立留出集确认。"
                if recommendation_status == "candidate"
                else "当前参数仅用于诊断；不要用于自动过滤、排序或替代线上投票。"
            ),
            "candidate_thresholds": {
                "scored_auc": 0.65,
                "scored_direction_accuracy": 0.62,
            },
        },
        "score_signal": _score_signal_report(dataset),
        "parameters": _parameter_report(final_parameters, items, fold_parameters),
        "optimizer": optimizer,
        "warnings": warnings,
        "training_score_difference_summary": {
            "mean": round(float(np.mean(differences)), 8),
            "mean_absolute": round(float(np.mean(np.abs(differences))), 8),
            "maximum_absolute": round(float(np.max(np.abs(differences))), 8),
        },
    }
