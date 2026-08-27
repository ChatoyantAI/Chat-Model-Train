"""Position-agnostic Bradley-Terry maximum-likelihood ranking.

The module deliberately accepts only winner/loser observations. UI position is
not part of the model and cannot affect the fitted scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence


ELO_CENTER = 1000.0
ELO_SCALE = 400.0 / math.log(10.0)


class BradleyTerryError(ValueError):
    """Raised when the vote graph cannot produce a finite ranking."""


@dataclass(frozen=True)
class Vote:
    """One valid user decision after left/right has been discarded."""

    user_id: str
    winner_model: str
    loser_model: str
    comparison_id: str = ""
    campaign_id: str = ""
    selected_at: str = ""


@dataclass(frozen=True)
class FitResult:
    """Point estimate returned by a Bradley-Terry fit."""

    models: tuple[str, ...]
    theta: tuple[float, ...]
    scores: tuple[float, ...]
    log_likelihood: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ModelRanking:
    rank: int
    model: str
    score: float
    theta: float
    wins: int
    losses: int
    battles: int
    raw_win_rate: float
    score_ci_low: float | None = None
    score_ci_high: float | None = None
    rank_ci_low: int | None = None
    rank_ci_high: int | None = None


@dataclass(frozen=True)
class PairSummary:
    model_a: str
    model_b: str
    model_a_wins: int
    model_b_wins: int
    total: int
    model_a_win_rate: float


@dataclass(frozen=True)
class RankingReport:
    models: tuple[ModelRanking, ...]
    pairs: tuple[PairSummary, ...]
    vote_count: int
    model_count: int
    pair_count: int
    log_likelihood: float
    iterations: int
    bootstrap_samples: int
    bootstrap_seed: int


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with partial-pivot Gaussian elimination."""

    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-14:
            raise BradleyTerryError(
                "Bradley-Terry Hessian is singular; check that the comparison graph is connected"
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]

        pivot_value = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= pivot_value

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for item in range(column, size + 1):
                augmented[row][item] -= factor * augmented[column][item]

    return [augmented[row][size] for row in range(size)]


def _is_strongly_connected(wins: Sequence[Sequence[int]]) -> bool:
    """A finite unregularized BT MLE requires a strongly connected win graph."""

    size = len(wins)
    if size <= 1:
        return True

    def reachable(reverse: bool) -> set[int]:
        seen = {0}
        pending = [0]
        while pending:
            current = pending.pop()
            for other in range(size):
                edge = wins[other][current] if reverse else wins[current][other]
                if edge > 0 and other not in seen:
                    seen.add(other)
                    pending.append(other)
        return seen

    return len(reachable(False)) == size and len(reachable(True)) == size


def _validate_win_matrix(wins: Sequence[Sequence[int]]) -> None:
    size = len(wins)
    if size < 2:
        raise BradleyTerryError("at least two models are required")
    if any(len(row) != size for row in wins):
        raise BradleyTerryError("win matrix must be square")
    for i, row in enumerate(wins):
        for j, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BradleyTerryError("win counts must be non-negative integers")
            if i == j and value != 0:
                raise BradleyTerryError("a model cannot beat itself")
    if not _is_strongly_connected(wins):
        raise BradleyTerryError(
            "win graph is not strongly connected, so the unregularized ranking is not finite"
        )


def _objective(
    parameters: Sequence[float], wins: Sequence[Sequence[int]], *, include_hessian: bool
) -> tuple[float, list[float], list[list[float]] | None]:
    """Return negative log-likelihood, gradient, and optional Hessian.

    The final model is anchored at theta=0 while optimizing. The result is
    centered after fitting, so the choice of anchor has no scoring effect.
    """

    size = len(wins)
    free_size = size - 1
    theta = list(parameters) + [0.0]
    loss = 0.0
    gradient = [0.0] * free_size
    hessian = [[0.0] * free_size for _ in range(free_size)] if include_hessian else None

    for i in range(size):
        for j in range(i + 1, size):
            i_wins = wins[i][j]
            j_wins = wins[j][i]
            total = i_wins + j_wins
            if total == 0:
                continue

            difference = theta[i] - theta[j]
            probability = _sigmoid(difference)
            loss += i_wins * _softplus(-difference)
            loss += j_wins * _softplus(difference)

            error = total * probability - i_wins
            curvature = total * probability * (1.0 - probability)

            if i < free_size:
                gradient[i] += error
                if hessian is not None:
                    hessian[i][i] += curvature
            if j < free_size:
                gradient[j] -= error
                if hessian is not None:
                    hessian[j][j] += curvature
            if hessian is not None and i < free_size and j < free_size:
                hessian[i][j] -= curvature
                hessian[j][i] -= curvature

    return loss, gradient, hessian


def fit_bradley_terry(
    models: Sequence[str],
    wins: Sequence[Sequence[int]],
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> FitResult:
    """Fit a standard, position-agnostic Bradley-Terry model.

    Every model starts at theta=0, meaning every matchup initially predicts
    50:50. Newton iterations minimize the joint negative log-likelihood of all
    pair outcomes. No chronological Elo update or position term is used.
    """

    if len(models) != len(wins):
        raise BradleyTerryError("model count and win-matrix size do not match")
    if len(set(models)) != len(models):
        raise BradleyTerryError("model identifiers must be unique")
    _validate_win_matrix(wins)

    parameters = [0.0] * (len(models) - 1)
    converged = False
    final_loss = math.inf
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        loss, gradient, hessian = _objective(parameters, wins, include_hessian=True)
        assert hessian is not None

        if max(abs(value) for value in gradient) <= tolerance:
            converged = True
            final_loss = loss
            break

        # A tiny diagonal term only protects the numerical solve. It is not a
        # statistical prior and is not included in the objective.
        for i in range(len(hessian)):
            hessian[i][i] += 1e-12

        step = _solve_linear_system(hessian, [-value for value in gradient])
        directional_derivative = sum(g * s for g, s in zip(gradient, step))

        step_scale = 1.0
        while step_scale >= 2.0**-30:
            candidate = [
                value + step_scale * change
                for value, change in zip(parameters, step)
            ]
            candidate_loss, _, _ = _objective(candidate, wins, include_hessian=False)
            if candidate_loss <= loss + 1e-4 * step_scale * directional_derivative:
                parameters = candidate
                final_loss = candidate_loss
                break
            step_scale *= 0.5
        else:
            raise BradleyTerryError("line search failed while fitting Bradley-Terry model")

        if max(abs(step_scale * value) for value in step) <= tolerance:
            converged = True
            break

    if not converged:
        _, gradient, _ = _objective(parameters, wins, include_hessian=False)
        converged = max(abs(value) for value in gradient) <= tolerance * 10.0
    if not converged:
        raise BradleyTerryError(
            f"Bradley-Terry fit did not converge within {max_iterations} iterations"
        )

    theta = parameters + [0.0]
    mean_theta = sum(theta) / len(theta)
    theta = [value - mean_theta for value in theta]
    scores = [ELO_CENTER + ELO_SCALE * value for value in theta]
    final_loss, _, _ = _objective(parameters, wins, include_hessian=False)

    return FitResult(
        models=tuple(models),
        theta=tuple(theta),
        scores=tuple(scores),
        log_likelihood=-final_loss,
        iterations=iteration,
        converged=True,
    )


def build_win_matrix(
    votes: Iterable[Vote], models: Sequence[str] | None = None
) -> tuple[tuple[str, ...], list[list[int]], int]:
    vote_list = list(votes)
    if models is None:
        model_set = {
            model
            for vote in vote_list
            for model in (vote.winner_model, vote.loser_model)
        }
        model_list = sorted(model_set)
    else:
        model_list = list(models)

    if len(set(model_list)) != len(model_list):
        raise BradleyTerryError("model identifiers must be unique")

    index = {model: position for position, model in enumerate(model_list)}
    wins = [[0] * len(model_list) for _ in model_list]

    for vote in vote_list:
        if vote.winner_model == vote.loser_model:
            raise BradleyTerryError(
                f"comparison {vote.comparison_id or '<unknown>'} compares a model with itself"
            )
        try:
            winner = index[vote.winner_model]
            loser = index[vote.loser_model]
        except KeyError as exc:
            raise BradleyTerryError(f"unknown model in vote: {exc.args[0]}") from exc
        wins[winner][loser] += 1

    return tuple(model_list), wins, len(vote_list)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap(
    models: Sequence[str],
    wins: Sequence[Sequence[int]],
    *,
    samples: int,
    seed: int,
) -> tuple[list[list[float]], list[list[int]]]:
    """Stratified bootstrap that preserves every unordered pair's sample size."""

    score_samples = [[] for _ in models]
    rank_samples = [[] for _ in models]
    rng = random.Random(seed)
    completed = 0
    attempts = 0
    maximum_attempts = max(samples * 5, samples + 10)

    while completed < samples and attempts < maximum_attempts:
        attempts += 1
        sampled_wins = [[0] * len(models) for _ in models]
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                i_wins = wins[i][j]
                j_wins = wins[j][i]
                total = i_wins + j_wins
                if total == 0:
                    continue
                probability = i_wins / total
                sampled_i_wins = sum(rng.random() < probability for _ in range(total))
                sampled_wins[i][j] = sampled_i_wins
                sampled_wins[j][i] = total - sampled_i_wins

        try:
            fit = fit_bradley_terry(models, sampled_wins)
        except BradleyTerryError:
            # A degenerate resample can lose graph connectivity. Retry rather
            # than introducing a prior that the main estimate does not use.
            continue

        ordered_indices = sorted(
            range(len(models)), key=lambda i: (-fit.scores[i], models[i])
        )
        ranks = [0] * len(models)
        for rank, model_index in enumerate(ordered_indices, start=1):
            ranks[model_index] = rank

        for i in range(len(models)):
            score_samples[i].append(fit.scores[i])
            rank_samples[i].append(ranks[i])
        completed += 1

    if completed != samples:
        raise BradleyTerryError(
            f"only {completed} of {samples} bootstrap samples produced a finite fit"
        )
    return score_samples, rank_samples


def build_ranking_report(
    votes: Iterable[Vote],
    *,
    expected_model_count: int | None = 11,
    require_complete_pairs: bool = True,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260827,
) -> RankingReport:
    vote_list = list(votes)
    models, wins, vote_count = build_win_matrix(vote_list)

    if expected_model_count is not None and len(models) != expected_model_count:
        raise BradleyTerryError(
            f"expected {expected_model_count} models, found {len(models)}"
        )

    pairs: list[PairSummary] = []
    missing_pairs: list[tuple[str, str]] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            i_wins = wins[i][j]
            j_wins = wins[j][i]
            total = i_wins + j_wins
            if total == 0:
                missing_pairs.append((models[i], models[j]))
                continue
            pairs.append(
                PairSummary(
                    model_a=models[i],
                    model_b=models[j],
                    model_a_wins=i_wins,
                    model_b_wins=j_wins,
                    total=total,
                    model_a_win_rate=i_wins / total,
                )
            )

    if require_complete_pairs and missing_pairs:
        preview = ", ".join(f"{a} vs {b}" for a, b in missing_pairs[:5])
        raise BradleyTerryError(
            f"missing {len(missing_pairs)} model pairs; first missing pairs: {preview}"
        )

    fit = fit_bradley_terry(models, wins)
    if bootstrap_samples < 0:
        raise BradleyTerryError("bootstrap sample count cannot be negative")

    if bootstrap_samples:
        score_samples, rank_samples = _bootstrap(
            models,
            wins,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    else:
        score_samples = [[] for _ in models]
        rank_samples = [[] for _ in models]

    ordered_indices = sorted(
        range(len(models)), key=lambda i: (-fit.scores[i], models[i])
    )
    point_ranks = [0] * len(models)
    for rank, model_index in enumerate(ordered_indices, start=1):
        point_ranks[model_index] = rank

    model_rows: list[ModelRanking] = []
    for i, model in enumerate(models):
        model_wins = sum(wins[i])
        model_losses = sum(row[i] for row in wins)
        battles = model_wins + model_losses
        if bootstrap_samples:
            score_low = _quantile(score_samples[i], 0.025)
            score_high = _quantile(score_samples[i], 0.975)
            rank_low = max(1, math.floor(_quantile(rank_samples[i], 0.025)))
            rank_high = min(len(models), math.ceil(_quantile(rank_samples[i], 0.975)))
        else:
            score_low = score_high = None
            rank_low = rank_high = None

        model_rows.append(
            ModelRanking(
                rank=point_ranks[i],
                model=model,
                score=fit.scores[i],
                theta=fit.theta[i],
                wins=model_wins,
                losses=model_losses,
                battles=battles,
                raw_win_rate=model_wins / battles,
                score_ci_low=score_low,
                score_ci_high=score_high,
                rank_ci_low=rank_low,
                rank_ci_high=rank_high,
            )
        )

    model_rows.sort(key=lambda row: row.rank)
    return RankingReport(
        models=tuple(model_rows),
        pairs=tuple(pairs),
        vote_count=vote_count,
        model_count=len(models),
        pair_count=len(pairs),
        log_likelihood=fit.log_likelihood,
        iterations=fit.iterations,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
