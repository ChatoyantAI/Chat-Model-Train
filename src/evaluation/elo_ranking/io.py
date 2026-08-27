"""CSV input validation and deterministic ranking report output."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from .ranking import BradleyTerryError, RankingReport, Vote


REQUIRED_COLUMNS = {"user_id", "winner_model", "loser_model"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_votes_csv(
    path: Path,
    *,
    campaign_id: str | None = None,
    duplicate_user_policy: Literal["error", "first"] = "error",
) -> list[Vote]:
    """Load the normalized winner/loser export.

    Position columns are intentionally unsupported: the database extraction is
    responsible for resolving the selected response to winner_model first.
    """

    votes: list[Vote] = []
    seen_users: dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise BradleyTerryError(
                f"input CSV is missing required columns: {', '.join(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            user_id = (row.get("user_id") or "").strip()
            winner = (row.get("winner_model") or "").strip()
            loser = (row.get("loser_model") or "").strip()
            row_campaign = (row.get("campaign_id") or "").strip()

            if campaign_id is not None and row_campaign != campaign_id:
                continue
            if not user_id or not winner or not loser:
                raise BradleyTerryError(
                    f"row {row_number} has an empty user_id, winner_model, or loser_model"
                )
            if winner == loser:
                raise BradleyTerryError(f"row {row_number} compares a model with itself")

            if user_id in seen_users:
                if duplicate_user_policy == "first":
                    continue
                raise BradleyTerryError(
                    f"user {user_id!r} appears more than once "
                    f"(rows {seen_users[user_id]} and {row_number})"
                )
            seen_users[user_id] = row_number

            votes.append(
                Vote(
                    comparison_id=(row.get("comparison_id") or "").strip(),
                    user_id=user_id,
                    campaign_id=row_campaign,
                    selected_at=(row.get("selected_at") or "").strip(),
                    winner_model=winner,
                    loser_model=loser,
                )
            )

    if not votes:
        suffix = f" for campaign {campaign_id!r}" if campaign_id else ""
        raise BradleyTerryError(f"input CSV contains no valid votes{suffix}")
    return votes


def write_report(
    output_dir: Path,
    report: RankingReport,
    *,
    input_path: Path,
    campaign_id: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = output_dir / "ranking.csv"
    with ranking_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rank",
            "model",
            "score",
            "theta",
            "wins",
            "losses",
            "battles",
            "raw_win_rate",
            "score_ci_low",
            "score_ci_high",
            "rank_ci_low",
            "rank_ci_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.models:
            data = asdict(row)
            for field in ("score", "theta", "raw_win_rate", "score_ci_low", "score_ci_high"):
                if data[field] is not None:
                    data[field] = f"{data[field]:.6f}"
            writer.writerow(data)

    pairwise_path = output_dir / "pairwise.csv"
    with pairwise_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_a",
            "model_b",
            "model_a_wins",
            "model_b_wins",
            "total",
            "model_a_win_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.pairs:
            data = asdict(row)
            data["model_a_win_rate"] = f"{data['model_a_win_rate']:.6f}"
            writer.writerow(data)

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "input": {
            "filename": input_path.name,
            "sha256": sha256_file(input_path),
        },
        "method": {
            "name": "standard_bradley_terry_mle",
            "position_effect": "ignored",
            "vote_weight": "one_valid_user_vote_equals_one_observation",
            "initial_theta": 0.0,
            "score_center": 1000.0,
            "score_scale": "400 / ln(10)",
            "sequential_elo_updates": False,
        },
        "counts": {
            "votes": report.vote_count,
            "models": report.model_count,
            "pairs": report.pair_count,
        },
        "fit": {
            "log_likelihood": report.log_likelihood,
            "iterations": report.iterations,
        },
        "uncertainty": {
            "method": "stratified_nonparametric_bootstrap_by_model_pair",
            "samples": report.bootstrap_samples,
            "seed": report.bootstrap_seed,
        },
        "ranking": [asdict(row) for row in report.models],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
