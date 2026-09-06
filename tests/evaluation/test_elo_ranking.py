from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from src.evaluation.elo_ranking.io import load_votes_csv
from src.evaluation.elo_ranking.ranking import (
    BradleyTerryError,
    Vote,
    build_ranking_report,
    fit_bradley_terry,
)


class BradleyTerryFitTests(unittest.TestCase):
    def test_two_model_sixty_forty_has_expected_elo_gap(self) -> None:
        models = ("model-a", "model-b")
        wins = [[0, 120], [80, 0]]

        result = fit_bradley_terry(models, wins)

        self.assertAlmostEqual(sum(result.scores) / 2, 1000.0, places=9)
        self.assertAlmostEqual(result.scores[0] - result.scores[1], 70.436504, places=5)
        self.assertAlmostEqual(result.theta[0] - result.theta[1], 0.405465108, places=7)

    def test_vote_input_order_does_not_change_scores(self) -> None:
        votes = [
            Vote("u1", "a", "b"),
            Vote("u2", "a", "b"),
            Vote("u3", "b", "a"),
            Vote("u4", "a", "c"),
            Vote("u5", "a", "c"),
            Vote("u6", "c", "a"),
            Vote("u7", "b", "c"),
            Vote("u8", "b", "c"),
            Vote("u9", "c", "b"),
        ]

        report_a = build_ranking_report(
            votes,
            expected_model_count=3,
            bootstrap_samples=0,
        )
        report_b = build_ranking_report(
            reversed(votes),
            expected_model_count=3,
            bootstrap_samples=0,
        )

        scores_a = {row.model: row.score for row in report_a.models}
        scores_b = {row.model: row.score for row in report_b.models}
        self.assertEqual(scores_a, scores_b)
        self.assertEqual([row.model for row in report_a.models], ["a", "b", "c"])

    def test_missing_pair_is_rejected_for_complete_campaign(self) -> None:
        votes = [
            Vote("u1", "a", "b"),
            Vote("u2", "b", "a"),
            Vote("u3", "a", "c"),
            Vote("u4", "c", "a"),
        ]

        with self.assertRaisesRegex(BradleyTerryError, "missing 1 model pairs"):
            build_ranking_report(
                votes,
                expected_model_count=3,
                bootstrap_samples=0,
            )

    def test_bootstrap_is_reproducible(self) -> None:
        votes = []
        user_number = 0
        outcomes = {
            ("a", "b"): (7, 3),
            ("a", "c"): (8, 2),
            ("b", "c"): (6, 4),
        }
        for (model_a, model_b), (a_wins, b_wins) in outcomes.items():
            for _ in range(a_wins):
                user_number += 1
                votes.append(Vote(f"u{user_number}", model_a, model_b))
            for _ in range(b_wins):
                user_number += 1
                votes.append(Vote(f"u{user_number}", model_b, model_a))

        first = build_ranking_report(
            votes,
            expected_model_count=3,
            bootstrap_samples=30,
            bootstrap_seed=42,
        )
        second = build_ranking_report(
            votes,
            expected_model_count=3,
            bootstrap_samples=30,
            bootstrap_seed=42,
        )

        self.assertEqual(first.models, second.models)


class VoteCsvTests(unittest.TestCase):
    def test_duplicate_user_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "votes.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["user_id", "winner_model", "loser_model"],
                )
                writer.writeheader()
                writer.writerow(
                    {"user_id": "same", "winner_model": "a", "loser_model": "b"}
                )
                writer.writerow(
                    {"user_id": "same", "winner_model": "b", "loser_model": "a"}
                )

            with self.assertRaisesRegex(BradleyTerryError, "appears more than once"):
                load_votes_csv(path)


if __name__ == "__main__":
    unittest.main()
