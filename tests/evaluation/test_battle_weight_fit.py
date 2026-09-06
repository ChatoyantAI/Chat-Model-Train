from __future__ import annotations

import unittest

from src.evaluation.roleplay_annotation_platform.battle_weight_fit import (
    build_fit_dataset,
    fit_battle_weights,
    score_battle_pair,
)


def rubric() -> dict:
    return {
        "dimensions": [
            {
                "id": "C",
                "name": "角色",
                "items": [
                    {"id": "C1", "name": "事实", "scale": ["0", "1", "2", "3"]},
                    {"id": "C2", "name": "性格", "scale": ["0", "1", "2", "3"]},
                ],
            },
            {
                "id": "H",
                "name": "张力",
                "items": [
                    {"id": "H1", "name": "承接", "scale": ["0", "1", "2"]},
                ],
            },
        ]
    }


def score(value: str) -> dict:
    return {"value": value, "evidence": [], "reason": "test"}


def parameters(*, weights: tuple[float, float, float] = (0.25, 0.25, 0.5)) -> dict:
    return {
        "items": [
            {"id": item_id, "overall_weight": weight}
            for item_id, weight in zip(("C1", "C2", "H1"), weights, strict=True)
        ],
        "temperature": 4.0,
        "gate_coefficient": 1.0,
    }


class BattleWeightFitTests(unittest.TestCase):
    def test_saved_parameters_reproduce_a_symmetric_pair_probability(self) -> None:
        pair = {
            "left_status": "completed",
            "right_status": "completed",
            "left_scores": {
                "C1": score("3"),
                "C2": score("1"),
                "H1": score("1"),
            },
            "right_scores": {
                "C1": score("0"),
                "C2": score("1"),
                "H1": score("2"),
            },
        }
        report = parameters(weights=(0.7, 0.1, 0.2))
        result = score_battle_pair(rubric(), pair, report)
        swapped = score_battle_pair(
            rubric(),
            {
                **pair,
                "left_scores": pair["right_scores"],
                "right_scores": pair["left_scores"],
            },
            report,
        )
        self.assertAlmostEqual(0.6, result["weighted_score_difference"])
        self.assertEqual("A", result["predicted_winner"])
        self.assertAlmostEqual(
            1.0,
            result["a_win_probability"] + swapped["a_win_probability"],
            places=7,
        )

    def test_pair_scoring_removes_shared_na_and_shrinks_one_sided_na(self) -> None:
        result = score_battle_pair(
            rubric(),
            {
                "left_status": "completed",
                "right_status": "completed",
                "left_scores": {
                    "C1": score("3"),
                    "C2": score("N/A"),
                    "H1": score("N/A"),
                },
                "right_scores": {
                    "C1": score("0"),
                    "C2": score("N/A"),
                    "H1": score("1"),
                },
            },
            parameters(),
        )
        self.assertAlmostEqual(
            0.25 / 0.75,
            result["weighted_score_difference"],
            places=7,
        )
        self.assertAlmostEqual(1 / 3, result["comparable_weight_coverage"], places=7)
        self.assertEqual(1, result["both_na_items"])
        self.assertEqual(1, result["one_sided_na_items"])

    def test_gate_is_hard_for_filtering_and_soft_for_vote_probability(self) -> None:
        result = score_battle_pair(
            rubric(),
            {
                "left_status": "completed",
                "right_status": "discarded",
                "left_scores": None,
                "right_scores": None,
            },
            parameters(),
        )
        self.assertEqual("gate", result["pair_kind"])
        self.assertEqual("A", result["filter_winner"])
        self.assertEqual("A", result["predicted_winner"])
        self.assertAlmostEqual(0.73105858, result["a_win_probability"])

    def test_dataset_treats_shared_and_one_sided_na_differently(self) -> None:
        pairs = []
        for index in range(20):
            pairs.append(
                {
                    "pair_seq": index + 1,
                    "fold_key": f"pair-{index}",
                    "online_selected": "A" if index % 2 == 0 else "B",
                    "left_status": "completed",
                    "right_status": "completed",
                    "left_scores": {
                        "C1": score("3"),
                        "C2": score("N/A"),
                        "H1": score("N/A"),
                    },
                    "right_scores": {
                        "C1": score("2"),
                        "C2": score("N/A"),
                        "H1": score("1"),
                    },
                }
            )
        dataset, _ = build_fit_dataset(rubric(), pairs)
        self.assertEqual([0.25, 0.25, 0.5], dataset.prior_weights.tolist())
        self.assertAlmostEqual(1 / 3, dataset.deltas[0, 0])
        self.assertEqual([1.0, 0.0, 1.0], dataset.active[0].tolist())
        self.assertEqual(20, dataset.diagnostics["both_na_items"])
        self.assertEqual(20, dataset.diagnostics["one_sided_na_items"])
        self.assertEqual(20, dataset.diagnostics["pairs_with_one_sided_na"])

    def test_joint_fit_recovers_predictive_item_and_reports_weights(self) -> None:
        fit_rubric = {
            "dimensions": [
                {
                    "id": "C",
                    "name": "角色",
                    "items": [
                        {"id": "C1", "name": "有效项", "scale": ["0", "1", "2", "3"]}
                    ],
                },
                {
                    "id": "L",
                    "name": "语言",
                    "items": [
                        {"id": "L1", "name": "反向噪声", "scale": ["0", "1", "2", "3"]}
                    ],
                },
            ]
        }
        pairs = []
        for index in range(80):
            a_wins = index % 2 == 0
            pairs.append(
                {
                    "pair_seq": index + 1,
                    "fold_key": f"synthetic-context-{index}",
                    "online_selected": "A" if a_wins else "B",
                    "left_status": "completed",
                    "right_status": "completed",
                    "left_scores": {
                        "C1": score("3" if a_wins else "0"),
                        "L1": score("0" if a_wins else "3"),
                    },
                    "right_scores": {
                        "C1": score("0" if a_wins else "3"),
                        "L1": score("3" if a_wins else "0"),
                    },
                }
            )
        report = fit_battle_weights(fit_rubric, pairs)
        item_weights = {
            entry["id"]: entry["overall_weight"]
            for entry in report["parameters"]["items"]
        }
        self.assertGreater(item_weights["C1"], item_weights["L1"])
        self.assertGreater(report["metrics"]["cross_validation"]["auc"], 0.95)
        self.assertFalse(report["method"]["logprobs_used"])
        self.assertEqual("candidate", report["recommendation"]["status"])
        signal = {entry["id"]: entry for entry in report["score_signal"]["items"]}
        self.assertEqual(80, signal["C1"]["different_pairs"])
        self.assertEqual(1.0, signal["C1"]["direction_accuracy_when_different"])

    def test_uninformative_scores_are_marked_diagnostic_only(self) -> None:
        pairs = []
        for index in range(40):
            pairs.append(
                {
                    "pair_seq": index + 1,
                    "fold_key": f"uninformative-{index}",
                    "online_selected": "A" if index % 2 == 0 else "B",
                    "left_status": "completed",
                    "right_status": "completed",
                    "left_scores": {
                        "C1": score("2"),
                        "C2": score("2"),
                        "H1": score("2"),
                    },
                    "right_scores": {
                        "C1": score("2"),
                        "C2": score("2"),
                        "H1": score("2"),
                    },
                }
            )

        report = fit_battle_weights(rubric(), pairs)

        self.assertEqual("diagnostic_only", report["recommendation"]["status"])
        self.assertTrue(any("低于 0.60" in message for message in report["warnings"]))
        self.assertEqual(40, report["score_signal"]["equal_weight_pair_ties"])


if __name__ == "__main__":
    unittest.main()
