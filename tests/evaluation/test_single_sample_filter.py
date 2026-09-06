from __future__ import annotations

import copy
import unittest

from src.evaluation.roleplay_annotation_platform.single_sample_filter import (
    filter_single_sample,
    resolve_targeted_verification,
)


def rubric() -> dict:
    return {
        "dimensions": [
            {
                "id": "character",
                "items": [
                    {
                        "id": "C1",
                        "name": "事实",
                        "scale": ["0", "2"],
                        "allows_na": False,
                    },
                    {
                        "id": "C2",
                        "name": "人设",
                        "scale": ["0", "1", "2", "3"],
                        "starred": True,
                    },
                ],
            },
            {
                "id": "language",
                "items": [
                    {
                        "id": "L1",
                        "name": "语言",
                        "scale": ["0", "1", "2", "3"],
                        "starred": True,
                    },
                    {"id": "L2", "name": "通顺", "scale": ["0", "1", "2"]},
                ],
            },
        ]
    }


def judgment(judge_id: str, values: dict[str, str], *, gate: bool = False) -> dict:
    return {
        "judge_id": judge_id,
        "status": "discarded" if gate else "completed",
        "failed_gate_ids": ["G-P1"] if gate else [],
        "scores": {item_id: {"value": value} for item_id, value in values.items()},
    }


class SingleSampleFilterTests(unittest.TestCase):
    def test_one_clean_judge_can_keep(self) -> None:
        result = filter_single_sample(
            rubric(),
            [judgment("a", {"C1": "2", "C2": "2", "L1": "3", "L2": "2"})],
        )
        self.assertEqual("keep", result["decision"])
        self.assertEqual("single_judge", result["assurance"])
        self.assertEqual("standout_candidate", result["quality_tier"])

    def test_one_negative_routes_to_targeted_verifier(self) -> None:
        result = filter_single_sample(
            rubric(),
            [judgment("a", {"C1": "2", "C2": "1", "L1": "2", "L2": "2"})],
        )
        self.assertEqual("review", result["decision"])
        self.assertEqual(["C2"], result["verification_targets"]["items"])

    def test_two_matching_negatives_exclude_without_compensation(self) -> None:
        high = {"C1": "2", "C2": "1", "L1": "3", "L2": "2"}
        result = filter_single_sample(
            rubric(),
            [judgment("a", high), judgment("b", copy.deepcopy(high))],
        )
        self.assertEqual("exclude", result["decision"])
        self.assertEqual(["C2"], result["confirmed_deficiencies"])

    def test_two_matching_gates_reject(self) -> None:
        result = filter_single_sample(
            rubric(),
            [judgment("a", {}, gate=True), judgment("b", {}, gate=True)],
        )
        self.assertEqual("reject", result["decision"])
        self.assertEqual(["G-P1"], result["confirmed_gates"])

    def test_one_gate_requires_verification(self) -> None:
        result = filter_single_sample(rubric(), [judgment("a", {}, gate=True)])
        self.assertEqual("review", result["decision"])
        self.assertEqual(["G-P1"], result["verification_targets"]["gates"])

    def test_two_threes_confirm_standout(self) -> None:
        values = {"C1": "2", "C2": "3", "L1": "2", "L2": "2"}
        result = filter_single_sample(
            rubric(),
            [judgment("a", values), judgment("b", copy.deepcopy(values))],
        )
        self.assertEqual("keep", result["decision"])
        self.assertEqual("confirmed_standout", result["quality_tier"])
        self.assertEqual(["C2"], result["metrics"]["confirmed_three_items"])

    def test_na_numeric_conflict_requires_review(self) -> None:
        first = {"C1": "2", "C2": "N/A", "L1": "2", "L2": "2"}
        second = {"C1": "2", "C2": "2", "L1": "2", "L2": "2"}
        result = filter_single_sample(
            rubric(), [judgment("a", first), judgment("b", second)]
        )
        self.assertEqual("review", result["decision"])
        self.assertEqual(["C2"], result["applicability_disagreements"])

    def test_no_valid_judgment_retries(self) -> None:
        result = filter_single_sample(rubric(), [{"judge_id": "a", "status": "failed"}])
        self.assertEqual("retry", result["decision"])

    def test_targeted_verifier_can_clear_a_single_low_score(self) -> None:
        primary = filter_single_sample(
            rubric(),
            [judgment("a", {"C1": "2", "C2": "1", "L1": "2", "L2": "2"})],
        )
        resolved = resolve_targeted_verification(
            primary,
            {
                "items": {
                    "C2": {
                        "applicable": True,
                        "complete": True,
                        "clearly_failed": False,
                    }
                }
            },
        )
        self.assertEqual("keep", resolved["decision"])
        self.assertEqual(["C2"], resolved["targeted_verification"]["cleared_items"])

    def test_targeted_verifier_can_confirm_exclusion(self) -> None:
        primary = filter_single_sample(
            rubric(),
            [judgment("a", {"C1": "2", "C2": "1", "L1": "2", "L2": "2"})],
        )
        resolved = resolve_targeted_verification(
            primary,
            {
                "items": {
                    "C2": {
                        "applicable": True,
                        "complete": False,
                        "clearly_failed": False,
                    }
                }
            },
        )
        self.assertEqual("exclude", resolved["decision"])
        self.assertEqual(["C2"], resolved["confirmed_deficiencies"])


if __name__ == "__main__":
    unittest.main()
