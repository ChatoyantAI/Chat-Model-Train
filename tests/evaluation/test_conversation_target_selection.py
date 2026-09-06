import unittest
from src.evaluation.roleplay_annotation_platform.select_ranked_conversation_targets import (
    plan_selection,
    quantiles,
)


def row(seq, score_sum=4, count=2, candidates=3, **extra):
    return {
        "seq": seq,
        "score_sum": score_sum,
        "numeric_count": count,
        "candidate_count": candidates,
        "status": "completed",
        "complete_scores": True,
        "suspected_refusal": False,
        **extra,
    }


class TargetSelectionTests(unittest.TestCase):
    def test_exact_mean_ties_and_partial_final_conversation(self):
        source = [row(3, 6, 3), row(2, 9, 3), row(1, 4, 2)]
        result = plan_selection(source, 5)
        self.assertEqual([(2, 3), (1, 2)], [(r["seq"], n) for r, n in result])
        self.assertEqual([3, 2, 1], [r["seq"] for r in source])

    def test_failed_refused_and_no_numeric_are_not_selected(self):
        source = [
            row(1, status="failed"),
            row(2, suspected_refusal=True),
            row(3, count=0),
            row(4),
        ]
        self.assertEqual(4, plan_selection(source, 2)[0][0]["seq"])
        with self.assertRaises(ValueError):
            plan_selection(source, 4)

    def test_duplicate_ids_and_invalid_limits(self):
        with self.assertRaises(ValueError):
            plan_selection([row(1), row(1)], 1)
        with self.assertRaises(ValueError):
            plan_selection([row(1)], 0)
        self.assertEqual(1.5, quantiles([1, 2])["median"])

    def test_character_cap_spans_users_languages_and_partial_conversations(self):
        rows = [
            row(1, 6, 2, 4, character_id="popular", language="en"),
            row(2, 5, 2, 4, character_id="popular", language="es"),
            row(3, 4, 2, 4, character_id="other", language="en"),
        ]
        plan = plan_selection(rows, 8, character_cap=5)
        self.assertEqual([(1, 4), (2, 1), (3, 3)], [(r["seq"], n) for r, n in plan])
        with self.assertRaises(ValueError):
            plan_selection(rows, 10, character_cap=4)
        with self.assertRaises(ValueError):
            plan_selection(rows, 3, character_cap=0)
        with self.assertRaises(ValueError):
            plan_selection([row(1)], 2, character_cap=2)

    def test_capped_selection_keeps_highest_scores_and_skips_exhausted_role(self):
        rows = [
            row(3, 4, 2, 3, character_id="b"),
            row(2, 5, 2, 3, character_id="a"),
            row(1, 6, 2, 3, character_id="a"),
        ]
        plan = plan_selection(rows, 5, character_cap=3)
        self.assertEqual([(1, 3), (3, 2)], [(r["seq"], n) for r, n in plan])


if __name__ == "__main__":
    unittest.main()
