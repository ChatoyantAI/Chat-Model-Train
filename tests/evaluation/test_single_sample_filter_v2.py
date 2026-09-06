from __future__ import annotations

import json
import math
import unittest

from src.evaluation.roleplay_annotation_platform.single_sample_filter_v2 import (
    attach_score_distributions,
    build_gate_messages,
    build_gate_schema,
    build_score_messages,
    build_score_schema,
    decode_score_output,
    deficient_item_ids,
    load_filter_rubric_v2,
    rubric_items,
    summarize_scores,
    validate_gate_output,
    validate_score_output,
)


class SingleSampleFilterV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rubric = load_filter_rubric_v2()

    def test_rubric_preserves_full_scored_scale(self) -> None:
        items = rubric_items(self.rubric)
        self.assertEqual("roleplay-single-sample-filter-v2.2", self.rubric["version"])
        self.assertEqual("roleplay-rubric-v0.18", self.rubric["source_rubric_version"])
        self.assertEqual(7, len(self.rubric["gates"]))
        self.assertEqual(20, len(items))
        h2 = next(item for item in items if item["id"] == "H2")
        self.assertEqual(["0", "1", "2", "3"], h2["scale"])
        self.assertTrue(h2["allows_na"])
        self.assertIn("3", h2["anchors"])
        self.assertIn("案例", h2["reference_markdown"])

    def test_gate_schema_and_boolean_direction_are_explicit(self) -> None:
        schema = build_gate_schema(self.rubric)
        self.assertEqual(
            {gate["id"] for gate in self.rubric["gates"]},
            set(schema["properties"]["gates"]["required"]),
        )
        messages = build_gate_messages(
            self.rubric, {"user_query": "Stay.", "candidate_response": "I will."}
        )
        self.assertIn("true=明确发现该错误", messages[0]["content"])
        self.assertIn("true 绝不表示通过", messages[0]["content"])
        self.assertIn("不做安全政策审核", messages[0]["content"])

    def test_score_schema_uses_single_character_codes(self) -> None:
        schema = build_score_schema(self.rubric, ["H1", "H2"])
        properties = schema["properties"]["scores"]["properties"]
        self.assertEqual(["A", "B", "C", "D", "X"], properties["H1"]["enum"])
        self.assertEqual(["A", "B", "C", "D", "X"], properties["H2"]["enum"])
        messages = build_score_messages(
            self.rubric,
            {"user_query": "Stay.", "candidate_response": "I will."},
            item_ids=["H1", "H2"],
        )
        system = messages[0]["content"]
        self.assertIn("A=0，B=1，C=2，D=3，X=N/A", system)
        self.assertIn("不要在评分前生成分析", system)
        self.assertIn("Gate 已由单独请求完成", system)
        self.assertNotIn("reason", json.dumps(schema))

    def test_validation_and_decoding(self) -> None:
        gates = {gate["id"]: False for gate in self.rubric["gates"]}
        validate_gate_output({"gates": gates}, self.rubric)
        output = {"scores": {"H1": "C", "H2": "D"}}
        validate_score_output(output, self.rubric, ["H1", "H2"])
        decoded = decode_score_output(output, self.rubric, ["H1", "H2"])
        self.assertEqual("2", decoded["H1"]["value"])
        self.assertEqual("3", decoded["H2"]["value"])

    @staticmethod
    def _entry(
        token: str, logprob: float, alternatives: dict[str, float] | None = None
    ) -> dict:
        top = alternatives or {token: logprob}
        return {
            "token": token,
            "bytes": list(token.encode("utf-8")),
            "logprob": logprob,
            "top_logprobs": [
                {"token": candidate, "logprob": probability}
                for candidate, probability in top.items()
            ],
        }

    def test_complete_legal_logits_produce_expected_score(self) -> None:
        raw = '{"scores":{"H2":"C"}}'
        entries = []
        for character in raw:
            if character == "C":
                entries.append(
                    self._entry(
                        "C",
                        -0.4,
                        {"A": -3.0, "B": -2.0, "C": -0.4, "D": -0.8, "X": -4.0},
                    )
                )
            else:
                entries.append(self._entry(character, -0.01))
        payload = {
            "choices": [{"message": {"content": raw}, "logprobs": {"content": entries}}]
        }
        scores = decode_score_output({"scores": {"H2": "C"}}, self.rubric, ["H2"])
        count = attach_score_distributions(scores, payload, raw, self.rubric)
        self.assertEqual(1, count)
        self.assertEqual(
            "complete_legal_distribution", scores["H2"]["distribution_method"]
        )
        expected = scores["H2"]["expected_score"]
        self.assertGreater(expected, 1.5)
        self.assertLess(expected, 3.0)
        self.assertTrue(
            math.isclose(1.0, sum(scores["H2"]["distribution"].values()), abs_tol=1e-7)
        )
        self.assertEqual(1.0, summarize_scores(scores, self.rubric)["logprob_coverage"])

    def test_missing_logits_keep_valid_discrete_score(self) -> None:
        raw = '{"scores":{"H2":"B"}}'
        scores = decode_score_output({"scores": {"H2": "B"}}, self.rubric, ["H2"])
        self.assertEqual(0, attach_score_distributions(scores, {}, raw, self.rubric))
        summary = summarize_scores(scores, self.rubric)
        self.assertEqual(0.0, summary["logprob_coverage"])
        self.assertAlmostEqual(100 / 3, summary["quality_score"], places=5)
        self.assertEqual(["H2"], deficient_item_ids(scores))

    def test_provider_may_omit_item_key_tokens_but_keep_score_token(self) -> None:
        raw = '{"scores":{"H1":"C"}}'
        entries = [
            self._entry('{"', -0.01),
            self._entry("scores", -0.01),
            self._entry('":{"', -0.01),
            self._entry('":"', -0.01),
            self._entry(
                "C",
                -0.2,
                {"A": -3.0, "B": -1.2, "C": -0.2, "D": -2.0, "X": -4.0},
            ),
            self._entry('"}}', -0.01),
        ]
        payload = {
            "choices": [{"message": {"content": raw}, "logprobs": {"content": entries}}]
        }
        scores = decode_score_output({"scores": {"H1": "C"}}, self.rubric, ["H1"])
        self.assertEqual(
            1, attach_score_distributions(scores, payload, raw, self.rubric)
        )
        self.assertEqual(
            "complete_legal_distribution", scores["H1"]["distribution_method"]
        )

    def test_position_mismatch_is_diagnostic_not_invalid(self) -> None:
        raw = '{"scores":{"H2":"C"}}'
        entries = [self._entry(character, -0.1) for character in raw]
        score_position = raw.index('"C"') + 1
        entries[score_position]["top_logprobs"] = [{"token": "C", "logprob": -9.0}]
        payload = {
            "choices": [{"message": {"content": raw}, "logprobs": {"content": entries}}]
        }
        scores = decode_score_output({"scores": {"H2": "C"}}, self.rubric, ["H2"])
        self.assertEqual(
            0, attach_score_distributions(scores, payload, raw, self.rubric)
        )
        self.assertEqual(
            "provider_position_mismatch", scores["H2"]["distribution_method"]
        )
        self.assertAlmostEqual(
            200 / 3, summarize_scores(scores, self.rubric)["quality_score"], places=5
        )


if __name__ == "__main__":
    unittest.main()
