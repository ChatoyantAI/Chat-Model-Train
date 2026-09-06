"""Offline checks for versioned documents, deployment examples and input contracts."""

import json
from pathlib import Path
import re
import unittest

from src.evaluation.roleplay_annotation_platform import build_filter_v2_docs
from src.evaluation.roleplay_annotation_platform import build_filter_v2_rubric
from src.evaluation.roleplay_annotation_platform.build_conversation_docs import TARGET
from src.evaluation.roleplay_annotation_platform.conversation_filter import (
    rubric_document,
)
from src.evaluation.roleplay_annotation_platform.llm_judge import (
    SAMPLE_PACKET_FIELD_NOTE,
)


ROOT = Path(__file__).resolve().parents[2]


class EvaluationReleaseContracts(unittest.TestCase):
    def test_conversation_document_matches_runtime_without_omissions(self):
        self.assertEqual(TARGET.read_text(encoding="utf-8"), rubric_document() + "\n")

    def test_v2_json_and_complete_document_match_sources(self):
        self.assertEqual(
            build_filter_v2_rubric.build(),
            json.loads(build_filter_v2_rubric.TARGET_PATH.read_text(encoding="utf-8")),
        )
        self.assertEqual(
            build_filter_v2_docs.TARGET.read_text(encoding="utf-8"),
            build_filter_v2_docs.PREAMBLE
            + build_filter_v2_docs.SOURCE.read_text(encoding="utf-8").rstrip()
            + "\n",
        )

    def test_n1_prompt_defers_numeric_scale_to_current_rubric(self):
        self.assertIn("当前量表允许的数字档位", SAMPLE_PACKET_FIELD_NOTE)
        self.assertNotIn("0、1、2 评分", SAMPLE_PACKET_FIELD_NOTE)
        self.assertIn("不允许使用 N/A", SAMPLE_PACKET_FIELD_NOTE)

    def test_environment_examples_do_not_enable_paid_keys(self):
        examples = list((ROOT / "configs/evaluation").glob("*.env.example"))
        examples.append(
            ROOT / "src/evaluation/roleplay_annotation_platform/llm-judge.env.example"
        )
        for path in examples:
            for line in path.read_text(encoding="utf-8").splitlines():
                if "API_KEY=" in line and not line.startswith("#"):
                    with self.subTest(file=path.name):
                        self.assertEqual("", line.split("=", 1)[1].strip())

    def test_current_document_links_resolve(self):
        paths = list((ROOT / "docs/evaluation").glob("*.md"))
        paths += [
            ROOT / "src/evaluation/README.md",
            ROOT / "src/evaluation/roleplay_annotation_platform/README.md",
            ROOT / "configs/evaluation/README.md",
        ]
        for path in paths:
            for target in re.findall(
                r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")
            ):
                target = target.split("#", 1)[0]
                if not target or re.match(r"^[a-z]+:", target):
                    continue
                with self.subTest(file=path.name, target=target):
                    self.assertTrue((path.parent / target).exists())


if __name__ == "__main__":
    unittest.main()
