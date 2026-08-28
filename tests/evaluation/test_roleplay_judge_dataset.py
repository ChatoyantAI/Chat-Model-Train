from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.roleplay_judge_dataset.builder import build_judge_dataset
from src.evaluation.roleplay_judge_dataset.context_builder import (
    CONTINUE_USER_QUERY,
    RestoredHistory,
    build_candidate_context,
    restore_history,
    select_mode_by_tokens,
)


def comparison_fixture(
    comparison_id: str = "comparison-a",
    *,
    trigger_type: str = "user_message",
    summary: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "comparison_id": comparison_id,
        "campaign_id": "test-campaign",
        "timestamps": {
            "started_at": "2026-08-27T01:00:00",
            "completed_at": "2026-08-27T01:00:02",
        },
        "trigger": {
            "type": trigger_type,
            "content": "Please continue from here.",
            "language": "en",
        },
        "candidates": {
            "left": {
                "model_id": 1,
                "model_name": "model-one",
                "content": "Left response",
            },
            "right": {
                "model_id": 2,
                "model_name": "model-two",
                "content": "Right response",
            },
        },
        "selection": {"preferred_side": "left"},
        "scoring": {"eligible": True, "exclusion_reason": None},
        "conversation": {"initial_greeting": "Hello {{user}}. [site](https://x.test)"},
        "character_snapshot_at_export": {
            "age": 27,
            "third_person_perspective": False,
            "card_prompt_description": None,
            "card_lorebook": None,
            "description": {
                "name": "Ari",
                "gender": "Female",
                "description": "A careful archivist.",
                "personality": "Proud and concise.",
                "scenario": "An archive at night.",
                "qa_examples": "",
                "greeting": "Fallback greeting",
            },
        },
        "persona_snapshot_at_export": {
            "nickname": "Lin",
            "gender": "Non-binary",
            "age": 24,
            "likes": "old maps",
            "dislikes": "spoilers",
            "personality": "patient",
            "pinned_memories": ["Keeps a brass key."],
        },
        "latest_memory_summary_before_generation": summary,
    }


def runtime_fixture(
    comparison_id: str = "comparison-a",
    *,
    left_context: int = 128,
    right_context: int = 128,
) -> dict:
    return {
        "schema_version": 1,
        "comparison_id": comparison_id,
        "chat_mod": 250,
        "user_gender": 2,
        "is_test_account": False,
        "prompt_experiment": {
            "name": "prompt_priority",
            "variant_value": None,
        },
        "candidates": {
            "left": {"context_window_k": left_context},
            "right": {"context_window_k": right_context},
        },
        "snapshot": {
            "captured_at_utc": "2026-08-28T00:00:00Z",
            "historical_snapshot": False,
        },
    }


def restored_turns(count: int) -> RestoredHistory:
    messages = []
    for turn in range(1, count + 1):
        message_key = f"turn-{turn}"
        messages.extend(
            [
                {
                    "increase_id": turn * 2 - 1,
                    "message_key": message_key,
                    "role": "user",
                    "content": f"User {turn}",
                    "media_type": 0,
                },
                {
                    "increase_id": turn * 2,
                    "message_key": message_key,
                    "role": "assistant",
                    "content": f"Assistant {turn}",
                    "media_type": 0,
                },
            ]
        )
    return RestoredHistory(
        messages=tuple(messages),
        image_contexts_by_message_key={},
        trigger_message_key=None,
        audit={"physical_rows": len(messages), "active_text_rows": len(messages)},
    )


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ContextBuilderTests(unittest.TestCase):
    def test_mode_boundaries_match_production(self) -> None:
        self.assertEqual(select_mode_by_tokens(50), "chat")
        self.assertEqual(select_mode_by_tokens(51), "scene")
        self.assertEqual(select_mode_by_tokens(449), "scene")
        self.assertEqual(select_mode_by_tokens(450), "longText")

    def test_small_context_ignores_summary_and_keeps_25_turns(self) -> None:
        summary = {
            "summary_text": "Earlier memory",
            "state_json": {"place": "archive"},
            "compressed_until_increase_id": 20,
        }
        result = build_candidate_context(
            comparison_fixture(summary=summary),
            "left",
            restored_turns(30),
            runtime_fixture(left_context=8),
        )

        self.assertFalse(result.audit["summary_used"])
        self.assertTrue(result.audit["summary_ignored_for_small_context"])
        self.assertEqual(result.audit["history_retained_physical_messages"], 50)
        self.assertNotIn(
            "Long-term conversation memory",
            "\n".join(message["content"] for message in result.input_messages),
        )

    def test_large_context_uses_summary_and_preserves_order(self) -> None:
        summary = {
            "summary_text": "Earlier memory",
            "state_json": {"place": "archive"},
            "compressed_until_increase_id": 4,
        }
        result = build_candidate_context(
            comparison_fixture(summary=summary),
            "left",
            restored_turns(4),
            runtime_fixture(left_context=128),
        )

        self.assertTrue(result.audit["summary_used"])
        self.assertEqual(
            [message["role"] for message in result.input_messages[:3]],
            ["system", "system", "assistant"],
        )
        self.assertIn("<user_profile>", result.input_messages[0]["content"])
        self.assertEqual(result.input_messages[2]["content"], "Hello Lin.")
        self.assertIn("<guideline>", result.input_messages[-1]["content"])

    def test_continue_uses_transient_production_query(self) -> None:
        result = build_candidate_context(
            comparison_fixture(trigger_type="continue"),
            "left",
            restored_turns(1),
            runtime_fixture(),
        )
        self.assertTrue(result.input_messages[-1]["content"].startswith(CONTINUE_USER_QUERY))

    def test_restore_history_reverses_later_edit_and_uses_completed_image(self) -> None:
        comparison = comparison_fixture()
        context_rows = [
            {
                "is_trigger_message": False,
                "active_at_generation_heuristic": True,
                "message": {
                    "increase_id": 1,
                    "message_key": "message-1",
                    "role": "user",
                    "content": "edited later",
                    "media_type": 0,
                },
            }
        ]
        image_rows = [
            {
                "message_key": "message-1",
                "sort_order": 1,
                "analysis_status": "completed",
                "analysis_text": "A brass key.",
            },
            {
                "message_key": "message-1",
                "sort_order": 2,
                "analysis_status": "failed",
                "analysis_text": "ignored",
            },
        ]
        mutation_rows = [
            {
                "event_type": "edit",
                "target_increase_id": 1,
                "before_content": "generation-time text",
                "created_at": "2026-08-27T01:01:00",
            }
        ]

        restored = restore_history(
            comparison, context_rows, image_rows, mutation_rows
        )

        self.assertEqual(restored.messages[0]["content"], "generation-time text")
        self.assertEqual(
            restored.image_contexts_by_message_key["message-1"][0]["analysis_text"],
            "A brass key.",
        )
        self.assertEqual(restored.audit["reversed_edits"], 1)


class DatasetBuilderTests(unittest.TestCase):
    def test_batch_outputs_are_blind_counted_and_non_adjacent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            output_dir = root / "output"
            raw_dir.mkdir()
            comparisons = [
                comparison_fixture("comparison-a"),
                comparison_fixture("comparison-b", trigger_type="continue"),
            ]
            write_jsonl_gz(raw_dir / "comparisons.jsonl.gz", comparisons)
            write_jsonl_gz(raw_dir / "context_messages.jsonl.gz", [])
            write_jsonl_gz(raw_dir / "message_image_contexts.jsonl.gz", [])
            write_jsonl_gz(raw_dir / "message_mutation_events.jsonl.gz", [])
            (raw_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "campaign_id": "test-campaign",
                        "cutoff_utc": "2026-08-27 00:00:00",
                        "counts": {"selected": 2, "scoring_eligible": 2},
                    }
                ),
                encoding="utf-8",
            )
            runtime_path = root / "runtime.jsonl.gz"
            write_jsonl_gz(
                runtime_path,
                [runtime_fixture("comparison-a"), runtime_fixture("comparison-b")],
            )

            manifest = build_judge_dataset(
                raw_dir=raw_dir,
                runtime_path=runtime_path,
                output_dir=output_dir,
            )

            self.assertEqual(manifest["counts"]["judge_inputs"], 4)
            with gzip.open(
                output_dir / "judge_inputs.jsonl.gz", "rt", encoding="utf-8"
            ) as handle:
                judge_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(judge_rows), 4)
            self.assertEqual(len({row["evaluation_id"] for row in judge_rows}), 4)
            forbidden = {
                "comparison_id",
                "pair_group_id",
                "model_id",
                "model_name",
                "human_selected",
                "left",
                "right",
            }
            for row in judge_rows:
                self.assertTrue(forbidden.isdisjoint(row))

            with (output_dir / "evaluation_labels.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                labels = list(csv.DictReader(handle))
            self.assertEqual(len(labels), 4)
            label_pair_by_evaluation = {
                row["evaluation_id"]: row["pair_group_id"] for row in labels
            }
            ordered_pairs = [
                label_pair_by_evaluation[row["evaluation_id"]] for row in judge_rows
            ]
            self.assertTrue(
                all(
                    ordered_pairs[index] != ordered_pairs[index - 1]
                    for index in range(1, len(ordered_pairs))
                )
            )


if __name__ == "__main__":
    unittest.main()
