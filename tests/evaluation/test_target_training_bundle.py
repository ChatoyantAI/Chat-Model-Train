import copy
import unittest

from src.evaluation.roleplay_annotation_platform.export_target_training_bundle import (
    make_example,
    digest,
)


def fixtures():
    packet = {
        "conversation_id": "conversation-1",
        "character_id": "role-1",
        "user_key": "hash-1",
        "role_profile": {"name": "Navigator"},
        "platform_assembly": {"version": "platform-assembly-v1"},
        "greeting": {
            "content": "Welcome aboard.",
            "source": "character_descriptions.greeting",
        },
        "system_contexts": [
            {
                "id": "S1",
                "content": "Earlier system",
                "response_language": "English",
                "template": "plain",
                "applies_to": ["A1"],
            },
            {
                "id": "S2",
                "content": "Frozen Spanish system",
                "response_language": "Spanish",
                "template": "scene",
                "applies_to": ["A2"],
            },
            {
                "id": "S3",
                "content": "Future system",
                "response_language": "English",
                "applies_to": ["A3"],
            },
        ],
        "messages": [
            {
                "id": "U1",
                "role": "user",
                "content": "Where are we going?",
                "scope": "context_only",
                "source_increase_id": 1,
            },
            {
                "id": "A1",
                "role": "assistant",
                "content": "North.",
                "scope": "context_only",
                "source_increase_id": 2,
            },
            {
                "id": "U2",
                "role": "user",
                "content": "Show me the chart.",
                "scope": "context_only",
                "source_increase_id": 3,
            },
            {
                "id": "A2",
                "role": "assistant",
                "content": "Aquí está el mapa.",
                "scope": "candidate",
                "source_increase_id": 4,
                "candidate_message_id": "target-1",
                "platform_output_requirements": {"response_language": "Spanish"},
            },
            {
                "id": "U3",
                "role": "user",
                "content": "Future fact.",
                "scope": "context_only",
                "source_increase_id": 5,
            },
            {
                "id": "A3",
                "role": "assistant",
                "content": "Future response.",
                "scope": "context_only",
                "source_increase_id": 6,
            },
        ],
    }
    ranking = {
        "selection_order": 1,
        "conversation_id": "conversation-1",
        "message_id": "target-1",
        "conversation_seq": 1,
        "target_increase_id": 4,
        "language": "es",
        "model": "source-model",
        "conversation_mean_score": 2.5,
        "conversation_numeric_items": 18,
        "conversation_na_items": 2,
        "conversation_low_items": 1,
        "conversation_major_issue": False,
    }
    return packet, ranking


class TrainingBundleTests(unittest.TestCase):
    def test_correct_system_strict_prefix_and_target_only(self):
        packet, rank = fixtures()
        original = copy.deepcopy(packet)
        ex = make_example(packet, "target-1", rank, "context-hash")
        self.assertEqual("Frozen Spanish system", ex["messages"][0]["content"])
        self.assertEqual(
            ["system", "assistant", "user", "assistant", "user"],
            [m["role"] for m in ex["messages"]],
        )
        self.assertEqual("Aquí está el mapa.", ex["target"])
        self.assertNotIn(ex["target"], [m["content"] for m in ex["messages"]])
        self.assertNotIn("Future fact.", [m["content"] for m in ex["messages"]])
        self.assertEqual(1, ex["metadata"]["background_assistant_count"])
        self.assertEqual(["U1", "A1", "U2"], ex["metadata"]["history_turn_ids"])
        self.assertEqual("Spanish", ex["metadata"]["response_language"])
        self.assertEqual(digest(ex["messages"]), ex["metadata"]["input_sha256"])
        self.assertEqual(original, packet)

    def test_missing_greeting_is_not_synthesized(self):
        packet, rank = fixtures()
        packet["greeting"]["content"] = ""
        ex = make_example(packet, "target-1", rank, "hash")
        self.assertFalse(ex["metadata"]["greeting_included"])
        self.assertEqual("user", ex["messages"][1]["role"])

    def test_duplicate_greeting_not_inserted_twice(self):
        packet, rank = fixtures()
        packet["messages"].insert(
            0,
            {
                "id": "A0",
                "role": "assistant",
                "content": "Welcome aboard.",
                "scope": "context_only",
                "source_increase_id": 0,
                "displayed_as_greeting": True,
            },
        )
        ex = make_example(packet, "target-1", rank, "hash")
        self.assertEqual(
            1, sum(m["content"] == "Welcome aboard." for m in ex["messages"])
        )
        self.assertNotIn("A0", ex["metadata"]["history_turn_ids"])

    def test_reject_ambiguous_system_language_or_background_target(self):
        packet, rank = fixtures()
        packet["system_contexts"].append(copy.deepcopy(packet["system_contexts"][1]))
        with self.assertRaises(ValueError):
            make_example(packet, "target-1", rank, "hash")
        packet, rank = fixtures()
        packet["messages"][3]["platform_output_requirements"]["response_language"] = (
            "English"
        )
        with self.assertRaises(ValueError):
            make_example(packet, "target-1", rank, "hash")
        packet, rank = fixtures()
        packet["messages"][3]["scope"] = "context_only"
        with self.assertRaises(ValueError):
            make_example(packet, "target-1", rank, "hash")

    def test_reject_old_prompt_and_invalid_current_user(self):
        packet, rank = fixtures()
        packet["role_profile"]["prompt"] = "Legacy block"
        with self.assertRaises(ValueError):
            make_example(packet, "target-1", rank, "hash")
        packet, rank = fixtures()
        packet["messages"][2]["role"] = "assistant"
        with self.assertRaises(ValueError):
            make_example(packet, "target-1", rank, "hash")


if __name__ == "__main__":
    unittest.main()
