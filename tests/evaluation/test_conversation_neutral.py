import unittest
import jsonschema
from tests.evaluation.test_conversation_filter import fixture
from src.evaluation.roleplay_annotation_platform.conversation_neutral_review import (
    SELECTED,
    build_neutral_request,
    validate_neutral_output,
)


class NeutralTests(unittest.TestCase):
    def test_scope_and_unassessed_not_na_or_zero(self):
        body, _ = build_neutral_request(fixture())
        schema = body["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            set(SELECTED), set(schema["properties"]["scores"]["properties"])
        )
        self.assertNotIn("H1", schema["properties"]["scores"]["properties"])
        result = {
            "status": "completed",
            "scores": {
                k: {
                    "value": 2,
                    "reason": "Facts remain consistent.",
                    "turn_ids": ["A1"],
                }
                for k in SELECTED
            },
            "limitations": [],
        }
        result["scores"]["M1"]["value"] = None
        result["scores"]["L2"]["value"] = 0
        jsonschema.validate(result, schema)
        normalized = validate_neutral_output(result)
        self.assertEqual("partial", normalized["status"])
        self.assertEqual(["M1"], normalized["unassessed_items"])
        self.assertEqual(0, normalized["scores"]["L2"]["value"])
        for invalid in (True, 4, 2.5):
            result["scores"]["C1"]["value"] = invalid
            with self.assertRaises(ValueError):
                validate_neutral_output(result)


if __name__ == "__main__":
    unittest.main()
