from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx
import jsonschema
from fastapi.testclient import TestClient

from src.evaluation.roleplay_annotation_platform import conversation_filter as rubric
from src.evaluation.roleplay_annotation_platform import conversation_filter_app as web
from src.evaluation.roleplay_annotation_platform.conversation_filter_data import (
    DDL,
    pack_conversation,
    insert_conversation,
)
from src.evaluation.roleplay_annotation_platform.conversation_filter_enrich import (
    enrich,
)
from src.evaluation.roleplay_annotation_platform.llm_judge import load_judge_config
from src.evaluation.roleplay_annotation_platform.conversation_platform_context import (
    assemble_platform_context,
)
from src.evaluation.roleplay_annotation_platform.migrate_conversation_platform import (
    clean_attempts,
)


def fixture():
    turns = [
        {
            "turn_order": i,
            "user_increase_id": i * 2,
            "assistant_increase_id": i * 2 + 1,
            "user_content": f"Question {i}",
            "assistant_content": f"Answer {i}",
            "assistant_model": "hidden-model",
        }
        for i in range(1, 5)
    ]
    bundle = {
        "conversation_id": "conversation-1",
        "character_id": "role-1",
        "user_id": "private-user",
        "turns": turns,
        "character_name": "Rowan",
        "character_personality": "Patient",
        "character_greeting": "DO NOT SUBSTITUTE THIS OPENING",
    }
    candidates = [
        {
            "target_increase_id": i * 2 + 1,
            "query_increase_id": i * 2,
            "query_text": f"Question {i}",
            "target_text": f"Answer {i}",
            "message_id": f"target-{i}",
            "language": "en",
            "model": "hidden-model",
        }
        for i in (1, 3)
    ]
    return pack_conversation(bundle, candidates)


def output():
    return {
        "scores": {
            item["id"]: {
                "value": 2,
                "reason": "The reply follows the established context.",
                "turn_ids": ["A1"],
            }
            for item in rubric.ITEMS
        },
        "issues": [],
        "data_limitations": [],
    }


class DomainTests(unittest.TestCase):
    def test_stored_prompt_is_excluded_at_all_boundaries(self):
        p = fixture()
        p["role_profile"]["prompt"] = "FORBIDDEN_DATABASE_PROMPT under 100 tokens"
        self.assertNotIn(
            "FORBIDDEN_DATABASE_PROMPT", json.dumps(rubric.judge_packet(p))
        )
        enriched = enrich(
            p,
            {},
            None,
            {"prompt": "NEW_FORBIDDEN_DATABASE_PROMPT", "personality": "Patient"},
            "reader",
            {},
        )
        self.assertNotIn("prompt", enriched["role_profile"])
        bundle = {
            "conversation_id": "x",
            "user_id": "u",
            "character_id": "r",
            "character_prompt": "ONLY_PROMPT",
            "turns": [
                {
                    "turn_order": 1,
                    "user_increase_id": 2,
                    "assistant_increase_id": 3,
                    "user_content": "Hi",
                    "assistant_content": "Hello",
                }
            ],
        }
        candidates = [
            {
                "target_increase_id": 3,
                "query_increase_id": 2,
                "query_text": "Hi",
                "target_text": "Hello",
                "message_id": "a",
                "language": "en",
            }
        ]
        packed = pack_conversation(bundle, candidates)
        self.assertNotIn("prompt", packed["role_profile"])

    def test_shared_platform_templates_language_persona_and_greeting(self):
        for maximum, third, test, variant, expected in [
            (50, False, False, None, "prompt_chat"),
            (250, False, False, None, "prompt_scene"),
            (650, False, False, "user_priority", "prompt_long_text_user_priority"),
            (250, True, False, None, "prompt_collaborative"),
            (250, False, True, None, "prompt_green"),
            (650, True, True, None, "prompt_green_collaborative"),
        ]:
            with self.subTest(template=expected):
                p = fixture()
                p["role_profile"]["prompt"] = "FORBIDDEN_STORED_PROMPT"
                p["persona"] = {"nickname": "Alex", "pinned_memories": "[]"}
                p["greeting"]["content"] = (
                    "*{{char}} waves to {{user}}.* ![](https://example.test/image.png)"
                )
                p["messages"][0]["content"] = "Merhaba"
                assembled = assemble_platform_context(
                    p,
                    {
                        "chat_mod": maximum,
                        "third_person_perspective": third,
                        "is_test_account": test,
                        "prompt_variant": variant,
                        "user_gender": 0,
                    },
                )
                self.assertEqual(expected, assembled["system_contexts"][0]["template"])
                self.assertEqual(
                    "English",
                    assembled["messages"][1]["platform_output_requirements"][
                        "response_language"
                    ],
                )
                self.assertIn("Alex", assembled["system_contexts"][0]["content"])
                self.assertNotIn(
                    "[Pinned Memories]", assembled["system_contexts"][0]["content"]
                )
                self.assertEqual(
                    "*Rowan waves to Alex.*", assembled["greeting"]["content"]
                )
                self.assertEqual(
                    p["greeting"]["content"], assembled["greeting"]["source_content"]
                )
                self.assertNotIn(
                    "FORBIDDEN_STORED_PROMPT",
                    json.dumps(rubric.judge_packet(assembled)),
                )
                self.assertEqual(
                    ["A1", "A3"], assembled["system_contexts"][0]["applies_to"]
                )
                self.assertIn("FORBIDDEN_STORED_PROMPT", p["role_profile"]["prompt"])

    def test_old_request_cleanup_keeps_score_and_marks_modified_display(self):
        c = sqlite3.connect(":memory:")
        c.execute(
            "CREATE TABLE attempts(id INTEGER PRIMARY KEY,request_json TEXT,request_hash TEXT,raw_json TEXT)"
        )
        source = fixture()
        source["role_profile"]["prompt"] = "FORBIDDEN_STORED_PROMPT"
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "请评价以下对话的候选回复组：\n" + json.dumps(source),
                }
            ]
        }
        raw = {"parsed_json": output()}
        c.execute(
            "INSERT INTO attempts VALUES(1,?,?,?)",
            (json.dumps(body), "original", json.dumps(raw)),
        )
        self.assertEqual(1, clean_attempts(c))
        request, fingerprint, saved = c.execute(
            "SELECT request_json,request_hash,raw_json FROM attempts"
        ).fetchone()
        self.assertNotIn("FORBIDDEN_STORED_PROMPT", request)
        self.assertEqual(output(), json.loads(saved)["parsed_json"])
        self.assertTrue(json.loads(saved)["request_display_sanitized"])
        self.assertEqual("original", json.loads(saved)["original_request_hash"])
        self.assertEqual(0, clean_attempts(c))
        c.close()

    def test_source_scope_and_no_synthetic_greeting(self):
        p = fixture()
        self.assertEqual(2, p["candidate_count"])
        self.assertEqual(
            ["U1", "A1", "U2", "A2", "U3", "A3"], [m["id"] for m in p["messages"]]
        )
        self.assertEqual("context_only", p["messages"][3]["scope"])
        self.assertEqual("", p["greeting"]["content"])
        raw = json.dumps(rubric.judge_packet(p))
        self.assertNotIn("private-user", raw)
        self.assertNotIn("hidden-model", raw)
        self.assertNotIn("target-1", raw)

    def test_role_card_opening_is_sent_verbatim_after_user_approval(self):
        p = enrich(
            fixture(),
            {},
            None,
            {"personality": "Patient"},
            "canonical",
            {"greeting": "CURRENT CARD ONLY", "language": "en"},
        )
        self.assertEqual("CURRENT CARD ONLY", p["greeting"]["content"])
        self.assertIn("CURRENT CARD ONLY", json.dumps(rubric.judge_packet(p)))
        self.assertFalse(any("Greeting" in g for g in p["context_basis"]["known_gaps"]))
        p = enrich(
            fixture(),
            {"initial_greeting": "SAVED CONVERSATION"},
            None,
            {},
            "canonical",
            {},
        )
        self.assertTrue(p["greeting"]["historical_snapshot"])
        self.assertEqual("SAVED CONVERSATION", p["greeting"]["content"])

    def test_schema_and_lenient_validation(self):
        o = output()
        jsonschema.validate(o, rubric.output_schema())
        o["scores"]["C4"]["value"] = "N/A"
        o["scores"]["C2"]["value"] = "3"
        o["scores"]["C2"]["turn_ids"] = ["unrecognized"]
        result = rubric.validate_output(o, fixture())
        self.assertEqual(3, result["scores"]["C2"]["value"])
        self.assertTrue(result["warnings"])
        for bad in (True, 4, "N/A"):
            o["scores"]["N1"]["value"] = bad
            with self.assertRaises(ValueError):
                rubric.validate_output(o, fixture())

    def test_na_is_not_zero_and_local_issues_do_not_delete(self):
        o = output()
        o["scores"]["H2"]["value"] = "N/A"
        result = rubric.validate_output(o, fixture())
        self.assertEqual("eligible", rubric.screening(result)["label"])
        result["scores"]["L1"]["value"] = 1
        self.assertEqual("needs_review", rubric.screening(result)["label"])
        self.assertEqual("eligible", rubric.screening(result, minimum=1)["label"])


class FakeCatalog:
    def __init__(self, config):
        self.model_ids = ("test/model",)
        self.reasoning_model_ids = ()

    async def refresh(self):
        return self.public_snapshot()

    def public_snapshot(self):
        return {
            "models": [
                {"id": "test/model", "label": "Mock model", "gateway": "openrouter"}
            ]
        }


async def no_start(self):
    pass


class AppTests(unittest.TestCase):
    def test_deepinfra_direct_routing_estimated_cost_and_secret_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conversations.sqlite3"
            c = sqlite3.connect(source)
            c.executescript(DDL)
            insert_conversation(c, 1, fixture())
            c.execute(
                "INSERT INTO metadata VALUES ('manifest',?)",
                (
                    json.dumps(
                        {
                            "ready": True,
                            "source_sha256": "coarse",
                            "dataset_sha256": "snapshot",
                            "report": {},
                            "languages": {"en": 1},
                        }
                    ),
                ),
            )
            c.commit()
            c.close()
            config = replace(
                load_judge_config(),
                api_key="mock-key-1",
                models=("test/model",),
                tatu_api_key="",
            )
            direct_key = "mock-deepinfra-secret"
            headers = {"X-Requested-With": "conversation-filter"}
            web.STATE.clear()
            with (
                patch.dict(os.environ, {"DEEPINFRA_API_KEY": direct_key}),
                patch.object(web, "SAMPLE_DB", source),
                patch.object(web, "RESULT_DB", root / "results/jobs.sqlite3"),
                patch.object(web, "load_judge_config", return_value=config),
                patch.object(web, "OpenRouterModelCatalog", FakeCatalog),
                patch.object(web.Runtime, "start", no_start),
                TestClient(web.app) as client,
            ):
                client.post(
                    "/api/login", json={"username": "direct-tester"}, headers=headers
                ).raise_for_status()
                public = client.get("/api/models").json()
                self.assertEqual(rubric.DEEPINFRA_MODEL, public["default_model"])
                self.assertEqual("deepinfra", public["models"][0]["gateway"])
                self.assertNotIn(direct_key, json.dumps(public))
                preview = client.get(
                    "/api/request-preview/1", params={"model": rubric.DEEPINFRA_MODEL}
                ).json()
                self.assertEqual(rubric.DEEPINFRA_URL, preview["endpoint"])
                self.assertEqual("zai-org/GLM-5.3-Flash", preview["body"]["model"])
                self.assertNotIn("provider", preview["body"])
                self.assertNotIn("reasoning", preview["body"])
                self.assertNotIn(direct_key, json.dumps(preview))
                self.assertNotIn(config.api_key, json.dumps(preview))
                job = client.post(
                    "/api/jobs",
                    json={
                        "model": rubric.DEEPINFRA_MODEL,
                        "scope": {"mode": "current", "seq": 1},
                        "budget": 1,
                        "concurrency": 1,
                        "nonce": "direct-1",
                        "acknowledge_snapshot": True,
                    },
                    headers=headers,
                ).json()["job_id"]
                runtime = web.STATE["runtime"]
                client.portal.call(runtime.client.aclose)

                def handle(request):
                    self.assertEqual(rubric.DEEPINFRA_URL, str(request.url))
                    self.assertEqual(
                        "Bearer " + direct_key, request.headers["Authorization"]
                    )
                    self.assertNotIn(config.api_key, str(request.headers))
                    o = output()
                    o["scores"]["C1"]["reason"] = "echoed secret " + direct_key
                    return httpx.Response(
                        200,
                        json={
                            "model": "zai-org/GLM-5.3-Flash",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "content": json.dumps(o),
                                        "reasoning": "not for display",
                                    },
                                }
                            ],
                            "usage": {
                                "estimated_cost": 0.0123,
                                "prompt_tokens": 120,
                                "completion_tokens": 25,
                            },
                        },
                    )

                runtime.client = httpx.AsyncClient(
                    transport=httpx.MockTransport(handle)
                )
                client.portal.call(runtime.evaluate, *runtime.claim())
                runtime.claim()
                info = client.get(f"/api/jobs/{job}").json()
                self.assertEqual("completed", info["status"])
                self.assertEqual(0, info["unknown_cost_attempts"])
                self.assertAlmostEqual(0.0123, info["spent"])
                result = client.get(f"/api/jobs/{job}/result/1").json()
                self.assertNotIn(direct_key, json.dumps(result))
                self.assertEqual(
                    "usage.estimated_cost", result["attempts"][0]["raw"]["cost_basis"]
                )
                self.assertEqual(
                    "DeepInfra (direct)", result["attempts"][0]["raw"]["provider"]
                )
                # The OpenRouter option remains on its own endpoint and key.
                endpoint, key, gateway = web.model_transport("test/model")
                self.assertEqual("openrouter", gateway)
                self.assertEqual(config.api_key, key)
                self.assertNotEqual(rubric.DEEPINFRA_URL, endpoint)
                web.STATE["config"] = replace(config, api_key="")
                self.assertTrue(client.get("/api/me").json()["api_configured"])
                self.assertEqual(
                    [rubric.DEEPINFRA_MODEL],
                    [m["id"] for m in client.get("/api/models").json()["models"]],
                )

    def test_isolated_preview_queue_retry_export_and_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "conversations.sqlite3"
            with sqlite3.connect(source) as c:
                c.executescript(DDL)
                for seq in (1, 2, 3):
                    p = fixture()
                    p["conversation_id"] = f"conversation-{seq}"
                    insert_conversation(c, seq, p)
                manifest = {
                    "ready": True,
                    "source_sha256": "coarse-hash",
                    "dataset_sha256": "enriched-hash",
                    "report": {"conversations": 3},
                    "languages": {"en": 3},
                }
                c.execute(
                    "INSERT INTO metadata VALUES ('manifest',?)",
                    (json.dumps(manifest),),
                )
            c.close()
            config = replace(
                load_judge_config(),
                api_key="mock-key-2",
                models=("test/model",),
                openrouter_url="https://mock.invalid/v1/chat/completions",
                tatu_api_key="",
            )
            web.STATE.clear()
            headers = {"X-Requested-With": "conversation-filter"}
            with (
                patch.object(web, "SAMPLE_DB", source),
                patch.object(web, "RESULT_DB", root / "results/jobs.sqlite3"),
                patch.object(web, "load_judge_config", return_value=config),
                patch.object(web, "OpenRouterModelCatalog", FakeCatalog),
                patch.object(web.Runtime, "start", no_start),
                patch.object(web, "retry_delay", return_value=0),
                TestClient(web.app) as client,
            ):
                self.assertEqual(401, client.get("/api/conversations").status_code)
                self.assertEqual(
                    403,
                    client.post("/api/login", json={"username": "Alice"}).status_code,
                )
                self.assertEqual(
                    200,
                    client.post(
                        "/api/login", json={"username": "Alice"}, headers=headers
                    ).status_code,
                )
                alice = dict(client.cookies)
                preview = client.get("/api/request-preview/1?model=test/model").json()
                self.assertNotIn(config.api_key, json.dumps(preview))
                self.assertNotIn("temperature", preview["body"])
                self.assertNotIn("max_tokens", preview["body"])
                self.assertNotIn("logprobs", preview["body"])
                self.assertEqual(
                    20, len(preview["schema"]["properties"]["scores"]["properties"])
                )
                scope = {
                    "mode": "random",
                    "count": 2,
                    "seed": 5,
                    "character_cap": 0,
                    "user_cap": 0,
                }
                a = client.post(
                    "/api/selection-preview", json=scope, headers=headers
                ).json()
                b = client.post(
                    "/api/selection-preview", json=scope, headers=headers
                ).json()
                self.assertEqual(a, b)
                self.assertFalse(a["paid_request"])
                self.assertEqual([], client.get("/api/jobs").json()["jobs"])
                payload = {
                    "scope": {"mode": "range", "start": 1, "end": 3},
                    "model": "test/model",
                    "budget": 0.001,
                    "concurrency": 1,
                    "acknowledge_snapshot": True,
                    "nonce": "unique-1",
                }
                created = client.post("/api/jobs", json=payload, headers=headers)
                self.assertEqual(200, created.status_code, created.text)
                job = created.json()["job_id"]
                self.assertEqual(
                    job,
                    client.post("/api/jobs", json=payload, headers=headers).json()[
                        "job_id"
                    ],
                )
                runtime = web.STATE["runtime"]
                client.portal.call(runtime.client.aclose)
                calls = []

                def handle(request):
                    calls.append(json.loads(request.content))
                    if len(calls) == 1:
                        return httpx.Response(
                            429,
                            json={"error": {"message": "Rate limited"}},
                            headers={"Retry-After": "0"},
                        )
                    return httpx.Response(
                        200,
                        json={
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "content": json.dumps(output()),
                                        "reasoning": "HIDDEN REASONING",
                                    },
                                }
                            ],
                            "provider": "mock",
                            "usage": {
                                "cost": 0.001,
                                "prompt_tokens": 100,
                                "completion_tokens": 50,
                            },
                        },
                    )

                runtime.client = httpx.AsyncClient(
                    transport=httpx.MockTransport(handle)
                )
                client.portal.call(runtime.evaluate, *runtime.claim())
                r = client.get(f"/api/jobs/{job}/result/1").json()
                self.assertEqual("queued", r["status"])
                client.portal.call(runtime.evaluate, *runtime.claim())
                r = client.get(f"/api/jobs/{job}/result/1").json()
                self.assertEqual("completed", r["status"])
                self.assertEqual(2, len(r["attempts"]))
                self.assertNotIn("HIDDEN REASONING", json.dumps(r))
                self.assertIsNone(runtime.claim())
                detail = client.get(f"/api/jobs/{job}").json()
                self.assertEqual("paused", detail["status"])
                self.assertEqual("enriched-hash", detail["data_hash"])
                self.assertEqual(1, detail["eligible"])
                rows = client.get(
                    f"/api/jobs/{job}/export?eligible_only=true"
                ).text.splitlines()
                self.assertEqual(1, len(rows))
                self.assertEqual(
                    ["target-1", "target-3"],
                    json.loads(rows[0])["candidate_message_ids"],
                )
                self.assertEqual(
                    calls[-1],
                    client.get(f"/api/request-preview/1?job_id={job}").json()["body"],
                )
                client.post("/api/login", json={"username": "Bob"}, headers=headers)
                self.assertEqual(404, client.get(f"/api/jobs/{job}").status_code)
                self.assertEqual([], client.get("/api/jobs").json()["jobs"])
                client.cookies.clear()
                client.cookies.update(alice)
                # A 2xx score without reported cost is saved, but stops further
                # dispatch instead of pretending the request was free.
                resumed = client.post(
                    f"/api/jobs/{job}/control",
                    json={"action": "resume", "budget": 1},
                    headers=headers,
                )
                self.assertEqual(200, resumed.status_code)
                client.portal.call(runtime.client.aclose)
                runtime.client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(
                            200,
                            json={
                                "choices": [
                                    {
                                        "message": {"content": json.dumps(output())},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {},
                            },
                        )
                    )
                )
                client.portal.call(runtime.evaluate, *runtime.claim())
                detail = client.get(f"/api/jobs/{job}").json()
                self.assertEqual("paused", detail["status"])
                self.assertEqual(1, detail["unknown_cost_attempts"])
                self.assertEqual(
                    "completed",
                    client.get(f"/api/jobs/{job}/result/2").json()["status"],
                )
                # Restart does not turn a possibly billed in-flight request back
                # into queued work automatically.
                with web.results() as c:
                    c.execute("UPDATE jobs SET status='running' WHERE id=?", (job,))
                    c.execute(
                        "UPDATE job_items SET status='running' WHERE job_id=? AND seq=3",
                        (job,),
                    )
                web.initialize()
                self.assertEqual(
                    "interrupted",
                    client.get(f"/api/jobs/{job}/result/3").json()["status"],
                )
                self.assertEqual(
                    "paused", client.get(f"/api/jobs/{job}").json()["status"]
                )
                web.STATE["manifest"]["dataset_sha256"] = "changed-context"
                self.assertEqual(
                    409,
                    client.post(
                        f"/api/jobs/{job}/control",
                        json={"action": "resume", "budget": 1},
                        headers=headers,
                    ).status_code,
                )


if __name__ == "__main__":
    unittest.main()
