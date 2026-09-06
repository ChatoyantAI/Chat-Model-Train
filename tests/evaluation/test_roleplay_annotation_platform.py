from __future__ import annotations

import gzip
import http.cookiejar
import json
import math
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path

import httpx

from src.evaluation.roleplay_annotation_platform.build_rubric import build_rubric
from src.evaluation.roleplay_annotation_platform.app import (
    ensure_versioned_annotation_primary_key,
    merge_legacy_narrative_scores,
    remove_legacy_safety_gates,
)
from src.evaluation.roleplay_annotation_platform.llm_judge import (
    JudgeConfig,
    JudgeValidationError,
    attach_score_logprob_confidence,
    build_gate_request,
    build_gate_schema,
    build_judgment_schema,
    build_openrouter_request,
    build_score_request,
    build_score_schema,
    compact_openrouter_logprobs,
    parse_and_validate_gate_response,
    parse_and_validate_openrouter_response,
    provider_preferences_for_model,
    public_config,
    score_logprobs_enabled_for_model,
    validate_gate_output,
    validate_judgment,
    validate_score_output,
)
from src.evaluation.roleplay_annotation_platform.llm_runtime import (
    LLMJudgeRuntime,
    _rate_limit_retry_delay,
    _retry_after_seconds,
)
from src.evaluation.roleplay_annotation_platform.model_catalog import (
    parse_openrouter_models,
    parse_tatu_models,
)
from src.evaluation.roleplay_annotation_platform.prepare_data import (
    _validate_record,
    build_samples_db,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRIC_MARKDOWN = (
    REPO_ROOT / "docs" / "evaluation" / "ROLEPLAY_EVALUATION_DIMENSIONS_V0_2.md"
)
RUBRIC_JSON = (
    REPO_ROOT / "src" / "evaluation" / "roleplay_annotation_platform" / "rubric.json"
)
PLATFORM_JS = (
    REPO_ROOT / "src" / "evaluation" / "roleplay_annotation_platform" / "static" / "app.js"
)


def sample_record(evaluation_id: str, candidate: str) -> dict:
    return {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "rubric_version": "roleplay-rubric-v0.1",
        "trigger_type": "message",
        "input_messages": [
            {"role": "system", "content": "You are Character A."},
            {"role": "assistant", "content": "Greeting text"},
            {"role": "user", "content": "Continue."},
        ],
        "output_message": {"role": "assistant", "content": candidate},
    }


class RubricAndDataBuildTests(unittest.TestCase):
    def test_v02_rubric_is_migrated_without_omitting_examples_or_explanations(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        self.assertEqual("roleplay-rubric-v0.18", rubric["version"])
        items = [item for dimension in rubric["dimensions"] for item in dimension["items"]]
        cases = [case for item in items for case in item["cases"]]
        examples = [example for case in cases for example in case["examples"]]
        self.assertEqual(5, len(rubric["dimensions"]))
        self.assertEqual(20, len(items))
        self.assertEqual(20, len(cases))
        self.assertEqual(80, len(examples))
        self.assertTrue(all(len(item["cases"]) == 1 for item in items))
        self.assertTrue(all(example["reply"] for example in examples))
        self.assertTrue(all(example["explanation"].startswith("说明：") for example in examples))
        self.assertTrue(all(example["score"] != "N/A" for example in examples))
        self.assertTrue(all(not any("\u3400" <= char <= "\u9fff" for char in example["reply"]) for example in examples))
        self.assertTrue(all(item["decision_steps"] for item in items))
        self.assertTrue(
            all(
                example["score"]
                in ({*item["scale"], "N/A"} if item["allows_na"] else set(item["scale"]))
                for item in items
                for case in item["cases"]
                for example in case["examples"]
            )
        )
        self.assertTrue(
            all(item["na_condition"] for item in items if item["allows_na"])
        )
        n1 = next(item for item in items if item["id"] == "N1")
        self.assertFalse(n1["allows_na"])
        self.assertEqual("", n1["na_condition"])
        self.assertNotIn("N/A", n1["anchors"])
        self.assertTrue(all(item["reference_markdown"] in rubric["source_markdown"] for item in items))
        self.assertEqual(
            "4385f107a12d7bb052217580e68058b42542c839889ea69201f5df3672c41df8",
            rubric["source_sha256"],
        )
        self.assertEqual(RUBRIC_MARKDOWN.read_text(encoding="utf-8"), rubric["source_markdown"])

    def test_checked_in_rubric_json_exactly_matches_the_markdown_build(self) -> None:
        self.assertEqual(
            build_rubric(RUBRIC_MARKDOWN),
            json.loads(RUBRIC_JSON.read_text(encoding="utf-8")),
        )

    def test_v02_uses_each_items_exact_score_scale(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        items = {
            item["id"]: item
            for dimension in rubric["dimensions"]
            for item in dimension["items"]
        }
        self.assertTrue(
            all(item["scale"] == ["0", "1", "2", "3"] for item in items.values())
        )
        self.assertFalse(items["H1"]["starred"])
        self.assertTrue(items["H1"]["allows_na"])
        self.assertIn("3", items["H1"]["anchors"])
        self.assertEqual(["0", "1", "2", "3"], items["H2"]["scale"])
        self.assertIn("漏掉本轮明确表达的一部分要求", items["H1"]["anchors"]["1"])
        self.assertIn("核心亲密意图", items["H1"]["anchors"]["2"])
        self.assertIn("鲜明、连贯、可感的刺激", items["H2"]["anchors"]["3"])
        self.assertIn("欲望或情色张力", items["H2"]["anchors"]["3"])
        self.assertIn(
            "成人互动本身已经形成鲜明、可感的刺激或张力",
            items["H2"]["decision_steps"][-1],
        )
        self.assertNotIn("H3", items)
        self.assertEqual("性格与动机呈现", items["C2"]["name"])
        self.assertIn("充分塑造本轮一个核心判断、选择、行动或目标", items["C2"]["anchors"]["3"])
        self.assertIn(
            "把人设发挥得充分、具体",
            items["C2"]["decision_steps"][-1],
        )
        self.assertIn("不要求同时展示理解、决定和行动", items["C2"]["reference_markdown"])
        self.assertNotIn("两个互相拉扯的特质", items["C2"]["reference_markdown"])
        self.assertEqual(["0", "1", "2", "3"], items["C5"]["scale"])
        self.assertFalse(items["C5"]["starred"])
        self.assertEqual("有一处归属需要猜测", items["C5"]["anchors"]["1"])
        self.assertIn("用户或玩家角色", items["C5"]["reference_markdown"])
        self.assertIn("一个主要角色加欢呼群众", items["C5"]["na_condition"])
        self.assertIn("信息不要求足够久远", items["M1"]["reference_markdown"])
        self.assertIn("使用更多条或更久远的信息不会自动升分", items["M1"]["not_counted"])
        self.assertNotIn("两轮之外", items["M1"]["reference_markdown"])
        self.assertEqual("角色台词与语感", items["L1"]["name"])
        self.assertIn("个人语感贯穿主要台词", items["L1"]["anchors"]["3"])
        self.assertIn("个人语感还自然、鲜明地贯穿主要台词", items["L1"]["decision_steps"][-1])
        self.assertIn("不要求引用用户原话、本轮特有物件", items["L1"]["reference_markdown"])
        self.assertEqual(["0", "1", "2", "3"], items["L2"]["scale"])
        self.assertIn("唯一的目标语言", items["L2"]["intro"])
        self.assertIn("不得因为用户正文使用其他语言", items["L2"]["intro"])
        self.assertEqual("情绪表现力", items["L3"]["name"])
        self.assertIn("准确、鲜明、有感染力", items["L3"]["anchors"]["3"])
        self.assertIn("情绪表现还准确、鲜明、有感染力", items["L3"]["decision_steps"][-1])
        self.assertIn("3 分不要求复杂情绪、潜台词", items["L3"]["reference_markdown"])
        self.assertEqual("动作与画面描写", items["L4"]["name"])
        self.assertIn("动作怎样发生并达到结果的物理过程", items["L4"]["anchors"]["3"])
        self.assertIn("鲜明、可信的物理过程", items["L4"]["decision_steps"][-1])
        self.assertIn("3 分不要求慢动作", items["L4"]["reference_markdown"])
        self.assertEqual("整体行文与收束", items["L5"]["name"])
        self.assertEqual(["0", "1", "2", "3"], items["L5"]["scale"])
        self.assertIn("取舍、顺序、轻重和结束位置相互配合", items["L5"]["anchors"]["3"])
        self.assertIn("整段组织还进一步增强了中心", items["L5"]["decision_steps"][-1])
        self.assertIn("3 分不要求铺垫与回收", items["L5"]["reference_markdown"])
        self.assertIn("是否华丽、含蓄、简洁、直白或小说化不会自动升分", items["L5"]["not_counted"])
        self.assertEqual("无效复用", items["L6"]["name"])
        self.assertEqual(["0", "1", "2", "3"], items["L6"]["scale"])
        self.assertFalse(items["L6"]["allows_na"])
        self.assertEqual("", items["L6"]["na_condition"])
        self.assertNotIn("N/A", items["L6"]["anchors"])
        self.assertIn("主要表达位置", items["L6"]["anchors"]["0"])
        self.assertIn(
            "Greeting 或当前上下文全部可见历史 Assistant 回复",
            items["L6"]["reference_markdown"],
        )
        self.assertNotIn("前五轮", items["L6"]["reference_markdown"])
        self.assertEqual("回应与推进", items["N2"]["name"])
        self.assertIn("用户输入后的状态和回复结束时的状态", items["N2"]["decision_steps"][-1])
        self.assertIn("核心互动原本是", items["N2"]["reference_markdown"])
        self.assertIn("准备动作，不等于事情已经进入下一阶段", items["N2"]["reference_markdown"])
        self.assertNotIn("前文已经明确规定下一步骤", items["N2"]["reference_markdown"])
        self.assertNotIn("新发展接口", items["N2"]["reference_markdown"])
        self.assertEqual(
            ["N1", "N2", "N3", "N4"],
            [item["id"] for item in rubric["dimensions"][-1]["items"]],
        )
        self.assertEqual("用户决定权", items["N3"]["name"])
        self.assertEqual("叙事控制", items["N4"]["name"])
        self.assertFalse(items["N4"]["allows_na"])
        self.assertEqual(8, sum(len(group["items"]) for group in rubric["gates"]))
        self.assertNotIn(
            "G-S",
            {
                gate["id"]
                for group in rubric["gates"]
                for gate in group["items"]
            },
        )
        self.assertNotIn("安全与上线资格 Gate", rubric["source_markdown"])

    def test_annotation_primary_key_migration_preserves_old_rubric_rows(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE TABLE annotations (
                username TEXT NOT NULL REFERENCES users(username),
                seq INTEGER NOT NULL,
                evaluation_id TEXT NOT NULL,
                rubric_version TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('draft', 'submitted')),
                gates_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                gate_notes TEXT NOT NULL,
                overall_notes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (username, seq)
            );
            INSERT INTO users VALUES ('reviewer', 'old', 'old');
            INSERT INTO annotations VALUES (
                'reviewer', 1, 'eval-1', 'roleplay-rubric-v0.8', 'submitted',
                '{}', '{"H3":{"value":3}}', '', '', 'old', 'old'
            );
            """
        )

        ensure_versioned_annotation_primary_key(connection)
        primary_key = [
            row["name"]
            for row in sorted(
                connection.execute("PRAGMA table_info(annotations)").fetchall(),
                key=lambda row: row["pk"],
            )
            if row["pk"]
        ]
        self.assertEqual(["username", "seq", "rubric_version"], primary_key)
        connection.execute(
            "INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "reviewer", 1, "eval-1", "roleplay-rubric-v0.11", "draft",
                "{}", "{}", "", "", "new", "new",
            ),
        )
        rows = connection.execute(
            "SELECT rubric_version, scores_json FROM annotations ORDER BY rubric_version"
        ).fetchall()
        self.assertEqual(
            ["roleplay-rubric-v0.11", "roleplay-rubric-v0.8"],
            [row["rubric_version"] for row in rows],
        )
        scores_by_version = {
            row["rubric_version"]: row["scores_json"] for row in rows
        }
        self.assertIn("H3", scores_by_version["roleplay-rubric-v0.8"])
        connection.close()

    def test_blind_data_validator_rejects_ground_truth_keys(self) -> None:
        record = sample_record("eval_x", "candidate")
        record["model_name"] = "secret-model"
        with self.assertRaisesRegex(ValueError, "blind-data violation"):
            _validate_record(record, 1)

    def test_all_score_standards_are_visible_before_selection(self) -> None:
        script = PLATFORM_JS.read_text(encoding="utf-8")
        self.assertIn("description.textContent = item.anchors[score]", script)
        self.assertIn("先读标准，再选择一项", script)
        self.assertIn(
            'item.allows_na?[...item.scale,"N/A"]:item.scale',
            ''.join(script.split()),
        )
        self.assertIn("item.decision_steps", script)
        self.assertIn("example.explanation", script)
        self.assertIn("state.rubric.source_markdown", script)
        self.assertNotIn("选择分数后显示对应判定标准", script)

    def test_gate_guidance_is_rendered_as_structure_instead_of_raw_markdown(self) -> None:
        script = PLATFORM_JS.read_text(encoding="utf-8")
        self.assertIn("function renderGuidanceMarkdown", script)
        self.assertIn('document.createElement("ol")', script)
        self.assertIn('document.createElement("blockquote")', script)
        self.assertIn("renderGuidanceMarkdown(guidance, group.guidance)", script)


class LLMJudgeSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rubric = build_rubric(RUBRIC_MARKDOWN)
        cls.config = JudgeConfig(
            api_key="test-key",
            models=("test/model",),
            openrouter_url="https://example.invalid/chat/completions",
            max_concurrency=2,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
            reasoning_models=("test/model",),
            provider_order=("modal",),
            provider_ignore_by_model=(("test/model", ("morph",)),),
        )
        cls.sample = {
            "trigger_type": "message",
            "messages": [
                {
                    "role": "system",
                    "display_role": "system",
                    "content": "You are Character A.",
                },
                {
                    "role": "assistant",
                    "display_role": "greeting",
                    "content": "Greeting text",
                },
                {
                    "role": "user",
                    "display_role": "user",
                    "content": "Continue.",
                },
            ],
            "candidate": {"role": "assistant", "content": "Candidate one"},
        }

    def valid_judgment(self) -> dict:
        gates = {}
        for group in self.rubric["gates"]:
            for gate in group["items"]:
                gates[gate["id"]] = {
                    "value": gate["default"],
                    "evidence": [],
                    "reason": "按当前对话内容判断。",
                }
        scores = {}
        for dimension in self.rubric["dimensions"]:
            for item in dimension["items"]:
                scores[item["id"]] = {
                    "value": item["scale"][0],
                    "evidence": [],
                    "reason": "按该项第一步判定。",
                }
        return {
            "gate_status": "completed",
            "gates": gates,
            "atomic_scoring_status": "completed",
            "scores": scores,
        }

    def test_public_config_never_serializes_server_key(self) -> None:
        config = replace(self.config, tatu_api_key="mock-key-1")
        payload = public_config(config)
        serialized = json.dumps(payload)
        self.assertNotIn(self.config.api_key, serialized)
        self.assertNotIn(config.tatu_api_key, serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("zdr", payload)
        self.assertEqual(["modal"], payload["provider_routing"]["preferred_order"])
        self.assertEqual(
            ["cloudflare"],
            payload["provider_routing"]["preferred_order_by_model"][
                "z-ai/glm-5.3"
            ],
        )
        self.assertEqual(
            ["alibaba", "morph"], payload["provider_routing"]["ignore"]
        )
        self.assertEqual(
            ["morph"],
            payload["provider_routing"]["ignore_by_model"]["test/model"],
        )
        self.assertEqual(
            ["qwen/"],
            payload["provider_routing"]["excluded_openrouter_model_prefixes"],
        )
        self.assertEqual(4, payload["rate_limit_retry"]["max_attempts"])
        self.assertEqual(5.0, payload["rate_limit_retry"]["backoff_seconds"])
        self.assertTrue(payload["gateways"]["tatu"]["enabled"])
        self.assertFalse(payload["gateways"]["tatu"]["reports_token_logprobs"])

    def test_openrouter_catalog_keeps_only_compatible_text_models(self) -> None:
        common = [
            "max_tokens",
            "response_format",
            "structured_outputs",
        ]
        payload = {
            "data": [
                {
                    "id": "vendor/good",
                    "name": "Vendor: Good",
                    "context_length": 131072,
                    "supported_parameters": common,
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                },
                {
                    "id": "vendor/short",
                    "context_length": 4096,
                    "supported_parameters": common,
                },
                {
                    "id": "vendor/no-schema",
                    "context_length": 131072,
                    "supported_parameters": ["max_tokens"],
                },
                {
                    "id": "vendor/free:free",
                    "context_length": 131072,
                    "supported_parameters": common,
                },
                {
                    "id": "openrouter/auto",
                    "context_length": 131072,
                    "supported_parameters": common,
                },
                {
                    "id": "vendor/image-output",
                    "context_length": 131072,
                    "supported_parameters": common,
                    "architecture": {"output_modalities": ["text", "image"]},
                },
            ]
        }
        models = parse_openrouter_models(payload, min_context_length=65536)
        self.assertEqual(["vendor/good"], [model["id"] for model in models])
        self.assertEqual(1.0, models[0]["prompt_price_per_million"])
        self.assertEqual(2.0, models[0]["completion_price_per_million"])
        self.assertIn("128K ctx", models[0]["label"])
        self.assertFalse(models[0]["reports_token_logprobs"])

    def test_openrouter_catalog_can_exclude_alibaba_model_family(self) -> None:
        common = ["response_format", "structured_outputs"]
        payload = {
            "data": [
                {
                    "id": "qwen/qwen3.8-27b",
                    "context_length": 131072,
                    "supported_parameters": common,
                    "architecture": {"output_modalities": ["text"]},
                },
                {
                    "id": "z-ai/glm-5.3",
                    "context_length": 131072,
                    "supported_parameters": common,
                    "architecture": {"output_modalities": ["text"]},
                },
            ]
        }
        models = parse_openrouter_models(
            payload,
            min_context_length=0,
            excluded_model_prefixes=("qwen/",),
        )
        self.assertEqual(["z-ai/glm-5.3"], [model["id"] for model in models])

    def test_tatu_catalog_excludes_unstable_glm_routes(self) -> None:
        payload = {
            "data": [
                {"id": model_id}
                for model_id in (
                    "glm-5.3",
                    "glm-5.3-flash",
                    "deepseek-v4-pro-0813",
                    "qwen3.8-27b",
                    "kimi-k3",
                    "gpt-5.4",
                )
            ]
        }
        models = parse_tatu_models(payload, min_context_length=0)
        self.assertEqual(3, len(models))
        self.assertEqual(
            "tatu::deepseek-v4-pro-0813@DEEPSEEK-default",
            models[0]["id"],
        )
        self.assertTrue(all(model["gateway"] == "tatu" for model in models))
        self.assertTrue(all(not model["reports_token_logprobs"] for model in models))
        self.assertEqual("CNY", models[0]["pricing"]["currency"])

    def test_openrouter_glm_is_pinned_to_cloudflare(self) -> None:
        preferences = provider_preferences_for_model(
            self.config, "z-ai/glm-5.3-flash"
        )
        self.assertEqual(["cloudflare"], preferences["order"])
        self.assertEqual(["cloudflare"], preferences["only"])
        self.assertFalse(preferences["allow_fallbacks"])
        self.assertFalse(
            score_logprobs_enabled_for_model(
                self.config, "z-ai/glm-5.3-flash"
            )
        )

    def test_tatu_request_uses_chat_completions_prompt_schema(self) -> None:
        model = "tatu::glm-5.3@ZAI-default"
        config = replace(
            self.config,
            models=(model,),
            tatu_api_key="tatu-secret",
            reasoning_models=(model,),
        )
        body, packet, _ = build_score_request(model, self.rubric, self.sample, config)
        self.assertEqual("glm-5.3@ZAI-default", body["model"])
        self.assertTrue(body["stream"])
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertNotIn("provider", body)
        self.assertNotIn("logprobs", body)
        self.assertNotIn("top_logprobs", body)
        self.assertNotIn("reasoning", body)
        self.assertIn("输出 JSON Schema 开始", body["messages"][0]["content"])
        self.assertIn('"scores"', body["messages"][0]["content"])
        self.assertIn(
            '{"value":"2","evidence":[],"reason":"..."}',
            body["messages"][0]["content"],
        )
        self.assertIn(
            "value、evidence、reason 三个字段",
            body["messages"][0]["content"],
        )

        deepseek_model = "tatu::deepseek-v4-pro-0813@DEEPSEEK-default"
        deepseek_config = replace(config, models=(deepseek_model,))
        deepseek_body, _, _ = build_score_request(
            deepseek_model, self.rubric, self.sample, deepseek_config
        )
        self.assertNotIn("response_format", deepseek_body)

        gate_output = {"gates": self.valid_judgment()["gates"]}
        fenced = "```json\n" + json.dumps(gate_output, ensure_ascii=False) + "\n```"
        gates, failed, raw = parse_and_validate_gate_response(
            {"choices": [{"message": {"content": fenced}}]},
            self.rubric,
            packet,
            allow_json_fence=True,
        )
        self.assertEqual([], failed)
        self.assertEqual(set(gate_output["gates"]), set(gates))
        self.assertEqual(fenced, raw)
        with self.assertRaises(JudgeValidationError):
            parse_and_validate_gate_response(
                {"choices": [{"message": {"content": fenced}}]},
                self.rubric,
                packet,
            )

    def test_schema_uses_each_atomic_items_exact_scale(self) -> None:
        schema = build_judgment_schema(self.rubric)
        properties = schema["properties"]["scores"]["properties"]
        self.assertEqual(
            ["0", "1", "2", "3", "N/A", "SKIP"],
            properties["C1"]["properties"]["value"]["enum"],
        )
        self.assertEqual(
            ["0", "1", "2", "3", "N/A", "SKIP"],
            properties["H1"]["properties"]["value"]["enum"],
        )
        self.assertEqual(
            ["0", "1", "2", "3", "N/A", "SKIP"],
            properties["H2"]["properties"]["value"]["enum"],
        )
        self.assertEqual(
            ["0", "1", "2", "3", "N/A", "SKIP"],
            properties["C5"]["properties"]["value"]["enum"],
        )
        self.assertEqual(
            ["0", "1", "2", "3", "SKIP"],
            properties["N1"]["properties"]["value"]["enum"],
        )
        score_schema = build_score_schema(self.rubric)
        self.assertEqual(
            ["0", "1", "2", "3", "N/A"],
            score_schema["properties"]["scores"]["properties"]["H1"]
            ["properties"]["value"]["enum"],
        )
        self.assertEqual(
            ["0", "1", "2", "3"],
            score_schema["properties"]["scores"]["properties"]["N1"]
            ["properties"]["value"]["enum"],
        )
        for record in score_schema["properties"]["scores"]["properties"].values():
            self.assertNotIn("confidence", record["required"])
            self.assertNotIn("confidence", record["properties"])
        self.assertEqual(20, len(properties))
        self.assertEqual(8, len(schema["properties"]["gates"]["properties"]))

    def test_boolean_gate_schema_and_prompt_define_true_as_error_triggered(self) -> None:
        schema = build_gate_schema(self.rubric)
        gate_properties = schema["properties"]["gates"]["properties"]
        for gate_id in ["G-D1", "G-D2", "G-D3", "G-P1", "G-P2", "G-P3", "G-P4", "G-P5"]:
            value_schema = gate_properties[gate_id]["properties"]["value"]
            self.assertEqual("boolean", value_schema["type"])
            self.assertIn("true 仅表示", value_schema["description"])
            self.assertIn("false 表示", value_schema["description"])
            self.assertIn("不得把 true 当作通过", value_schema["description"])

        body, _, _ = build_gate_request(
            "test/model", self.rubric, self.sample, self.config
        )
        prompt = body["messages"][0]["content"]
        self.assertIn("true=该错误或失败事件明确发生、会淘汰", prompt)
        self.assertIn("false=该错误没有发生、继续", prompt)
        self.assertIn("不得把 true 当作通过", prompt)
        self.assertNotIn("logprobs", body)
        self.assertNotIn("top_logprobs", body)

    def test_request_sends_full_source_but_only_a_normalized_sample_packet(self) -> None:
        body, packet, request_hash = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        self.assertIn(self.rubric["source_markdown"], body["messages"][0]["content"])
        self.assertEqual("greeting", packet["context_messages"][1]["role"])
        self.assertEqual("assistant_candidate", packet["candidate_response"]["role"])
        serialized_packet = json.dumps(packet)
        self.assertNotIn("evaluation_id", serialized_packet)
        self.assertNotIn("model_name", serialized_packet)
        self.assertEqual(64, len(request_hash))
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertNotIn("max_tokens", body)
        self.assertEqual(
            {"effort": "low", "exclude": True},
            body["reasoning"],
        )
        self.assertEqual(["modal"], body["provider"]["order"])
        self.assertEqual(["alibaba", "morph"], body["provider"]["ignore"])
        self.assertTrue(body["provider"]["allow_fallbacks"])
        self.assertNotIn("zdr", body["provider"])
        self.assertNotIn("data_collection", body["provider"])

    def test_openrouter_uses_model_specific_success_first_provider_order(self) -> None:
        model = "z-ai/glm-5.3-flash"
        config = replace(
            self.config,
            models=(model,),
            provider_order=(),
            provider_ignore_by_model=(),
        )
        body, _, _ = build_score_request(model, self.rubric, self.sample, config)
        self.assertEqual(
            ["cloudflare"],
            body["provider"]["order"],
        )
        self.assertEqual(["cloudflare"], body["provider"]["only"])
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertEqual(["alibaba", "morph"], body["provider"]["ignore"])
        self.assertNotIn("logprobs", body)
        self.assertNotIn("top_logprobs", body)

    def test_platform_guideline_is_not_sent_as_part_of_the_user_query(self) -> None:
        sample = json.loads(json.dumps(self.sample))
        sample["messages"][-1]["content"] = """Continue the scene.

<guideline>
- Target length: 200-250 words total
- Interweave dialogue and narration naturally for engaging rhythm
- Reply in this format: Dialogue in \"quotes\", narration in *asterisks*.
- Never include word count, token count, or meta commentary
- All output must be in English language
</guideline>"""

        body, packet, _ = build_score_request(
            "test/model", self.rubric, sample, self.config
        )

        self.assertEqual("Continue the scene.", packet["context_messages"][-1]["content"])
        self.assertNotIn("<guideline>", json.dumps(packet))
        requirements = packet["platform_output_requirements"]
        self.assertEqual("annotation_platform", requirements["origin"])
        self.assertEqual("200-250 words", requirements["approximate_length"])
        self.assertEqual("English", requirements["response_language"])
        self.assertIn(
            "Interweave dialogue and narration naturally for engaging rhythm",
            requirements["format_and_style_notes"],
        )
        prompt = body["messages"][0]["content"]
        self.assertIn("N1 只能根据 context_messages 中用户自己提出的要求判断", prompt)
        self.assertIn("字数仅表示大致生成目标", requirements["description"])
        self.assertIn("不得因为候选回复未落在该范围内而扣分", prompt)
        self.assertIn("它是本轮候选回复唯一的目标语言", prompt)
        self.assertIn("唯一的目标语言", requirements["description"])
        self.assertNotIn("<guideline>", json.dumps(body))
        self.assertTrue(body["logprobs"])
        self.assertEqual(20, body["top_logprobs"])

    def test_score_logprobs_are_reduced_to_per_item_probability(self) -> None:
        raw = (
            '{"scores":{"H2":{"value":"2",'
            '"evidence":[],"reason":"ok"}}}'
        )
        prefix, suffix = raw.split('"2"', 1)
        selected_logprob = -0.02
        payload = {
            "choices": [
                {
                    "message": {"content": raw},
                    "logprobs": {
                        "content": [
                            {"token": prefix, "logprob": 0.0},
                            {
                                "token": '"2"',
                                "logprob": selected_logprob,
                                "top_logprobs": [
                                    {"token": '"2"', "logprob": selected_logprob},
                                    {"token": '"1"', "logprob": -4.0},
                                    {"token": '"3"', "logprob": -6.0},
                                    {"token": '"0"', "logprob": -7.0},
                                    {"token": '"N/A"', "logprob": -9.0},
                                ],
                            },
                            {"token": suffix, "logprob": 0.0},
                        ]
                    },
                }
            ]
        }
        scores = {
            "H2": {
                "value": "2",
                "evidence": [],
                "reason": "ok",
            }
        }

        attached = attach_score_logprob_confidence(
            scores, payload, raw, self.rubric
        )

        weights = [
            math.exp(value - selected_logprob)
            for value in (selected_logprob, -4.0, -6.0, -7.0, -9.0)
        ]
        expected = weights[0] / sum(weights)
        self.assertEqual(1, attached)
        self.assertAlmostEqual(expected, scores["H2"]["logprob_confidence"], 6)
        self.assertEqual("complete_allowed_values", scores["H2"]["logprob_method"])
        self.assertGreater(scores["H2"]["logprob_margin"], 0.9)
        self.assertEqual(
            ["0", "1", "2", "3", "N/A"],
            list(scores["H2"]["logprob_distribution"]),
        )
        self.assertAlmostEqual(
            sum(scores["H2"]["logprob_distribution"].values()), 1.0, 5
        )
        self.assertAlmostEqual(
            sum(
                int(value) * probability
                for value, probability in scores["H2"][
                    "logprob_distribution"
                ].items()
                if value != "N/A"
            ),
            scores["H2"]["logprob_expected_score"]
            * (1.0 - scores["H2"]["logprob_na_probability"]),
            5,
        )
        self.assertGreater(scores["H2"]["logprob_na_probability"], 0.0)
        self.assertTrue(scores["H2"]["logprob_distribution_complete"])

        compact_openrouter_logprobs(payload, score_confidence_count=attached)
        choice = payload["choices"][0]
        self.assertNotIn("logprobs", choice)
        self.assertEqual(
            {
                "returned": True,
                "token_count": 3,
                "score_confidence_count": 1,
            },
            choice["logprobs_summary"],
        )

    def test_score_logprobs_align_when_provider_reformats_json_content(self) -> None:
        raw = '{"scores":{"H2":{"value":"2","evidence":[],"reason":"ok"}}}'
        tokenized = '''{
  "scores": {
    "H2": {
      "value": "2",
      "evidence": [],
      "reason": "ok"
    }
  }
}'''
        prefix, suffix = tokenized.split('"2"', 1)
        payload = {
            "choices": [
                {
                    "message": {"content": raw},
                    "logprobs": {
                        "content": [
                            {"token": prefix, "logprob": 0.0},
                            {
                                "token": '"2"',
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": '"2"', "logprob": -0.1},
                                    {"token": '"3"', "logprob": -1.1},
                                ],
                            },
                            {"token": suffix, "logprob": 0.0},
                        ]
                    },
                }
            ]
        }
        scores = {
            "H2": {"value": "2", "evidence": [], "reason": "ok"}
        }

        attached = attach_score_logprob_confidence(
            scores, payload, raw, self.rubric
        )

        self.assertEqual(0, attached)
        self.assertEqual(
            "partial_allowed_values_unusable", scores["H2"]["logprob_method"]
        )
        self.assertNotIn("logprob_confidence", scores["H2"])
        self.assertNotIn("logprob_expected_score", scores["H2"])
        self.assertEqual(
            {"2", "3"}, set(scores["H2"]["logprob_distribution"])
        )

    def test_score_logprobs_do_not_treat_selected_token_probability_as_confidence(self) -> None:
        raw = '{"scores":{"H2":{"value":"2","evidence":[],"reason":"ok"}}}'
        prefix, suffix = raw.split('"2"', 1)
        payload = {
            "choices": [
                {
                    "message": {"content": raw},
                    "logprobs": {
                        "content": [
                            {"token": prefix, "logprob": 0.0},
                            {"token": '"2"', "logprob": -0.000001},
                            {"token": suffix, "logprob": 0.0},
                        ]
                    },
                }
            ]
        }
        scores = {"H2": {"value": "2", "evidence": [], "reason": "ok"}}

        attached = attach_score_logprob_confidence(scores, payload, raw, self.rubric)

        self.assertEqual(0, attached)
        self.assertEqual(
            "top_logprobs_missing_unusable", scores["H2"]["logprob_method"]
        )
        self.assertNotIn("logprob_confidence", scores["H2"])
        self.assertNotIn("logprob_distribution", scores["H2"])

    def test_score_logprobs_reject_provider_token_position_mismatch(self) -> None:
        raw = '{"scores":{"H2":{"value":"2","evidence":[],"reason":"ok"}}}'
        prefix, suffix = raw.split('"2"', 1)
        first_token_alternatives = [
            {"token": prefix, "logprob": -0.01},
            {"token": "{", "logprob": -5.0},
        ]
        payload = {
            "choices": [
                {
                    "message": {"content": raw},
                    "logprobs": {
                        "content": [
                            {
                                "token": prefix,
                                "logprob": -0.01,
                                "top_logprobs": first_token_alternatives,
                            },
                            {
                                "token": '"2"',
                                "logprob": -0.02,
                                # Reproduces a real provider bug: alternatives
                                # from the first token are repeated here.
                                "top_logprobs": first_token_alternatives,
                            },
                            {"token": suffix, "logprob": 0.0},
                        ]
                    },
                }
            ]
        }
        scores = {"H2": {"value": "2", "evidence": [], "reason": "ok"}}

        attached = attach_score_logprob_confidence(scores, payload, raw, self.rubric)

        self.assertEqual(0, attached)
        self.assertEqual(
            "provider_position_mismatch_unusable",
            scores["H2"]["logprob_method"],
        )
        self.assertNotIn("logprob_confidence", scores["H2"])

    def test_provider_routing_is_part_of_the_request_hash(self) -> None:
        _, _, modal_hash = build_gate_request(
            "test/model", self.rubric, self.sample, self.config
        )
        other_config = replace(self.config, provider_order=("together",))
        _, _, other_hash = build_gate_request(
            "test/model", self.rubric, self.sample, other_config
        )
        self.assertNotEqual(modal_hash, other_hash)

    def test_valid_judgment_round_trips_through_json_and_semantic_validation(self) -> None:
        body, packet, _ = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        judgment = self.valid_judgment()
        openrouter_payload = {
            "choices": [{"message": {"content": json.dumps(judgment)}}]
        }
        validated, raw = parse_and_validate_openrouter_response(
            openrouter_payload, self.rubric, packet
        )
        self.assertEqual(judgment, validated)
        self.assertEqual(judgment, json.loads(raw))
        self.assertIn("scores", body["response_format"]["json_schema"]["schema"]["properties"])

    def test_structured_object_content_is_accepted_without_string_reencoding(self) -> None:
        _, packet, _ = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        judgment = self.valid_judgment()
        payload = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": judgment}}
            ]
        }
        validated, raw = parse_and_validate_openrouter_response(
            payload, self.rubric, packet
        )
        self.assertEqual(judgment, validated)
        self.assertEqual(judgment, json.loads(raw))

    def test_null_content_at_length_limit_reports_the_actual_failure(self) -> None:
        _, packet, _ = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning": "thinking"},
                }
            ],
            "usage": {
                "completion_tokens_details": {"reasoning_tokens": 6000}
            },
        }
        with self.assertRaisesRegex(
            JudgeValidationError,
            "达到输出长度上限.*reasoning_tokens=6000",
        ):
            parse_and_validate_openrouter_response(payload, self.rubric, packet)

    def test_invalid_per_item_score_is_rejected(self) -> None:
        _, packet, _ = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        judgment = self.valid_judgment()
        judgment["scores"]["C1"]["value"] = "4"
        with self.assertRaisesRegex(JudgeValidationError, "C1"):
            validate_judgment(judgment, self.rubric, packet)

    def test_score_schema_rejects_model_reported_confidence(self) -> None:
        judgment = self.valid_judgment()
        self.assertEqual(
            judgment["scores"],
            validate_score_output(
                {"scores": judgment["scores"]}, self.rubric, self.sample
            ),
        )

        judgment["scores"]["L1"]["confidence"] = "high"
        with self.assertRaisesRegex(JudgeValidationError, "scores.L1.*confidence"):
            validate_score_output(
                {"scores": judgment["scores"]}, self.rubric, self.sample
            )

    def test_missing_item_and_markdown_fence_are_not_silently_repaired(self) -> None:
        _, packet, _ = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        judgment = self.valid_judgment()
        del judgment["scores"]["N4"]
        with self.assertRaisesRegex(JudgeValidationError, "N4"):
            validate_judgment(judgment, self.rubric, packet)
        fenced = {"choices": [{"message": {"content": "```json\n{}\n```"}}]}
        with self.assertRaisesRegex(JudgeValidationError, "合法 JSON"):
            parse_and_validate_openrouter_response(fenced, self.rubric, packet)

    def test_g_d1_requires_every_atomic_item_to_be_skip(self) -> None:
        _, packet, _ = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        judgment = self.valid_judgment()
        judgment["gates"]["G-D1"]["value"] = True
        judgment["gate_status"] = "stopped_at_g_d1"
        judgment["atomic_scoring_status"] = "skipped_g_d1"
        with self.assertRaisesRegex(JudgeValidationError, "必须为 SKIP"):
            validate_judgment(judgment, self.rubric, packet)
        for score in judgment["scores"].values():
            score["value"] = "SKIP"
        self.assertEqual(
            judgment, validate_judgment(judgment, self.rubric, packet)
        )

    def test_evidence_may_be_a_paraphrase_or_location_hint(self) -> None:
        _, packet, _ = build_openrouter_request(
            "test/model", self.rubric, self.sample, self.config
        )
        judgment = self.valid_judgment()
        judgment["scores"]["H1"]["evidence"] = [
            "Character Identity ... Age(18)"
        ]
        self.assertEqual(
            judgment,
            validate_judgment(judgment, self.rubric, packet),
        )

    def test_llm_score_schema_does_not_request_decision_step(self) -> None:
        schema = build_score_schema(self.rubric)
        records = schema["properties"]["scores"]["properties"].values()
        self.assertTrue(
            all("decision_step" not in record["properties"] for record in records)
        )
        self.assertNotIn(
            "record.decision_step",
            (REPO_ROOT / "src" / "evaluation" / "roleplay_annotation_platform" / "static" / "llm_judge.js").read_text(encoding="utf-8"),
        )

    def test_n1_na_is_rejected_after_gate_pass(self) -> None:
        judgment = self.valid_judgment()
        judgment["scores"]["N1"]["value"] = "N/A"

        with self.assertRaisesRegex(JudgeValidationError, "scores.N1.value"):
            validate_score_output(
                {"scores": judgment["scores"]}, self.rubric, self.sample
            )
        with self.assertRaisesRegex(JudgeValidationError, "scores.N1.value"):
            validate_judgment(judgment, self.rubric, self.sample)

    def test_score_request_includes_validated_gate_values_and_forbids_whole_refusal(self) -> None:
        gates = self.valid_judgment()["gates"]
        body, _, _ = build_score_request(
            "test/model",
            self.rubric,
            self.sample,
            self.config,
            validated_gates=gates,
        )
        prompt = body["messages"][0]["content"]
        sample_message = body["messages"][1]["content"]
        self.assertIn("20 个原子项不得全部为 N/A", prompt)
        self.assertNotIn("confidence", prompt)
        self.assertIn("后台已验证 Gate 结果", sample_message)
        self.assertIn('"G-D1":false', sample_message)
        self.assertNotIn('"G-S"', sample_message)

    def test_two_stage_schemas_are_smaller_and_gate_failures_stop_scoring(self) -> None:
        full_schema = build_judgment_schema(self.rubric)
        gate_schema = build_gate_schema(self.rubric)
        score_schema = build_score_schema(self.rubric)
        self.assertLess(len(json.dumps(gate_schema)), len(json.dumps(full_schema)))
        self.assertLess(len(json.dumps(score_schema)), len(json.dumps(full_schema)))
        self.assertEqual(8, len(gate_schema["properties"]["gates"]["required"]))
        self.assertEqual(20, len(score_schema["properties"]["scores"]["required"]))
        for record in score_schema["properties"]["scores"]["properties"].values():
            self.assertNotIn("SKIP", record["properties"]["value"]["enum"])

        _, packet, _ = build_gate_request(
            "test/model", self.rubric, self.sample, self.config
        )
        gates = self.valid_judgment()["gates"]
        output = {"gates": gates}
        validated, failed = validate_gate_output(output, self.rubric, packet)
        self.assertEqual(gates, validated)
        self.assertEqual([], failed)
        gates["G-P2"]["value"] = True
        _, failed = validate_gate_output(output, self.rubric, packet)
        self.assertEqual(["G-P2"], failed)

    def test_legacy_safety_answers_are_removed_without_losing_other_gates(self) -> None:
        base = {"G-D1": False, "G-P1": False}
        self.assertEqual(
            base,
            remove_legacy_safety_gates(
                {
                    **base,
                    "G-S": "fail",
                    "G-S1": "yes",
                    "G-S2": "fail",
                    "G-S3": "N/A",
                }
            ),
        )

    def test_legacy_narrative_scores_are_merged_and_renumbered(self) -> None:
        migrated = merge_legacy_narrative_scores(
            {
                "C1": {"value": 2, "evidence": "character"},
                "N2": {"value": 2, "evidence": "回应完整"},
                "N3": {"value": 3, "evidence": "出现新线索"},
                "N4": {"value": 0, "evidence": "代写用户"},
                "N5": {"value": 2, "evidence": "推进正常"},
            }
        )
        self.assertEqual({"C1", "N2", "N3", "N4"}, set(migrated))
        self.assertEqual(3, migrated["N2"]["value"])
        self.assertIn("回应完整", migrated["N2"]["evidence"])
        self.assertIn("出现新线索", migrated["N2"]["evidence"])
        self.assertEqual(0, migrated["N3"]["value"])
        self.assertEqual(2, migrated["N4"]["value"])

        self.assertEqual(
            2,
            merge_legacy_narrative_scores(
                {
                    "N2": {"value": 3, "evidence": ""},
                    "N3": {"value": 2, "evidence": ""},
                }
            )["N2"]["value"],
        )
        self.assertEqual(
            1,
            merge_legacy_narrative_scores(
                {
                    "N2": {"value": 2, "evidence": ""},
                    "N3": {"value": 0, "evidence": ""},
                }
            )["N2"]["value"],
        )
        llm_migrated = merge_legacy_narrative_scores(
            {
                "N2": {
                    "value": "2",
                    "reason": "回应完整",
                    "evidence": ["接过匕首"],
                },
                "N3": {
                    "value": "3",
                    "reason": "留下新计划",
                    "evidence": ["明天同行"],
                },
                "N4": {"value": "2", "reason": "未代写", "evidence": []},
                "N5": {"value": "2", "reason": "推进适中", "evidence": []},
            }
        )
        self.assertEqual("3", llm_migrated["N2"]["value"])
        self.assertEqual(["接过匕首", "明天同行"], llm_migrated["N2"]["evidence"])
        self.assertIn("回应完整", llm_migrated["N2"]["reason"])
        self.assertIn("留下新计划", llm_migrated["N2"]["reason"])


class LLMJudgeRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_tatu_transport_uses_server_key_and_tatu_endpoint(self) -> None:
        model = "tatu::glm-5.3@ZAI-default"
        config = JudgeConfig(
            api_key="mock-key-2",
            models=(model,),
            openrouter_url="https://openrouter.test/chat/completions",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
            tatu_api_key="tatu-secret",
            tatu_url="https://tatu.test/v1/chat/completions",
        )
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["authorization"] = request.headers.get("Authorization")
            seen["referer"] = request.headers.get("HTTP-Referer")
            seen["body"] = json.loads(request.content)
            seen["read_timeout"] = request.extensions["timeout"]["read"]
            first = {
                "id": "chatcmpl-test",
                "model": "glm-5.3",
                "choices": [{"delta": {"content": "{"}, "finish_reason": None}],
            }
            second = {
                "id": "chatcmpl-test",
                "model": "glm-5.3",
                "choices": [{"delta": {"content": "}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            }
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=(
                    f"data: {json.dumps(first)}\n\n"
                    f"data: {json.dumps(second)}\n\n"
                    "data: [DONE]\n\n"
                ),
            )

        runtime = LLMJudgeRuntime(
            config=config,
            rubric=build_rubric(RUBRIC_MARKDOWN),
            connection_factory=lambda: None,
            sample_loader=lambda seq: (str(seq), {}),
            utc_now=lambda: "now",
        )
        runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            payload = await runtime._post_phase(
                {"model": "glm-5.3@ZAI-default"},
                requested_model=model,
            )
        finally:
            await runtime._client.aclose()
            runtime._client = None
        self.assertEqual("https://tatu.test/v1/chat/completions", seen["url"])
        self.assertEqual("Bearer tatu-secret", seen["authorization"])
        self.assertIsNone(seen["referer"])
        self.assertEqual("glm-5.3@ZAI-default", seen["body"]["model"])
        self.assertTrue(seen["body"]["stream"])
        self.assertEqual(600.0, seen["read_timeout"])
        self.assertEqual("{}", payload["choices"][0]["message"]["content"])
        self.assertEqual("stop", payload["choices"][0]["finish_reason"])
        self.assertEqual(12, payload["usage"]["prompt_tokens"])

    def test_rate_limit_delay_uses_exponential_jitter_and_retry_after(self) -> None:
        config = JudgeConfig(
            api_key="key",
            models=("test/model",),
            openrouter_url="https://openrouter.test/chat/completions",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
        )
        self.assertIsNone(_retry_after_seconds(None))
        self.assertIsNone(_retry_after_seconds("tomorrow"))
        self.assertEqual(0.0, _retry_after_seconds("-1"))
        self.assertEqual(600.0, _retry_after_seconds("700"))

        delays = [
            _rate_limit_retry_delay(
                job_id="job-rate-limit",
                seq=7,
                attempt_number=attempt,
                config=config,
                retry_after_seconds=None,
            )
            for attempt in (1, 2, 3)
        ]
        self.assertGreaterEqual(delays[0], 4.0)
        self.assertLessEqual(delays[0], 6.0)
        self.assertGreaterEqual(delays[1], 12.0)
        self.assertLessEqual(delays[1], 18.0)
        self.assertGreaterEqual(delays[2], 36.0)
        self.assertLessEqual(delays[2], 54.0)
        self.assertEqual(
            20.0,
            _rate_limit_retry_delay(
                job_id="job-rate-limit",
                seq=7,
                attempt_number=1,
                config=config,
                retry_after_seconds=20.0,
            ),
        )
        self.assertEqual(
            100.0,
            _rate_limit_retry_delay(
                job_id="job-rate-limit",
                seq=7,
                attempt_number=3,
                config=config,
                retry_after_seconds=100.0,
            ),
        )

    def test_initialize_retires_old_rubric_jobs_and_recovers_current_jobs(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        config = JudgeConfig(
            api_key="key",
            models=("test/model",),
            openrouter_url="https://example.invalid",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            database = Path(tempdir) / "versioned-jobs.sqlite3"

            def connection_factory():
                connection = sqlite3.connect(database, timeout=30)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys=ON")
                return connection

            connection = connection_factory()
            connection.execute(
                "CREATE TABLE users(username TEXT PRIMARY KEY, created_at TEXT, last_seen_at TEXT)"
            )
            connection.execute("INSERT INTO users VALUES ('reviewer', 'now', 'now')")
            connection.commit()
            connection.close()

            runtime = LLMJudgeRuntime(
                config=config,
                rubric=rubric,
                connection_factory=connection_factory,
                sample_loader=lambda seq: (f"eval-{seq}", {}),
                utc_now=lambda: "2026-09-01T00:00:00+00:00",
            )
            runtime.initialize()
            retired_job = runtime.create_job(
                username="reviewer",
                model="test/model",
                sequences=[1],
                scope={"type": "current", "seq": 1},
                budget_limit_usd=1,
            )
            current_job = runtime.create_job(
                username="reviewer",
                model="test/model",
                sequences=[2],
                scope={"type": "current", "seq": 2},
                budget_limit_usd=1,
            )
            connection = connection_factory()
            connection.execute(
                "UPDATE llm_jobs SET rubric_version = 'roleplay-rubric-v0.8', "
                "rubric_sha256 = 'old-sha', status = 'running' WHERE job_id = ?",
                (retired_job["job_id"],),
            )
            connection.execute(
                "UPDATE llm_job_items SET status = 'running', locked_at = 'old-lock' "
                "WHERE job_id = ?",
                (retired_job["job_id"],),
            )
            connection.execute(
                "UPDATE llm_jobs SET status = 'running' WHERE job_id = ?",
                (current_job["job_id"],),
            )
            connection.execute(
                "UPDATE llm_job_items SET status = 'running', locked_at = 'new-lock' "
                "WHERE job_id = ?",
                (current_job["job_id"],),
            )
            connection.commit()
            connection.close()

            runtime.initialize()
            self.assertEqual(
                "cancelled",
                runtime.get_job("reviewer", retired_job["job_id"])["status"],
            )
            retired_item = runtime.list_items(
                "reviewer", retired_job["job_id"]
            )["items"][0]
            self.assertEqual("cancelled", retired_item["status"])
            self.assertEqual("rubric_retired", retired_item["error_code"])

            self.assertEqual(
                "queued",
                runtime.get_job("reviewer", current_job["job_id"])["status"],
            )
            current_item = runtime.list_items(
                "reviewer", current_job["job_id"]
            )["items"][0]
            self.assertEqual("pending", current_item["status"])
            claimed = runtime._claim_item()
            self.assertEqual(current_job["job_id"], claimed["job_id"])

    async def test_http_429_waits_and_gets_four_attempts(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        config = JudgeConfig(
            api_key="server-only-key",
            models=("test/model",),
            openrouter_url="https://openrouter.test/chat/completions",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            database = Path(tempdir) / "rate-limit.sqlite3"

            def connection_factory():
                connection = sqlite3.connect(database, timeout=30)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys=ON")
                return connection

            connection = connection_factory()
            connection.execute(
                "CREATE TABLE users(username TEXT PRIMARY KEY, created_at TEXT, last_seen_at TEXT)"
            )
            connection.execute("INSERT INTO users VALUES ('rate-user', 'now', 'now')")
            connection.commit()
            connection.close()

            sample = {
                "trigger_type": "message",
                "messages": [
                    {"role": "user", "display_role": "user", "content": "Continue"}
                ],
                "candidate": {"role": "assistant", "content": "Candidate"},
            }
            request_count = 0

            def handler(request: httpx.Request) -> httpx.Response:
                nonlocal request_count
                request_count += 1
                return httpx.Response(
                    429,
                    headers={"Retry-After": "7"},
                    json={"error": {"message": "rate limited"}},
                )

            runtime = LLMJudgeRuntime(
                config=config,
                rubric=rubric,
                connection_factory=connection_factory,
                sample_loader=lambda seq: ("eval-rate", sample),
                utc_now=lambda: "2026-08-31T00:00:00+00:00",
            )
            runtime.initialize()
            runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            delays: list[float] = []

            async def record_delay(delay: float) -> None:
                delays.append(delay)

            runtime._retry_sleep = record_delay
            job = runtime.create_job(
                username="rate-user",
                model="test/model",
                sequences=[1],
                scope={"type": "current", "seq": 1},
                budget_limit_usd=1,
            )
            for attempt in range(1, 5):
                item = runtime._claim_item()
                self.assertIsNotNone(item)
                self.assertEqual(attempt, item["attempt_number"])
                await runtime._execute_item(item)

            final_job = runtime.get_job("rate-user", job["job_id"])
            self.assertEqual("completed_with_errors", final_job["status"])
            self.assertEqual(1, final_job["counts"]["failed"])
            self.assertEqual(4, request_count)
            self.assertEqual(3, len(delays))
            self.assertTrue(all(delay >= 7.0 for delay in delays))
            items = runtime.list_items("rate-user", job["job_id"])["items"]
            self.assertEqual(4, items[0]["attempt_count"])
            self.assertEqual("openrouter_http_429", items[0]["error_code"])
            connection = connection_factory()
            try:
                attempts = connection.execute(
                    "SELECT status, error_code FROM llm_judgments WHERE job_id = ? "
                    "ORDER BY attempt_number",
                    (job["job_id"],),
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(4, len(attempts))
            self.assertTrue(
                all(row["error_code"] == "openrouter_http_429" for row in attempts)
            )
            await runtime._client.aclose()
            runtime._client = None

    async def test_background_item_is_validated_persisted_and_costed(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        config = JudgeConfig(
            api_key="server-only-key",
            models=("test/model",),
            openrouter_url="https://openrouter.test/chat/completions",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
            reasoning_models=("test/model",),
        )
        tempdir = tempfile.TemporaryDirectory()
        database = Path(tempdir.name) / "runtime.sqlite3"

        def connection_factory():
            connection = sqlite3.connect(database, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            return connection

        connection = connection_factory()
        try:
            connection.execute(
                "CREATE TABLE users(username TEXT PRIMARY KEY, created_at TEXT, last_seen_at TEXT)"
            )
            connection.execute(
                "INSERT INTO users VALUES ('runtime-user', 'now', 'now')"
            )
            connection.commit()
        finally:
            connection.close()

        sample = {
            "trigger_type": "message",
            "messages": [
                {"role": "system", "display_role": "system", "content": "Role card"},
                {"role": "user", "display_role": "user", "content": "Continue"},
            ],
            "candidate": {"role": "assistant", "content": "Candidate response"},
        }

        def sample_loader(seq: int):
            self.assertEqual(1, seq)
            return "eval-runtime", sample

        judgment = {"gate_status": "completed", "gates": {}, "atomic_scoring_status": "completed", "scores": {}}
        for group in rubric["gates"]:
            for gate in group["items"]:
                judgment["gates"][gate["id"]] = {
                    "value": gate["default"],
                    "evidence": [],
                    "reason": "按待评数据判断。",
                }
        for dimension in rubric["dimensions"]:
            for item in dimension["items"]:
                judgment["scores"][item["id"]] = {
                    "value": item["scale"][0],
                    "evidence": [],
                    "reason": "按该项第一步判定。",
                }

        phase_names: list[str] = []
        discard_mode = {"enabled": False}

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual("Bearer server-only-key", request.headers["Authorization"])
            request_body = json.loads(request.content)
            self.assertNotIn("max_tokens", request_body)
            self.assertEqual(
                {"effort": "low", "exclude": True},
                request_body["reasoning"],
            )
            phase_name = request_body["response_format"]["json_schema"]["name"]
            phase_names.append(phase_name)
            if phase_name == "roleplay_gate_v14":
                self.assertIn("## Gate 层", request_body["messages"][0]["content"])
                self.assertNotIn("## 维度一", request_body["messages"][0]["content"])
                gate_values = json.loads(json.dumps(judgment["gates"]))
                if discard_mode["enabled"]:
                    gate_values["G-P2"]["value"] = True
                content = {"gates": gate_values}
            else:
                self.assertEqual("roleplay_scores_v16", phase_name)
                self.assertIn("## 维度一", request_body["messages"][0]["content"])
                self.assertNotIn("## Gate 层", request_body["messages"][0]["content"])
                self.assertIn(
                    '"G-D1":false', request_body["messages"][1]["content"]
                )
                content = {"scores": judgment["scores"]}
            return httpx.Response(
                200,
                json={
                    "model": "test/model-resolved",
                    "provider": "test-provider",
                    "choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 100,
                        "cost": 0.006,
                    },
                },
            )

        runtime = LLMJudgeRuntime(
            config=config,
            rubric=rubric,
            connection_factory=connection_factory,
            sample_loader=sample_loader,
            utc_now=lambda: "2026-08-29T00:00:00+00:00",
        )
        runtime.initialize()
        runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        job = runtime.create_job(
            username="runtime-user",
            model="test/model",
            sequences=[1],
            scope={"type": "current", "seq": 1},
            budget_limit_usd=1,
        )
        item = runtime._claim_item()
        self.assertIsNotNone(item)
        await runtime._execute_item(item)
        completed = runtime.get_job("runtime-user", job["job_id"])
        self.assertEqual("completed", completed["status"])
        self.assertEqual(1, completed["counts"]["completed"])
        self.assertAlmostEqual(0.012, completed["actual_cost_usd"])
        self.assertEqual(["roleplay_gate_v14", "roleplay_scores_v16"], phase_names)
        items = runtime.list_items("runtime-user", job["job_id"])["items"]
        result = runtime.get_judgment(
            "runtime-user", job["job_id"], items[0]["run_id"]
        )
        self.assertEqual("completed", result["status"])
        self.assertEqual("pass", result["judgment"]["gate_decision"])
        self.assertEqual("0", result["judgment"]["scores"]["H1"]["value"])
        self.assertNotIn("confidence", result["judgment"]["scores"]["H1"])
        self.assertEqual("str", result["raw_output"]["gate"]["content_type"])
        self.assertEqual(
            judgment["gates"],
            result["raw_output"]["gate"]["parsed_json"]["gates"],
        )

        discard_mode["enabled"] = True
        phase_names.clear()
        discarded_job = runtime.create_job(
            username="runtime-user",
            model="test/model",
            sequences=[1],
            scope={"type": "current", "seq": 1},
            budget_limit_usd=1,
        )
        discarded_item = runtime._claim_item()
        self.assertIsNotNone(discarded_item)
        await runtime._execute_item(discarded_item)
        discarded = runtime.get_job("runtime-user", discarded_job["job_id"])
        self.assertEqual("completed", discarded["status"])
        self.assertEqual(1, discarded["counts"]["discarded"])
        self.assertAlmostEqual(0.006, discarded["actual_cost_usd"])
        self.assertEqual(["roleplay_gate_v14"], phase_names)
        discarded_items = runtime.list_items(
            "runtime-user", discarded_job["job_id"]
        )["items"]
        discarded_result = runtime.get_judgment(
            "runtime-user", discarded_job["job_id"], discarded_items[0]["run_id"]
        )
        self.assertEqual("discarded", discarded_result["status"])
        self.assertEqual("discard", discarded_result["judgment"]["gate_decision"])
        self.assertEqual(["G-P2"], discarded_result["judgment"]["failed_gate_ids"])
        self.assertIsNone(discarded_result["judgment"]["scores"])
        await runtime._client.aclose()
        runtime._client = None
        tempdir.cleanup()

    async def test_comparison_groups_the_same_sequences_for_multiple_models(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        config = JudgeConfig(
            api_key="server-only-key",
            models=("test/model-a", "test/model-b"),
            openrouter_url="https://openrouter.test/chat/completions",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=1000,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            database = Path(tempdir) / "comparison.sqlite3"

            def connection_factory():
                connection = sqlite3.connect(database, timeout=30)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys=ON")
                return connection

            connection = connection_factory()
            connection.execute(
                "CREATE TABLE users(username TEXT PRIMARY KEY, created_at TEXT, last_seen_at TEXT)"
            )
            connection.execute("INSERT INTO users VALUES ('aligner', 'now', 'now')")
            connection.commit()
            connection.close()

            def sample_loader(seq: int):
                if seq < 1 or seq > 10:
                    raise IndexError(seq)
                return f"eval-{seq}", {
                    "trigger_type": "message",
                    "messages": [{"role": "user", "display_role": "user", "content": "Continue"}],
                    "candidate": {"role": "assistant", "content": f"Candidate {seq}"},
                }

            runtime = LLMJudgeRuntime(
                config=config,
                rubric=rubric,
                connection_factory=connection_factory,
                sample_loader=sample_loader,
                utc_now=lambda: "2026-08-30T00:00:00+00:00",
            )
            runtime.initialize()
            comparison = runtime.create_comparison(
                username="aligner",
                models=["test/model-a", "test/model-b"],
                sequences=[3, 1],
                scope={"type": "random", "count": 2},
                budget_per_model_usd=2,
            )
            self.assertEqual(["test/model-a", "test/model-b"], comparison["models"])
            self.assertEqual(2, comparison["total_count"])
            self.assertEqual(4, comparison["total_budget_limit_usd"])
            self.assertEqual(2, len(comparison["jobs"]))
            self.assertEqual(2, len(comparison["metrics"]["samples"]))
            for job in comparison["jobs"]:
                items = runtime.list_items("aligner", job["job_id"])["items"]
                self.assertEqual([3, 1], [item["seq"] for item in items])
            listed = runtime.list_comparisons("aligner")
            self.assertEqual(comparison["comparison_id"], listed[0]["comparison_id"])
            with self.assertRaisesRegex(ValueError, "2–20"):
                runtime.create_comparison(
                    username="aligner",
                    models=["test/model-a"],
                    sequences=[1],
                    scope={"type": "current", "seq": 1},
                    budget_per_model_usd=1,
                )

    async def test_comparison_metrics_keep_missing_results_out_of_agreement(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        config = JudgeConfig(
            api_key="key",
            models=("model/a", "model/b", "model/c"),
            openrouter_url="https://example.invalid",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
        )
        runtime = LLMJudgeRuntime(
            config=config,
            rubric=rubric,
            connection_factory=lambda: None,
            sample_loader=lambda seq: (str(seq), {}),
            utc_now=lambda: "now",
        )
        gates = {
            gate["id"]: {"value": gate["default"], "reason": "ok", "evidence": []}
            for group in rubric["gates"]
            for gate in group["items"]
        }
        scores_a = {
            item["id"]: {"value": item["scale"][-1], "reason": "a", "evidence": []}
            for dimension in rubric["dimensions"]
            for item in dimension["items"]
        }
        scores_b = json.loads(json.dumps(scores_a))
        scores_b["C1"]["value"] = "0"
        rows = [
            {
                "seq": 1,
                "sample_position": 1,
                "model": "model/a",
                "status": "completed",
                "gates_json": json.dumps(gates),
                "scores_json": json.dumps(scores_a),
            },
            {
                "seq": 1,
                "sample_position": 1,
                "model": "model/b",
                "status": "completed",
                "gates_json": json.dumps(gates),
                "scores_json": json.dumps(scores_b),
            },
            {
                "seq": 1,
                "sample_position": 1,
                "model": "model/c",
                "status": "invalid",
                "gates_json": None,
                "scores_json": None,
            },
            {
                "seq": 2,
                "sample_position": 2,
                "model": "model/a",
                "status": "completed",
                "gates_json": json.dumps(gates),
                "scores_json": json.dumps(scores_a),
            },
            {
                "seq": 2,
                "sample_position": 2,
                "model": "model/b",
                "status": "pending",
                "gates_json": None,
                "scores_json": None,
            },
        ]
        metrics = runtime._comparison_metrics(
            ["model/a", "model/b", "model/c"], rows
        )
        item_count = sum(
            len(dimension["items"]) for dimension in rubric["dimensions"]
        )
        gate_count = sum(len(group["items"]) for group in rubric["gates"])
        self.assertEqual(20, metrics["atomic"]["comparable"])
        self.assertEqual(19, metrics["atomic"]["agreed"])
        self.assertEqual(1, metrics["atomic"]["disagreed"])
        self.assertEqual(3, metrics["atomic"]["excluded_model_samples"])
        self.assertEqual(0, metrics["atomic"]["full_coverage_comparable"])
        self.assertIsNone(metrics["atomic"]["full_coverage_agreement_rate"])
        self.assertEqual(0, metrics["atomic"]["full_coverage_within_one"])
        self.assertIsNone(metrics["atomic"]["full_coverage_within_one_rate"])
        self.assertEqual(
            2 * 3 * item_count,
            metrics["atomic"]["coverage"]["possible_ratings"],
        )
        self.assertEqual(
            3 * item_count,
            metrics["atomic"]["coverage"]["available_ratings"],
        )
        self.assertEqual(0.5, metrics["atomic"]["coverage"]["rate"])
        self.assertEqual(item_count, metrics["atomic"]["coverage"]["partial_units"])
        self.assertEqual(
            item_count,
            metrics["atomic"]["coverage"]["insufficient_units"],
        )
        self.assertEqual(20, metrics["atomic"]["pairwise"]["comparable"])
        self.assertEqual(19, metrics["atomic"]["pairwise"]["exact"])
        self.assertEqual(19, metrics["atomic"]["pairwise"]["adjacent"])
        self.assertEqual(1, metrics["atomic"]["pairwise"]["severe"])
        self.assertEqual(1, metrics["atomic"]["pairwise"]["numeric_far"])
        self.assertEqual(0, metrics["atomic"]["pairwise"]["na_mismatch"])
        self.assertEqual(8, metrics["gates"]["comparable"])
        self.assertEqual(3, metrics["gates"]["excluded_model_samples"])
        self.assertEqual(
            2 * 3 * gate_count,
            metrics["gates"]["coverage"]["possible_ratings"],
        )
        self.assertEqual(2, metrics["samples"][0]["atomic_participant_count"])
        self.assertEqual(2, metrics["samples"][0]["gate_participant_count"])
        self.assertEqual(
            3 * item_count,
            metrics["samples"][0]["atomic_expected_ratings"],
        )
        self.assertEqual(
            2 * item_count,
            metrics["samples"][0]["atomic_available_ratings"],
        )
        self.assertEqual(0, metrics["samples"][1]["atomic_comparable"])
        self.assertEqual(1, metrics["samples"][1]["atomic_participant_count"])
        self.assertEqual(20, metrics["atomic_pairwise"][0]["comparable"])
        self.assertEqual(2 * item_count, metrics["atomic_pairwise"][0]["possible"])
        self.assertEqual(8, metrics["gate_pairwise"][0]["comparable"])
        self.assertEqual(metrics["atomic_pairwise"], metrics["pairwise"])
        self.assertEqual(item_count, len(metrics["atomic_items"]))
        c1 = next(item for item in metrics["atomic_items"] if item["id"] == "C1")
        self.assertTrue(c1["name"])
        self.assertEqual(["0", "1", "2", "3"], c1["scale"])
        self.assertEqual(1, c1["pairwise"]["severe"])
        self.assertEqual("current", metrics["rubric_metadata"]["status"])
        self.assertEqual(
            rubric["source_sha256"],
            metrics["rubric_metadata"]["comparison_sha256"],
        )

        malformed_gate_rows = [dict(row) for row in rows]
        malformed_gate_a = json.loads(malformed_gate_rows[0]["gates_json"])
        malformed_gate_a["G-D1"]["value"] = []
        malformed_gate_rows[0]["gates_json"] = json.dumps(malformed_gate_a)
        malformed_gate_b = json.loads(malformed_gate_rows[1]["gates_json"])
        malformed_gate_b["G-D1"]["value"] = 0
        malformed_gate_rows[1]["gates_json"] = json.dumps(malformed_gate_b)
        malformed_gate_metrics = runtime._comparison_metrics(
            ["model/a", "model/b", "model/c"], malformed_gate_rows
        )
        malformed_gd1 = next(
            item
            for item in malformed_gate_metrics["gate_items"]
            if item["id"] == "G-D1"
        )
        self.assertEqual(1, malformed_gd1["coverage"]["available_ratings"])
        self.assertEqual(5, malformed_gd1["coverage"]["missing_ratings"])
        self.assertEqual(0, malformed_gd1["comparable"])

    async def test_comparison_metrics_treat_na_as_a_real_comparison_category(self) -> None:
        rubric = build_rubric(RUBRIC_MARKDOWN)
        config = JudgeConfig(
            api_key="key",
            models=("model/a", "model/b", "model/c"),
            openrouter_url="https://example.invalid",
            max_concurrency=1,
            max_output_tokens=12000,
            max_job_samples=100,
            max_job_budget_usd=10,
            request_timeout_seconds=30,
        )
        runtime = LLMJudgeRuntime(
            config=config,
            rubric=rubric,
            connection_factory=lambda: None,
            sample_loader=lambda seq: (str(seq), {}),
            utc_now=lambda: "now",
        )
        base_scores = {
            item["id"]: {
                "value": item["scale"][-1],
                "reason": "ok",
                "evidence": [],
            }
            for dimension in rubric["dimensions"]
            for item in dimension["items"]
        }

        def scores_with_h2(
            value: str,
            logprob_confidence: float = 0.9,
        ) -> dict:
            scores = json.loads(json.dumps(base_scores))
            scores["H2"]["value"] = value
            scores["H2"]["logprob_confidence"] = logprob_confidence
            return scores

        h2_values = [
            ("N/A", "N/A", "2"),
            ("1", "2", "3"),
            ("2", "2", "2"),
        ]
        rows = []
        for position, values in enumerate(h2_values, 1):
            for model, value in zip(("model/a", "model/b", "model/c"), values):
                logprob_confidence = (
                    0.4 if position == 1 and model == "model/c" else 0.9
                )
                rows.append(
                    {
                        "seq": position,
                        "sample_position": position,
                        "model": model,
                        "status": "completed",
                        "gates_json": None,
                        "scores_json": json.dumps(
                            scores_with_h2(
                                value,
                                logprob_confidence,
                            )
                        ),
                    }
                )

        metrics = runtime._comparison_metrics(
            ["model/a", "model/b", "model/c"], rows
        )
        h2 = next(item for item in metrics["atomic_items"] if item["id"] == "H2")
        self.assertEqual(3, h2["comparable"])
        self.assertEqual(1, h2["agreed"])
        self.assertEqual(3, h2["full_coverage_comparable"])
        self.assertEqual(1, h2["full_coverage_agreed"])
        self.assertEqual(1, h2["full_coverage_within_one"])
        self.assertAlmostEqual(
            1 / 3, h2["full_coverage_within_one_rate"], places=6
        )
        self.assertEqual(9, h2["coverage"]["possible_ratings"])
        self.assertEqual(9, h2["coverage"]["available_ratings"])
        self.assertEqual(2, h2["coverage"]["na_ratings"])
        self.assertEqual(4, h2["pairwise"]["exact"])
        self.assertEqual(6, h2["pairwise"]["adjacent"])
        self.assertEqual(3, h2["pairwise"]["severe"])
        self.assertEqual(1, h2["pairwise"]["numeric_far"])
        self.assertEqual(2, h2["pairwise"]["na_mismatch"])
        self.assertAlmostEqual(4 / 9, h2["pairwise"]["exact_rate"], places=6)
        self.assertAlmostEqual(6 / 9, h2["pairwise"]["adjacent_rate"], places=6)
        self.assertAlmostEqual(3 / 9, h2["pairwise"]["severe_rate"], places=6)
        logprob = h2["confidence"]["logprob"]
        self.assertEqual({"logprob"}, set(h2["confidence"]))
        self.assertEqual(9, logprob["recorded_ratings"])
        self.assertEqual(1.0, logprob["coverage_rate"])
        self.assertEqual(9, logprob["distance"]["comparable"])
        self.assertAlmostEqual(10 / 27, logprob["distance"]["raw_mean"], 6)
        self.assertAlmostEqual(2.4 / 9, logprob["distance"]["adjusted_mean"], 6)

        historical = runtime._comparison_metrics(
            ["model/a", "model/b", "model/c"],
            rows,
            rubric_version="roleplay-rubric-historical",
        )
        self.assertEqual(
            "historical_snapshot_unavailable",
            historical["rubric_metadata"]["status"],
        )
        stale_same_version = runtime._comparison_metrics(
            ["model/a", "model/b", "model/c"],
            rows,
            rubric_version=rubric["version"],
            rubric_sha256="stale-sha256",
        )
        self.assertEqual(
            "historical_snapshot_unavailable",
            stale_same_version["rubric_metadata"]["status"],
        )
        self.assertEqual(
            "stale-sha256",
            stale_same_version["rubric_metadata"]["comparison_sha256"],
        )

        malformed_rows = [dict(row) for row in rows]
        malformed_l6 = json.loads(malformed_rows[1]["scores_json"])
        malformed_l6["L6"]["value"] = "N/A"
        malformed_rows[1]["scores_json"] = json.dumps(malformed_l6)
        malformed = runtime._comparison_metrics(
            ["model/a", "model/b", "model/c"], malformed_rows
        )
        malformed_h1_row = next(
            item for item in malformed["atomic_items"] if item["id"] == "H1"
        )
        malformed_l6_row = next(
            item for item in malformed["atomic_items"] if item["id"] == "L6"
        )
        self.assertEqual(9, malformed_h1_row["coverage"]["available_ratings"])
        self.assertEqual(0, malformed_h1_row["coverage"]["missing_ratings"])
        self.assertEqual(8, malformed_l6_row["coverage"]["available_ratings"])
        self.assertEqual(1, malformed_l6_row["coverage"]["missing_ratings"])
        historical_scale = runtime._comparison_metrics(
            ["model/a", "model/b", "model/c"],
            malformed_rows,
            rubric_version="roleplay-rubric-historical",
        )
        historical_l6 = next(
            item for item in historical_scale["atomic_items"] if item["id"] == "L6"
        )
        historical_h1 = next(
            item for item in historical_scale["atomic_items"] if item["id"] == "H1"
        )
        self.assertEqual(9, historical_h1["coverage"]["available_ratings"])
        self.assertEqual(0, historical_h1["coverage"]["missing_ratings"])
        self.assertEqual(9, historical_l6["coverage"]["available_ratings"])
        self.assertEqual(0, historical_l6["coverage"]["missing_ratings"])


class PlatformApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tempdir.name)
        source = cls.root / "samples.jsonl.gz"
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(sample_record("eval_one", "Candidate one")) + "\n")
            handle.write(json.dumps(sample_record("eval_two", "Candidate two")) + "\n")
        cls.sample_db = cls.root / "samples.sqlite3"
        build_samples_db(source, cls.sample_db)
        cls.labels_csv = cls.root / "evaluation_labels.csv"
        cls.labels_csv.write_text(
            "evaluation_id,pair_group_id,model_id,model_name,human_selected,source_completed_at\n"
            "eval_one,pair_one,1,model-one,1,2026-08-26 12:00:00\n"
            "eval_two,pair_one,2,model-two,0,2026-08-26 12:00:00\n",
            encoding="utf-8",
        )
        cls.rubric_json = cls.root / "rubric.json"
        cls.rubric_json.write_text(
            RUBRIC_JSON.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        cls.rubric = json.loads(cls.rubric_json.read_text(encoding="utf-8"))
        os.environ["ROLEPLAY_SAMPLE_DB"] = str(cls.sample_db)
        os.environ["ROLEPLAY_RESULTS_DB"] = str(cls.root / "results" / "annotations.sqlite3")
        os.environ["ROLEPLAY_RESULTS_DIR"] = str(cls.root / "results" / "by_user")
        os.environ["ROLEPLAY_RUBRIC_JSON"] = str(cls.rubric_json)
        os.environ["ROLEPLAY_LABELS_CSV"] = str(cls.labels_csv)
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("ROLEPLAY_LLM_MODELS", None)
        os.environ["ROLEPLAY_LLM_MODEL_SOURCE"] = "static"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        environment = os.environ.copy()
        cls.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.evaluation.roleplay_annotation_platform.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(cls.port),
                "--log-level",
                "warning",
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            if cls.server.poll() is not None:
                raise RuntimeError("local annotation server exited during startup")
            try:
                with urllib.request.urlopen(f"{cls.base_url}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(0.05)
        else:
            raise RuntimeError("local annotation server did not become ready")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server.kill()
        cls.tempdir.cleanup()

    @classmethod
    def client(cls):
        cookies = http.cookiejar.CookieJar()
        return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))

    @classmethod
    def request_json(cls, client, path: str, method: str = "GET", payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{cls.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with client.open(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    @staticmethod
    def gates() -> dict:
        return {
            "G-D1": False,
            "G-D2": False,
            "G-D3": False,
            "G-P1": False,
            "G-P2": False,
            "G-P3": False,
            "G-P4": False,
            "G-P5": False,
        }

    def score_payload(self, value: int = 2, evidence: str = "") -> dict:
        return {
            item["id"]: {"value": value, "evidence": evidence}
            for dimension in self.rubric["dimensions"]
            for item in dimension["items"]
        }

    def rubric_identity(self) -> dict[str, str]:
        return {
            "rubric_version": self.rubric["version"],
            "rubric_sha256": self.rubric["source_sha256"],
        }

    def test_same_samples_but_independent_user_results(self) -> None:
        alice = self.client()
        status, _ = self.request_json(
            alice, "/api/login", "POST", {"username": "alice"}
        )
        self.assertEqual(200, status)
        status, sample = self.request_json(alice, "/api/sample/1")
        self.assertEqual(200, status)
        self.assertEqual("Candidate one", sample["candidate"]["content"])
        self.assertNotIn("evaluation_id", sample)
        self.assertEqual("greeting", sample["messages"][1]["display_role"])
        status, saved = self.request_json(
            alice,
            "/api/annotation",
            "POST",
            {
                "seq": 1,
                "status": "submitted",
                **self.rubric_identity(),
                "gates": self.gates(),
                "scores": self.score_payload(),
                "gate_notes": "",
                "overall_notes": "",
            },
        )
        self.assertEqual(200, status, saved)
        self.assertEqual(1, saved["progress"]["submitted"])

        bob = self.client()
        self.request_json(bob, "/api/login", "POST", {"username": "bob"})
        status, sample = self.request_json(bob, "/api/sample/1")
        self.assertEqual(200, status)
        self.assertEqual("Candidate one", sample["candidate"]["content"])
        self.assertIsNone(sample["annotation"])
        self.assertEqual(0, sample["progress"]["submitted"])

        snapshot = self.root / "results" / "by_user" / "alice" / "000001.json"
        self.assertTrue(snapshot.is_file())
        record = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual("eval_one", record["evaluation_id"])
        self.assertEqual("submitted", record["status"])

    def test_score_three_allows_empty_evidence(self) -> None:
        scores = self.score_payload()
        scores["H2"] = {"value": 3, "evidence": ""}
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "charlie"})
        status, response = self.request_json(
            client,
            "/api/annotation",
            "POST",
            {
                "seq": 1,
                "status": "submitted",
                **self.rubric_identity(),
                "gates": self.gates(),
                "scores": scores,
                "gate_notes": "",
                "overall_notes": "",
            },
        )
        self.assertEqual(200, status)
        self.assertEqual("submitted", response["status"])

    def test_item_rejects_a_score_outside_the_common_scale(self) -> None:
        scores = self.score_payload()
        scores["C1"] = {"value": 4, "evidence": ""}
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "delta"})
        status, response = self.request_json(
            client,
            "/api/annotation",
            "POST",
            {
                "seq": 1,
                "status": "submitted",
                **self.rubric_identity(),
                "gates": self.gates(),
                "scores": scores,
                "gate_notes": "",
                "overall_notes": "",
            },
        )
        self.assertEqual(400, status)
        self.assertIn("C1 分数无效", response["detail"])

    def test_stale_human_page_cannot_save_under_a_new_rubric(self) -> None:
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "stale-page"})
        status, response = self.request_json(
            client,
            "/api/annotation",
            "POST",
            {
                "seq": 1,
                "status": "draft",
                "rubric_version": "roleplay-rubric-stale",
                "rubric_sha256": "0" * 64,
                "gates": {},
                "scores": {},
                "gate_notes": "",
                "overall_notes": "",
            },
        )
        self.assertEqual(409, status)
        self.assertIn("完整刷新页面", response["detail"])

    def test_battle_replays_pair_and_hides_model_identity(self) -> None:
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "viewer"})
        status, response = self.request_json(client, "/api/battle/1")
        self.assertEqual(200, status)
        self.assertEqual(1, response["pair_seq"])
        self.assertEqual(1, response["total"])
        self.assertTrue(response["context_match"])
        self.assertEqual({"A", "B"}, {candidate["label"] for candidate in response["candidates"]})
        self.assertEqual(
            {"Candidate one", "Candidate two"},
            {candidate["content"] for candidate in response["candidates"]},
        )
        self.assertEqual(1, sum(candidate["online_selected"] for candidate in response["candidates"]))
        serialized = json.dumps(response)
        self.assertNotIn("model-one", serialized)
        self.assertNotIn("model-two", serialized)
        self.assertNotIn("pair_one", serialized)
        self.assertNotIn("eval_one", serialized)
        self.assertTrue(all("seq" not in candidate for candidate in response["candidates"]))

        status, models = self.request_json(client, "/api/battle/1/models")
        self.assertEqual(200, status)
        self.assertEqual(
            {"model-one", "model-two"},
            {model["model_name"] for model in models["models"]},
        )

    def test_llm_judge_exposes_schema_but_never_a_server_key(self) -> None:
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "judge-viewer"})
        status, config = self.request_json(client, "/api/llm-judge/config")
        self.assertEqual(200, status)
        self.assertFalse(config["enabled"])
        self.assertNotIn("api_key", json.dumps(config))
        self.assertEqual("static", config["model_catalog"]["source"])
        self.assertGreaterEqual(config["model_catalog"]["count"], 1)
        self.assertEqual(
            {"min_models": 2, "max_models": 20, "max_samples": 500},
            config["comparison_limits"],
        )
        self.assertEqual(1, config["battle_limits"]["pair_count"])
        self.assertEqual(
            config["max_job_samples"] // 2,
            config["battle_limits"]["max_pairs_per_job"],
        )
        status, schema = self.request_json(client, "/api/llm-judge/schema")
        self.assertEqual(200, status)
        self.assertEqual(
            8,
            len(
                schema["schema"]["phase_1_gate"]["properties"]["gates"]
                ["properties"]
            ),
        )
        self.assertEqual(
            20,
            len(
                schema["schema"]["phase_2_dimensions"]["properties"]["scores"]
                ["properties"]
            ),
        )
        self.assertEqual(64, len(schema["schema_sha256"]))

    def test_security_policy_prevents_browser_calls_to_openrouter(self) -> None:
        client = self.client()
        request = urllib.request.Request(f"{self.base_url}/llm-judge")
        with client.open(request, timeout=5) as response:
            policy = response.headers["Content-Security-Policy"]
            body = response.read().decode("utf-8")
        self.assertIn("connect-src 'self'", policy)
        self.assertNotIn("sk-or-v1-", body)

    def test_multi_model_compare_page_and_api_are_available(self) -> None:
        client = self.client()
        with client.open(f"{self.base_url}/llm-compare", timeout=5) as response:
            body = response.read().decode("utf-8")
        self.assertIn("多模型评分对齐", body)
        self.assertIn("原子评分一致率", body)
        self.assertIn("Gate 一致率", body)
        self.assertIn("只显示实际参与模型之间的分歧", body)
        self.assertIn("/static/llm_compare.js", body)
        self.request_json(client, "/api/login", "POST", {"username": "align-viewer"})
        status, payload = self.request_json(client, "/api/llm-judge/comparisons")
        self.assertEqual(200, status)
        self.assertEqual([], payload["comparisons"])

    def test_llm_judge_battle_page_and_api_are_available(self) -> None:
        client = self.client()
        with client.open(f"{self.base_url}/llm-battle", timeout=5) as response:
            body = response.read().decode("utf-8")
        self.assertIn("LLM Judge Battle", body)
        self.assertIn("/static/llm_battle.js", body)

        self.request_json(client, "/api/login", "POST", {"username": "battle-judge"})
        status, config = self.request_json(client, "/api/llm-judge/config")
        self.assertEqual(200, status)
        status, payload = self.request_json(client, "/api/llm-judge/battles")
        self.assertEqual(200, status)
        self.assertEqual([], payload["battles"])

        status, error = self.request_json(
            client,
            "/api/llm-judge/battles",
            "POST",
            {
                "model": config["models"][0]["id"],
                "scope_type": "current",
                "pair_seq": 1,
                "budget_limit_usd": 1,
            },
        )
        self.assertEqual(400, status)
        self.assertIn("OPENROUTER_API_KEY", error["detail"])

    def test_llm_request_preview_matches_real_builder_and_redacts_secrets(self) -> None:
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "previewer"})
        status, config = self.request_json(client, "/api/llm-judge/config")
        self.assertEqual(200, status)
        self.assertGreaterEqual(len(config["models"]), 1)
        model = config["models"][0]["id"]
        query_model = urllib.parse.quote(model, safe="")
        status, preview = self.request_json(
            client,
            f"/api/llm-judge/request-preview?seq=1&model={query_model}",
        )
        self.assertEqual(200, status)
        self.assertEqual(model, preview["model"])
        self.assertEqual("POST", preview["method"])
        self.assertEqual(64, len(preview["request_hashes"]["gate"]))
        self.assertEqual(64, len(preview["request_hashes"]["dimensions"]))
        self.assertEqual(
            "Bearer <SERVER_ONLY_REDACTED>", preview["headers"]["Authorization"]
        )
        self.assertEqual(
            "Candidate one",
            preview["sample_packet"]["candidate_response"]["content"],
        )
        self.assertEqual(2, len(preview["phases"]))
        gate_body = preview["phases"][0]["request_body"]
        score_body = preview["phases"][1]["request_body"]
        self.assertTrue(gate_body["response_format"]["json_schema"]["strict"])
        self.assertEqual(
            8,
            len(
                gate_body["response_format"]["json_schema"]["schema"]
                ["properties"]["gates"]["properties"]
            ),
        )
        self.assertEqual(
            20,
            len(
                score_body["response_format"]["json_schema"]["schema"]
                ["properties"]["scores"]["properties"]
            ),
        )
        serialized = json.dumps(preview)
        self.assertNotIn("OPENROUTER_API_KEY", serialized)
        self.assertNotIn("eval_one", serialized)
        self.assertNotIn("model-one", serialized)
        self.assertNotIn("human_selected", serialized)

    def test_llm_job_creation_fails_cleanly_when_server_key_is_not_configured(self) -> None:
        client = self.client()
        self.request_json(client, "/api/login", "POST", {"username": "judge-disabled"})
        status, response = self.request_json(
            client,
            "/api/llm-judge/jobs",
            "POST",
            {
                "model": "test/model",
                "scope_type": "current",
                "seq": 1,
                "budget_limit_usd": 1,
            },
        )
        self.assertEqual(400, status)
        self.assertIn("OPENROUTER_API_KEY", response["detail"])


if __name__ == "__main__":
    unittest.main()
