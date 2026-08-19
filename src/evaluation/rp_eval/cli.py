from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, candidate_readiness, load_config, load_openrouter_key, load_secret_value
from .dataset import load_seeds, preflight
from .ledger import append_jsonl, utc_now, write_new_json
from .prompts import all_prompt_hashes
from .schemas import SmokeResult


def _write_preflight(report: dict[str, Any], plan: list[dict[str, Any]]) -> None:
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report_path = artifacts / "preflight.json"
    plan_path = artifacts / "staged-plan.jsonl"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with plan_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in plan:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    if os.name == "posix":
        os.chmod(report_path, 0o600)
        os.chmod(plan_path, 0o600)


def command_preflight(config: dict[str, Any]) -> int:
    report, plan = preflight(config)
    _write_preflight(report, plan)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


async def _smoke(config: dict[str, Any]) -> int:
    from .openrouter import OpenRouterClient

    nonce = secrets.token_hex(8)
    messages = [
        {"role": "system", "content": "Return exactly one JSON object and no commentary."},
        {"role": "user", "content": json.dumps({"task": "transport_smoke_test", "required": {"status": "ok", "nonce": nonce}})},
    ]
    async with OpenRouterClient(config) as client:
        record = await client.complete_json(messages=messages, schema=SmokeResult, role="verifier", request_id="transport-smoke")
    parsed = record.parsed
    provider = (record.provider or "").lower()
    model = (record.model or "").lower()
    checks = {
        "schema_valid": parsed.status == "ok" and parsed.nonce == nonce,
        "model_valid": "glm-5.2" in model,
        "provider_streamlake": "streamlake" in provider,
        "attempt_count": record.attempt_count,
    }
    output = {
        "ok": all(value is True for key, value in checks.items() if key != "attempt_count"),
        "checks": checks,
        "response_id": record.response_id,
        "model": record.model,
        "provider": record.provider,
        "usage": record.usage,
        "latency_seconds": round(record.latency_seconds, 3),
        "prompt_sha256": record.prompt_sha256,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["ok"] else 3


async def _smoke_scorers(config: dict[str, Any]) -> int:
    from .openrouter import OpenRouterClient
    from .workflow import blind_judge_pair

    character = {
        "name": "林夏",
        "description": "虚构的旧书店店员，冷静但友善。",
        "personality": "观察细致，不替对方决定行动。",
        "scenario": "雨夜的旧书店。",
        "language": "zh",
        "nsfw": False,
    }
    history = [
        {"role": "assistant", "content": "雨点敲着玻璃。我把灯调亮了一些：‘进来避避雨吧。’"},
        {"role": "user", "content": "你这里有旅行相关的旧书吗？"},
    ]
    candidate_a = "有。我从靠窗的架子上抽出一本旧游记，放在你看得清的位置：‘想看山川，还是城市？’"
    candidate_b = "当然有！你一定想看城市旅行。我替你拿起书并决定让你坐下阅读。"
    async with OpenRouterClient(config) as client:
        judgment = await blind_judge_pair(
            client,
            request_id="scorer-smoke:judge",
            random_seed=20260811,
            character_reference=character,
            history=history,
            current_user=history[-1]["content"],
            candidate_name_1="candidate_one",
            candidate_response_1=candidate_a,
            candidate_name_2="candidate_two",
            candidate_response_2=candidate_b,
        )
    records = [judgment["first"], judgment["second"]]
    provider_checks = ["streamlake" in (record.provider or "").lower() for record in records]
    output = {
        "ok": all(provider_checks),
        "judge_order_1": judgment["first"].parsed.model_dump(),
        "judge_order_2": judgment["second"].parsed.model_dump(),
        "aggregate": judgment["aggregate"],
        "assignments": {
            "order_1": judgment["first_assignment"],
            "order_2": judgment["second_assignment"],
        },
        "transport": [
            {
                "provider": record.provider,
                "model": record.model,
                "attempt_count": record.attempt_count,
                "latency_seconds": round(record.latency_seconds, 3),
                "usage": record.usage,
                "prompt_sha256": record.prompt_sha256,
            }
            for record in records
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if output["ok"] else 3


def _candidate_meta(record: Any) -> dict[str, Any]:
    return {
        "response_id": record.response_id,
        "model": record.model,
        "provider": record.provider,
        "usage": record.usage,
        "latency_seconds": round(record.latency_seconds, 3),
        "attempt_count": record.attempt_count,
        "characters": len(record.text),
        "content_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
    }


def _seed_is_nsfw(seed: dict[str, Any]) -> bool:
    value = seed.get("metadata", {}).get("nsfw")
    if isinstance(value, str):
        return value.strip().lower() == "nsfw"
    if isinstance(value, bool):
        return value
    return bool(seed.get("character", {}).get("role_card", {}).get("nsfw"))


async def _smoke_pipeline(config: dict[str, Any]) -> int:
    from .candidate_clients import chat_completion
    from .openrouter import OpenRouterClient
    from .prompts import bridge_messages, normalize_bridge_text, prompt_hash
    from .roleplay_prompts import build_deepseek_system, build_gemma_system, normalized_character_reference
    from .workflow import blind_judge_pair

    data_dir = Path(config["benchmark"]["data_dir"])
    seeds = load_seeds(data_dir / "rollout-seeds.jsonl.gz")
    eligible = [s for s in seeds if not _seed_is_nsfw(s) and len(s["ordered_user_messages"]) >= 2]
    if not eligible:
        eligible = [s for s in seeds if len(s["ordered_user_messages"]) >= 2]
    seed = eligible[0]
    sample_id = seed["sample_id"]
    role_card = seed["character"]["role_card"]
    first_user = seed["ordered_user_messages"][0]
    original_next_user = seed["ordered_user_messages"][1]
    greeting = seed["greeting"]

    gemma_system = build_gemma_system(role_card)
    deepseek_system = build_deepseek_system(role_card)
    gemma_config = config["candidates"]["gemma"]
    deepseek_config = config["candidates"]["deepseek_v4_flash"]
    bridge_config = config["bridge"]
    gemma_key = load_secret_value(gemma_config["secret_file"], gemma_config["api_key_env"])
    openrouter_key = load_openrouter_key(config)

    run_name = "smoke-pipeline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    contract = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "kind": "two-turn-single-sample-pipeline-smoke",
        "sample_id": sample_id,
        "holdout_content_used": True,
        "sample_policy": "first SFW sample in immutable order with at least two real user messages",
        "models": {
            "gemma": gemma_config["model"],
            "deepseek": deepseek_config["model"],
            "bridge": bridge_config["model"],
            "judge": config["openrouter"]["model"],
        },
        "candidate_generation": {
            name: {
                key: candidate[key]
                for key in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens")
                if key in candidate
            }
            for name, candidate in config["candidates"].items()
        },
        "prompt_sha256": {
            "gemma_system": hashlib.sha256(gemma_system.encode("utf-8")).hexdigest(),
            "deepseek_system": hashlib.sha256(deepseek_system.encode("utf-8")).hexdigest(),
            **all_prompt_hashes(),
        },
        "formal_benchmark": False,
    }
    write_new_json(run_dir / "run_contract.json", contract)
    append_jsonl(run_dir / "events.jsonl", {"at": utc_now(), "event": "run_started", "sample_id": sample_id})

    gemma_messages = [
        {"role": "system", "content": gemma_system},
        {"role": "assistant", "content": greeting},
        {"role": "user", "content": first_user},
    ]
    deepseek_messages = [
        {"role": "system", "content": deepseek_system},
        {"role": "assistant", "content": greeting},
        {"role": "user", "content": first_user},
    ]
    gemma_1, deepseek_1 = await asyncio.gather(
        chat_completion(
            url=gemma_config["base_url"], api_key=gemma_key, model=gemma_config["model"],
            messages=gemma_messages, temperature=gemma_config["temperature"], max_tokens=gemma_config["max_tokens"],
            top_p=gemma_config.get("top_p"), top_k=gemma_config.get("top_k"),
            repetition_penalty=gemma_config.get("repetition_penalty"),
        ),
        chat_completion(
            url=deepseek_config["base_url"], api_key=openrouter_key, model=deepseek_config["model"],
            messages=deepseek_messages, temperature=deepseek_config["temperature"], max_tokens=deepseek_config["max_tokens"],
            top_p=deepseek_config.get("top_p"), top_k=deepseek_config.get("top_k"),
            repetition_penalty=deepseek_config.get("repetition_penalty"),
        ),
    )
    for branch, record in (("gemma", gemma_1), ("deepseek", deepseek_1)):
        append_jsonl(run_dir / "responses.jsonl", {
            "at": utc_now(), "kind": "candidate", "branch": branch, "turn": 0,
            "content": record.text, **_candidate_meta(record),
        })

    reference = normalized_character_reference(role_card)
    visible_before_reply = [{"role": "assistant", "content": greeting}]
    random_seed = int(hashlib.sha256(sample_id.encode()).hexdigest()[:16], 16) ^ int(config["benchmark"]["random_seed"])
    branch_history_gemma = visible_before_reply + [{"role": "user", "content": first_user}]
    branch_history_deepseek = visible_before_reply + [{"role": "user", "content": first_user}]
    bridge_messages_gemma = bridge_messages(
        history=branch_history_gemma,
        candidate_reply=gemma_1.text,
        original_next_user=original_next_user,
    )
    bridge_messages_deepseek = bridge_messages(
        history=branch_history_deepseek,
        candidate_reply=deepseek_1.text,
        original_next_user=original_next_user,
    )
    async with OpenRouterClient(config) as glm:
        judgment = await blind_judge_pair(
            glm, request_id=f"{run_name}:{sample_id}:turn0", random_seed=random_seed,
            character_reference=reference, history=visible_before_reply, current_user=first_user,
            candidate_name_1="gemma", candidate_response_1=gemma_1.text,
            candidate_name_2="deepseek", candidate_response_2=deepseek_1.text,
        )

    bridge_gemma, bridge_deepseek = await asyncio.gather(
        chat_completion(
            url=bridge_config["base_url"], api_key=openrouter_key, model=bridge_config["model"],
            messages=bridge_messages_gemma, temperature=bridge_config.get("temperature"),
            max_tokens=bridge_config["max_tokens"], provider=bridge_config.get("provider"),
            reasoning_effort=bridge_config.get("reasoning_effort"),
            request_id=f"{run_name}:{sample_id}:bridge:gemma",
            timeout_seconds=bridge_config.get("timeout_seconds", 240),
            max_retries=bridge_config.get("max_retries", 2),
        ),
        chat_completion(
            url=bridge_config["base_url"], api_key=openrouter_key, model=bridge_config["model"],
            messages=bridge_messages_deepseek, temperature=bridge_config.get("temperature"),
            max_tokens=bridge_config["max_tokens"], provider=bridge_config.get("provider"),
            reasoning_effort=bridge_config.get("reasoning_effort"),
            request_id=f"{run_name}:{sample_id}:bridge:deepseek",
            timeout_seconds=bridge_config.get("timeout_seconds", 240),
            max_retries=bridge_config.get("max_retries", 2),
        ),
    )
    bridge_gemma = replace(bridge_gemma, text=normalize_bridge_text(bridge_gemma.text))
    bridge_deepseek = replace(bridge_deepseek, text=normalize_bridge_text(bridge_deepseek.text))

    scorer_records = {
        "judge_order_1": judgment["first"], "judge_order_2": judgment["second"],
    }
    for kind, record in scorer_records.items():
        append_jsonl(run_dir / "responses.jsonl", {
            "at": utc_now(), "kind": kind, "result": record.parsed.model_dump(),
            "model": record.model, "provider": record.provider, "usage": record.usage,
            "latency_seconds": record.latency_seconds, "attempt_count": record.attempt_count,
            "prompt_sha256": record.prompt_sha256,
        })
    for branch, messages, record in (
        ("gemma", bridge_messages_gemma, bridge_gemma),
        ("deepseek", bridge_messages_deepseek, bridge_deepseek),
    ):
        append_jsonl(run_dir / "responses.jsonl", {
            "at": utc_now(), "kind": "bridge", "branch": branch, "turn": 0,
            "content": record.text.strip(), "model": record.model, "provider": record.provider,
            "usage": record.usage, "latency_seconds": record.latency_seconds,
            "attempt_count": record.attempt_count, "prompt_sha256": prompt_hash(messages),
            "content_sha256": hashlib.sha256(record.text.strip().encode("utf-8")).hexdigest(),
        })

    second_calls = []
    for branch, system, messages, first_reply, bridge in (
        ("gemma", gemma_system, gemma_messages, gemma_1, bridge_gemma),
        ("deepseek", deepseek_system, deepseek_messages, deepseek_1, bridge_deepseek),
    ):
        second_messages = messages + [
            {"role": "assistant", "content": first_reply.text},
            {"role": "user", "content": bridge.text.strip()},
        ]
        cfg = gemma_config if branch == "gemma" else deepseek_config
        key = gemma_key if branch == "gemma" else openrouter_key
        second_calls.append((branch, chat_completion(
            url=cfg["base_url"], api_key=key, model=cfg["model"], messages=second_messages,
            temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
            top_p=cfg.get("top_p"), top_k=cfg.get("top_k"),
            repetition_penalty=cfg.get("repetition_penalty"),
        )))
    second_results = await asyncio.gather(*(call for _, call in second_calls)) if second_calls else []
    for (branch, _), record in zip(second_calls, second_results):
        append_jsonl(run_dir / "responses.jsonl", {
            "at": utc_now(), "kind": "candidate", "branch": branch, "turn": 1,
            "content": record.text, **_candidate_meta(record),
        })

    summary = {
        "ok": True,
        "run_dir": str(run_dir),
        "sample_id": sample_id,
        "sample_nsfw": _seed_is_nsfw(seed),
        "turns_generated": 2,
        "candidate_turn_0": {"gemma": _candidate_meta(gemma_1), "deepseek": _candidate_meta(deepseek_1)},
        "judge": judgment["aggregate"],
        "bridge": {
            "gemma": {"content": bridge_gemma.text.strip(), **_candidate_meta(bridge_gemma)},
            "deepseek": {"content": bridge_deepseek.text.strip(), **_candidate_meta(bridge_deepseek)},
        },
        "second_turn_branches": [branch for branch, _ in second_calls],
        "glm_providers": sorted({record.provider for record in scorer_records.values()}),
        "bridge_models": sorted({record.model for record in (bridge_gemma, bridge_deepseek)}),
        "bridge_providers": sorted({record.provider for record in (bridge_gemma, bridge_deepseek)}),
    }
    write_new_json(run_dir / "summary.json", summary)
    append_jsonl(run_dir / "events.jsonl", {"at": utc_now(), "event": "run_completed", "ok": True})
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_status(config: dict[str, Any]) -> int:
    try:
        load_openrouter_key(config)
        judge_key = True
    except RuntimeError:
        judge_key = False
    preflight_path = PROJECT_ROOT / "artifacts" / "preflight.json"
    preflight_ready = False
    if preflight_path.exists():
        preflight_ready = bool(json.loads(preflight_path.read_text(encoding="utf-8")).get("ready"))
    candidates = candidate_readiness(config)
    status = {
        "preflight_ready": preflight_ready,
        "judge": {
            "model": config["openrouter"]["model"],
            "provider": config["openrouter"]["provider"],
            "api_key_present": judge_key,
            "prompt_hashes": all_prompt_hashes(),
        },
        "candidates": candidates,
        "formal_run_enabled": bool(config["policy"].get("formal_run_enabled")),
        "formal_run_ready": preflight_ready and judge_key and all(x["ready"] for x in candidates.values()) and bool(config["policy"].get("formal_run_enabled")),
    }
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_init_run(config: dict[str, Any], name: str) -> int:
    if not config["policy"].get("formal_run_enabled"):
        print(json.dumps({"error": "formal benchmark is disabled; complete and approve pipeline smoke first"}, indent=2), file=sys.stderr)
        return 5
    readiness = candidate_readiness(config)
    if not all(x["ready"] for x in readiness.values()):
        print(json.dumps({"error": "candidate adapters are not ready", "candidates": readiness}, indent=2), file=sys.stderr)
        return 4
    run_dir = PROJECT_ROOT / "runs" / name
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    contract = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": name,
        "benchmark": config["benchmark"],
        "openrouter": {k: v for k, v in config["openrouter"].items() if k not in {"secret_file", "api_key_env"}},
        "glm_roles": config["glm_roles"],
        "candidates": config["candidates"],
        "policy": config["policy"],
        "prompt_hashes": all_prompt_hashes(),
        "config_sha256": hashlib.sha256(Path(config["_config_path"]).read_bytes()).hexdigest(),
    }
    write_new_json(run_dir / "run_contract.json", contract)
    print(str(run_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed 500-conversation RP benchmark evaluator")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("status")
    sub.add_parser("smoke-openrouter")
    sub.add_parser("smoke-scorers")
    sub.add_parser("smoke-pipeline")
    pilot = sub.add_parser("run-pilot20")
    pilot.add_argument("--count", type=int, default=20)
    pilot.add_argument("--concurrency", type=int, default=20)
    pilot.add_argument("--max-turns", type=int, default=10)
    pilot.add_argument("--run-name", default=None)
    formal = sub.add_parser("run-formal500")
    formal.add_argument("--concurrency", type=int, default=20)
    formal.add_argument("--run-name", default=None)
    replay = sub.add_parser("run-replay-completed")
    replay.add_argument("--source-run", required=True)
    replay.add_argument("--count", type=int, required=True)
    replay.add_argument("--concurrency", type=int, default=8)
    replay.add_argument("--run-name", default=None)
    sub.add_parser("selftest")
    init = sub.add_parser("init-run")
    init.add_argument("name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "preflight":
        return command_preflight(config)
    if args.command == "status":
        return command_status(config)
    if args.command == "smoke-openrouter":
        return asyncio.run(_smoke(config))
    if args.command == "smoke-scorers":
        return asyncio.run(_smoke_scorers(config))
    if args.command == "smoke-pipeline":
        return asyncio.run(_smoke_pipeline(config))
    if args.command == "run-pilot20":
        from .pilot import run_pilot20

        summary = asyncio.run(run_pilot20(
            config,
            count=args.count,
            concurrency=args.concurrency,
            max_turns=args.max_turns,
            run_name=args.run_name,
        ))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if summary["failed_conversations"] == 0 else 6
    if args.command == "run-formal500":
        from .pilot import run_formal500

        summary = asyncio.run(run_formal500(
            config,
            concurrency=args.concurrency,
            run_name=args.run_name,
        ))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if summary["failed_conversations"] == 0 else 6
    if args.command == "run-replay-completed":
        from .pilot import run_replay_completed

        summary = asyncio.run(run_replay_completed(
            config,
            source_run=args.source_run,
            count=args.count,
            concurrency=args.concurrency,
            run_name=args.run_name,
        ))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if summary["failed_conversations"] == 0 else 6
    if args.command == "init-run":
        return command_init_run(config, args.name)
    if args.command == "selftest":
        from .selftest import run
        return run()
    raise AssertionError(args.command)
