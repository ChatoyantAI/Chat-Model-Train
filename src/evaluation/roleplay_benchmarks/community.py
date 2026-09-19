"""Runnable, versioned reproductions for benchmarks releasing data but no harness."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

from .benchmark_api import Client, CallFailed, RequestLimit, endpoint, object_hash
from .community_protocols import (build_cases, actor_system, expected_judgments,
    rmt_judge_jobs, parse_rmt, parse_mini, mini_chunks, MINI_DIMS)
from .paper_templates import load_templates
from .sources import ROOT, LOCK, digest, git, inside, read_json, registry, verify_one, write_json

ENGINE_FILES = ("community.py", "community_protocols.py", "benchmark_api.py", "paper_templates.py")


def validate(config):
    name = config.get("benchmark")
    if name not in ("rmtbench", "minimax"):
        raise ValueError("Expected rmtbench or minimax")
    allowed = {"benchmark", "protocol_label", "languages", "version", "layout", "repeats", "seed", "limit_cases",
               "concurrency", "max_attempts", "timeout_seconds", "max_requests_per_run", "roles",
               "generated_messages", "chunk_messages", "prompt_file"}
    if set(config) - allowed:
        raise ValueError("Unknown configuration keys; credentials belong in environment variables")
    if not isinstance(config.get("protocol_label"), str) or not config["protocol_label"].strip():
        raise ValueError("protocol_label is required")
    languages = config.get("languages")
    if not isinstance(languages, list) or not languages or any(x not in ("en", "zh") for x in languages) or len(set(languages)) != len(languages):
        raise ValueError("languages must be a unique nonempty subset of en, zh")
    for key in ("concurrency", "max_attempts", "timeout_seconds", "max_requests_per_run", "repeats"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if config.get("limit_cases") is not None and (type(config["limit_cases"]) is not int or config["limit_cases"] < 1):
        raise ValueError("limit_cases must be null or a positive integer")
    roles = {"actor", "judge"} | ({"simulator"} if name == "minimax" else set())
    if set(config.get("roles", {})) != roles:
        raise ValueError("Supply exactly the required actor/judge/simulator roles")
    for role in config["roles"].values():
        if set(role) != {"model", "revision", "base_url_env", "api_key_env", "sampling"}:
            raise ValueError("Each role needs model, revision, base_url_env, api_key_env, sampling")
        for key in ("model", "revision", "base_url_env", "api_key_env"):
            if not isinstance(role[key], str) or not role[key].strip():
                raise ValueError(f"Missing role field {key}")
        for key in ("base_url_env", "api_key_env"):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role[key]):
                raise ValueError("Use environment variable names, not credential values")
        sampling = role["sampling"]
        if not isinstance(sampling, dict) or set(sampling) - {"temperature", "top_p", "seed", "max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported sampling fields")
        for key, value in sampling.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Sampling values must be finite numbers")
            if key in ("seed", "max_tokens", "max_completion_tokens") and type(value) is not int:
                raise ValueError("Token limits and seed must be integers")
            if key.startswith("max_") and value < 1:
                raise ValueError("Token limits must be positive")
            if key == "temperature" and not 0 <= value <= 2 or key == "top_p" and not 0 < value <= 1:
                raise ValueError("Invalid temperature/top_p")
        if "max_tokens" in sampling and "max_completion_tokens" in sampling:
            raise ValueError("Choose only one output token limit")
    if name == "rmtbench":
        if config.get("version") not in ("claude35", "claude45") or config.get("layout") not in ("mixed_four_blocks", "single_blocks"):
            raise ValueError("Choose RMT version and layout")
    else:
        for key in ("generated_messages", "chunk_messages"):
            if type(config.get(key)) is not int or config[key] < 2 or config[key] % 2:
                raise ValueError(f"{key} must be positive and even; counts individual messages, not pairs")
        if not isinstance(config.get("prompt_file"), str):
            raise ValueError("prompt_file is required")


def prepare(config_path, run_dir, allow_external=False):
    config = read_json(config_path)
    validate(config)
    name = config["benchmark"]
    spec = registry()[name]
    source = verify_one(name, spec)
    directory = Path(run_dir).resolve()
    if directory.exists() or (not allow_external and not directory.is_relative_to(ROOT / "outputs")) or directory == ROOT / "outputs":
        raise ValueError("Use a new run directory under repository outputs/")
    cases = build_cases(config, inside(ROOT, spec["path"]))
    if not cases:
        raise ValueError("No cases selected")
    if name == "rmtbench":
        prompts = load_templates()
        prompts["actor"] = "Role-play {name}. Use {language}. Stay in character and respond to the current user message.\nCharacter profile:\n{profile}"
        choices = ["Actor system wrapper is locally authored; judge templates/rubrics extracted from Appendix E.",
                   "Fixed user scripts; no adaptive user model; prior actor replies stay in generation history.",
                   "mixed_four_blocks selects one block from each public category per repeat, then shuffles blocks with a fixed seed; not the authors' unavailable exact ordering.",
                   "UPA reference questions map to the final N turns, in source order; reference answers are judge-only.",
                   "Judge receives profile/current query/current response as in published templates; no additional history.",
                   "CM is scored only on character_maintenance blocks; UPA only at annotated final questions; SEC excluded.",
                   "Raw 1-5 means and binary success rates reported separately; display quality normalization is score/5*100.",
                   "Judge temperature/output cap are explicit local settings; actor defaults left to backend when sampling is empty."]
    else:
        prompts = read_json(inside(ROOT, config["prompt_file"]))
        for key in ("actor", "simulator", "judge", "version", "provenance"):
            if not isinstance(prompts.get(key), str) or not prompts[key].strip():
                raise ValueError(f"Missing prompt field {key}")
        choices = ["Actor, user simulator and judge prompts are locally authored from public descriptions.",
                   "100 generated messages = 50 actor + 50 simulator messages, plus supplied opening/initial user = 102 messages.",
                   "20-message target chunks exclude the two seed messages; prior history is context only; only actor output is scored.",
                   "100/20 message lengths and three repetitions are configurable and recorded.",
                   "Six dimensions use local 0-100 defect-severity anchors; no human calibration has been performed.",
                   "Chunks weighted by actor-message count; repeats averaged within seed; seeds macro-averaged per language.",
                   "Within Worlds/Stories dimensions are equally weighted (local choice); category weights are published 50/25/25."]
    # Validate all format placeholders before any paid requests.
    for case in cases:
        actor_system(case, prompts)
        if name == "minimax":
            prompts["simulator"].format(language="English", scene="{}")
            prompts["judge"].format(language="English")
        else:
            dummy = [m for t in case["turns"] for m in ({"role": "user", "content": t["user"]}, {"role": "assistant", "content": "test"})]
            list(rmt_judge_jobs(case, dummy, prompts))
    directory.mkdir(parents=True)
    write_json(directory / "cases.json", cases)
    write_json(directory / "prompts.json", prompts)
    hashes = {filename: digest(Path(__file__).with_name(filename)) for filename in ENGINE_FILES}
    manifest = {"schema_version": 1, "runner": "public-method-v1", "benchmark": name,
                "status": "prepared-not-evaluated", "upstream_revision": spec["revision"],
                "config": config, "config_sha256": object_hash(config), "source_snapshot": source,
                "suite_commit": git("rev-parse", "HEAD"), "suite_dirty": bool(git("status", "--porcelain")),
                "sources_lock_sha256": digest(LOCK), "engine_hashes": hashes,
                "cases_sha256": digest(directory / "cases.json"), "prompts_sha256": digest(directory / "prompts.json"),
                "protocol_choices": choices, "completed_stages": [],
                "inventory": {"cases": len(cases), "generation_requests": sum(len(c["turns"]) if name == "rmtbench" else config["generated_messages"] for c in cases),
                              "judge_requests": sum(expected_judgments(c, config) for c in cases)}}
    manifest["engineering_patch_sha256"] = object_hash(hashes)
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_run(directory, allow_external=False):
    directory = Path(directory).resolve()
    if directory == ROOT / "outputs" or (not allow_external and not directory.is_relative_to(ROOT / "outputs")):
        raise ValueError("Run directory must be under repository outputs/")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("runner") != "public-method-v1":
        raise ValueError("Not a public-method run")
    validate(manifest["config"])
    if object_hash(manifest["config"]) != manifest["config_sha256"]:
        raise ValueError("Run configuration changed; prepare a new run")
    for name in ("cases", "prompts"):
        if digest(directory / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError(f"{name} snapshot changed")
    for filename, expected in manifest["engine_hashes"].items():
        if digest(Path(__file__).with_name(filename)) != expected:
            raise ValueError("Runner implementation changed; prepare a new run")
    return directory, manifest, read_json(directory / "cases.json"), read_json(directory / "prompts.json")


def generate_case(case, config, prompts, directory, client):
    system = {"role": "system", "content": actor_system(case, prompts)}
    history = []
    output = directory / "trajectory.json"
    if "scene" not in case:
        for turn in case["turns"]:
            history.append({"role": "user", "content": turn["user"]})
            content, _ = client.call(directory / "requests", "actor", [system, *history])
            history.append({"role": "assistant", "content": content})
            write_json(output, {"complete": False, "messages": history})
    else:
        scene = case["scene"]
        history = [{"role": "assistant", "content": scene["ai_prologue"]}, {"role": "user", "content": scene["initial_user_input"]}]
        language = "English" if case["language"] == "en" else "Chinese"
        simulator = prompts["simulator"].format(language=language, scene=json.dumps(scene, ensure_ascii=False))
        for i in range(config["generated_messages"]):
            if i % 2 == 0:
                content, _ = client.call(directory / "requests", "actor", [system, *history])
                role = "assistant"
            else:
                content, _ = client.call(directory / "requests", "simulator", [
                    {"role": "system", "content": simulator},
                    {"role": "user", "content": json.dumps({"dialogue": history}, ensure_ascii=False)}])
                role = "user"
            history.append({"role": role, "content": content})
            write_json(output, {"complete": False, "messages": history})
    write_json(output, {"complete": True, "messages": history, "messages_sha256": object_hash(history)})


def trajectory(case, config, directory):
    file = directory / "trajectory.json"
    if not file.exists():
        return None
    result = read_json(file)
    expected = 2 * len(case["turns"]) if "scene" not in case else config["generated_messages"] + 2
    messages = result.get("messages", [])
    if not result.get("complete") or len(messages) != expected or result.get("messages_sha256") != object_hash(messages):
        return None
    first = "assistant" if "scene" in case else "user"
    if not all(m.get("role") == (first if i % 2 == 0 else ("user" if first == "assistant" else "assistant")) and isinstance(m.get("content"), str) for i, m in enumerate(messages)):
        return None
    return messages


def judge_case(case, config, prompts, directory, client):
    messages = trajectory(case, config, directory)
    if messages is None:
        return {"status": "missing-generation", "id": case["id"]}
    result = {"id": case["id"], "status": "incomplete", "judgments": [], "errors": []}
    if "scene" not in case:
        jobs = [(key, dim, index, [{"role": "user", "content": prompt}])
                for key, dim, index, prompt in rmt_judge_jobs(case, messages, prompts)]
    else:
        language = "English" if case["language"] == "en" else "Chinese"
        jobs = [(f"chunk-{index:04}", "all", index, [
            {"role": "system", "content": prompts["judge"].format(language=language)},
            {"role": "user", "content": json.dumps({"scene": case["scene"], "prior_context": prior, "target_chunk": chunk}, ensure_ascii=False)}])
            for index, prior, chunk in mini_chunks(messages, config["chunk_messages"])]
    for key, dimension, index, request in jobs:
        try:
            parser = parse_mini if dimension == "all" else lambda text, dim=dimension: parse_rmt(text, dim)
            content, score = client.call(directory / "requests", "judge", request, parser)
            result["judgments"].append({"key": key, "dimension": dimension, "index": index, "value": score, "raw": content})
        except CallFailed as exc:
            result["errors"].append({"key": key, "error": str(exc)})
            if isinstance(exc, RequestLimit):
                break
        write_json(directory / "judgments.json", result)
    result["status"] = "completed" if len(result["judgments"]) == expected_judgments(case, config) else "incomplete"
    result["trajectory_sha256"] = object_hash(messages)
    write_json(directory / "judgments.json", result)
    return {"id": case["id"], "status": result["status"]}


def aggregate(directory, manifest, cases):
    config = manifest["config"]
    by_language = {}
    total_valid = generated = completed = 0
    case_rows = []
    for language in config["languages"]:
        values = defaultdict(list)
        seeds = defaultdict(list)
        for case in (c for c in cases if c["language"] == language):
            work = directory / "cases" / case["id"]
            messages = trajectory(case, config, work)
            generated += messages is not None
            file = work / "judgments.json"
            data = read_json(file) if file.exists() else {}
            units = data.get("judgments", []) if messages is not None and data.get("trajectory_sha256") == object_hash(messages) else []
            # Reparse persisted raw text so invalid/missing scores are never zeros.
            valid = []
            expected_units = ({key: (dim, index) for key, dim, index, _ in rmt_judge_jobs(case, messages, read_json(directory / "prompts.json"))} if messages and "scene" not in case
                              else {f"chunk-{i:04}": ("all", i) for i, *_ in mini_chunks(messages, config["chunk_messages"])} if messages else {})
            seen = set()
            for unit in units:
                try:
                    if unit["key"] not in expected_units or unit["key"] in seen or expected_units[unit["key"]] != (unit["dimension"], unit["index"]):
                        continue
                    parsed = parse_mini(unit["raw"]) if "scene" in case else parse_rmt(unit["raw"], unit["dimension"])
                    if parsed != unit["value"]:
                        continue
                    valid.append(unit)
                    seen.add(unit["key"])
                except (ValueError, KeyError, TypeError):
                    continue
            total_valid += len(valid)
            full = messages is not None and len(valid) == expected_judgments(case, config)
            completed += full
            row = {"id": case["id"], "language": language, "source_id": case["source_id"], "complete": full,
                   "valid_judgments": len(valid), "expected_judgments": expected_judgments(case, config)}
            if "scene" not in case:
                for unit in valid:
                    values[unit["dimension"]].append(unit["value"])
                row["scores"] = {dim: statistics.mean([u["value"] for u in valid if u["dimension"] == dim]) for dim in {u["dimension"] for u in valid}}
            elif full:
                weights = {i: sum(m["role"] == "assistant" for m in chunk) for i, _, chunk in mini_chunks(messages, config["chunk_messages"])}
                row["scores"] = {dim: sum(u["value"][dim]["score"] * weights[u["index"]] for u in valid) / sum(weights.values()) for dim in MINI_DIMS}
                seeds[case["source_id"]].append(row["scores"])
            case_rows.append(row)
        if config["benchmark"] == "rmtbench":
            by_language[language] = {dim: {"mean": statistics.mean(v), "count": len(v),
                                                "scale": "0/1" if dim in ("CM", "UPA") else "1-5",
                                                "display_0_100": statistics.mean(v) * (100 if dim in ("CM", "UPA") else 20)} for dim, v in sorted(values.items())}
        else:
            seed_means = [{dim: statistics.mean(s[dim] for s in repeats) for dim in MINI_DIMS} for repeats in seeds.values()]
            scores = {dim: statistics.mean(s[dim] for s in seed_means) for dim in MINI_DIMS} if seed_means else {}
            by_language[language] = {"dimensions": scores, "complete_seeds_or_partial_repeats": len(seed_means),
                                     "included_sessions": sum(len(v) for v in seeds.values())}
            if scores:
                worlds = statistics.mean(scores[d] for d in ("basics", "logic", "knowledge"))
                stories = statistics.mean(scores[d] for d in ("diversity", "content_logic"))
                by_language[language].update(Worlds=worlds, Stories=stories, Preferences=scores["interaction"],
                                            overall=worlds * .5 + stories * .25 + scores["interaction"] * .25)
    usage = defaultdict(float)
    attempts = failures = cost_observed = 0
    for file in (directory / "cases").glob("*/requests/*.attempt-*.json"):
        item = read_json(file)
        attempts += 1
        failures += item["status"] != "completed"
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = (item.get("usage") or {}).get(key)
            if type(value) in (float, int) and math.isfinite(value):
                usage[key] += value
        # DeepInfra reports estimated_cost; prefer cost when both are present,
        # and never double count or convert missing cost into a reported zero.
        provider_usage = item.get("usage") or {}
        for key in ("cost", "estimated_cost"):
            value = provider_usage.get(key)
            if type(value) in (float, int) and math.isfinite(value) and value >= 0:
                usage["cost"] += value
                cost_observed += 1
                break
    summary = {"benchmark": config["benchmark"], "protocol_label": config["protocol_label"],
               "coverage": {"cases": len(cases), "generated_cases": generated, "completed_cases": completed,
                            "valid_judgments": total_valid, "expected_judgments": sum(expected_judgments(c, config) for c in cases),
                            "complete": completed == len(cases)},
               "scores": by_language, "cases": case_rows, "usage": dict(usage), "attempts": attempts,
               "failed_attempts": failures, "attempts_with_reported_cost": cost_observed,
               "cost_note": "Cost is provider-reported only; absence means unknown, not zero. Timed-out requests can be billed without returning usage.",
               "partial_results": completed != len(cases), "protocol_choices": manifest["protocol_choices"]}
    write_json(directory / "summary.json", summary)
    lines = [f"# {config['benchmark']} results", "", f"Protocol: {config['protocol_label']}", "",
             f"Completed cases: {completed}/{len(cases)}; valid judgments: {total_valid}/{summary['coverage']['expected_judgments']}", "",
             "Partial results exclude missing judgments (not zero scores). Compare full matched coverage.", "", "```json", json.dumps(by_language, ensure_ascii=False, indent=2), "```", ""]
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(run_dir, stage, execute=False, max_requests=None, sender=None, allow_external=False):
    directory, manifest, cases, prompts = load_run(run_dir, allow_external=allow_external)
    if stage not in ("generate", "judge", "aggregate", "evaluate"):
        raise ValueError("Stages: generate, judge, aggregate, evaluate")
    if not execute:
        return {"dry_run": True, "stage": stage, "inventory": manifest["inventory"],
                "request_limit": max_requests or manifest["config"]["max_requests_per_run"],
                "protocol_choices": manifest["protocol_choices"], "roles": manifest["config"]["roles"]}
    config = json.loads(json.dumps(manifest["config"]))
    if max_requests is not None:
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_requests must be positive")
        config["max_requests_per_run"] = max_requests
    if stage != "aggregate":
        if "REPLACE_WITH" in json.dumps(config):
            raise ValueError("Fill in model IDs and revisions before execution")
        active_roles = (["actor"] + (["simulator"] if config["benchmark"] == "minimax" else [])) if stage == "generate" else ["judge"] if stage == "judge" else list(config["roles"])
        fingerprints = dict(manifest.get("endpoint_fingerprints", {}))
        for name in active_roles:
            if sender is None:
                url, _ = endpoint(config["roles"][name])
                fingerprint = object_hash(url)
            else:
                fingerprint = "offline-test-transport"
            if name in fingerprints and fingerprints[name] != fingerprint:
                raise ValueError("Endpoint changed; use a new run directory")
            fingerprints[name] = fingerprint
        manifest["endpoint_fingerprints"] = fingerprints
    lock = directory / ".run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Run already locked; if a process crashed, verify it stopped before removing .run.lock") from None
    os.close(descriptor)
    client = Client(config, sender=sender)
    try:
        manifest["status"] = f"running-{stage}"
        write_json(directory / "manifest.json", manifest)
        stages = ["generate", "judge", "aggregate"] if stage == "evaluate" else [stage]
        for step in stages:
            if step != "aggregate":
                def worker(case):
                    work = directory / "cases" / case["id"]
                    work.mkdir(parents=True, exist_ok=True)
                    try:
                        if step == "generate":
                            generate_case(case, config, prompts, work, client)
                            result = {"id": case["id"], "status": "completed"}
                        else:
                            result = judge_case(case, config, prompts, work, client)
                    except CallFailed as exc:
                        result = {"id": case["id"], "status": "failed", "error": str(exc)}
                    write_json(work / f"{step}-status.json", result)
                    return result
                progress = []
                with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
                    futures = [pool.submit(worker, case) for case in cases]
                    for future in as_completed(futures):
                        progress.append(future.result())
                        write_json(directory / "progress.json", {"stage": step, "done": len(progress), "total": len(cases), "requests_this_run": client.count, "cases": progress})
                if all(r["status"] == "completed" for r in progress):
                    manifest["completed_stages"] = list(dict.fromkeys([*manifest["completed_stages"], step]))
        # Accounting scans every saved request. Do it once after the requested
        # stages, so network-volume reads never block generation -> judging.
        summary = aggregate(directory, manifest, cases)
        if summary["coverage"]["complete"]:
            manifest["completed_stages"] = list(dict.fromkeys([*manifest["completed_stages"], "aggregate"]))
        success = summary["coverage"]["generated_cases"] == len(cases) if stage == "generate" else summary["coverage"]["complete"]
        manifest["status"] = f"completed-{stage}" if success else f"incomplete-{stage}"
        manifest["last_execution"] = {"stage": stage, "requests": client.count, "request_limit": config["max_requests_per_run"], "finished_at": time.time()}
        return {"status": manifest["status"], "coverage": summary["coverage"], "requests_this_run": client.count, "summary": str(directory / "summary.md")}
    except BaseException:
        manifest["status"] = f"failed-{stage}"
        raise
    finally:
        write_json(directory / "manifest.json", manifest)
        lock.unlink()
