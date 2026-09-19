"""Run a copied, pinned upstream tree with minimal, fully recorded patches.

No upstream file is changed. Prompt/rubric/score code is not rewritten here.
"""
from __future__ import annotations

import ast
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .sources import ROOT, LOCK, digest, git, inside, inventory, read_json, registry, verify_one, write_json


def replace_exact(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Upstream patch anchor has changed: {old[:65]!r}")
    return text.replace(old, new, 1)


def patched_sources(name, upstream):
    """Return filename -> new source; no execution or network access."""
    if name == "coser":
        relative = "gca_evaluation/utils.py"
        text = (upstream / relative).read_text(encoding="utf-8")
        text = replace_exact(text, "with open('config.json', 'r') as f:\n\tconfig = json.load(f)",
            "config = {'api_key': os.environ['BENCH_RUNTIME_KEY'], 'base_url': os.environ['BENCH_RUNTIME_URL']}\n_runtime_endpoints_by_model = json.loads(os.environ.get('BENCH_RUNTIME_ENDPOINTS_BY_MODEL', '{}'))\ndef _runtime_endpoint(model):\n\tvalue = _runtime_endpoints_by_model.get(model, {})\n\treturn value if isinstance(value, dict) else config\n_runtime_options_by_model = json.loads(os.environ.get('BENCH_RUNTIME_REQUEST_OPTIONS_BY_MODEL', '{}'))\ndef _runtime_options(model):\n\tvalue = _runtime_options_by_model.get(model, {})\n\treturn value if isinstance(value, dict) else {}")
        text = text.replace("client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url'], timeout=180)",
                            "endpoint = _runtime_endpoint(model)\n\t\tclient = openai.OpenAI(api_key=endpoint.get('api_key', config['api_key']), base_url=endpoint.get('base_url', config['base_url']), timeout=180)")
        text = text.replace("\t\t\t\tmodel=model,\n",
                            "\t\t\t\tmodel=model,\n\t\t\t\textra_body=_runtime_options(model),\n")
        return {relative: text}
    if name != "characterbench":
        raise ValueError("No official runnable harness is available for this benchmark")
    text = (upstream / "process.py").read_text(encoding="utf-8")
    # literal_eval preserves the accepted Python/JSON-like literals without executing code.
    text = "import ast\n" + text.replace("eval(response_", "ast.literal_eval(response_")
    text = replace_exact(text, "                for retry in range(max_retry):", "                succeeded = False\n                for retry in range(max_retry):")
    text = replace_exact(text, "                        break\n", "                        succeeded = True\n                        break\n")
    text = replace_exact(text, "                if retry == max_retry-1:\n                    continue", "                if not succeeded:\n                    raise RuntimeError(f'Generation failed: {filename} row {i}; no row was silently dropped')")
    judge = (upstream / "model_generation_qwen_chat.py").read_text(encoding="utf-8")
    judge = replace_exact(judge, 'model_path = "COAI/CharacterJudge"', 'model_path = os.environ["BENCH_RUNTIME_JUDGE_PATH"]')
    # API plumbing only: honor the requested actor rather than hard-coded gpt-4o.
    # Bounded API failure retries already exist in process.py; no second infinite loop.
    gpt_call = '''from openai import OpenAI
import json
import os
client = OpenAI(base_url=os.environ["BENCH_RUNTIME_URL"], api_key=os.environ["BENCH_RUNTIME_KEY"], timeout=180, max_retries=0)
_request_options = json.loads(os.environ.get("BENCH_RUNTIME_REQUEST_OPTIONS", "{}"))
def gpt_call(prompt, model_name):
    options = _request_options if isinstance(_request_options, dict) else {}
    response = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], extra_body=options)
    return response.choices[0].message.content
'''
    return {"process.py": text, "gpt_call.py": gpt_call, "model_generation_qwen_chat.py": judge}


def validate_config(config):
    spec = registry().get(config.get("benchmark"))
    if spec is None:
        raise ValueError("Unknown benchmark")
    if spec["blockers"]:
        raise ValueError("Exact reproduction blocked: " + " ".join(spec["blockers"]))
    for key in ("actor_model", "model_revision", "precision", "protocol_label", "base_url_env", "api_key_env"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Missing configuration field: {key}")
    # Configs must contain environment variable names, never credentials/endpoints.
    if any(k in config for k in ("api_key", "token", "password", "base_url")):
        raise ValueError("Store API credentials/endpoints in environment variables, not run configs")
    if config["benchmark"] == "coser":
        if config.get("retrieval") is not None:
            raise ValueError("RAG needs a separately locked full-book/embedding protocol; this recipe is no-RAG")
        if config.get("nth_exp", 0) not in (0, 1, 2):
            raise ValueError("Use one isolated run per repeat: nth_exp must be 0, 1, or 2")
        for key in ("judge_model", "env_model", "nsp_model"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"Missing {key}")
        for key in ("actor_model", "judge_model", "env_model", "nsp_model"):
            if not re.fullmatch(r"[A-Za-z0-9_.:/@+-]+", config[key]) or ".." in config[key]:
                raise ValueError(f"Unsafe model ID in upstream filesystem paths: {key}")
    else:
        if not re.fullmatch(r"[A-Za-z0-9-]+", config.get("actor_alias", "")):
            raise ValueError("actor_alias must contain ASCII letters, digits or hyphens (no underscores/slashes)")
        for key in ("judge_path_env", "judge_revision"):
            if not config.get(key):
                raise ValueError(f"Missing {key}")
    return spec


def prepare(config_path, run_dir, allow_external=False):
    config = read_json(config_path)
    if config.get("benchmark") in ("rmtbench", "minimax"):
        from .community import prepare as prepare_public
        return prepare_public(config_path, run_dir, allow_external=allow_external)
    spec = validate_config(config)
    source_manifest = verify_one(config["benchmark"], spec)
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise ValueError("Run directory already exists; use a new ID or resume with run")
    if not allow_external and not run_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Run directories must be under repository outputs/")
    upstream = inside(ROOT, spec["path"])
    run_dir.mkdir(parents=True)
    work = run_dir / "upstream"
    shutil.copytree(upstream, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    diffs = []
    for relative, new in patched_sources(config["benchmark"], upstream).items():
        path = work / relative
        old = path.read_text(encoding="utf-8")
        ast.parse(new)
        diffs.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        path.write_text(new, encoding="utf-8")
    (run_dir / "engineering.patch").write_text("".join(diffs), encoding="utf-8")
    # Upstream writes model IDs (which can contain '/') into its cache path.
    if config["benchmark"] == "coser":
        (work / ".cache" / config["actor_model"]).parent.mkdir(parents=True, exist_ok=True)
    else:
        for language in ("en", "zh"):
            (work / f"eval_data/evaluation_data_{language}").mkdir(parents=True, exist_ok=True)
    prepared_files = {p.relative_to(work).as_posix(): digest(p) for p in work.rglob("*") if p.is_file()}
    manifest = {"schema_version": 1, "benchmark": config["benchmark"], "status": "prepared-not-evaluated",
        "suite_commit": git("rev-parse", "HEAD"), "suite_dirty": bool(git("status", "--porcelain")),
        "sources_lock_sha256": digest(LOCK), "upstream_revision": spec["revision"],
        "config": config, "source_snapshot": source_manifest, "prepared_files": prepared_files,
        "inventory": inventory(config["benchmark"], spec), "completed_stages": [],
        "engineering_patch_sha256": digest(run_dir / "engineering.patch"),
        "claim": "Pinned official source with recorded engineering patches; not a completed leaderboard reproduction."}
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def commands(config, stage):
    if config["benchmark"] == "coser":
        if stage != "evaluate":
            raise ValueError("CoSER stages: evaluate (official generation + judging)")
        args = ["gca_evaluation/main.py", "--test_file", "data/test/test_set.json"]
        for key in ("actor_model", "judge_model", "env_model", "nsp_model", "nth_exp", "num_workers", "continue_from"):
            args += [f"--{key}", str(config.get(key, 0 if key != "num_workers" else 1))]
        if config.get("wo_thought"):
            args.append("--wo_thought")
        return [(".", args)]
    alias = config["actor_alias"]
    if stage == "generate":
        return [(".", ["process.py", "--model_name", config["actor_model"]])]
    if stage == "prompts":
        return [("construct_prompts", [f"process_wo_context_{lang}_all.py", "--model_name", alias]) for lang in ("zh", "en")]
    if stage == "judge":
        return [(".", ["model_generation_qwen_chat.py", "--input_path", f"eval_data/evaluation_data_{lang}",
                       "--output_path", f"results/{lang}"]) for lang in ("zh", "en")]
    if stage == "aggregate":
        return [(".", ["all_eval.py", "--result_path", f"results/{lang}", "--output_path", f"results/{lang}.json"]) for lang in ("zh", "en")]
    raise ValueError("CharacterBench stages: generate, prompts, judge, aggregate")


def audit_character(work):
    """Independent coverage audit; does not alter official scores."""
    result = {"languages": {}, "complete": True}
    for lang in ("zh", "en"):
        report = {"expected": 0, "generated": 0, "judged": 0, "parse_errors": 0, "duplicate_ids": 0, "out_of_range": 0}
        for source in sorted((work / "eval_data/raw_data").glob("*.json")):
            raw = read_json(source)
            report["expected"] += len(raw)
            response = work / "eval_data/response_data" / source.name
            rows = read_json(response) if response.exists() else []
            report["generated"] += sum(bool(r.get("response_zh") if lang == "zh" else r.get("translation_en", {}).get("response_en")) for r in rows)
            files = list((work / f"results/{lang}").glob(f"*_{source.name}"))
            if len(files) > 1:
                raise ValueError("Mixed model results in a single run")
            if not files:
                continue
            ids = set()
            maximum = max(float(r["annotation_score"]) for r in raw)
            for line in files[0].read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["id"] in ids:
                    report["duplicate_ids"] += 1
                ids.add(row["id"])
                try:
                    score = float(str(row["generated"]).split("\n")[0])
                    if not (1 <= score <= maximum):
                        report["out_of_range"] += 1
                except (ValueError, KeyError, IndexError):
                    report["parse_errors"] += 1
            report["judged"] += len(ids)
        report["complete"] = report["expected"] > 0 and report["expected"] == report["generated"] == report["judged"] and not any(report[k] for k in ("parse_errors", "duplicate_ids", "out_of_range"))
        result["languages"][lang] = report
        result["complete"] &= report["complete"]
    return result


def run(run_dir, stage, python=sys.executable, execute=False, max_requests=None, allow_external=False):
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("runner") == "public-method-v1":
        from .community import run as run_public
        return run_public(run_dir, stage, execute, max_requests, allow_external=allow_external)
    if max_requests is not None:
        raise ValueError("--max-requests applies to RMTBench/MiniMax only")
    config = manifest["config"]
    validate_config(config)
    work = run_dir / "upstream"
    for relative, expected in manifest["prepared_files"].items():
        if digest(inside(work, relative)) != expected:
            raise ValueError(f"Prepared source changed: {relative}")
    sequence = commands(config, stage)
    if not execute:
        return {"dry_run": True, "commands": [{"cwd": str(work / cwd), "argv": [python, *args]} for cwd, args in sequence]}
    if stage == "generate" and config["benchmark"] == "characterbench":
        from .preflight import character_inputs
        input_audit = character_inputs(work)
        write_json(run_dir / "input_audit.json", input_audit)
        if not input_audit["complete"]:
            raise ValueError("Official CharacterBench preprocessing rejects source rows; inspect input_audit.json before any paid requests. No samples were silently removed.")
    if any("REPLACE_WITH" in str(v) for v in config.values()):
        raise ValueError("Replace all example configuration placeholders before executing")
    required_stage = {"prompts": "generate", "judge": "prompts", "aggregate": "judge"}.get(stage)
    if required_stage and required_stage not in manifest["completed_stages"]:
        raise ValueError(f"Run stage {required_stage} successfully first")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    if stage in ("generate", "evaluate"):
        for name, key in (("BENCH_RUNTIME_URL", "base_url_env"), ("BENCH_RUNTIME_KEY", "api_key_env")):
            value = os.environ.get(config[key])
            if not value:
                raise ValueError(f"Missing environment variable: {config[key]}")
            env[name] = value
        options_by_model = config.get("request_options_by_model", {})
        if options_by_model:
            env["BENCH_RUNTIME_REQUEST_OPTIONS_BY_MODEL"] = json.dumps(options_by_model, ensure_ascii=False)
        endpoint_env_by_model = config.get("endpoint_env_by_model", {})
        if endpoint_env_by_model:
            resolved = {}
            for model, names in endpoint_env_by_model.items():
                resolved[model] = {
                    "base_url": os.environ.get(names["base_url_env"], ""),
                    "api_key": os.environ.get(names["api_key_env"], ""),
                }
            env["BENCH_RUNTIME_ENDPOINTS_BY_MODEL"] = json.dumps(resolved, ensure_ascii=False)
        if config.get("request_options"):
            env["BENCH_RUNTIME_REQUEST_OPTIONS"] = json.dumps(config["request_options"], ensure_ascii=False)
    if stage == "judge":
        judge_path = Path(os.environ.get(config["judge_path_env"], ""))
        if not judge_path.is_absolute() or not (judge_path / "config.json").exists():
            raise ValueError("CHARACTERJUDGE_PATH must point to a local, downloaded judge snapshot")
        env["BENCH_RUNTIME_JUDGE_PATH"] = str(judge_path)
    manifest["status"] = f"running-{stage}"
    write_json(run_dir / "manifest.json", manifest)
    try:
        # The upstream judge imports CUDA packages; install only in its isolated environment.
        with (run_dir / f"{stage}.log").open("a", encoding="utf-8") as log:
            for cwd, args in sequence:
                result = subprocess.run([python, *args], cwd=work / cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Upstream {stage} exited {result.returncode}; see {stage}.log")
        if config["benchmark"] == "characterbench" and stage in ("judge", "aggregate"):
            audit = audit_character(work)
            write_json(run_dir / "coverage.json", audit)
            if not audit["complete"]:
                raise RuntimeError("Missing/invalid CharacterBench outputs; coverage.json records the failures")
        if config["benchmark"] == "coser":
            from .reports import audit_coser
            audit = audit_coser(work)
            write_json(run_dir / "coverage.json", audit)
            if not audit["complete"]:
                raise RuntimeError("Missing/invalid CoSER outputs; coverage.json records the failures")
        manifest["completed_stages"] = list(dict.fromkeys([*manifest["completed_stages"], stage]))
        manifest["status"] = f"completed-{stage}"
    except BaseException:
        manifest["status"] = f"failed-{stage}"
        raise
    finally:
        write_json(run_dir / "manifest.json", manifest)
    return {"status": manifest["status"], "run_dir": str(run_dir)}
