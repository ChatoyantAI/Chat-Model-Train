"""Offline whole-dataset checks. Never queries a model or alters source rows."""
import ast
from collections import Counter
import json
from pathlib import Path

from .sources import ROOT, inside, inventory, read_json, registry, verify_one


def character_inputs(base):
    base = Path(base)
    tree = ast.parse((base / "process.py").read_text(encoding="utf-8"))
    # Use only the reviewed, pure upstream normalization functions. Do not import
    # process.py, which imports an API client and may resolve GPU/network modules.
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("norm_session_en", "norm_session_zh")]
    if len(functions) != 2:
        raise ValueError("Upstream normalizers changed")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "upstream-normalizers", "exec"), namespace)
    prompt_tree = ast.parse((base / "roleplay_prompt.py").read_text(encoding="utf-8"))
    prompts = {node.targets[0].id: ast.literal_eval(node.value) for node in prompt_tree.body if isinstance(node, ast.Assign)}
    result = {"expected": {"en": 0, "zh": 0}, "valid": {"en": 0, "zh": 0}, "errors": [], "row_indices": "zero-based"}
    for source in sorted((base / "eval_data/raw_data").glob("*.json")):
        for index, row in enumerate(read_json(source)):
            for lang in ("en", "zh"):
                result["expected"][lang] += 1
                try:
                    session = namespace[f"norm_session_{lang}"](row)
                    settings = session["dialogue_setting"]
                    for message in session["messages"]:
                        if not isinstance(message["content"], str):
                            raise TypeError("Message content is not a string")
                    prompts[f"Role_Play_PROMPT_{lang.upper()}"].format(
                        character_profile=settings["bot_profile"], user_profile=settings["user_profile"],
                        dialogue="\n".join(f"{m['role']}: {m['content']}" for m in session["messages"]))
                    result["valid"][lang] += 1
                except (AssertionError, ValueError, KeyError, TypeError, IndexError) as error:
                    result["errors"].append({"file": source.name, "row_index": index, "language": lang, "error_type": type(error).__name__})
    result["complete"] = not result["errors"] and sum(result["expected"].values()) > 0
    return result


def inspect_source(name):
    spec = registry()[name]
    verify_one(name, spec)
    base = inside(ROOT, spec["path"])
    report = {"benchmark": name, "revision": spec["revision"], "data_verified": True,
              "inventory": inventory(name, spec), "protocol": spec["protocol"],
              "official_reproduction_blockers": list(spec["blockers"]), "model_calls": 0}
    if name == "characterbench":
        report["input_audit"] = character_inputs(base)
        if not report["input_audit"]["complete"]:
            report["official_reproduction_blockers"].append("Upstream preprocessing rejects source rows; see input_audit. No automatic exclusion or relabeling.")
        report["stages"] = ["generate", "prompts", "judge", "aggregate"]
        report["needs"] = ["Candidate OpenAI-compatible API", "Revision-pinned local CharacterJudge snapshot", "Isolated compatible GPU environment"]
    elif name == "coser":
        report["stages"] = ["evaluate"]
        report["needs"] = ["Actor API", "GPT-4o 2024-08-06 judge for published-score comparisons", "Environment and next-speaker models", "Compatible legacy LangChain imports", "NLTK punkt/punkt_tab resources"]
        report["notes"] = ["evaluate runs simulation then judging.", "Upstream num_workers is across models, not 200 scenarios; a single actor does not gain scenario concurrency."]
    elif name == "rmtbench":
        versions = {}
        for filename in spec["files"]:
            if not filename.endswith(".json"):
                continue
            counts = Counter()
            errors = []
            for index, row in enumerate(read_json(base / filename)):
                for key in ("character_understanding", "character_maintenance", "implicit_motivations", "user_preference"):
                    for block_index, block in enumerate(row[key]):
                        chat = block["chat"] if key == "user_preference" else block
                        if not isinstance(chat, list) or not all(isinstance(message, str) for message in chat):
                            errors.append({"character_index": index, "field": key, "block_index": block_index})
                        counts["blocks"] += 1
                        counts["stored_user_utterances"] += len(chat)
            versions[filename] = {**counts, "errors": errors}
        report["versions"] = versions
        report["stages"] = []
        report["needs"] = ["Official extended-dialogue ordering/seed and actor prompt", "Official executable harness", "Unreleased sensitive scenario data for the full benchmark"]
    else:
        stats = {}
        for lang in ("en", "zh"):
            seed_ids = {row["id"] for row in _jsonl(base / f"data/{lang}/seeds.jsonl")}
            keys = set()
            counts = Counter()
            errors = []
            for index, row in enumerate(_jsonl(base / f"data/{lang}/dialogues.jsonl")):
                key = (row["seed_id"], row["model_name"], row["run_id"])
                messages = json.loads(row["dialogue"]) if isinstance(row["dialogue"], str) else row["dialogue"]
                if key in keys or key[0] not in seed_ids or len(messages) != row["num_turns"]:
                    errors.append(index)
                keys.add(key)
                counts[str(row["num_turns"])] += 1
            evaluation_keys = [(r["seed_id"], r["model_name"], r["run_id"]) for r in _jsonl(base / f"data/{lang}/evaluations.jsonl")]
            stats[lang] = {"num_turns_histogram": dict(counts), "trajectory_count": len(keys), "structural_errors": errors,
                           "evaluation_keys_match": len(evaluation_keys) == len(set(evaluation_keys)) and set(evaluation_keys) == keys}
        report["published_trajectory_audit"] = stats
        report["stages"] = []
        report["needs"] = ["Exact User Simulator and actor prompts", "Exact judge identity/prompts", "Complete scoring/aggregation implementation"]
    report["source_preflight_passed"] = not report["official_reproduction_blockers"]
    if name in ("rmtbench", "minimax"):
        report["stages"] = ["generate", "judge", "aggregate", "evaluate"]
        report["local_runner"] = "public-method-v1"
        report["runner_note"] = "Local runnable reproduction; official release gaps remain listed separately."
    # Availability of code is not evidence of a working model service/environment.
    report["runtime_validated"] = False
    return report


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
