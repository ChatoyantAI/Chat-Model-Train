"""Report native scores separately; never manufacture a cross-benchmark average."""
import hashlib
import json
from pathlib import Path

from .sources import digest, read_json, write_json
from .runtime import audit_character


def protocol_id(manifest):
    config = manifest["config"].copy()
    if manifest.get("runner") == "public-method-v1":
        config = json.loads(json.dumps(config))
        # Exclude actor identity and operational limits, retain actor sampling.
        for key in ("concurrency", "max_requests_per_run"):
            config.pop(key, None)
        for name, role in config["roles"].items():
            role.pop("api_key_env", None)
            role.pop("base_url_env", None)
            if name == "actor":
                role.pop("model", None)
                role.pop("revision", None)
        payload = {"config": config, "revision": manifest["upstream_revision"],
                   "prompts": manifest["prompts_sha256"], "cases": manifest["cases_sha256"],
                   "engine": manifest["engine_hashes"],
                   "helper_endpoints": {k: v for k, v in manifest.get("endpoint_fingerprints", {}).items() if k != "actor"}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    # Model identities differ intentionally; sampling/protocol settings must not.
    for key in ("actor_model", "actor_alias", "model_revision", "runtime_notes", "api_key_env"):
        config.pop(key, None)
    payload = {"config": config, "revision": manifest["upstream_revision"], "patch": manifest["engineering_patch_sha256"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def audit_coser(work):
    source = read_json(work / "data/test/test_set.json")
    expected = {f"{r['book']}-{r['plot']['i_p']}-{r['i_c']}" for r in source}
    files = list((work / "exp/evaluation").rglob("*.json"))
    if len(files) != 1:
        return {"complete": False, "expected": len(expected), "files": len(files), "reason": "Expected exactly one result file"}
    data = read_json(files[0])
    cases = data["cases"]
    dimensions = ["Storyline Consistency", "Anthropomorphism", "Character Fidelity", "Storyline Quality"]
    invalid = []
    for key, case in cases.items():
        try:
            values = [float(case["critique"][dim]["score"]) for dim in dimensions]
            if not all(0 <= value <= 100 for value in values) or abs(sum(values)/4 - float(case["score"])) > 1e-6:
                invalid.append(key)
        except (TypeError, KeyError, ValueError):
            invalid.append(key)
    return {"complete": set(cases) == expected and len(expected) == len(source) and not invalid,
            "expected": len(source), "judged": len(cases), "invalid": invalid,
            "missing": sorted(expected - set(cases)), "unexpected": sorted(set(cases) - expected)}


def describe_run(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    benchmark = manifest["benchmark"]
    if manifest.get("runner") == "public-method-v1":
        from .community import load_run, aggregate
        directory, manifest, cases, _ = load_run(directory, allow_external=True)
        summary = aggregate(directory, manifest, cases)
        actor = manifest["config"]["roles"]["actor"]
        return {"benchmark": benchmark, "actor": actor["model"], "model_revision": actor["revision"],
                "protocol_id": protocol_id(manifest), "status": manifest["status"], "run_dir": str(directory),
                "native_scores": summary["scores"], "score_provenance": "local-public-method-reproduction",
                "coverage": summary["coverage"], "score_files": {"summary.json": digest(directory / "summary.json")},
                "eligible_for_comparison": summary["coverage"]["complete"] and "aggregate" in manifest["completed_stages"]}
    row = {"benchmark": benchmark, "actor": manifest["config"]["actor_model"],
           "model_revision": manifest["config"]["model_revision"], "protocol_id": protocol_id(manifest),
           "status": manifest["status"], "run_dir": str(directory), "native_scores": {}, "score_files": {}}
    work = directory / "upstream"
    if benchmark == "coser":
        row["coverage"] = audit_coser(work)
        for file in (work / "exp/evaluation").rglob("*.json"):
            row["native_scores"][file.relative_to(work).as_posix()] = read_json(file)["scores"]
            row["score_files"][file.relative_to(work).as_posix()] = digest(file)
    else:
        row["coverage"] = audit_character(work)
        for lang in ("en", "zh"):
            file = work / f"results/{lang}.json"
            if file.exists():
                row["native_scores"][lang] = read_json(file)
                row["score_files"][lang] = digest(file)
    final_stage = "evaluate" if benchmark == "coser" else "aggregate"
    row["eligible_for_comparison"] = row["coverage"]["complete"] and bool(row["native_scores"]) and final_stage in manifest["completed_stages"]
    return row


def report(directories, output):
    rows = [describe_run(path) for path in directories]
    result = {"schema_version": 1, "runs": rows, "comparison_groups": {},
              "note": "Only eligible runs with the same protocol_id may be compared. Native scores remain separate by benchmark and language. No benchmark-wide average is defined."}
    for index, row in enumerate(rows):
        if row["eligible_for_comparison"]:
            result["comparison_groups"].setdefault(row["protocol_id"], []).append(index)
    write_json(output, result)
    markdown = ["# Role-play benchmark results", "", result["note"], "", "| Benchmark | Model | Status | Complete coverage | Comparable |", "|---|---|---|---|---|"]
    for row in rows:
        fields = [row["benchmark"], row["actor"], row["status"], row["coverage"]["complete"], row["eligible_for_comparison"]]
        markdown.append("| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in fields) + " |")
    for row in rows:
        markdown.extend(["", f"## {row['benchmark']} / {row['actor']}", "", "```json", json.dumps(row["native_scores"], ensure_ascii=False, indent=2), "```"])
    Path(output).with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return result
