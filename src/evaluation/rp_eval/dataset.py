from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seeds(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_selection(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["conversations"]


def verify_sha256s(data_dir: Path) -> dict[str, dict[str, Any]]:
    sums_path = data_dir / "SHA256SUMS"
    results: dict[str, dict[str, Any]] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        actual = sha256_file(data_dir / name)
        results[name] = {"expected": expected.lower(), "actual": actual, "ok": actual == expected.lower()}
    return results


def _depth(seed: dict[str, Any]) -> int:
    return len(seed["ordered_user_messages"])


def staged_cohorts(seeds: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    indexed = list(enumerate(seeds))
    ranked = sorted(indexed, key=lambda item: (-_depth(item[1]), item[0], item[1]["sample_id"]))
    cohort30 = {seed["sample_id"] for _, seed in ranked[:150]}
    cohort60 = {seed["sample_id"] for _, seed in ranked[:50]}
    return cohort30, cohort60


def iter_staged_plan(seeds: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    cohort30, cohort60 = staged_cohorts(seeds)
    for source_index, seed in enumerate(seeds):
        sample_id = seed["sample_id"]
        target_depth = 60 if sample_id in cohort60 else 30 if sample_id in cohort30 else 10
        available = _depth(seed)
        for user_turn_index in range(min(available, target_depth)):
            yield {
                "sample_id": sample_id,
                "source_index": source_index,
                "user_turn_index": user_turn_index,
                "checkpoint_cohort": target_depth,
                "language": seed.get("metadata", {}).get("language"),
                "segment": seed.get("user", {}).get("segment"),
                "user_id": seed.get("user", {}).get("id"),
                "character_id": seed.get("character", {}).get("id"),
            }


def preflight(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = config["benchmark"]
    data_dir = Path(benchmark["data_dir"])
    sha_results = verify_sha256s(data_dir)
    selection = load_selection(data_dir / "dialogue-selection.json.gz")
    seeds = load_seeds(data_dir / "rollout-seeds.jsonl.gz")
    selection_ids = [item["id"] for item in selection]
    seed_ids = [item["sample_id"] for item in seeds]
    plan = list(iter_staged_plan(seeds))
    users = [item["user"]["id"] for item in seeds]
    characters = [item["character"]["id"] for item in seeds]
    checks = {
        "all_sha256_ok": bool(sha_results) and all(v["ok"] for v in sha_results.values()),
        "selection_count_ok": len(selection) == benchmark["expected_selection_count"],
        "seed_count_ok": len(seeds) == benchmark["expected_seed_count"],
        "selection_seed_order_identical": selection_ids == seed_ids,
        "sample_ids_unique": len(set(seed_ids)) == len(seed_ids),
        "user_ids_unique": len(set(users)) == len(users),
        "character_ids_unique": len(set(characters)) == len(characters),
        "staged_anchor_count_ok": len(plan) == benchmark["expected_staged_anchor_count"],
    }
    report = {
        "schema_version": "1.0",
        "data_dir": str(data_dir),
        "sha256": sha_results,
        "counts": {
            "selection": len(selection),
            "seeds": len(seeds),
            "users_unique": len(set(users)),
            "characters_unique": len(set(characters)),
            "staged_anchors": len(plan),
        },
        "checks": checks,
        "ready": all(checks.values()),
    }
    return report, plan
