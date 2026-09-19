import argparse
import json
from pathlib import Path
import sys

from .sources import ROOT, inventory, registry, sync_one, verify_one, write_json, load_env_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download and verify pinned benchmark sources; evaluation uses run_public_evaluation.py")
    parser.add_argument("--env-file", help="Local ignored file containing HF_TOKEN; no credential values are printed")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("catalog", "sync", "verify"):
        command = commands.add_parser(name)
        command.add_argument("benchmarks", nargs="*", default=["all"], help="coser characterbench rmtbench minimax, or all")
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            load_env_file(args.env_file)
        if args.command in ("catalog", "sync", "verify"):
            specs = registry()
            names = args.benchmarks or ["all"]
            if set(names) - set(specs) - {"all"}:
                raise ValueError("Unknown benchmark; choose coser, characterbench, rmtbench, minimax, all")
            selected = list(specs) if "all" in names else list(dict.fromkeys(names))
            results = {}
            failed = False
            for name in selected:
                spec = specs[name]
                print(f"{args.command}: {name}", file=sys.stderr, flush=True)
                try:
                    details = None if args.command == "catalog" else (sync_one(name, spec) if args.command == "sync" else verify_one(name, spec))
                    results[name] = {"revision": spec["revision"], "protocol": spec["protocol"], "blockers": spec["blockers"], "inventory": inventory(name, spec)}
                    if details:
                        results[name]["source_status"] = details["status"]
                        results[name]["file_count"] = len(details["files"])
                        results[name]["errors"] = details.get("errors", {})
                        failed |= details["status"] == "incomplete"
                except (ValueError, OSError, RuntimeError) as error:
                    failed = True
                    results[name] = {"source_status": "unavailable", "error": str(error), "blockers": spec["blockers"]}
            if args.command != "catalog":
                write_json(ROOT / f"data/external/roleplay_benchmarks/{args.command}-report.json", results)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 2 if failed else 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
