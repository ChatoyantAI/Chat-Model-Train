import argparse
import json
from pathlib import Path
import sys

from .sources import ROOT, inventory, registry, sync_one, verify_one, write_json, load_env_file
from .runtime import prepare, run
from .reports import report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pinned official role-play benchmark suite (no silent protocol substitutions)")
    parser.add_argument("--env-file", help="Local ignored file containing HF_TOKEN; no credential values are printed")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("catalog", "sync", "verify"):
        command = commands.add_parser(name)
        command.add_argument("benchmarks", nargs="*", default=["all"], help="coser characterbench rmtbench minimax, or all")
    command = commands.add_parser("prepare")
    command.add_argument("--config", required=True)
    command.add_argument("--run-dir", required=True)
    command = commands.add_parser("run")
    command.add_argument("--run-dir", required=True)
    command.add_argument("--stage", required=True, choices=["evaluate", "generate", "prompts", "judge", "aggregate"])
    command.add_argument("--python", default=sys.executable)
    command.add_argument("--execute", action="store_true", help="Without this flag, only show the exact upstream commands")
    command.add_argument("--max-requests", type=int, help="RMTBench/MiniMax API attempt limit for this invocation")
    command = commands.add_parser("report")
    command.add_argument("--runs", nargs="+", required=True)
    command.add_argument("--output", required=True)
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
        if args.command == "prepare":
            manifest = prepare(args.config, args.run_dir)
            result = {"status": manifest["status"], "inventory": manifest["inventory"], "run_dir": str(Path(args.run_dir).resolve())}
        elif args.command == "run":
            result = run(args.run_dir, args.stage, args.python, args.execute, args.max_requests)
        else:
            result = report(args.runs, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status", "").startswith(("incomplete-", "failed-")) else 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
