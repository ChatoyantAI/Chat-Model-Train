"""Small per-benchmark command interface, shared by the four thin scripts."""
import argparse
import json
from pathlib import Path
import sys

from .preflight import inspect_source
from .runtime import prepare, run
from .sources import ROOT, read_json, write_json


def entry(benchmark, argv=None):
    parser = argparse.ArgumentParser(description=f"{benchmark}: inspect official protocol, prepare isolated run, execute native stages")
    parser.add_argument("action", choices=["inspect", "prepare", "run"])
    parser.add_argument("--config")
    parser.add_argument("--run-dir")
    parser.add_argument("--stage")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-requests", type=int, help="RMTBench/MiniMax: API attempt limit for this invocation, including retries")
    parser.add_argument("--output", help="Optional inspection JSON, recommended under outputs/")
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            result = inspect_source(benchmark)
            if args.output:
                write_json(args.output, result)
        else:
            if not args.run_dir:
                raise ValueError("--run-dir is required")
            if args.action == "prepare":
                config = args.config or str(ROOT / f"configs/evaluation/roleplay_benchmarks/{benchmark}.example.json")
                if read_json(config).get("benchmark") != benchmark:
                    raise ValueError("Config belongs to another benchmark")
                manifest = prepare(config, args.run_dir)
                result = {"status": manifest["status"], "inventory": manifest["inventory"], "run_dir": str(Path(args.run_dir).resolve())}
            else:
                if read_json(Path(args.run_dir) / "manifest.json")["benchmark"] != benchmark:
                    raise ValueError("Run directory belongs to another benchmark")
                stage = args.stage or ("generate" if benchmark == "characterbench" else "evaluate")
                result = run(args.run_dir, stage, args.python, args.execute, args.max_requests)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get("status", "").startswith(("incomplete-", "failed-")) else 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
