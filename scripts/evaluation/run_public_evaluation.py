"""Single entry point for resumable public role-play benchmark runs."""
import argparse
import json
import os
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.evaluation.roleplay_benchmarks import unified
from src.evaluation.roleplay_benchmarks import bootstrap


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run fixed role-play benchmarks against any OpenAI-compatible endpoint")
    parser.add_argument('--env-file', help='Read endpoint/key environment variables without shell execution')
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser('init', help='create a model config with the fixed benchmark settings')
    init.add_argument('--config', required=True)
    init.add_argument('--actor-model', required=True)
    init.add_argument('--actor-kind', choices=['local', 'remote'], default='local')
    init.add_argument('--actor-revision')
    init.add_argument('--judge-model', default='zai-org/GLM-5.3-Flash')
    init.add_argument('--benchmarks', nargs='+', default=['characterbench', 'coser'])
    check = commands.add_parser('check', help='check dependencies, sources, and endpoint configuration')
    check.add_argument('--config', required=True)
    check.add_argument('--check-api', action='store_true', help='also GET /models; sends no paid completions')
    prepare = commands.add_parser("prepare", help="freeze benchmark inputs and endpoint identities")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--run-dir", required=True)
    run = commands.add_parser("run", help="resume the prepared run")
    run.add_argument("--run-dir", required=True)
    run.add_argument('--config', help='prepare a new run automatically; existing runs must match this config')
    run.add_argument("--execute", action="store_true", help="send requests; without it only show the plan")
    run.add_argument("--max-requests", type=int)
    worker = commands.add_parser('coser-worker', help=argparse.SUPPRESS)
    worker.add_argument('--run-dir', required=True)
    worker.add_argument('--port', required=True, type=int)
    worker.add_argument('--case', required=True, type=int)
    worker.add_argument('--attempt', required=True, type=int)
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            bootstrap.load_environment(args.env_file)
        if args.command == 'coser-worker':
            from src.evaluation.roleplay_benchmarks.frozen_backend import worker
            worker(Path(args.run_dir), args.case, args.attempt, args.port)
            return 0
        if args.command == 'init':
            result = bootstrap.initialize(args.config, args.actor_model, args.actor_kind,
                                          args.benchmarks, args.actor_revision, args.judge_model)
        elif args.command == 'check':
            result = bootstrap.check(unified.read_json(args.config), args.check_api)
        elif args.command == "prepare":
            result = unified.prepare(args.config, args.run_dir)
        else:
            if args.config:
                config = unified.read_json(args.config)
                manifest = Path(args.run_dir) / 'manifest.json'
                if manifest.exists():
                    if config != unified.read_json(manifest)['config']:
                        raise ValueError('Config differs from the saved run; choose a new run directory')
                else:
                    bootstrap.check(config)
                    unified.prepare(config, args.run_dir)
            result = unified.run(args.run_dir, args.execute, args.max_requests)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status", "").startswith(("completed", "prepared")) or result.get("dry_run") else 2
    except (ValueError, RuntimeError, OSError, LookupError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
