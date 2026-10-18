"""Single entry point for resumable public role-play benchmark runs."""
import argparse
import json
from pathlib import Path
import sys
import time

from . import unified, bootstrap
from .sources import read_json


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
    init.add_argument('--her-system-thinking', choices=['with-system', 'without-system'],
                      help='HER official CoSER mode; with-system uses official Qwen ChatML completions')
    init.add_argument('--assessment', choices=['reviewed', 'legacy'], default='reviewed',
                      help='new runs use reviewed scoring; legacy retains the historical protocol')
    init.add_argument('--coser-role-thoughts', choices=['include', 'exclude'], default='include',
                      help='use the SAME explicit judge view for paired native/HER runs')
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
    ownership = run.add_mutually_exclusive_group()
    ownership.add_argument('--wait-for-endpoints', action='store_true',
                     help='queue behind an existing evaluation using the same endpoint; preserve shared limits')
    ownership.add_argument('--independent-endpoints', action='store_true',
                           help='explicitly allow overlapping evaluations; each retains its own configured limits')
    worker = commands.add_parser('coser-worker', help='internal CoSER scene worker, started by run')
    worker.add_argument('--run-dir', required=True)
    worker.add_argument('--port', required=True, type=int)
    worker.add_argument('--case', required=True, type=int)
    worker.add_argument('--attempt', required=True, type=int)
    rejudge = commands.add_parser('coser-rejudge', help='rejudge saved trajectories with explicit speaker labels; no actor calls')
    rejudge.add_argument('--source-runs', nargs=2, help='two source runs; prepare their common successful scenes')
    rejudge.add_argument('--run-dir', required=True)
    rejudge.add_argument('--execute', action='store_true')
    audit = commands.add_parser('audit-results', help='read-only checks of saved CoSER scores and judge transcripts; no model calls')
    audit.add_argument('--inputs', nargs='+', required=True)
    audit.add_argument('--output', required=True, help='new audit JSON; never overwrites existing evidence')
    args = parser.parse_args(argv)
    try:
        if args.env_file:
            bootstrap.load_environment(args.env_file)
        if args.command == 'audit-results':
            from src.evaluation.roleplay_benchmarks.assessment_audit import save_audit
            result = save_audit(args.inputs, args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'coser-rejudge':
            from src.evaluation.roleplay_benchmarks import coser_rejudge
            if args.source_runs:
                coser_rejudge.prepare(args.source_runs, args.run_dir)
            result = coser_rejudge.run(args.run_dir, args.execute)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get('dry_run') or result.get('status') == 'completed' else 2
        if args.command == 'coser-worker':
            from src.evaluation.roleplay_benchmarks.frozen_backend import worker
            worker(Path(args.run_dir), args.case, args.attempt, args.port)
            return 0
        if args.command == 'init':
            result = bootstrap.initialize(args.config, args.actor_model, args.actor_kind,
                                          args.benchmarks, args.actor_revision, args.judge_model,
                                          args.her_system_thinking, args.assessment, args.coser_role_thoughts)
        elif args.command == 'check':
            result = bootstrap.check(read_json(args.config), args.check_api)
        elif args.command == "prepare":
            result = unified.prepare(args.config, args.run_dir)
        else:
            if args.config:
                config = read_json(args.config)
                manifest = Path(args.run_dir) / 'manifest.json'
                if manifest.exists():
                    if config != read_json(manifest)['config']:
                        raise ValueError('Config differs from the saved run; choose a new run directory')
                else:
                    bootstrap.check(config)
                    unified.prepare(config, args.run_dir)
            announced_wait = False
            while True:
                try:
                    options = {'independent_endpoints': True} if args.independent_endpoints else {}
                    result = unified.run(args.run_dir, args.execute, args.max_requests, **options)
                    break
                except RuntimeError as exc:
                    busy = 'Another evaluation owns this endpoint; run selected benchmarks in one supervisor'
                    if not args.wait_for_endpoints or str(exc) != busy:
                        raise
                    if not announced_wait:
                        print('Waiting for the existing endpoint owner to finish; no new requests sent.',
                              file=sys.stderr, flush=True)
                        announced_wait = True
                    time.sleep(5)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status", "").startswith(("completed", "prepared")) or result.get("dry_run") else 2
    except (ValueError, RuntimeError, OSError, LookupError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
