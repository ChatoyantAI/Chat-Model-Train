"""Install the endpoint-only evaluator and pinned benchmark assets."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.roleplay_benchmarks.bootstrap import install, load_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path.home() / '.cache/chat-model-train/eval-venv')
    parser.add_argument('--benchmarks', nargs='+', default=['all'])
    parser.add_argument('--env-file')
    parser.add_argument('--cache-from', type=Path, help='Optional existing checkout: reuse hash-verified benchmark assets')
    args = parser.parse_args()
    try:
        if args.env_file:
            load_environment(args.env_file)
        install(args.venv, args.benchmarks, args.cache_from)
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
