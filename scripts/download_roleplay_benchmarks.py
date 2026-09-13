"""Download pinned public benchmark assets without installing GPU/LLM libraries."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.roleplay_benchmarks.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmarks", nargs="*", default=["all"])
    parser.add_argument("--env-file", help="Optional ignored file with your own HF_TOKEN")
    args = parser.parse_args()
    forwarded = ["--env-file", args.env_file] if args.env_file else []
    raise SystemExit(main([*forwarded, "sync", *(args.benchmarks or ["all"])]))
