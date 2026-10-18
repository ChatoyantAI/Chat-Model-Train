"""Stable launcher; implementation: src/evaluation/roleplay_benchmarks/cli.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.roleplay_benchmarks.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
