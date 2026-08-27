"""Thin local wrapper for ``python -m src.evaluation.elo_ranking``."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.elo_ranking.cli import main  # noqa: E402


raise SystemExit(main())
