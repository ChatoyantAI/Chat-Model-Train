"""Reconstruct one production context from frozen ELO data."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.evaluation.roleplay_judge_dataset.cli import main  # noqa: E402


raise SystemExit(main(["single", *sys.argv[1:]]))
