"""Command-line entry point for the ELO campaign ranking."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .io import load_votes_csv, write_report
from .ranking import BradleyTerryError, build_ranking_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a position-agnostic batch Bradley-Terry ranking from a "
            "winner/loser CSV export."
        )
    )
    parser.add_argument("input_csv", type=Path, help="normalized valid-vote CSV")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="directory for CSV and JSON reports"
    )
    parser.add_argument(
        "--campaign-id",
        help="only include rows with this campaign_id and record it in the report",
    )
    parser.add_argument(
        "--expected-model-count",
        type=int,
        default=11,
        help="expected unique model count (default: 11; use 0 to disable)",
    )
    parser.add_argument(
        "--allow-incomplete-pairs",
        action="store_true",
        help="allow missing unordered model pairs (not recommended for this campaign)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="stratified bootstrap repetitions; use 0 to skip (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260827, help="bootstrap random seed"
    )
    parser.add_argument(
        "--duplicate-user-policy",
        choices=("error", "first"),
        default="error",
        help="fail on duplicate users or keep their first CSV row (default: error)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        votes = load_votes_csv(
            args.input_csv,
            campaign_id=args.campaign_id,
            duplicate_user_policy=args.duplicate_user_policy,
        )
        expected_model_count = args.expected_model_count or None
        report = build_ranking_report(
            votes,
            expected_model_count=expected_model_count,
            require_complete_pairs=not args.allow_incomplete_pairs,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        )
        write_report(
            args.output_dir,
            report,
            input_path=args.input_csv,
            campaign_id=args.campaign_id,
        )
    except (BradleyTerryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"ranked {report.model_count} models from {report.vote_count} votes "
        f"across {report.pair_count} pairs"
    )
    print(f"wrote ranking.csv, pairwise.csv, and summary.json to {args.output_dir}")
    return 0
