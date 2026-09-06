"""Command-line interfaces for context reconstruction and Judge dataset builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .builder import BuildError, build_judge_dataset
from .context_builder import ContextBuildError, build_candidate_context, restore_history
from .io import find_json_group, find_json_record


def build_one_context(
    *,
    raw_dir: Path,
    runtime_path: Path,
    comparison_id: str,
    candidate_side: str,
) -> Mapping[str, Any]:
    comparison = find_json_record(raw_dir / "comparisons.jsonl.gz", comparison_id)
    if comparison is None:
        raise BuildError(f"comparison not found: {comparison_id}")
    runtime = find_json_record(runtime_path, comparison_id)
    if runtime is None:
        raise BuildError(f"runtime snapshot not found: {comparison_id}")
    context_rows = find_json_group(
        raw_dir / "context_messages.jsonl.gz", comparison_id
    )
    image_rows = find_json_group(
        raw_dir / "message_image_contexts.jsonl.gz", comparison_id
    )
    mutation_rows = find_json_group(
        raw_dir / "message_mutation_events.jsonl.gz", comparison_id
    )
    restored = restore_history(comparison, context_rows, image_rows, mutation_rows)
    result = build_candidate_context(
        comparison,
        candidate_side,
        restored,
        runtime,
    )
    return {
        "input_messages": list(result.input_messages),
        "audit": dict(result.audit),
    }


def _add_single_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--runtime-file", type=Path, required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--candidate", choices=("left", "right"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON to this local file instead of stdout.",
    )


def _add_batch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--runtime-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rubric-version",
        default="roleplay-rubric-v0.1",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct production roleplay contexts and build Judge data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_single_arguments(
        subparsers.add_parser("single", help="Reconstruct one candidate context.")
    )
    _add_batch_arguments(
        subparsers.add_parser("batch", help="Build the complete Judge dataset.")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "single":
            result = build_one_context(
                raw_dir=args.raw_dir.resolve(),
                runtime_path=args.runtime_file.resolve(),
                comparison_id=args.comparison_id,
                candidate_side=args.candidate,
            )
            serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized, encoding="utf-8")
            else:
                sys.stdout.write(serialized)
        else:
            result = build_judge_dataset(
                raw_dir=args.raw_dir,
                runtime_path=args.runtime_file,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                rubric_version=args.rubric_version,
            )
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    except (BuildError, ContextBuildError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
