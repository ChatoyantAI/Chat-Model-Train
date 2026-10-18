"""Standalone conversion CLI; the shared format implementation is her_format.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.roleplay_benchmarks.her_format import (
    main, convert_her_to_coser, remove_system_thinking, conversion_record,
    HER_OUTPUT_INSTRUCTION, UPSTREAM_COMMIT, UPSTREAM_URL,
)

if __name__ == '__main__':
    raise SystemExit(main())
