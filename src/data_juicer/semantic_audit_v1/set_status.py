#!/usr/bin/env python3
"""Small CLI used by the systemd wrapper to record terminal failures."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import update_status, utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--message")
    args = parser.parse_args()
    changes = {
        "status": args.status,
        "stage": args.stage,
        "message": args.message,
    }
    if args.status == "failed":
        changes["failed_at"] = utc_now()
    update_status(args.path, **changes)


if __name__ == "__main__":
    main()

