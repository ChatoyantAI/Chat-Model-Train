#!/usr/bin/env python3
"""Validate a pipeline artifact and print its record count."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def parquet_rows(path: Path) -> int:
    if path.is_file():
        return pq.ParquetFile(path).metadata.num_rows
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return ds.dataset([str(file) for file in files], format="parquet").count_rows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(parquet_rows(args.path))


if __name__ == "__main__":
    main()
