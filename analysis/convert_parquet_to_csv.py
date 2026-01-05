#!/usr/bin/env python
"""Convert parquet splits to CSV and preview their heads.

Usage:
    python analysis/convert_parquet_to_csv.py \
        --input-dir data/processed/BTCUSDT_v1 \
        --output-dir analysis/csv_preview \
        --head 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def convert_files(files: Iterable[Path], output_dir: Path, head_rows: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fp in files:
        if not fp.exists():
            print(f"Skipping missing file: {fp}")
            continue
        df = pd.read_parquet(fp)
        out_csv = output_dir / f"{fp.stem}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Converted {fp} -> {out_csv}")
        if head_rows > 0:
            print(f"\nHead of {fp.name}:")
            print(df.head(head_rows))
            print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert parquet splits to CSV and preview their heads.")
    parser.add_argument("--input-dir", default="data/processed/BTCUSDT_v1", help="Directory containing train/val/test parquet files.")
    parser.add_argument("--output-dir", default="analysis/csv_preview", help="Where to write CSV outputs.")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to preview per file (0 to skip).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = [input_dir / "train.parquet", input_dir / "val.parquet", input_dir / "test.parquet"]
    convert_files(files, output_dir, head_rows=args.head)


if __name__ == "__main__":
    main()
