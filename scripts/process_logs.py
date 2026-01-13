#!/usr/bin/env python
# ruff: noqa: E402
"""
process_logs.py

Convert raw log files into clean machine-learning ready datasets using the
project's built-in log parser.

For each .log file in `data/raw/`, this script parses and writes two CSVs
into `data/processed/`:

- `signals/<file_stem>.csv` - Model signals with entry decisions
- `trade_fills/<file_stem>.csv` - Binance order fills

Files that already exist in the output directory are skipped unless `--force`
is specified.

Example:

```bash
python scripts/process_logs.py
python scripts/process_logs.py --force  # overwrite existing
```
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Allow running the script without installing the package
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from myproj.config import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from myproj.data import parse_log_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Process raw log files into ML-ready data")
    parser.add_argument(
        "--force", action="store_true", help="Reprocess files even if outputs already exist"
    )
    parser.add_argument("--ext", default=".log", help="Extension of raw log files to look for (default: .log)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write processed CSVs (default: DATA_ROOT/processed)",
    )
    args = parser.parse_args()

    raw_dir = RAW_DATA_DIR
    out_dir = Path(args.output_dir).resolve() if args.output_dir else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(f for f in raw_dir.glob(f"*{args.ext}") if f.is_file())
    if not raw_files:
        print(f"No raw files with extension {args.ext} found in {raw_dir}")
        return

    for raw_file in raw_files:
        base_name = raw_file.stem
        signals_path = out_dir / "signals" / f"{base_name}.csv"
        fills_path = out_dir / "trade_fills" / f"{base_name}.csv"

        if signals_path.exists() and fills_path.exists() and not args.force:
            print(f"Skipping {base_name} (already parsed, use --force to overwrite)")
            continue

        signals_out, fills_out = parse_log_file(raw_file, output_dir=out_dir, base_name=base_name)
        print(f"Processed {raw_file.name} -> {signals_out.name}, {fills_out.name}")


if __name__ == "__main__":
    main()
