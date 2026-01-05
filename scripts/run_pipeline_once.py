#!/usr/bin/env python
# ruff: noqa: E402
"""
Run the end-to-end pipeline once: download raw logs from Drive and parse them into clean CSVs.

Example:
    python scripts/run_pipeline_once.py
    python scripts/run_pipeline_once.py --subset 2025-01-05 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
# Allow running without installing the package
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from download_data import download_with_gdown, download_with_pydrive  # type: ignore  # noqa: E402
from myproj.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, get_env  # noqa: E402
from myproj.data import parse_log_file  # noqa: E402


def download_from_drive(subset: str | None, backend: str) -> None:
    """Download raw files from Drive into RAW_DATA_DIR."""
    folder_id = get_env("DRIVE_FOLDER_ID")
    if not folder_id:
        raise RuntimeError("DRIVE_FOLDER_ID is not set in environment/.env")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if backend == "gdown":
        download_with_gdown(folder_id, RAW_DATA_DIR, pattern=subset)
    else:
        download_with_pydrive(folder_id, RAW_DATA_DIR, pattern=subset)


def parse_downloaded(ext: str, force: bool, output_dir: Path) -> None:
    """Parse downloaded raw files into processed CSVs."""
    raw_files = sorted(f for f in RAW_DATA_DIR.glob(f"*{ext}") if f.is_file())
    if not raw_files:
        print(f"No raw files with extension {ext} found in {RAW_DATA_DIR}")
        return

    for raw_file in raw_files:
        base_name = raw_file.stem
        strategy_path = output_dir / "strategy_updates" / f"{base_name}.csv"
        fills_path = output_dir / "trade_fills" / f"{base_name}.csv"
        if strategy_path.exists() and fills_path.exists() and not force:
            print(f"Skipping {base_name} (outputs exist; use --force to reparse)")
            continue

        strategy_out, fills_out = parse_log_file(raw_file, output_dir=output_dir, base_name=base_name)
        print(f"Parsed {raw_file.name} -> {strategy_out.name}, {fills_out.name}")


def main() -> None:
    load_dotenv()

    cli = argparse.ArgumentParser(description="Download raw logs from Drive and parse to clean CSVs.")
    cli.add_argument("--subset", help="Optional substring/regex to filter Drive files", default=None)
    cli.add_argument(
        "--backend",
        choices=["gdown", "pydrive"],
        default="gdown",
        help="Backend to use for downloading (default: gdown)",
    )
    cli.add_argument("--ext", default=".log", help="Extension of raw log files to parse (default: .log)")
    cli.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write processed CSVs (default: DATA_ROOT/processed)",
    )
    cli.add_argument("--force", action="store_true", help="Reparse even if outputs already exist")
    args = cli.parse_args()

    output_dir = Path(args.output_dir).resolve() if args.output_dir else PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    download_from_drive(args.subset, args.backend)
    parse_downloaded(args.ext, args.force, output_dir)


if __name__ == "__main__":
    main()
