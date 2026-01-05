#!/usr/bin/env python
"""
process_logs.py

Convert raw log files into clean machine‑learning ready datasets using your
existing log parser.  The parser script path is read from the `LOG_PARSER_PATH`
environment variable (see `.env.example`).

For each file in `data/raw/`, this script invokes the parser and writes the
output into `data/processed/` with the same base name and a `.csv` extension
(you can adjust the extension in the code).  Files that already exist in the
output directory are skipped unless `--force` is specified.

Example:

```bash
python scripts/process_logs.py
```

This script assumes that your log parser can be executed like:

```
python logparser.py --input path/to/raw.log --output path/to/out.csv
```

Adjust the `run_parser()` function if the interface differs.
"""

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def run_parser(parser_path: str, input_path: Path, output_path: Path) -> None:
    """Execute the external log parser.

    Modify this function if your parser accepts different arguments or if you
    want to import a Python function instead of invoking a subprocess.
    """
    cmd = ["python", parser_path, "--input", str(input_path), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error parsing {input_path}: {result.stderr}")
    else:
        print(f"Processed {input_path.name} -> {output_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Process raw log files into ML-ready data")
    parser.add_argument(
        "--force", action="store_true", help="Reprocess files even if outputs already exist"
    )
    parser.add_argument(
        "--ext", default=".log", help="Extension of raw log files to look for (default: .log)"
    )
    parser.add_argument(
        "--output-ext", default=".csv", help="Extension for processed files (default: .csv)"
    )
    args = parser.parse_args()

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT", "./data")
    parser_path = os.environ.get("LOG_PARSER_PATH")
    if not parser_path:
        raise RuntimeError("LOG_PARSER_PATH environment variable must be set to your log parser script")
    parser_path = os.path.expanduser(parser_path)

    raw_dir = Path(data_root) / "raw"
    out_dir = Path(data_root) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(f for f in raw_dir.glob(f"*{args.ext}") if f.is_file())
    if not raw_files:
        print(f"No raw files with extension {args.ext} found in {raw_dir}")
        return

    for raw_file in raw_files:
        output_file = out_dir / (raw_file.stem + args.output_ext)
        if output_file.exists() and not args.force:
            print(f"Skipping {output_file.name} (already exists, use --force to overwrite)")
            continue
        run_parser(parser_path, raw_file, output_file)


if __name__ == "__main__":
    main()