"""CLI to run EDA/quality checks/plots on dataset splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .config import EDAConfig, ensure_dir
from .io import load_table
from .plots import generate_plots
from .quality import compare_splits, quality_report
from .report import build_dashboard, profiling_report, write_summary
from .targets import target_report


def run(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    out_dir: Path,
    raw_plots_dir: Optional[Path] = None,
) -> None:
    cfg = EDAConfig()
    out_dir = ensure_dir(out_dir)

    print("Loading splits...")
    train = load_table(train_path)
    val = load_table(val_path)
    test = load_table(test_path)

    print("Running quality checks...")
    q_train = quality_report(train, cfg)
    q_val = quality_report(val, cfg)
    q_test = quality_report(test, cfg)
    split_cmp = compare_splits(train, val, test, cfg)

    print("Summarizing targets...")
    t_train = target_report(train, cfg)
    t_val = target_report(val, cfg)
    t_test = target_report(test, cfg)

    print("Generating plots...")
    plot_entries: List[Dict[str, str]] = []
    for name, df in [("train", train), ("val", val), ("test", test)]:
        paths = generate_plots(df, name, cfg, out_dir)
        for p in paths:
            plot_entries.append({"href": p.relative_to(out_dir).as_posix(), "label": p.name})

    print("Generating profiling reports (minimal)...")
    prof_entries = []
    for name, df in [("train", train), ("val", val), ("test", test)]:
        prof_path = profiling_report(df, out_dir / f"profile_{name}.html", title=f"{name} profile")
        prof_entries.append({"href": prof_path.relative_to(out_dir).as_posix(), "label": prof_path.name})

    summary = {
        "quality": {"train": q_train, "val": q_val, "test": q_test, "compare": split_cmp},
        "targets": {"train": t_train, "val": t_val, "test": t_test},
    }
    summary_path = out_dir / "summary.json"
    write_summary(summary, summary_path)

    # Optional raw plots section (e.g., from analysis/raw_plots)
    raw_plot_entries: List[Dict[str, str]] = []
    if raw_plots_dir:
        for p in sorted(Path(raw_plots_dir).glob("*.html")):
            raw_plot_entries.append({"href": Path(p).resolve().as_posix(), "label": p.name})

    build_dashboard(out_dir, summary_path, plot_entries, prof_entries, raw_plot_entries)
    print(f"EDA outputs written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on train/val/test splits.")
    parser.add_argument("--train", required=True, help="Path to train CSV/Parquet")
    parser.add_argument("--val", required=True, help="Path to val CSV/Parquet")
    parser.add_argument("--test", required=True, help="Path to test CSV/Parquet")
    parser.add_argument("--out", default="reports/eda", help="Output directory")
    parser.add_argument(
        "--raw-plots",
        default=None,
        help="Optional directory containing raw plot HTML files to include in dashboard (e.g., analysis/raw_plots)",
    )
    args = parser.parse_args()
    run(Path(args.train), Path(args.val), Path(args.test), Path(args.out), raw_plots_dir=Path(args.raw_plots) if args.raw_plots else None)


if __name__ == "__main__":
    main()
