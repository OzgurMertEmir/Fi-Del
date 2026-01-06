#!/usr/bin/env python
"""Quick inspection of raw BTCUSDT.store bid/ask/qty/volume over time.

Usage:
    python analysis/inspect_raw_store.py \
        --path data/raw/BTCUSDT.store \
        --skip-rows 1 \
        --out analysis/raw_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio


def load_store(path: Path, skip_rows: int) -> pd.DataFrame:
    cols = ["rec_type", "ts_ms", "bid_px", "bid_qty", "ask_px", "ask_qty", "vol_base"]
    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        skiprows=skip_rows,
        dtype={
            "rec_type": "int64",
            "ts_ms": "int64",
            "bid_px": "float64",
            "bid_qty": "float64",
            "ask_px": "float64",
            "ask_qty": "float64",
            "vol_base": "float64",
        },
    )
    df = df.dropna()
    df = df.sort_values("ts_ms").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=path, include_plotlyjs="cdn", auto_open=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw BTCUSDT store file.")
    parser.add_argument("--path", required=True, help="Path to BTCUSDT.store")
    parser.add_argument("--skip-rows", type=int, default=1, help="Rows to skip at top (default: 1 to drop first line)")
    parser.add_argument("--out", default="analysis/raw_plots", help="Output directory for HTML plots")
    parser.add_argument("--head", type=int, default=5, help="Print head rows")
    args = parser.parse_args()

    path = Path(args.path)
    out_dir = Path(args.out)
    df = load_store(path, skip_rows=args.skip_rows)

    print(f"Loaded {len(df):,} rows from {path}")
    print(df.head(args.head))

    # Basic summary
    summary = df[["bid_px", "ask_px", "bid_qty", "ask_qty", "vol_base"]].describe(percentiles=[0.01, 0.5, 0.99])
    print("\nSummary stats:\n", summary)

    # Plots
    save_fig(px.line(df, x="ts", y=["bid_px", "ask_px"], title="Bid/Ask over time"), out_dir / "bid_ask_timeseries.html")
    save_fig(px.line(df, x="ts", y=["bid_qty", "ask_qty"], title="Bid/Ask qty over time"), out_dir / "qty_timeseries.html")
    save_fig(px.line(df, x="ts", y="vol_base", title="Volume (base) over time"), out_dir / "vol_timeseries.html")
    save_fig(px.histogram(df, x="bid_px", nbins=100, title="Bid price histogram"), out_dir / "bid_px_hist.html")
    save_fig(px.histogram(df, x="ask_px", nbins=100, title="Ask price histogram"), out_dir / "ask_px_hist.html")
    save_fig(px.histogram(df, x="vol_base", nbins=100, title="Volume histogram"), out_dir / "vol_hist.html")

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
