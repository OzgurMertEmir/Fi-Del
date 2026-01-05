"""Plotly visualizations for EDA."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from .config import EDAConfig, ensure_dir


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=path, auto_open=False, include_plotlyjs="cdn")


def plot_time_series(df: pd.DataFrame, cols: Iterable[str], out_dir: Path, prefix: str) -> list[Path]:
    paths = []
    for c in cols:
        if c not in df.columns:
            continue
        fig = px.line(df, x="ts", y=c, title=f"{prefix} {c} over time")
        out_path = out_dir / f"{prefix}_{c}_timeseries.html"
        save_fig(fig, out_path)
        paths.append(out_path)
    return paths


def plot_histograms(df: pd.DataFrame, cols: Iterable[str], out_dir: Path, prefix: str) -> list[Path]:
    paths = []
    for c in cols:
        if c not in df.columns:
            continue
        fig = px.histogram(df, x=c, nbins=100, title=f"{prefix} {c} histogram")
        out_path = out_dir / f"{prefix}_{c}_hist.html"
        save_fig(fig, out_path)
        paths.append(out_path)
    return paths


def plot_dir_bars(df: pd.DataFrame, cols: Iterable[str], out_dir: Path, prefix: str) -> list[Path]:
    paths = []
    for c in cols:
        if c not in df.columns:
            continue
        vc = df[c].value_counts(normalize=True).reset_index()
        vc.columns = [c, "pct"]
        fig = px.bar(vc, x=c, y="pct", title=f"{prefix} {c} class balance")
        out_path = out_dir / f"{prefix}_{c}_bars.html"
        save_fig(fig, out_path)
        paths.append(out_path)
    return paths


def plot_corr(df: pd.DataFrame, targets: Iterable[str], out_dir: Path, prefix: str) -> Path | None:
    target_cols = [t for t in targets if t in df.columns]
    if not target_cols:
        return None
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric[target_cols].corr(method="spearman")
    fig = px.imshow(corr, text_auto=True, title=f"{prefix} Spearman corr (targets)")
    out_path = out_dir / f"{prefix}_corr_targets.html"
    save_fig(fig, out_path)
    return out_path


def generate_plots(df: pd.DataFrame, split_name: str, cfg: EDAConfig, out_base: Path) -> list[Path]:
    out_dir = ensure_dir(out_base / "plots")
    paths: list[Path] = []

    mid_col = pick_first_existing(df, ["mid", "mid_raw", "mid_z", "last_fill_price"])
    vol_col = pick_first_existing(df, ["vol_base", "sum_fill_qty", "total_fill_qty"])
    spread_col = pick_first_existing(df, ["spread", "rel_spread"])

    ts_cols = [c for c in [mid_col, vol_col, spread_col] if c]
    paths += plot_time_series(df, ts_cols, out_dir, prefix=split_name)

    hist_targets = list(cfg.ret_targets) + list(cfg.vol_targets)
    paths += plot_histograms(df, hist_targets, out_dir, prefix=split_name)

    paths += plot_dir_bars(df, cfg.dir_targets, out_dir, prefix=split_name)

    corr_path = plot_corr(df, hist_targets + list(cfg.dir_targets), out_dir, prefix=split_name)
    if corr_path:
        paths.append(corr_path)

    return paths
