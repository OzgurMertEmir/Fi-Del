"""Validation and cleaning of raw store data."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .schemas import PRICE_COLS, QTY_COLS


def validate_raw(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, Dict]:
    """Validate raw data frame according to cleaning config."""
    report: Dict[str, int | float] = {}
    total_rows = len(df)
    report["total_rows"] = total_rows

    # Drop nonpositive if requested
    if cfg.drop_nonpositive:
        mask = (df["bid_px"] > 0) & (df["ask_px"] > 0) & (df["bid_qty"] >= 0) & (df["ask_qty"] >= 0) & (df["vol_base"] >= 0)
        report["dropped_nonpositive"] = int((~mask).sum())
        df = df.loc[mask]
    else:
        report["dropped_nonpositive"] = 0

    if cfg.enforce_bid_leq_ask:
        mask = df["bid_px"] <= df["ask_px"]
        report["dropped_bid_gt_ask"] = int((~mask).sum())
        df = df.loc[mask]
    else:
        report["dropped_bid_gt_ask"] = 0

    # Remove NaNs in critical cols
    df = df.dropna(subset=PRICE_COLS + QTY_COLS + ["vol_base", "ts_ms"])
    report["dropped_nan"] = total_rows - len(df)

    # dt anomalies
    df = df.sort_values("ts_ms").reset_index(drop=True)
    dt = df["ts_ms"].diff()
    anomalies = (dt != cfg.expected_dt_ms) & (~dt.isna())
    report["dt_anomalies"] = int(anomalies.sum())

    return df.reset_index(drop=True), report
