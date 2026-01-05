"""Label generation for future returns/direction/volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        future_log_mid = df.groupby("segment_id")["log_mid"].shift(-h)
        df[f"y_ret_{h}"] = future_log_mid - df["log_mid"]
    return df


def add_direction_labels(df: pd.DataFrame, horizons: list[int], deadzone: str = "half_spread", fixed_bps: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    if deadzone == "half_spread":
        thr = np.log((df["mid"] + 0.5 * df["spread"]) / df["mid"])
    else:
        thr = fixed_bps * 1e-4
    for h in horizons:
        ret = df[f"y_ret_{h}"]
        df[f"y_dir_{h}"] = np.where(ret > thr, 1, np.where(ret < -thr, -1, 0))
    return df


def add_vol_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["r2"] = df["r1"] ** 2
    # per-segment cumulative sum
    df["r2_cum"] = df.groupby("segment_id")["r2"].cumsum()
    for h in horizons:
        future = df.groupby("segment_id")["r2_cum"].shift(-h) - df["r2_cum"]
        df[f"y_vol_{h}"] = np.sqrt(future.clip(lower=0.0))
    df = df.drop(columns=["r2", "r2_cum"])
    return df
