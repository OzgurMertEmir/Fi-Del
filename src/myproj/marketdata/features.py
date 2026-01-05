"""Feature engineering for L1 store data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_base_features(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    df = df.copy()
    df["mid"] = 0.5 * (df["bid_px"] + df["ask_px"])
    df["log_mid"] = np.log(df["mid"])
    df["spread"] = df["ask_px"] - df["bid_px"]
    df["rel_spread"] = df["spread"] / (df["mid"] + eps)
    df["imb"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + eps)
    df["microprice"] = (df["ask_px"] * df["bid_qty"] + df["bid_px"] * df["ask_qty"]) / (
        df["bid_qty"] + df["ask_qty"] + eps
    )
    df["micro_mid_dev"] = (df["microprice"] - df["mid"]) / (df["mid"] + eps)
    df["vol_log1p"] = np.log1p(df["vol_base"])
    df["notional"] = df["vol_base"] * df["mid"]
    df["notional_log1p"] = np.log1p(df["notional"])

    # segment-safe diff
    df["r1"] = df.groupby("segment_id")["log_mid"].diff()
    return df


def _segment_rolling(df: pd.DataFrame, col: str, window: int, func: str) -> pd.Series:
    return (
        df.groupby("segment_id")[col]
        .rolling(window, min_periods=1)
        .agg(func)
        .reset_index(level=0, drop=True)
    )


def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"r1_std_{w}"] = _segment_rolling(df, "r1", w, "std")
        df[f"vol_sum_{w}"] = _segment_rolling(df, "vol_base", w, "sum")
        df[f"not_sum_{w}"] = _segment_rolling(df, "notional", w, "sum")
        df[f"imb_mean_{w}"] = _segment_rolling(df, "imb", w, "mean")
        df[f"spr_mean_{w}"] = _segment_rolling(df, "rel_spread", w, "mean")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df
