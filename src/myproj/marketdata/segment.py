"""Segmentation to handle gaps safely."""

from __future__ import annotations

import pandas as pd


def add_segment_id(df: pd.DataFrame, expected_dt_ms: int, handle_bad_dt: str = "segment") -> pd.DataFrame:
    """Assign segment_id based on timestamp gaps."""
    df = df.sort_values("ts_ms").reset_index(drop=True)
    dt = df["ts_ms"].diff()
    breaks = (dt != expected_dt_ms) | dt.isna()

    if handle_bad_dt == "drop":
        df = df.loc[~breaks | dt.isna()].reset_index(drop=True)
        dt = df["ts_ms"].diff()
        breaks = (dt != expected_dt_ms) | dt.isna()

    seg_id = breaks.cumsum().astype(int)
    df = df.assign(segment_id=seg_id)
    return df


def drop_short_segments(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """Drop segments shorter than min_len rows."""
    counts = df.groupby("segment_id").size()
    keep_ids = counts[counts >= min_len].index
    return df[df["segment_id"].isin(keep_ids)].reset_index(drop=True)
