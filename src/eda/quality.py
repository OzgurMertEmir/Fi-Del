"""Validity checks and split comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EDAConfig


def quality_report(df: pd.DataFrame, cfg: EDAConfig) -> dict:
    out: dict = {}

    missing = [c for c in cfg.required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.isna().any().any():
        raise ValueError("Found NaNs in dataframe.")

    numeric = df.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Found inf values in numeric columns.")

    ts = df[cfg.ts_col]
    out["rows"] = int(len(df))
    out["ts_min"] = int(ts.min())
    out["ts_max"] = int(ts.max())
    out["ts_duplicates"] = int(ts.duplicated().sum())
    out["ts_monotonic_increasing"] = bool(ts.is_monotonic_increasing)
    if out["ts_duplicates"] > 0 or not out["ts_monotonic_increasing"]:
        raise ValueError("Timestamp integrity failed (duplicates or non-monotonic).")

    diffs = ts.diff().dropna()
    out["dt_ms_min"] = int(diffs.min()) if len(diffs) else None
    out["dt_ms_median"] = int(diffs.median()) if len(diffs) else None
    out["dt_ms_max"] = int(diffs.max()) if len(diffs) else None
    out["pct_dt_expected"] = float((diffs == cfg.expected_dt_ms).mean() * 100) if len(diffs) else None

    num_cols = [c for c in numeric.columns if c != cfg.ts_col]
    const_cols = [c for c in num_cols if df[c].nunique(dropna=False) == 1]
    out["constant_cols"] = const_cols

    stds = df[num_cols].std()
    near_const = stds[stds < cfg.near_constant_std].index.tolist()
    out["near_constant_cols"] = near_const

    q_low = df[num_cols].quantile(cfg.outlier_q_low).to_dict()
    q_high = df[num_cols].quantile(cfg.outlier_q_high).to_dict()
    out["quantiles"] = {
        "q_low": q_low,
        "q_high": q_high,
        "q_low_p": cfg.outlier_q_low,
        "q_high_p": cfg.outlier_q_high,
    }
    return out


def compare_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: EDAConfig) -> dict:
    out: dict = {}
    cols_train = list(train.columns)
    cols_val = list(val.columns)
    cols_test = list(test.columns)
    out["same_columns_train_val"] = cols_train == cols_val
    out["same_columns_train_test"] = cols_train == cols_test
    if not out["same_columns_train_val"] or not out["same_columns_train_test"]:
        out["train_cols"] = cols_train
        out["val_cols"] = cols_val
        out["test_cols"] = cols_test
        raise ValueError("Split schemas differ.")

    def bounds(df):
        return int(df[cfg.ts_col].min()), int(df[cfg.ts_col].max())

    tr0, tr1 = bounds(train)
    va0, va1 = bounds(val)
    te0, te1 = bounds(test)

    out["train_bounds"] = (tr0, tr1)
    out["val_bounds"] = (va0, va1)
    out["test_bounds"] = (te0, te1)

    out["train_val_overlap"] = not (tr1 < va0)
    out["val_test_overlap"] = not (va1 < te0)
    if out["train_val_overlap"] or out["val_test_overlap"]:
        raise ValueError("Time overlap detected between splits.")

    out["gap_train_to_val_ms"] = int(va0 - tr1)
    out["gap_val_to_test_ms"] = int(te0 - va1)
    return out
