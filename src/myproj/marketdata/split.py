"""Time-based splitting with embargo."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def _split_segment(df: pd.DataFrame, cfg) -> Dict[str, pd.DataFrame]:
    n = len(df)
    t_end = int(n * cfg.train_ratio)
    v_end = t_end + int(n * cfg.val_ratio)

    train = df.iloc[:t_end]
    val = df.iloc[t_end:v_end]
    test = df.iloc[v_end:]

    embargo = cfg.embargo_s
    if embargo > 0:
        if not train.empty:
            last_train_ts = train["ts_ms"].iloc[-1]
            val = val[val["ts_ms"] > last_train_ts + embargo * 1000]
        if not val.empty:
            last_val_ts = val["ts_ms"].iloc[-1]
            test = test[test["ts_ms"] > last_val_ts + embargo * 1000]

    return {"train": train, "val": val, "test": test}


def split_by_segment(df: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = {"train": [], "val": [], "test": []}
    for _, seg_df in df.groupby("segment_id"):
        seg_df = seg_df.sort_values("ts_ms")
        seg_splits = _split_segment(seg_df, cfg)
        for k, v in seg_splits.items():
            if not v.empty:
                parts[k].append(v)
    train = pd.concat(parts["train"]).reset_index(drop=True) if parts["train"] else pd.DataFrame()
    val = pd.concat(parts["val"]).reset_index(drop=True) if parts["val"] else pd.DataFrame()
    test = pd.concat(parts["test"]).reset_index(drop=True) if parts["test"] else pd.DataFrame()
    return train, val, test
