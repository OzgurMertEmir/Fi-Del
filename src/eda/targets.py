"""Target summaries and simple baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

from .config import EDAConfig


def _reg_stats(s: pd.Series) -> dict:
    q = s.quantile([0.01, 0.5, 0.99]).to_dict()
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "q01": float(q.get(0.01)),
        "q50": float(q.get(0.5)),
        "q99": float(q.get(0.99)),
    }


def target_report(df: pd.DataFrame, cfg: EDAConfig) -> dict:
    out: dict = {"ret": {}, "dir": {}, "vol": {}, "baselines": {}}

    for c in cfg.ret_targets:
        if c in df.columns:
            out["ret"][c] = _reg_stats(df[c])

    for c in cfg.vol_targets:
        if c in df.columns:
            out["vol"][c] = _reg_stats(df[c])

    for c in cfg.dir_targets:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False).to_dict()
            out["dir"][c] = {
                "value_counts": {str(k): int(v) for k, v in vc.items()},
                "unique": int(df[c].nunique()),
                "pct_majority": float(df[c].value_counts(normalize=True).iloc[0] * 100),
            }

    for c in cfg.ret_targets:
        if c in df.columns:
            y = df[c].to_numpy()
            out["baselines"][f"{c}_mse_pred0"] = float(mean_squared_error(y, np.zeros_like(y)))

    for c in cfg.vol_targets:
        if c in df.columns:
            y = df[c].to_numpy()
            pred = np.full_like(y, y.mean())
            out["baselines"][f"{c}_mae_predmean"] = float(mean_absolute_error(y, pred))

    for c in cfg.dir_targets:
        if c in df.columns:
            y = df[c]
            maj = y.value_counts().index[0]
            pred = pd.Series([maj] * len(y), index=y.index)
            out["baselines"][f"{c}_acc_majority"] = float(accuracy_score(y, pred))
            out["baselines"][f"{c}_bal_acc_majority"] = float(balanced_accuracy_score(y, pred))

    return out
