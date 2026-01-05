"""Feature normalization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def _get_scaler(method: str):
    if method == "standard":
        return StandardScaler()
    if method == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown normalization method: {method}")


def fit_transform(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], method: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    scaler = _get_scaler(method)
    scaler.fit(train[feature_cols])
    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        arr = scaler.transform(df[feature_cols])
        df_out = df.copy()
        df_out[feature_cols] = arr
        return df_out
    train_n = _apply(train)
    val_n = _apply(val)
    test_n = _apply(test)
    params = {
        "method": method,
        "feature_order": feature_cols,
        "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        "center_": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
        "median_": scaler.center_.tolist() if hasattr(scaler, "center_") else None,
    }
    return train_n, val_n, test_n, params


def save_scaler_params(params: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
