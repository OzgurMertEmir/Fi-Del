"""IO helpers for EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")

    if "ts_ms" not in df.columns:
        raise ValueError("Expected column ts_ms not found.")

    for c in df.columns:
        if c != "ts_ms":
            df[c] = pd.to_numeric(df[c], errors="raise")

    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df
