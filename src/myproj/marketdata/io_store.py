"""I/O utilities for .store CSV-style files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def read_store(
    path: str | Path,
    columns: Iterable[str],
    dtypes: Dict[str, str],
    has_header: bool = False,
    delimiter: str = ",",
    skip_rows: int = 0,
) -> pd.DataFrame:
    """Read a .store file with known schema."""
    path = Path(path)
    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=list(columns),
        dtype=dtypes,
        delimiter=delimiter,
        engine="c",
        skiprows=skip_rows,
    )
    df = df.dropna()
    df = df.sort_values("ts_ms").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
