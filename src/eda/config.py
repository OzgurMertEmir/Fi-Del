"""Configuration for EDA and validity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class EDAConfig:
    ts_col: str = "ts_ms"
    ts_dt_col: str = "ts"
    required_cols: Sequence[str] = ("ts_ms",)

    ret_targets: Sequence[str] = ("y_ret_5", "y_ret_30", "y_ret_60")
    dir_targets: Sequence[str] = ("y_dir_5", "y_dir_30", "y_dir_60")
    vol_targets: Sequence[str] = ("y_vol_5", "y_vol_30", "y_vol_60")

    expected_dt_ms: int = 1000
    near_constant_std: float = 1e-8

    outlier_q_low: float = 0.001
    outlier_q_high: float = 0.999


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
