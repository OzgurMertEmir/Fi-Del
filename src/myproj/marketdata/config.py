"""Configuration utilities for the market data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DatasetConfig:
    symbol: str
    raw_path: Path
    out_dir: Path
    file_format: str = "parquet"


@dataclass
class IngestConfig:
    has_header: bool
    delimiter: str
    columns: List[str]
    dtypes: Dict[str, Any]
    skip_rows: int = 0


@dataclass
class CleaningConfig:
    enforce_monotonic_ts: bool = True
    expected_dt_ms: int = 1000
    handle_bad_dt: str = "segment"
    min_segment_len: int = 600
    enforce_bid_leq_ask: bool = True
    drop_nonpositive: bool = True


@dataclass
class FeaturesConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [5, 15, 60, 300])
    add_time_features: bool = True
    eps: float = 1e-12


@dataclass
class LabelsConfig:
    horizons_s: List[int] = field(default_factory=lambda: [5, 30, 60])
    direction_deadzone: str = "half_spread"
    fixed_bps: float = 1.0


@dataclass
class SplitConfig:
    method: str = "time_ratio"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_s: int = 60


@dataclass
class NormalizeConfig:
    enabled: bool = True
    method: str = "standard"
    fit_on: str = "train"
    save_params: bool = True


@dataclass
class OutputConfig:
    save_feature_list: bool = True
    save_label_list: bool = True
    save_summary_stats: bool = True
    save_config_snapshot: bool = True


@dataclass
class Config:
    dataset: DatasetConfig
    ingest: IngestConfig
    cleaning: CleaningConfig
    features: FeaturesConfig
    labels: LabelsConfig
    split: SplitConfig
    normalize: NormalizeConfig
    output: OutputConfig
    raw: Dict[str, Any]


def _dict_to_dataclass(cfg: Dict[str, Any]) -> Config:
    dataset = DatasetConfig(
        symbol=cfg["dataset"]["symbol"],
        raw_path=Path(cfg["dataset"]["raw_path"]),
        out_dir=Path(cfg["dataset"]["out_dir"]),
        file_format=cfg["dataset"].get("file_format", "parquet"),
    )
    ingest = IngestConfig(
        has_header=cfg["ingest"]["has_header"],
        delimiter=cfg["ingest"]["delimiter"],
        columns=list(cfg["ingest"]["columns"]),
        dtypes=dict(cfg["ingest"]["dtypes"]),
        skip_rows=int(cfg["ingest"].get("skip_rows", 0)),
    )
    cleaning = CleaningConfig(**cfg["cleaning"])
    features = FeaturesConfig(**cfg["features"])
    labels = LabelsConfig(**cfg["labels"])
    split = SplitConfig(**cfg["split"])
    normalize = NormalizeConfig(**cfg["normalize"])
    output = OutputConfig(**cfg["output"])
    return Config(
        dataset=dataset,
        ingest=ingest,
        cleaning=cleaning,
        features=features,
        labels=labels,
        split=split,
        normalize=normalize,
        output=output,
        raw=cfg,
    )


def load_config(path: str | Path) -> Config:
    """Load YAML config into dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _dict_to_dataclass(cfg)
