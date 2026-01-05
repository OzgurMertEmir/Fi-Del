"""Orchestrate dataset building from raw store to ML-ready splits."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import Config
from .features import add_rolling_features, add_time_features, compute_base_features
from .io_store import read_store, write_json, write_parquet
from .labels import add_direction_labels, add_return_labels, add_vol_labels
from .normalize import fit_transform, save_scaler_params
from .segment import add_segment_id, drop_short_segments
from .split import split_by_segment
from .validate import validate_raw


def _drop_unusable(df: pd.DataFrame, label_cols: List[str], feature_cols: List[str]) -> pd.DataFrame:
    keep_cols = ["ts_ms", "ts", "segment_id"] + feature_cols + label_cols
    df = df[keep_cols]
    df = df.dropna(subset=feature_cols + label_cols)
    return df.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame, label_cols: List[str]) -> List[str]:
    exclude = set(label_cols + ["ts_ms", "ts", "segment_id"])
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]


def get_label_columns(label_cfg) -> List[str]:
    return (
        [f"y_ret_{h}" for h in label_cfg.horizons_s]
        + [f"y_dir_{h}" for h in label_cfg.horizons_s]
        + [f"y_vol_{h}" for h in label_cfg.horizons_s]
    )


def build_dataset(cfg: Config) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    raw_df = read_store(cfg.dataset.raw_path, cfg.ingest.columns, cfg.ingest.dtypes, cfg.ingest.has_header, cfg.ingest.delimiter)
    raw_df, val_report = validate_raw(raw_df, cfg.cleaning)

    df = add_segment_id(raw_df, cfg.cleaning.expected_dt_ms, cfg.cleaning.handle_bad_dt)
    df = drop_short_segments(df, cfg.cleaning.min_segment_len)

    df = compute_base_features(df, eps=cfg.features.eps)
    df = add_rolling_features(df, cfg.features.rolling_windows)
    if cfg.features.add_time_features:
        df = add_time_features(df)

    df = add_return_labels(df, cfg.labels.horizons_s)
    df = add_direction_labels(df, cfg.labels.horizons_s, cfg.labels.direction_deadzone, cfg.labels.fixed_bps)
    df = add_vol_labels(df, cfg.labels.horizons_s)

    label_cols = get_label_columns(cfg.labels)
    feature_cols = get_feature_columns(df, label_cols)
    df = _drop_unusable(df, label_cols, feature_cols)

    train, val, test = split_by_segment(df, cfg.split)

    if cfg.normalize.enabled and cfg.normalize.method != "none":
        train, val, test, scaler_params = fit_transform(train, val, test, feature_cols, cfg.normalize.method)
    else:
        scaler_params = None

    out_dir = Path(cfg.dataset.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train, out_dir / "train.parquet")
    write_parquet(val, out_dir / "val.parquet")
    write_parquet(test, out_dir / "test.parquet")

    write_json(val_report, out_dir / "validation_report.json")
    write_json(feature_cols, out_dir / "feature_columns.json")
    write_json(label_cols, out_dir / "label_columns.json")
    write_json(cfg.raw if cfg.output.save_config_snapshot else {}, out_dir / "config_snapshot.json")

    summary = {
        "rows_train": len(train),
        "rows_val": len(val),
        "rows_test": len(test),
        "segments": int(df["segment_id"].nunique()) if "segment_id" in df.columns else 0,
    }
    write_json(summary, out_dir / "summary_stats.json")

    if scaler_params and cfg.normalize.save_params:
        save_scaler_params(scaler_params, out_dir / "scaler.json")

    paths.update(
        {
            "train": out_dir / "train.parquet",
            "val": out_dir / "val.parquet",
            "test": out_dir / "test.parquet",
        }
    )
    return paths
