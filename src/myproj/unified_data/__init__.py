"""Unified data pipeline for multi-source timeseries data.

This module provides tools to combine:
- Order book snapshots (.store files)
- Trading signals and fills (parsed log data)
- Binance public historical data

Into aligned, per-symbol ML-ready datasets.

Example usage:

    from myproj.unified_data import (
        MultiSourceLoader,
        UnifiedDataConfig,
        MLDatasetConfig,
        build_ml_dataset,
    )

    # Initialize loader
    loader = MultiSourceLoader(
        raw_dir="data/raw",
        processed_dir="data/processed",
    )

    # Load data sources
    loader.load_store_data()
    loader.load_signals()
    loader.load_fills()

    # Build ML dataset for a symbol
    config = MLDatasetConfig(
        interval="1s",
        label_type="fill_occurred",
        label_horizon=30,
    )
    dataset = build_ml_dataset(loader, "BTCUSDT", config)

    # Use for training
    model.fit(dataset.X_train, dataset.y_train)
"""

from .store_parser import (
    parse_store_file,
    load_all_store_files,
    resample_store_data,
    get_store_summary,
)

from .alignment import (
    normalize_timestamp_precision,
    create_time_grid,
    align_to_grid,
    find_overlapping_range,
    resample_signals,
    resample_fills,
    compute_time_features,
)

from .multi_source import (
    DataSourceConfig,
    UnifiedDataConfig,
    MultiSourceLoader,
    build_unified_dataset,
    build_multi_symbol_dataset,
    add_lagged_features,
    add_rolling_features,
    create_labels,
)

from .dataset_builder import (
    UnifiedMLDataset,
    MLDatasetConfig,
    get_feature_columns,
    prepare_ml_features,
    build_ml_dataset,
    build_multi_symbol_ml_datasets,
    get_dataset_summary,
    save_dataset,
)

__all__ = [
    # Store parser
    "parse_store_file",
    "load_all_store_files",
    "resample_store_data",
    "get_store_summary",
    # Alignment
    "normalize_timestamp_precision",
    "create_time_grid",
    "align_to_grid",
    "find_overlapping_range",
    "resample_signals",
    "resample_fills",
    "compute_time_features",
    # Multi-source
    "DataSourceConfig",
    "UnifiedDataConfig",
    "MultiSourceLoader",
    "build_unified_dataset",
    "build_multi_symbol_dataset",
    "add_lagged_features",
    "add_rolling_features",
    "create_labels",
    # Dataset builder
    "UnifiedMLDataset",
    "MLDatasetConfig",
    "get_feature_columns",
    "prepare_ml_features",
    "build_ml_dataset",
    "build_multi_symbol_ml_datasets",
    "get_dataset_summary",
    "save_dataset",
]
