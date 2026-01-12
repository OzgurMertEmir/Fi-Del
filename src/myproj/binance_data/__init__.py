"""Binance public data pipeline for historical market data."""

from .config import BinanceDataConfig
from .downloader import BinanceDownloader, download_binance_data
from .parsers import parse_trades, parse_aggTrades, parse_klines, load_multiple_files
from .aggregators import (
    aggregate_trades_to_ohlcv,
    compute_trade_flow_features,
    add_rolling_flow_features,
    add_advanced_flow_features,
    add_return_labels,
    add_direction_labels,
    add_volatility_labels,
    build_ml_dataset,
    merge_multi_symbol,
)
from .pipeline import BinancePipelineConfig, run_binance_pipeline, quick_load
from .ml_integration import (
    BinanceMLDataset,
    build_binance_ml_dataset,
    run_binance_baseline,
    load_processed_binance_dataset,
    get_feature_importance,
)

__all__ = [
    # Config
    "BinanceDataConfig",
    "BinancePipelineConfig",
    # Downloader
    "BinanceDownloader",
    "download_binance_data",
    # Parsers
    "parse_trades",
    "parse_aggTrades",
    "parse_klines",
    "load_multiple_files",
    # Aggregators
    "aggregate_trades_to_ohlcv",
    "compute_trade_flow_features",
    "add_rolling_flow_features",
    "add_advanced_flow_features",
    "add_return_labels",
    "add_direction_labels",
    "add_volatility_labels",
    "build_ml_dataset",
    "merge_multi_symbol",
    # Pipeline
    "run_binance_pipeline",
    "quick_load",
    # ML Integration
    "BinanceMLDataset",
    "build_binance_ml_dataset",
    "run_binance_baseline",
    "load_processed_binance_dataset",
    "get_feature_importance",
]
