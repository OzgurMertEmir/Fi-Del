"""EDA utilities for dataset inspection and reporting."""

from .config import EDAConfig, ensure_dir
from .io import load_table
from .quality import quality_report, compare_splits
from .targets import target_report
from .binance_plots import generate_binance_plots, plot_feature_importance
from .run_binance_eda import run_binance_eda
from .log_plots import generate_log_plots
from .run_log_eda import run_log_eda
from .unified_plots import generate_unified_plots
from .run_unified_eda import run_unified_eda

__all__ = [
    "EDAConfig",
    "ensure_dir",
    "load_table",
    "quality_report",
    "compare_splits",
    "target_report",
    # Binance EDA
    "generate_binance_plots",
    "plot_feature_importance",
    "run_binance_eda",
    # Log EDA
    "generate_log_plots",
    "run_log_eda",
    # Unified EDA
    "generate_unified_plots",
    "run_unified_eda",
]
