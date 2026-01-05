"""EDA utilities for dataset inspection and reporting."""

from .config import EDAConfig, ensure_dir
from .io import load_table
from .quality import quality_report, compare_splits
from .targets import target_report

__all__ = [
    "EDAConfig",
    "ensure_dir",
    "load_table",
    "quality_report",
    "compare_splits",
    "target_report",
]
