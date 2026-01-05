"""Top-level package for the crypto order book project.

This package exposes configuration helpers and reusable functionality for
working with raw and processed data.  See `myproj.config` for centralised
paths and environment variables, and `myproj.data` for convenience I/O
functions.
"""

from .config import DATA_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR, get_env

__all__ = ["DATA_ROOT", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "get_env"]