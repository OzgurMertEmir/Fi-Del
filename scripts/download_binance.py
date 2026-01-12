#!/usr/bin/env python3
"""Script to download Binance public data.

Usage:
    python scripts/download_binance.py --config configs/binance_download.yaml

    # Or with command-line arguments:
    python scripts/download_binance.py -s BTCUSDT ETHUSDT -d trades --start 2024-01-01

See `python scripts/download_binance.py --help` for all options.
"""

from myproj.binance_data.cli import main

if __name__ == "__main__":
    main()
