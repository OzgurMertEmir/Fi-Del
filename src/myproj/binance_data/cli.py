"""Command-line interface for Binance data pipeline."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path

from .config import BinanceDataConfig, KLINE_INTERVALS, MARKET_TYPES, DATA_TYPES
from .downloader import BinanceDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Download historical data from Binance public data repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download daily trades for BTCUSDT and ETHUSDT
  python -m myproj.binance_data.cli -s BTCUSDT ETHUSDT -d trades --start 2024-01-01 --end 2024-01-31

  # Download 1-minute klines for all major pairs
  python -m myproj.binance_data.cli -s BTCUSDT ETHUSDT BNBUSDT -d klines -i 1m --start 2024-01-01

  # Download monthly aggregated trades for futures
  python -m myproj.binance_data.cli -s BTCUSDT -d aggTrades -t um --period monthly

  # Use a config file
  python -m myproj.binance_data.cli --config configs/binance_download.yaml
""",
    )

    parser.add_argument(
        "-s", "--symbols",
        nargs="+",
        help="Trading pair symbols to download (e.g., BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "-d", "--data-type",
        choices=DATA_TYPES,
        default="trades",
        help="Type of data to download",
    )
    parser.add_argument(
        "-t", "--market-type",
        choices=MARKET_TYPES,
        default="spot",
        help="Market type (spot, um=USD-M futures, cm=COIN-M futures)",
    )
    parser.add_argument(
        "-i", "--interval",
        choices=KLINE_INTERVALS,
        help="Kline interval (required for klines data type)",
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--period",
        choices=["daily", "monthly"],
        default="daily",
        help="File period (daily or monthly files)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("data/binance"),
        help="Output directory for downloaded files",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip checksum verification",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Load configuration from YAML file",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="List available symbols and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List symbols if requested
    if args.list_symbols:
        symbols = BinanceDownloader.fetch_symbols(args.market_type)
        print(f"\nAvailable {args.market_type.upper()} symbols ({len(symbols)}):")
        print("-" * 50)
        for i, sym in enumerate(sorted(symbols)):
            print(f"  {sym}", end="")
            if (i + 1) % 5 == 0:
                print()
        print()
        return

    # Load from config file if provided
    if args.config:
        config = BinanceDataConfig.from_yaml(args.config)
    else:
        if not args.symbols:
            parser.error("--symbols is required unless using --config or --list-symbols")

        config = BinanceDataConfig(
            symbols=args.symbols,
            data_type=args.data_type,
            market_type=args.market_type,
            interval=args.interval,
            start_date=args.start if isinstance(args.start, date) else parse_date(args.start),
            end_date=args.end if isinstance(args.end, date) else parse_date(args.end),
            period=args.period,
            output_dir=args.output_dir,
            verify_checksum=not args.no_checksum,
            skip_existing=not args.force,
        )

    logger.info(f"Starting download: {config.data_type} for {len(config.symbols)} symbols")
    logger.info(f"Date range: {config.start_date} to {config.end_date}")
    logger.info(f"Output directory: {config.output_dir}")

    downloader = BinanceDownloader(config)
    results = downloader.download_all()

    # Print summary
    successful = [r for r in results if r.success and not r.skipped]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if not r.success and not r.skipped]

    print(f"\n{'='*50}")
    print("Download Summary")
    print(f"{'='*50}")
    print(f"  Successful: {len(successful)}")
    print(f"  Skipped:    {len(skipped)}")
    print(f"  Failed:     {len(failed)}")

    if failed:
        print("\nFailed downloads:")
        for r in failed[:10]:  # Show first 10
            print(f"  - {r.symbol} {r.file_date}: {r.error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
