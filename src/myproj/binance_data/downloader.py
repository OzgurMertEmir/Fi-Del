"""Download manager for Binance public data with retry logic and checksum verification."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Generator
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from tqdm import tqdm

from .config import BinanceDataConfig, API_ENDPOINTS

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download attempt."""

    symbol: str
    file_date: date
    success: bool
    path: Path | None = None
    error: str | None = None
    skipped: bool = False


class BinanceDownloader:
    """Downloads historical data from Binance public data repository."""

    def __init__(self, config: BinanceDataConfig) -> None:
        self.config = config
        self._session_headers = {
            "User-Agent": "Mozilla/5.0 (compatible; FiDel/1.0; +https://github.com/fidel)"
        }

    def download_all(self) -> list[DownloadResult]:
        """Download all files according to configuration."""
        results: list[DownloadResult] = []
        dates = list(self._generate_dates())

        total_files = len(self.config.symbols) * len(dates)
        logger.info(
            f"Downloading {total_files} files for {len(self.config.symbols)} symbols "
            f"from {self.config.start_date} to {self.config.end_date}"
        )

        with tqdm(total=total_files, desc="Downloading", unit="file") as pbar:
            for symbol in self.config.symbols:
                for file_date in dates:
                    result = self._download_file(symbol, file_date)
                    results.append(result)
                    pbar.update(1)

                    if not result.success and not result.skipped:
                        pbar.set_postfix_str(f"Failed: {symbol} {file_date}")

        # Summary
        successful = sum(1 for r in results if r.success)
        skipped = sum(1 for r in results if r.skipped)
        failed = sum(1 for r in results if not r.success and not r.skipped)
        logger.info(f"Download complete: {successful} succeeded, {skipped} skipped, {failed} failed")

        return results

    def _generate_dates(self) -> Generator[date, None, None]:
        """Generate dates to download based on period configuration."""
        current = self.config.start_date

        if self.config.period == "monthly":
            # Align to first of month
            current = current.replace(day=1)
            while current <= self.config.end_date:
                yield current
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        else:
            # Daily
            while current <= self.config.end_date:
                yield current
                current += timedelta(days=1)

    def _download_file(self, symbol: str, file_date: date) -> DownloadResult:
        """Download a single file with retries."""
        output_path = self.config.get_output_path(symbol, file_date)

        # Skip if exists
        if self.config.skip_existing and output_path.exists():
            return DownloadResult(
                symbol=symbol,
                file_date=file_date,
                success=True,
                path=output_path,
                skipped=True,
            )

        url = self.config.get_download_url(symbol, file_date)
        checksum_url = self.config.get_checksum_url(symbol, file_date)

        last_error: str | None = None

        for attempt in range(self.config.max_retries):
            try:
                # Download main file
                content = self._fetch_url(url)

                # Verify checksum if enabled
                if self.config.verify_checksum:
                    try:
                        checksum_content = self._fetch_url(checksum_url)
                        if not self._verify_checksum(content, checksum_content):
                            raise ValueError("Checksum verification failed")
                    except HTTPError as e:
                        if e.code == 404:
                            logger.debug(f"No checksum file for {symbol} {file_date}")
                        else:
                            raise

                # Save file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(content)

                return DownloadResult(
                    symbol=symbol,
                    file_date=file_date,
                    success=True,
                    path=output_path,
                )

            except HTTPError as e:
                if e.code == 404:
                    # File doesn't exist (e.g., future date or delisted symbol)
                    return DownloadResult(
                        symbol=symbol,
                        file_date=file_date,
                        success=False,
                        error="File not found (404)",
                        skipped=True,
                    )
                last_error = f"HTTP {e.code}: {e.reason}"

            except URLError as e:
                last_error = f"URL error: {e.reason}"

            except Exception as e:
                last_error = str(e)

            # Retry delay with exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                logger.debug(f"Retry {attempt + 1}/{self.config.max_retries} for {symbol} after {delay}s")
                time.sleep(delay)

        return DownloadResult(
            symbol=symbol,
            file_date=file_date,
            success=False,
            error=last_error,
        )

    def _fetch_url(self, url: str) -> bytes:
        """Fetch content from URL."""
        request = Request(url, headers=self._session_headers)
        with urlopen(request, timeout=30) as response:
            return response.read()

    def _verify_checksum(self, content: bytes, checksum_content: bytes) -> bool:
        """Verify SHA256 checksum of content."""
        # Parse checksum file (format: "hash  filename")
        checksum_line = checksum_content.decode("utf-8").strip()
        expected_hash = checksum_line.split()[0].lower()

        # Compute actual hash
        actual_hash = hashlib.sha256(content).hexdigest().lower()

        return actual_hash == expected_hash

    @staticmethod
    def fetch_symbols(market_type: str) -> list[str]:
        """Fetch available trading symbols from Binance API."""
        endpoint = API_ENDPOINTS.get(market_type)
        if not endpoint:
            raise ValueError(f"Unknown market type: {market_type}")

        request = Request(
            endpoint,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FiDel/1.0)"}
        )

        try:
            with urlopen(request, timeout=30) as response:
                import json
                data = json.loads(response.read().decode("utf-8"))

                if market_type == "spot":
                    return [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]
                else:
                    return [s["symbol"] for s in data["symbols"]]

        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            raise


def download_binance_data(
    symbols: list[str],
    data_type: str,
    start_date: date,
    end_date: date,
    market_type: str = "spot",
    interval: str | None = None,
    period: str = "daily",
    output_dir: Path | str = "data/binance",
    verify_checksum: bool = True,
    skip_existing: bool = True,
) -> list[DownloadResult]:
    """Convenience function to download Binance data.

    Args:
        symbols: List of trading pair symbols (e.g., ["BTCUSDT", "ETHUSDT"])
        data_type: Type of data ("trades", "aggTrades", "klines")
        start_date: Start date for download
        end_date: End date for download
        market_type: Market type ("spot", "um", "cm")
        interval: Kline interval (required if data_type is "klines")
        period: File period ("daily" or "monthly")
        output_dir: Output directory for downloaded files
        verify_checksum: Whether to verify file checksums
        skip_existing: Whether to skip already downloaded files

    Returns:
        List of DownloadResult objects
    """
    config = BinanceDataConfig(
        symbols=symbols,
        data_type=data_type,
        market_type=market_type,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        period=period,
        output_dir=Path(output_dir),
        verify_checksum=verify_checksum,
        skip_existing=skip_existing,
    )

    downloader = BinanceDownloader(config)
    return downloader.download_all()
