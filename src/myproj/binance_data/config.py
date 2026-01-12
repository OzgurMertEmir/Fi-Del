"""Configuration for Binance public data downloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

# Base URL for Binance public data
BASE_URL = "https://data.binance.vision"

# API endpoints for fetching available symbols
API_ENDPOINTS = {
    "spot": "https://api.binance.com/api/v3/exchangeInfo",
    "um": "https://fapi.binance.com/fapi/v1/exchangeInfo",
    "cm": "https://dapi.binance.com/dapi/v1/exchangeInfo",
}

# Available data types
DATA_TYPES = ["trades", "aggTrades", "klines"]

# Available market types
MARKET_TYPES = ["spot", "um", "cm"]

# Kline intervals
KLINE_INTERVALS = [
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1mo",
]

# Intervals valid for daily downloads (subset)
DAILY_INTERVALS = [
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h", "1d",
]

# Data availability start date
DATA_START_DATE = date(2020, 1, 1)

# Timestamp format change date (microseconds from this date for SPOT)
MICROSECONDS_START_DATE = date(2025, 1, 1)


MarketType = Literal["spot", "um", "cm"]
DataType = Literal["trades", "aggTrades", "klines"]
Period = Literal["daily", "monthly"]


@dataclass
class BinanceDataConfig:
    """Configuration for a Binance data download job."""

    symbols: list[str]
    data_type: DataType
    market_type: MarketType = "spot"
    start_date: date = field(default_factory=lambda: DATA_START_DATE)
    end_date: date = field(default_factory=lambda: date.today())
    interval: str | None = None  # Required for klines
    period: Period = "daily"  # daily or monthly files
    output_dir: Path = field(default_factory=lambda: Path("data/binance"))
    verify_checksum: bool = True
    skip_existing: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.market_type not in MARKET_TYPES:
            raise ValueError(f"Invalid market_type: {self.market_type}. Must be one of {MARKET_TYPES}")
        if self.data_type not in DATA_TYPES:
            raise ValueError(f"Invalid data_type: {self.data_type}. Must be one of {DATA_TYPES}")
        if self.data_type == "klines" and not self.interval:
            raise ValueError("interval is required for klines data_type")
        if self.interval and self.interval not in KLINE_INTERVALS:
            raise ValueError(f"Invalid interval: {self.interval}. Must be one of {KLINE_INTERVALS}")
        if self.start_date > self.end_date:
            raise ValueError(f"start_date ({self.start_date}) must be before end_date ({self.end_date})")
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(cls, path: Path) -> "BinanceDataConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert date strings to date objects
        if isinstance(data.get("start_date"), str):
            data["start_date"] = datetime.strptime(data["start_date"], "%Y-%m-%d").date()
        if isinstance(data.get("end_date"), str):
            data["end_date"] = datetime.strptime(data["end_date"], "%Y-%m-%d").date()
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def get_download_url(self, symbol: str, file_date: date) -> str:
        """Build the download URL for a specific symbol and date."""
        market_path = self._get_market_path()
        period_path = self.period

        if self.data_type == "klines":
            data_path = f"klines/{symbol}/{self.interval}"
        else:
            data_path = f"{self.data_type}/{symbol}"

        filename = self._get_filename(symbol, file_date)
        return f"{BASE_URL}/data/{market_path}/{period_path}/{data_path}/{filename}"

    def get_checksum_url(self, symbol: str, file_date: date) -> str:
        """Get the checksum file URL."""
        return self.get_download_url(symbol, file_date) + ".CHECKSUM"

    def get_output_path(self, symbol: str, file_date: date) -> Path:
        """Get the local output path for a downloaded file."""
        market_path = self._get_market_path()
        if self.data_type == "klines":
            subdir = f"{market_path}/{self.data_type}/{symbol}/{self.interval}"
        else:
            subdir = f"{market_path}/{self.data_type}/{symbol}"

        filename = self._get_filename(symbol, file_date)
        return self.output_dir / subdir / filename

    def _get_market_path(self) -> str:
        """Get the market path component for URLs."""
        if self.market_type == "spot":
            return "spot"
        elif self.market_type == "um":
            return "futures/um"
        else:
            return "futures/cm"

    def _get_filename(self, symbol: str, file_date: date) -> str:
        """Generate the filename for a given symbol and date."""
        if self.period == "monthly":
            date_str = file_date.strftime("%Y-%m")
        else:
            date_str = file_date.strftime("%Y-%m-%d")

        if self.data_type == "klines":
            return f"{symbol}-{self.interval}-{date_str}.zip"
        else:
            return f"{symbol}-{self.data_type}-{date_str}.zip"
