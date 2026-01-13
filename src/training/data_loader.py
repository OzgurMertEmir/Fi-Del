"""
Data loader for training pipeline.

Loads processed parquet files from S3 and prepares them for training.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from training.config import DataConfig

logger = logging.getLogger(__name__)


class S3DataLoader:
    """Load and prepare training data from S3."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._s3_client = None

        # Resolve bucket from environment if not set
        if not self.config.bucket:
            self.config.bucket = os.environ.get(
                "FIDEL_PROCESSED_BUCKET",
                "fidel-processed-data-794531974380",
            )

    @property
    def s3(self):
        """Lazy-load S3 client."""
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3")
        return self._s3_client

    def list_files(self, symbol: str) -> list[str]:
        """List all parquet files for a symbol within date range."""
        prefix = f"{self.config.prefix}{symbol}/"

        # Determine date range
        if self.config.start_date and self.config.end_date:
            start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        else:
            end = datetime.now()
            start = end - timedelta(days=self.config.days_back)

        logger.info(f"Listing files for {symbol} from {start.date()} to {end.date()}")

        # List objects
        files = []
        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".parquet"):
                    continue

                # Parse date from path: features/BTCUSDT/2026-01-13/01/file.parquet
                try:
                    parts = key.split("/")
                    if len(parts) >= 3:
                        date_str = parts[2]  # 2026-01-13
                        file_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if start <= file_date <= end:
                            files.append(key)
                except (ValueError, IndexError):
                    continue

        logger.info(f"Found {len(files)} parquet files for {symbol}")
        return sorted(files)

    def load_parquet(self, key: str) -> pd.DataFrame:
        """Load a single parquet file from S3."""
        response = self.s3.get_object(Bucket=self.config.bucket, Key=key)
        buffer = io.BytesIO(response["Body"].read())
        return pd.read_parquet(buffer)

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load all data for a symbol."""
        files = self.list_files(symbol)

        if not files:
            logger.warning(f"No files found for {symbol}")
            return pd.DataFrame()

        # Load and concatenate
        dfs = []
        for key in files:
            try:
                df = self.load_parquet(key)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows for {symbol}")
        return df

    def load_all(self) -> pd.DataFrame:
        """Load data for all configured symbols."""
        dfs = []

        for symbol in self.config.symbols:
            df = self.load_symbol(symbol)
            if not df.empty:
                df["symbol"] = symbol
                dfs.append(df)

        if not dfs:
            raise ValueError("No data loaded for any symbol")

        combined = pd.concat(dfs, ignore_index=True)

        # Sample if configured
        if self.config.sample_rate < 1.0:
            n_samples = int(len(combined) * self.config.sample_rate)
            combined = combined.sample(n=n_samples, random_state=42)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Sampled to {len(combined)} rows ({self.config.sample_rate:.0%})")

        # Cap rows if configured
        if self.config.max_rows and len(combined) > self.config.max_rows:
            combined = combined.tail(self.config.max_rows).reset_index(drop=True)
            logger.info(f"Capped to {self.config.max_rows} rows")

        logger.info(f"Total loaded: {len(combined)} rows, {len(combined.columns)} columns")
        return combined


class LocalDataLoader:
    """Load data from local parquet files (for testing without S3)."""

    def __init__(self, config: DataConfig, data_dir: str = "data/processed"):
        self.config = config
        self.data_dir = data_dir

    def load_all(self) -> pd.DataFrame:
        """Load all parquet files from local directory."""
        import glob
        from pathlib import Path

        dfs = []
        base_path = Path(self.data_dir)

        for symbol in self.config.symbols:
            pattern = str(base_path / "**" / f"*{symbol}*.parquet")
            files = glob.glob(pattern, recursive=True)

            for f in files:
                try:
                    df = pd.read_parquet(f)
                    df["symbol"] = symbol
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")

        if not dfs:
            raise ValueError(f"No data found in {self.data_dir}")

        return pd.concat(dfs, ignore_index=True)


def create_data_loader(config: DataConfig, local: bool = False) -> S3DataLoader | LocalDataLoader:
    """Factory function to create appropriate data loader."""
    if local:
        return LocalDataLoader(config)
    return S3DataLoader(config)
