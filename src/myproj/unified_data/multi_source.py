"""Multi-source data joining and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import polars as pl

from .store_parser import load_all_store_files, resample_store_data
from .alignment import (
    create_time_grid,
    find_overlapping_range,
    resample_signals,
    resample_fills,
    compute_time_features,
    normalize_timestamp_precision,
)


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    path: Optional[Path] = None
    enabled: bool = True
    priority: int = 0  # Higher priority sources fill gaps first


@dataclass
class UnifiedDataConfig:
    """Configuration for unified dataset creation."""
    # Time settings
    interval: str = "1s"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    use_overlap_only: bool = True

    # Symbol filtering
    symbols: Optional[List[str]] = None

    # Feature settings
    include_time_features: bool = True
    include_lagged_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 30, 60])

    # Label settings
    label_horizon: int = 30  # Periods ahead for prediction target
    label_type: str = "fill_occurred"  # or "price_direction"


class MultiSourceLoader:
    """Load and combine data from multiple sources."""

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        binance_dir: Optional[Path] = None,
        config: Optional[UnifiedDataConfig] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.binance_dir = Path(binance_dir) if binance_dir else Path("data/binance")
        self.config = config or UnifiedDataConfig()

        # Data storage
        self._store_data: Optional[pl.DataFrame] = None
        self._signals_data: Optional[pl.DataFrame] = None
        self._fills_data: Optional[pl.DataFrame] = None
        self._binance_data: Dict[str, pl.DataFrame] = {}

    def load_store_data(self, symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """Load order book data from .store files."""
        symbols = symbols or self.config.symbols
        self._store_data = load_all_store_files(self.raw_dir, symbols=symbols)
        return self._store_data

    def load_signals(self, symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """Load parsed signal data."""
        signals_dir = self.processed_dir / "signals"
        if not signals_dir.exists():
            raise FileNotFoundError(f"Signals directory not found: {signals_dir}")

        dfs = []
        for csv_file in signals_dir.glob("*.csv"):
            df = pl.read_csv(csv_file)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(f"No signal CSV files found in {signals_dir}")

        self._signals_data = pl.concat(dfs)

        # Filter symbols
        symbols = symbols or self.config.symbols
        if symbols:
            self._signals_data = self._signals_data.filter(
                pl.col("symbol").is_in(symbols)
            )

        return self._signals_data

    def load_fills(self, symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """Load parsed fill data."""
        fills_dir = self.processed_dir / "trade_fills"
        if not fills_dir.exists():
            raise FileNotFoundError(f"Fills directory not found: {fills_dir}")

        dfs = []
        for csv_file in fills_dir.glob("*.csv"):
            df = pl.read_csv(csv_file)
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(f"No fill CSV files found in {fills_dir}")

        self._fills_data = pl.concat(dfs)

        # Filter symbols
        symbols = symbols or self.config.symbols
        if symbols:
            self._fills_data = self._fills_data.filter(
                pl.col("symbol").is_in(symbols)
            )

        return self._fills_data

    def load_binance_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1s",
        include_advanced_features: bool = True,
    ) -> pl.DataFrame:
        """Load Binance public data for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_date: Start date YYYY-MM-DD (default: infer from other sources)
            end_date: End date YYYY-MM-DD (default: infer from other sources)
            interval: Aggregation interval (default: "1s")
            include_advanced_features: Whether to add advanced order flow features

        Returns:
            DataFrame with Binance order flow features
        """
        try:
            from myproj.binance_data import (
                BinancePipelineConfig,
                compute_trade_flow_features,
                add_rolling_flow_features,
                add_advanced_flow_features,
            )
            from myproj.binance_data.pipeline import _load_symbol_data
        except ImportError as e:
            print(f"Warning: binance_data module not available: {e}")
            return None

        from datetime import datetime as dt

        # Try to infer dates from other data sources if not provided
        if start_date is None or end_date is None:
            time_ranges = self.get_time_range()
            if time_ranges:
                all_starts = [r["start"] for r in time_ranges.values()]
                all_ends = [r["end"] for r in time_ranges.values()]
                if start_date is None and all_starts:
                    start_dt = min(all_starts)
                    start_date = start_dt.strftime("%Y-%m-%d") if hasattr(start_dt, "strftime") else str(start_dt)[:10]
                if end_date is None and all_ends:
                    end_dt = max(all_ends)
                    end_date = end_dt.strftime("%Y-%m-%d") if hasattr(end_dt, "strftime") else str(end_dt)[:10]

        if start_date is None or end_date is None:
            print("Warning: Cannot infer date range for Binance data")
            return None

        print(f"  Loading Binance data: {symbol} from {start_date} to {end_date}")

        # Create config
        config = BinancePipelineConfig(
            symbols=[symbol],
            start_date=dt.strptime(start_date, "%Y-%m-%d").date(),
            end_date=dt.strptime(end_date, "%Y-%m-%d").date(),
            interval=interval,
            raw_data_dir=self.binance_dir,
            download_if_missing=True,
        )

        # Load raw data
        try:
            df = _load_symbol_data(config, symbol)
            if df is None or len(df) == 0:
                print(f"  Warning: No Binance data found for {symbol}")
                return None
        except Exception as e:
            print(f"  Warning: Failed to load Binance data for {symbol}: {e}")
            return None

        print(f"    Loaded {len(df):,} raw trades")

        # Compute features
        agg = compute_trade_flow_features(df, interval=interval)
        agg = add_rolling_flow_features(agg, windows=[5, 15, 60])

        if include_advanced_features:
            agg = add_advanced_flow_features(agg, windows=[5, 15, 60])

        # Rename ts to timestamp for consistency
        if "ts" in agg.columns:
            agg = agg.rename({"ts": "timestamp"})

        # Add symbol column if not present
        if "symbol" not in agg.columns:
            agg = agg.with_columns(pl.lit(symbol).alias("symbol"))

        # Prefix columns to avoid conflicts
        cols_to_prefix = [c for c in agg.columns if c not in ["timestamp", "symbol"]]
        agg = agg.rename({c: f"binance_{c}" for c in cols_to_prefix})

        self._binance_data[symbol] = agg
        print(f"    Computed {len(agg.columns) - 2} Binance features")

        return agg

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols available across all data sources."""
        symbols = set()

        # From store files
        try:
            store_files = list(self.raw_dir.glob("*.store"))
            symbols.update(f.stem for f in store_files)
        except Exception:
            pass

        # From signals
        if self._signals_data is not None:
            symbols.update(self._signals_data["symbol"].unique().to_list())

        # From fills
        if self._fills_data is not None:
            symbols.update(self._fills_data["symbol"].unique().to_list())

        return sorted(symbols)

    def get_time_range(self) -> Dict[str, Dict[str, datetime]]:
        """Get time ranges for each data source."""
        ranges = {}

        if self._store_data is not None and len(self._store_data) > 0:
            ranges["store"] = {
                "start": self._store_data["timestamp"].min(),
                "end": self._store_data["timestamp"].max(),
            }

        if self._signals_data is not None and len(self._signals_data) > 0:
            # Parse timestamp if needed
            if self._signals_data["timestamp"].dtype == pl.Utf8:
                ts = self._signals_data.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f")
                )["timestamp"]
            else:
                ts = self._signals_data["timestamp"]
            ranges["signals"] = {
                "start": ts.min(),
                "end": ts.max(),
            }

        if self._fills_data is not None and len(self._fills_data) > 0:
            if self._fills_data["timestamp"].dtype == pl.Utf8:
                ts = self._fills_data.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f")
                )["timestamp"]
            else:
                ts = self._fills_data["timestamp"]
            ranges["fills"] = {
                "start": ts.min(),
                "end": ts.max(),
            }

        return ranges


def build_unified_dataset(
    loader: MultiSourceLoader,
    symbol: str,
    config: Optional[UnifiedDataConfig] = None,
) -> pl.DataFrame:
    """Build a unified dataset for a single symbol.

    Combines:
    - Order book data (.store files)
    - Trading signals (log data)
    - Fill data (log data)

    Into a single time-aligned DataFrame.

    Args:
        loader: MultiSourceLoader instance with data loaded
        symbol: Symbol to build dataset for
        config: Dataset configuration

    Returns:
        Unified DataFrame with all features aligned
    """
    config = config or loader.config

    # Get time ranges
    time_ranges = loader.get_time_range()

    # Determine effective time range
    if config.use_overlap_only and len(time_ranges) > 1:
        dfs_for_overlap = []
        if loader._store_data is not None:
            dfs_for_overlap.append(
                loader._store_data.filter(pl.col("symbol") == symbol)
            )
        if loader._signals_data is not None:
            signals = loader._signals_data.filter(pl.col("symbol") == symbol)
            if signals["timestamp"].dtype == pl.Utf8:
                signals = signals.with_columns(
                    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.3f")
                )
            dfs_for_overlap.append(signals)

        if len(dfs_for_overlap) >= 2:
            start, end = find_overlapping_range(*dfs_for_overlap)
        else:
            start = config.start_time or min(r["start"] for r in time_ranges.values())
            end = config.end_time or max(r["end"] for r in time_ranges.values())
    else:
        start = config.start_time or min(r["start"] for r in time_ranges.values())
        end = config.end_time or max(r["end"] for r in time_ranges.values())

    print(f"Building dataset for {symbol}: {start} to {end}")

    # Create time grid
    grid = create_time_grid(start, end, config.interval, symbols=[symbol])

    # Normalize grid timestamps to microseconds
    grid = normalize_timestamp_precision(grid, "timestamp", "us")

    # Start with grid
    result = grid.clone()

    # Add store data if available
    if loader._store_data is not None:
        store_symbol = loader._store_data.filter(pl.col("symbol") == symbol)
        if len(store_symbol) > 0:
            # Resample to interval
            store_resampled = resample_store_data(store_symbol, config.interval)
            # Normalize timestamps
            store_resampled = normalize_timestamp_precision(store_resampled, "timestamp", "us")
            # Align to grid
            result = result.join_asof(
                store_resampled.drop("symbol"),
                on="timestamp",
                strategy="backward",
            )
            print(f"  Added store data: {len(store_resampled)} rows")

    # Add signals data if available
    if loader._signals_data is not None:
        signals_symbol = loader._signals_data.filter(pl.col("symbol") == symbol)
        if len(signals_symbol) > 0:
            # Resample signals
            signals_resampled = resample_signals(signals_symbol, config.interval)
            # Normalize timestamps
            signals_resampled = normalize_timestamp_precision(signals_resampled, "timestamp", "us")
            # Align to grid
            result = result.join_asof(
                signals_resampled.drop("symbol"),
                on="timestamp",
                strategy="backward",
            )
            print(f"  Added signals data: {len(signals_resampled)} rows")

    # Add fills data if available
    if loader._fills_data is not None:
        fills_symbol = loader._fills_data.filter(pl.col("symbol") == symbol)
        if len(fills_symbol) > 0:
            # Resample fills
            fills_resampled = resample_fills(fills_symbol, config.interval)
            # Normalize timestamps
            fills_resampled = normalize_timestamp_precision(fills_resampled, "timestamp", "us")
            # Align to grid
            result = result.join_asof(
                fills_resampled.drop("symbol"),
                on="timestamp",
                strategy="backward",
            )
            print(f"  Added fills data: {len(fills_resampled)} rows")

    # Add Binance public data if available
    if symbol in loader._binance_data:
        binance_df = loader._binance_data[symbol]
        if len(binance_df) > 0:
            # Normalize timestamps
            binance_df = normalize_timestamp_precision(binance_df, "timestamp", "us")
            # Align to grid
            result = result.join_asof(
                binance_df.drop("symbol"),
                on="timestamp",
                strategy="backward",
            )
            print(f"  Added Binance data: {len(binance_df)} rows, {len(binance_df.columns)-2} features")

    # Add time features
    if config.include_time_features:
        result = compute_time_features(result)

    # Fill nulls in numeric columns
    numeric_cols = [
        c for c in result.columns
        if result[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
    ]
    result = result.with_columns([
        pl.col(c).fill_null(0) for c in numeric_cols
    ])

    return result.sort("timestamp")


def build_multi_symbol_dataset(
    loader: MultiSourceLoader,
    symbols: Optional[List[str]] = None,
    config: Optional[UnifiedDataConfig] = None,
) -> Dict[str, pl.DataFrame]:
    """Build unified datasets for multiple symbols.

    Args:
        loader: MultiSourceLoader instance
        symbols: List of symbols (None = all available)
        config: Dataset configuration

    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    config = config or loader.config
    symbols = symbols or config.symbols or loader.get_available_symbols()

    datasets = {}
    for symbol in symbols:
        try:
            datasets[symbol] = build_unified_dataset(loader, symbol, config)
        except Exception as e:
            print(f"Warning: Could not build dataset for {symbol}: {e}")

    return datasets


def add_lagged_features(
    df: pl.DataFrame,
    columns: List[str],
    lags: List[int],
) -> pl.DataFrame:
    """Add lagged versions of specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with added lag columns
    """
    # Filter out columns that already have lag/rolling suffixes to avoid duplicates
    base_cols = []
    seen = set()
    for col in columns:
        if col not in df.columns:
            continue
        # Skip columns that already have lag/ma/std suffixes
        if any("_lag" in col or "_ma" in col or "_std" in col for _ in [1]):
            continue
        if col in seen:
            continue
        seen.add(col)
        base_cols.append(col)

    lag_exprs = []
    for col in base_cols:
        for lag in lags:
            new_col_name = f"{col}_lag{lag}"
            # Only add if column doesn't already exist
            if new_col_name not in df.columns:
                lag_exprs.append(
                    pl.col(col).shift(lag).alias(new_col_name)
                )

    if not lag_exprs:
        return df
    return df.with_columns(lag_exprs)


def add_rolling_features(
    df: pl.DataFrame,
    columns: List[str],
    windows: List[int],
) -> pl.DataFrame:
    """Add rolling statistics for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to compute rolling stats for
        windows: List of window sizes

    Returns:
        DataFrame with added rolling columns
    """
    # Filter out columns that already have lag/rolling suffixes
    base_cols = []
    seen = set()
    for col in columns:
        if col not in df.columns:
            continue
        if any("_lag" in col or "_ma" in col or "_std" in col for _ in [1]):
            continue
        if col in seen:
            continue
        seen.add(col)
        base_cols.append(col)

    roll_exprs = []
    for col in base_cols:
        for window in windows:
            ma_col = f"{col}_ma{window}"
            std_col = f"{col}_std{window}"
            if ma_col not in df.columns:
                roll_exprs.append(pl.col(col).rolling_mean(window).alias(ma_col))
            if std_col not in df.columns:
                roll_exprs.append(pl.col(col).rolling_std(window).alias(std_col))

    if not roll_exprs:
        return df
    return df.with_columns(roll_exprs)


def create_labels(
    df: pl.DataFrame,
    label_type: str = "fill_occurred",
    horizon: int = 30,
) -> pl.DataFrame:
    """Create prediction labels.

    Args:
        df: Input DataFrame
        label_type: Type of label ("fill_occurred", "price_direction", "auto")
        horizon: Number of periods ahead for label

    Returns:
        DataFrame with label column added
    """
    # Auto-detect best label type based on available columns
    if label_type == "auto":
        if "n_fills" in df.columns or "fill_volume" in df.columns:
            label_type = "fill_occurred"
        elif "close" in df.columns or "mid_price" in df.columns or "signal_mid_price" in df.columns:
            label_type = "price_direction"
        else:
            raise ValueError("No suitable columns found for label creation")

    if label_type == "fill_occurred":
        # Look ahead to see if a fill occurs in the next `horizon` periods
        if "n_fills" in df.columns:
            df = df.with_columns(
                pl.col("n_fills")
                .rolling_sum(window_size=horizon)
                .shift(-horizon)
                .gt(0)
                .cast(pl.Int32)
                .alias("label")
            )
        elif "fill_volume" in df.columns:
            df = df.with_columns(
                pl.col("fill_volume")
                .rolling_sum(window_size=horizon)
                .shift(-horizon)
                .gt(0)
                .cast(pl.Int32)
                .alias("label")
            )
        else:
            # Fall back to price_direction if no fill data
            print("  Warning: No fill columns found, falling back to price_direction label")
            return create_labels(df, "price_direction", horizon)

    elif label_type == "price_direction":
        # Predict if price goes up in next `horizon` periods
        if "close" in df.columns:
            price_col = "close"
        elif "mid_price" in df.columns:
            price_col = "mid_price"
        elif "signal_mid_price" in df.columns:
            price_col = "signal_mid_price"
        else:
            raise ValueError("No price column found for price_direction label")

        df = df.with_columns(
            (pl.col(price_col).shift(-horizon) > pl.col(price_col))
            .cast(pl.Int32)
            .alias("label")
        )

    else:
        raise ValueError(f"Unknown label type: {label_type}")

    return df
