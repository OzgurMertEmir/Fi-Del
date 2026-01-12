"""Tests for the Binance data pipeline module."""

from datetime import date

import polars as pl
import pytest

from myproj.binance_data.config import BinanceDataConfig
from myproj.binance_data.schemas import get_schema, SPOT_TRADES_COLUMNS
from myproj.binance_data.aggregators import (
    aggregate_trades_to_ohlcv,
    compute_trade_flow_features,
    add_rolling_flow_features,
    add_return_labels,
    add_direction_labels,
)


class TestBinanceDataConfig:
    """Tests for BinanceDataConfig."""

    def test_valid_config(self) -> None:
        """Test creating a valid configuration."""
        config = BinanceDataConfig(
            symbols=["BTCUSDT"],
            data_type="trades",
            market_type="spot",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        assert config.symbols == ["BTCUSDT"]
        assert config.data_type == "trades"

    def test_klines_requires_interval(self) -> None:
        """Test that klines data type requires an interval."""
        with pytest.raises(ValueError, match="interval is required"):
            BinanceDataConfig(
                symbols=["BTCUSDT"],
                data_type="klines",
            )

    def test_klines_with_interval(self) -> None:
        """Test klines config with valid interval."""
        config = BinanceDataConfig(
            symbols=["BTCUSDT"],
            data_type="klines",
            interval="1m",
        )
        assert config.interval == "1m"

    def test_invalid_market_type(self) -> None:
        """Test that invalid market type raises error."""
        with pytest.raises(ValueError, match="Invalid market_type"):
            BinanceDataConfig(
                symbols=["BTCUSDT"],
                data_type="trades",
                market_type="invalid",
            )

    def test_get_download_url(self) -> None:
        """Test URL generation."""
        config = BinanceDataConfig(
            symbols=["BTCUSDT"],
            data_type="trades",
            period="daily",
        )
        url = config.get_download_url("BTCUSDT", date(2024, 1, 15))
        assert "data.binance.vision" in url
        assert "BTCUSDT" in url
        assert "2024-01-15" in url


class TestSchemas:
    """Tests for data schemas."""

    def test_get_spot_trades_schema(self) -> None:
        """Test getting spot trades schema."""
        schema = get_schema("spot", "trades")
        assert schema.columns == SPOT_TRADES_COLUMNS
        assert "trade_id" in schema.dtypes
        assert "price" in schema.dtypes

    def test_get_klines_schema(self) -> None:
        """Test getting klines schema."""
        schema = get_schema("spot", "klines")
        assert "open" in schema.columns
        assert "high" in schema.columns
        assert "close" in schema.columns
        assert "volume" in schema.columns


class TestAggregators:
    """Tests for trade aggregation functions."""

    @pytest.fixture
    def sample_trades(self) -> pl.DataFrame:
        """Create sample trade data for testing."""
        return pl.DataFrame({
            "ts": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0, 0, 0),
                end=pl.datetime(2024, 1, 1, 0, 0, 59),
                interval="1s",
                eager=True,
            ),
            "price": [100.0 + i * 0.1 for i in range(60)],
            "qty": [1.0] * 60,
            "is_buyer_maker": [i % 2 == 0 for i in range(60)],
        })

    def test_aggregate_to_ohlcv(self, sample_trades: pl.DataFrame) -> None:
        """Test OHLCV aggregation."""
        agg = aggregate_trades_to_ohlcv(sample_trades, interval="10s")

        assert len(agg) == 6  # 60 seconds / 10 second buckets
        assert "open" in agg.columns
        assert "high" in agg.columns
        assert "low" in agg.columns
        assert "close" in agg.columns
        assert "volume" in agg.columns
        assert "vwap" in agg.columns

    def test_compute_trade_flow_features(self, sample_trades: pl.DataFrame) -> None:
        """Test order flow feature computation."""
        agg = compute_trade_flow_features(sample_trades, interval="10s")

        assert "buy_volume" in agg.columns
        assert "sell_volume" in agg.columns
        assert "volume_imbalance" in agg.columns
        assert "trade_imbalance" in agg.columns
        assert "net_volume" in agg.columns

        # Check imbalance is bounded [-1, 1]
        imb = agg["volume_imbalance"]
        assert imb.min() >= -1.0
        assert imb.max() <= 1.0

    def test_add_rolling_features(self, sample_trades: pl.DataFrame) -> None:
        """Test rolling feature computation."""
        agg = compute_trade_flow_features(sample_trades, interval="1s")
        agg = add_rolling_flow_features(agg, windows=[5, 10])

        assert "vol_sum_5" in agg.columns
        assert "vol_sum_10" in agg.columns
        assert "vol_imb_mean_5" in agg.columns
        assert "ret_std_5" in agg.columns

    def test_add_return_labels(self, sample_trades: pl.DataFrame) -> None:
        """Test return label generation."""
        agg = compute_trade_flow_features(sample_trades, interval="1s")
        agg = add_return_labels(agg, horizons=[5, 10])

        assert "y_ret_5" in agg.columns
        assert "y_ret_10" in agg.columns

        # Last 5/10 rows should have NaN labels
        assert agg["y_ret_5"].tail(5).null_count() == 5

    def test_add_direction_labels(self, sample_trades: pl.DataFrame) -> None:
        """Test direction label generation."""
        agg = compute_trade_flow_features(sample_trades, interval="1s")
        agg = add_return_labels(agg, horizons=[5])
        agg = add_direction_labels(agg, horizons=[5], deadzone_bps=0.0)

        assert "y_dir_5" in agg.columns

        # Direction should be -1, 0, or 1
        valid_labels = agg.filter(pl.col("y_dir_5").is_not_null())["y_dir_5"]
        assert set(valid_labels.unique().to_list()).issubset({-1, 0, 1})


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_aggregation_pipeline(self) -> None:
        """Test the full aggregation pipeline."""
        # Create synthetic data
        trades = pl.DataFrame({
            "ts": pl.datetime_range(
                start=pl.datetime(2024, 1, 1, 0, 0, 0),
                end=pl.datetime(2024, 1, 1, 0, 9, 59),
                interval="1s",
                eager=True,
            ),
            "price": [100.0 + (i % 100) * 0.01 for i in range(600)],
            "qty": [1.0 + (i % 10) * 0.1 for i in range(600)],
            "is_buyer_maker": [i % 3 != 0 for i in range(600)],
            "symbol": ["BTCUSDT"] * 600,
        })

        # Run aggregation
        agg = compute_trade_flow_features(trades, interval="1s")
        agg = add_rolling_flow_features(agg, windows=[5, 15, 60])
        agg = add_return_labels(agg, horizons=[5, 30])
        agg = add_direction_labels(agg, horizons=[5, 30], deadzone_bps=5.0)

        # Verify output
        assert len(agg) == 600
        assert "volume_imbalance" in agg.columns
        assert "vol_imb_mean_60" in agg.columns
        assert "y_dir_5" in agg.columns
        assert "y_dir_30" in agg.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
