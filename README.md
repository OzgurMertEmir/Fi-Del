# Crypto Order Book Project

This repository provides a clean and extensible starting point for training machine‑learning models on high frequency **level‑1 crypto order book** data. The goal is to separate **code** (which lives in Git) from **data** (which lives on Google Drive) so that your repository stays small and collaborators never accidentally commit huge files or secrets.

## Project layout

```
my‑crypto‑model/
├── README.md            # project overview and instructions
├── requirements.txt     # Python dependencies
├── .gitignore           # patterns of files that should never be committed
├── .pre‑commit‑config.yaml  # automated linters and checks
├── .env.example         # template for local environment variables
├── scripts/             # entry points for downloading, preprocessing and training
│   ├── download_data.py
│   └── process_logs.py
├── src/                 # importable Python package for your code
│   └── myproj/
│       ├── __init__.py
│       ├── config.py    # central location for paths and env vars
│       └── data/
│           ├── __init__.py
│           └── io.py    # helper functions for loading data
├── data/                # working folder for your raw/interim/processed data (ignored by Git)
│   ├── raw/
│   ├── processed/
│   └── logs/
└── artifacts/           # models, metrics and other outputs (ignored by Git)
```

## Installing dependencies

Create a virtual environment (recommended) and install the dependencies listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, replace `requirements.txt` with a modern build system like Poetry or a `pyproject.toml` when you are ready.

## Preparing environment variables

This project uses environment variables for configuration. Copy the provided `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Fields include the local data directory, the Google Drive folder ID and any API credentials needed to access Drive.

## Downloading data from Google Drive

The `scripts/download_data.py` script provides an example of downloading your raw datasets from a shared Google Drive folder. You can implement it using either **pydrive2**/Google Drive API or the simpler `gdown` package, depending on your preference. The script reads the Drive folder ID from your `.env` file, lists the files in that folder and downloads them into the `data/raw/` directory.

## Processing logs

`scripts/process_logs.py` uses `myproj.data.log_parser.LogParser` to parse each raw log in `data/raw/` and emit two CSVs per file into `data/processed/strategy_updates/` and `data/processed/trade_fills/`. Use `--force` to re‑parse if outputs already exist:

```bash
python scripts/process_logs.py          # parses *.log in data/raw
python scripts/process_logs.py --force  # overwrite existing outputs
```

## Prototype ML pipelines

Two simple baselines are provided that work directly off the parsed CSVs:

- Price move classification: `python scripts/train_price_baseline.py --symbol SOLUSDC --bucket 1s --horizon 5`
- Execution/fill probability: `python scripts/train_exec_baseline.py --symbol SOLUSDC --bucket 1s --horizon 5`

Both use 1-second bucketing of `data/processed/strategy_updates/` and `data/processed/trade_fills/`, then train a logistic regression with a time-based validation split. Adjust `--symbol`, `--bucket`, `--horizon`, or `--val-fraction` as needed.

## Binance Public Data Pipeline

This project includes a complete pipeline for downloading, processing, and analyzing historical trade data from [Binance Public Data](https://github.com/binance/binance-public-data/).

### Quick Start

```bash
# 1. Download historical trades (e.g., one week of BTCUSDT)
PYTHONPATH=src python -m myproj.binance_data.cli \
    --symbols BTCUSDT \
    --start 2024-12-01 \
    --end 2024-12-07 \
    --data-type trades

# 2. Generate EDA dashboard with visualizations
PYTHONPATH=src python -m eda.run_binance_eda \
    --symbol BTCUSDT \
    --start 2024-12-01 \
    --end 2024-12-07 \
    --out reports/binance_eda

# 3. View the dashboard
cd reports/binance_eda && python -m http.server 8080
# Open http://localhost:8080 in your browser
```

### Available Data

Binance provides historical data from **August 17, 2017** to present:

| Data Type | Description |
|-----------|-------------|
| `trades` | Individual trades with price, quantity, buyer/seller |
| `aggTrades` | Aggregated trades (smaller files) |
| `klines` | OHLCV candles (1m, 5m, 1h, 1d, etc.) |

### Order Flow Features

The pipeline computes 60+ features from raw trades:

**Basic Features:**
- OHLCV (open, high, low, close, volume)
- Buy/sell volume split, trade counts
- Volume imbalance, trade imbalance
- VWAP, microprice, log returns

**Advanced Microstructure Features:**
- `toxicity` - VPIN-inspired order flow toxicity
- `kyle_lambda` - Price impact proxy (Kyle's lambda)
- `momentum` - Cumulative returns over window
- `ret_autocorr` - Return serial correlation
- `intensity_momentum` - Trade arrival acceleration

### Python API

```python
from myproj.binance_data import (
    build_binance_ml_dataset,
    run_binance_baseline,
    get_feature_importance,
)

# Build ML-ready dataset
dataset = build_binance_ml_dataset(
    symbol="BTCUSDT",
    start_date="2024-12-01",
    end_date="2024-12-07",
    interval="1s",           # 1-second bars
    label_horizon=30,        # Predict 30 bars ahead
    include_advanced_features=True,
)

# Access the data
print(f"Features: {len(dataset.feature_cols)}")
print(f"Train: {len(dataset.X_train)}, Val: {len(dataset.X_val)}")

# Quick baseline evaluation
metrics, dataset = run_binance_baseline(
    symbol="BTCUSDT",
    start_date="2024-12-01",
    end_date="2024-12-07",
)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Dashboard Visualizations

The EDA dashboard (`reports/binance_eda/`) includes:

- **OHLCV Candlestick** - Interactive price chart with volume
- **Full History** - 7+ years of daily data (if downloaded)
- **Order Flow Imbalance** - Buy/sell pressure over time
- **Toxicity Timeseries** - Order flow toxicity patterns
- **Kyle's Lambda** - Price impact signals
- **Feature Importance** - Top predictive features
- **Label Distribution** - Up/Down/Neutral class balance

### Project Structure

```
src/myproj/binance_data/
├── config.py        # URL generation, download configuration
├── schemas.py       # Column definitions for all data types
├── downloader.py    # Download manager with retries & checksums
├── parsers.py       # ZIP/CSV parsing, timestamp handling
├── aggregators.py   # Feature engineering (60+ features)
├── pipeline.py      # High-level orchestration
├── ml_integration.py # ML training utilities
└── cli.py           # Command-line interface

src/eda/
├── binance_plots.py    # Order flow visualizations
└── run_binance_eda.py  # Dashboard generator
```

## Automated checks

This project ships with a `.pre‑commit‑config.yaml` file. [pre‑commit](https://pre-commit.com/) hooks run automatically when you try to commit code. To enable the hooks run:

```bash
pip install pre-commit gitleaks
pre-commit install
```
