# FiDel - Crypto Trading Data Pipeline

A production-ready pipeline for streaming, processing, and training ML models on high-frequency crypto order book data. The project supports both local development and AWS cloud deployment.

## Features

- **Real-time Data Streaming**: Live order book data from Binance via WebSocket
- **AWS Cloud Pipeline**: Serverless feature engineering with Lambda, S3 storage
- **ML-Ready Features**: 36+ engineered features including rolling statistics, order flow, spreads
- **Historical Data**: Integration with Binance Public Data for backtesting
- **EDA Dashboard**: Interactive visualizations for data exploration

## Project layout

```
FiDel/
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from Git
├── .pre‑commit‑config.yaml   # Automated linters and checks
│
├── scripts/                  # Entry points for data processing
│   ├── download_data.py      # Download from Google Drive
│   ├── process_logs.py       # Parse raw trading logs
│   └── train_*.py            # ML training scripts
│
├── src/                      # Python packages
│   ├── myproj/
│   │   ├── binance_data/     # Binance public data pipeline
│   │   ├── unified_data/     # Feature engineering library
│   │   └── data/             # Data loading utilities
│   ├── eda/                  # Exploratory data analysis
│   └── deployment/           # AWS Lambda functions
│       └── lambda_processor.py
│
├── infra/                    # Infrastructure as code
│   └── aws/
│       ├── setup-iam.sh      # IAM roles and policies
│       ├── s3-setup.sh       # S3 bucket configuration
│       ├── lambda-deploy.sh  # Lambda deployment
│       ├── ec2-java-streamer.sh  # EC2 streamer setup
│       ├── cloudwatch-setup.sh   # Monitoring setup
│       └── monitor-pipeline.sh   # Pipeline monitoring CLI
│
├── data/                     # Local data (ignored by Git)
│   ├── raw/
│   └── processed/
│
└── artifacts/                # Models and outputs (ignored by Git)
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

## AWS Cloud Pipeline

Deploy the full data pipeline to AWS for real-time streaming, feature engineering, and ML training.

### Architecture

```
[Binance WebSocket] --> [EC2 Spot Streamer] --> [S3 Raw Data]
                                                      |
                                                      v
                                            [Lambda: Feature Engineering]
                                                      |
                                                      v
                                            [S3 Processed Features]
```

### Components

| Component | Description | Cost |
|-----------|-------------|------|
| **EC2 Spot Instance** | Java streamer collecting order book data | ~$5/month |
| **S3 Buckets** | Raw data, processed features, models | ~$5/month |
| **Lambda Function** | Serverless feature engineering (pandas + pyarrow) | Free tier |
| **CloudWatch** | Monitoring, alerts, dashboards | Free tier |

### Deployment

```bash
# 1. Configure AWS CLI
aws configure

# 2. Run deployment scripts in order
cd infra/aws
./setup-iam.sh          # Create IAM roles and policies
./s3-setup.sh           # Create S3 buckets with lifecycle rules
./lambda-deploy.sh      # Deploy feature engineering Lambda
./cloudwatch-setup.sh   # Set up monitoring dashboard
./ec2-java-streamer.sh  # Launch data streamer on EC2 Spot

# 3. Monitor the pipeline
./monitor-pipeline.sh           # Full status check
./monitor-pipeline.sh --watch   # Continuous monitoring
./monitor-pipeline.sh --logs    # View streamer logs
./monitor-pipeline.sh --s3      # Check recent data files
```

### Data Flow

1. **Java Streamer** connects to Binance WebSocket and collects order book snapshots
2. **S3 Sync** uploads `.store` files to S3 every 5 minutes
3. **Lambda Trigger** automatically processes new files
4. **Feature Engineering** computes 36 features:
   - Mid price, spread, bid/ask quantities
   - Returns, log prices, dollar volume
   - Rolling MA, STD (windows: 5, 10, 30)
   - Lag features (1, 5, 10 periods)
   - Order flow imbalance
5. **Parquet Output** saved to processed bucket for ML training

### Symbols Tracked

- BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT

## Automated checks

This project ships with a `.pre‑commit‑config.yaml` file. [pre‑commit](https://pre-commit.com/) hooks run automatically when you try to commit code. To enable the hooks run:

```bash
pip install pre-commit gitleaks
pre-commit install
```
