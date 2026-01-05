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

## Automated checks

This project ships with a `.pre‑commit‑config.yaml` file. [pre‑commit](https://pre-commit.com/) hooks run automatically when you try to commit code. To enable the hooks run:

```bash
pip install pre-commit gitleaks
pre-commit install
```

