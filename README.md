# Crypto Order Book Project

This repository provides a clean and extensible starting point for training machine‑learning models on high frequency **level‑1 crypto order book** data. 

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

This project uses environment variables for configuration.  Copy the provided `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Fields include the local data directory, the Google Drive folder ID and any API credentials needed to access Drive.

## Downloading data from Google Drive

The `scripts/download_data.py` script provides an example of downloading your raw datasets from a shared Google Drive folder.  You can implement it using either **pydrive2**/Google Drive API or the simpler `gdown` package, depending on your preference.  The script reads the Drive folder ID from your `.env` file, lists the files in that folder and downloads them into the `data/raw/` directory.

## Processing logs

`scripts/process_logs.py` demonstrates how to call log parser (e.g. `logparser.py`) on each raw log file.  It writes cleaned machine‑learning ready files into `data/processed/`.

## Automated checks

This project ships with a `.pre‑commit‑config.yaml` file.  [pre‑commit](https://pre-commit.com/) hooks run automatically when you try to commit code.  In particular, the hooks included here:

To enable the hooks run:

```bash
pip install pre-commit gitleaks
pre-commit install
```
