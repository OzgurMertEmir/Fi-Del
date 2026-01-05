# Crypto Order Book Project

This repository provides a clean and extensible starting point for training machine‑learning models on high frequency **level‑1 crypto order book** data.  The goal is to separate **code** (which lives in Git) from **data** (which lives on Google Drive) so that your repository stays small and collaborators never accidentally commit huge files or secrets.

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

The structure is inspired by several common machine‑learning project templates.  A **`data/`** folder with sub‑folders for *raw* and *processed* data keeps your local working files organized.  According to best practices, large binary data files and logs should **not** be committed to version control【269884832086393†L239-L270】.  Instead they live in Google Drive and are downloaded into this folder when needed.

The **`src/`** directory is an installable Python package (`myproj`).  Putting your code in a package makes it easier to import from anywhere and helps avoid relative import chaos.  Configuration values such as the path to your data root live in `src/myproj/config.py`.

`scripts/` contains small command‑line interfaces that wrap the functionality in `src/`.  For example, `download_data.py` will fetch the required raw data from a Google Drive folder, and `process_logs.py` can convert raw logs into clean training data using your existing log parser.

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

Fields include the local data directory, the Google Drive folder ID and any API credentials needed to access Drive.  **Never commit your actual `.env` file**.  Keeping secrets out of your repository is a common recommendation when using Git【767455111253032†L165-L169】.

## Downloading data from Google Drive

The `scripts/download_data.py` script provides an example of downloading your raw datasets from a shared Google Drive folder.  You can implement it using either **pydrive2**/Google Drive API or the simpler `gdown` package, depending on your preference.  The script reads the Drive folder ID from your `.env` file, lists the files in that folder and downloads them into the `data/raw/` directory.

**Note:** storing a Git repository inside a cloud‑synchronised folder such as Google Drive is discouraged because concurrent file‑system operations can corrupt the `.git` directory【734821703252939†L1007-L1044】.  Instead, let GitHub host your repository and use Drive solely for data storage.

## Processing logs

`scripts/process_logs.py` demonstrates how to call your existing log parser (e.g. `logparser.py`) on each raw log file.  It writes cleaned machine‑learning ready files into `data/processed/`.

## Automated checks

This project ships with a `.pre‑commit‑config.yaml` file.  [pre‑commit](https://pre-commit.com/) hooks run automatically when you try to commit code.  In particular, the hooks included here:

- ensure you never accidentally add a file larger than 5 MB to the repository,
- trim trailing whitespace and add missing newlines at the end of files,
- scan staged changes for secrets using **gitleaks**.

To enable the hooks run:

```bash
pip install pre-commit gitleaks
pre-commit install
```

The pre‑commit team recommends using existing hooks to maintain code quality and catch issues early【273277953454760†L18-L41】.

## Version controlling large data

GitHub imposes a 100 MB per‑file limit, and Git itself performs poorly with large binary files.  If you need full version control of your datasets, consider using **Data Version Control (DVC)** with a Google Drive remote.  DVC stores the data in your Drive and keeps only small metadata files in Git【749473904798644†L15-L23】.  You can then fetch the correct version of the data with `dvc pull` and upload updates with `dvc push`【749473904798644†L31-L44】.

## Pushing to GitHub

1. Create a new, empty repository on GitHub (do **not** initialize it with a README or `.gitignore`).
2. In your local project root (`my‑crypto‑model`), run:

   ```bash
   git init
   git add .
   git commit -m "Initial project structure"
   git remote add origin https://github.com/your‑username/your‑repo.git
   git branch -M main
   git push -u origin main
   ```

3. After pushing, collaborators can clone the repository and run `scripts/download_data.py` to pull the raw data into their local `data/` folder.

See the Stack Overflow discussion for why using Google Drive to sync entire Git repositories is unsafe【734821703252939†L1007-L1044】; keep code and git history on GitHub and data on Drive instead.
