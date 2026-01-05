"""Configuration for the crypto order book project.

This module centralises access to environment variables and important
filesystem paths.  Loading environment variables from a `.env` file
eliminates the need to hardcode machine‑specific paths or secrets in your
source code.  For more information about `.env` files, see the README.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load variables from a `.env` file if present.  Loading is idempotent, so
# calling this multiple times is safe.
load_dotenv()


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable with an optional default.

    Parameters
    ----------
    name : str
        The name of the environment variable to retrieve.
    default : str | None
        A default value to return if the variable is not set.

    Returns
    -------
    str | None
        The value of the environment variable or the default.
    """
    return os.environ.get(name, default)


# Base directory for data.  Defaults to `./data`.  Use an absolute path in your
# `.env` file if you wish to store data outside the project directory.
DATA_ROOT = Path(get_env("DATA_ROOT", "./data")).resolve()

# Subdirectories for raw and processed data
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
LOGS_DIR = DATA_ROOT / "logs"

# Ensure that the directories exist when imported; they are created if
# necessary.  Note: these calls do nothing if the directory already exists.
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def get_drive_folder_id() -> Optional[str]:
    """Return the Google Drive folder ID from the environment.

    Returns
    -------
    str | None
        The folder ID or `None` if not set.
    """
    return get_env("DRIVE_FOLDER_ID")


def get_log_parser_path() -> Optional[Path]:
    """Return the path to the log parser script, if set.

    The value is expanded to handle `~` and relative paths.
    """
    path = get_env("LOG_PARSER_PATH")
    return Path(os.path.expanduser(path)).resolve() if path else None
