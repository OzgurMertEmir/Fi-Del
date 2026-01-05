#!/usr/bin/env python
"""
download_data.py

This script downloads raw crypto order book logs and data from a shared Google Drive
folder.  It reads configuration values from environment variables (`DATA_ROOT` and
`DRIVE_FOLDER_ID`) and writes all downloaded files into `data/raw/` by default.

Two download backends are supported:

1. **gdown** – if installed, the script uses `gdown.download_folder()` to fetch
   every file in a Drive folder via the publicly accessible link.  This method
   requires only the folder ID, and is suitable when the folder is shared
   publicly or within your domain.
2. **PyDrive2** – as a fallback, the script demonstrates how to authenticate
   against the Google Drive API using OAuth and download files programmatically.
   You must supply a client secret JSON and a refresh token in your `.env` file.

Usage:

```
python scripts/download_data.py
python scripts/download_data.py --subset 2025-01-05
```

The optional `--subset` argument can be used to filter files by name.  For
example, `--subset jan_2025` will download only files whose names contain
`jan_2025`.

Note: this script is intended as a template.  Depending on your security
policies and Drive configuration you may need to adjust the authentication
logic.  See the README for more details.
"""

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv


def download_with_gdown(folder_id: str, output_dir: Path, pattern: str | None = None) -> None:
    """Use gdown to download all files from a Google Drive folder.

    Parameters
    ----------
    folder_id : str
        The ID of the Google Drive folder (the last part of the URL).
    output_dir : Path
        Directory where downloaded files will be saved.
    pattern : str | None
        Optional substring or regex pattern to filter file names.
    """
    try:
        import gdown
    except ImportError:
        raise RuntimeError("gdown is not installed. Please add it to requirements.txt or install it manually.")

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    # gdown downloads all files into the specified output directory
    print(f"Downloading files from {url} into {output_dir}…")
    # gdown automatically handles listing; if pattern provided, filter afterwards
    downloaded = gdown.download_folder(url, output=str(output_dir), quiet=False)
    if not downloaded:
        print("No files were downloaded – please check the folder ID or permissions.")
        return

    if pattern:
        regex = re.compile(pattern)
        # remove any files that do not match the pattern
        for path in downloaded:
            name = os.path.basename(path)
            if not regex.search(name):
                print(f"Skipping {name} (does not match pattern)")
                os.remove(path)


def download_with_pydrive(folder_id: str, output_dir: Path, pattern: str | None = None) -> None:
    """Download files from Drive using PyDrive2.

    This method demonstrates how to authenticate using client credentials and a
    refresh token.  You need to store `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`
    and `GOOGLE_REFRESH_TOKEN` in your environment or `.env` file.
    """
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError:
        raise RuntimeError("pydrive2 is not installed. Please add it to requirements.txt or install it.")

    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if not (client_id and client_secret and refresh_token):
        raise RuntimeError("GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN must be set for PyDrive2 authentication.")

    # Authenticate silently using refresh token
    settings = {
        "client_config": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uri": "urn:ietf:wg:oauth:2.0:oob",
        },
        "save_credentials": False,
        "get_refresh_token": True,
    }
    gauth = GoogleAuth(settings=settings)
    # Provide the refresh token directly
    gauth.credentials = gauth._build_credentials({
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "token_uri": "https://oauth2.googleapis.com/token",
    })
    drive = GoogleDrive(gauth)

    query = f"'{folder_id}' in parents and trashed = false"
    file_list = drive.ListFile({'q': query}).GetList()
    regex = re.compile(pattern) if pattern else None
    for file1 in file_list:
        title = file1['title']
        if regex and not regex.search(title):
            continue
        dest_path = output_dir / title
        print(f"Downloading {title}…")
        f = drive.CreateFile({'id': file1['id']})
        f.GetContentFile(str(dest_path))


def main():
    parser = argparse.ArgumentParser(description="Download raw data from Google Drive.")
    parser.add_argument(
        "--subset", help="Optional substring or regex to filter file names", default=None
    )
    parser.add_argument(
        "--backend",
        choices=["gdown", "pydrive"],
        default="gdown",
        help="Which backend to use for downloading (default: gdown)",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    data_root = os.environ.get("DATA_ROOT", "./data")
    drive_folder = os.environ.get("DRIVE_FOLDER_ID")
    if not drive_folder:
        raise RuntimeError("DRIVE_FOLDER_ID environment variable is not set.")

    raw_dir = Path(data_root) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "gdown":
        download_with_gdown(drive_folder, raw_dir, pattern=args.subset)
    else:
        download_with_pydrive(drive_folder, raw_dir, pattern=args.subset)


if __name__ == "__main__":
    main()