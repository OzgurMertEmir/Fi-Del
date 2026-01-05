"""CLI entrypoint to build ML-ready dataset from a .store file."""

from __future__ import annotations

import argparse

from .build_dataset import build_dataset
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML-ready dataset from 1s L1 store file.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = build_dataset(cfg)
    print("Dataset built:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
