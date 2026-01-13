#!/usr/bin/env python3
"""
Training CLI for FiDel ML Pipeline.

Usage:
    # Quick experiment with XGBoost
    python scripts/run_training.py --symbols BTCUSDT --model xgboost

    # Train LSTM with custom settings
    python scripts/run_training.py \\
        --symbols BTCUSDT ETHUSDT \\
        --model lstm \\
        --sequence-length 60 \\
        --epochs 100 \\
        --days 7

    # Train for regression
    python scripts/run_training.py \\
        --symbols BTCUSDT \\
        --model ridge \\
        --label-type regression \\
        --horizon 60

    # Full configuration from YAML
    python scripts/run_training.py --config experiments/btc_transformer.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML models on FiDel processed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data arguments
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT"],
        help="Trading symbols to use (default: BTCUSDT)",
    )
    data_group.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of historical data (default: 7)",
    )
    data_group.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    data_group.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    data_group.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Data sampling rate (default: 1.0 = all data)",
    )
    data_group.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation set fraction (default: 0.2)",
    )

    # Label arguments
    label_group = parser.add_argument_group("Label Options")
    label_group.add_argument(
        "--label-type",
        choices=["direction", "direction_3", "regression", "volatility"],
        default="direction",
        help="Type of prediction target (default: direction)",
    )
    label_group.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Prediction horizon in steps (default: 30)",
    )
    label_group.add_argument(
        "--threshold",
        type=float,
        default=0.0001,
        help="Direction threshold for classification (default: 0.0001)",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model",
        choices=[
            "logistic",
            "ridge",
            "xgboost",
            "lightgbm",
            "random_forest",
            "lstm",
            "gru",
            "transformer",
        ],
        default="xgboost",
        help="Model type (default: xgboost)",
    )
    model_group.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden layer size for neural networks (default: 64)",
    )
    model_group.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of layers for neural networks (default: 2)",
    )
    model_group.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for LSTM/Transformer (default: 60)",
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    train_group.add_argument(
        "--early-stopping",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/training",
        help="Local output directory (default: artifacts/training)",
    )
    output_group.add_argument(
        "--experiment",
        type=str,
        help="Experiment name (auto-generated if not specified)",
    )
    output_group.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload results to S3 models bucket",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (overrides other arguments)",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def build_config(args):
    """Build TrainingConfig from arguments."""
    from training.config import (
        DataConfig,
        LabelType,
        ModelConfig,
        ModelType,
        ScalerType,
        TrainingConfig,
    )

    # If config file provided, load it
    if args.config:
        config_dict = load_yaml_config(args.config)
        return TrainingConfig.from_dict(config_dict)

    # Build from arguments
    data_config = DataConfig(
        symbols=args.symbols,
        days_back=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        sample_rate=args.sample_rate,
        val_fraction=args.val_fraction,
        label_type=LabelType(args.label_type),
        label_horizon=args.horizon,
        label_threshold=args.threshold,
        scaler_type=ScalerType.STANDARD,
    )

    model_config = ModelConfig(
        model_type=ModelType(args.model),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.sequence_length,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
    )

    # Get S3 bucket from environment
    import os

    output_bucket = ""
    if args.upload_s3:
        output_bucket = os.environ.get("FIDEL_MODELS_BUCKET", "fidel-models-794531974380")

    config = TrainingConfig(
        data=data_config,
        model=model_config,
        output_bucket=output_bucket,
        local_output_dir=args.output_dir,
        experiment_name=args.experiment or "",
        seed=args.seed,
        verbose=args.verbose and not args.quiet,
    )

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build config
    config = build_config(args)

    # Run pipeline
    from training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline(config)
    results = pipeline.run()

    # Print results
    if not args.quiet:
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Experiment: {results['experiment_name']}")
        print(f"Run: {results['run_name']}")
        print(f"Model: {results['model_type']}")
        print(f"Features: {results['n_features']}")
        print(f"Train samples: {results['n_train']}")
        print(f"Val samples: {results['n_val']}")
        print("\nMetrics:")
        for metric, value in results["metrics"].items():
            print(f"  {metric}: {value:.4f}")
        print("\nSaved to:")
        for name, path in results["saved_files"].items():
            print(f"  {name}: {path}")
        print("=" * 60)

    # Return exit code based on success
    return 0


if __name__ == "__main__":
    sys.exit(main())
