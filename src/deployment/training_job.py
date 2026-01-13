#!/usr/bin/env python3
"""
training_job.py - ML training job for AWS EC2 GPU instances

This script runs on an EC2 Spot GPU instance and:
1. Downloads processed features from S3
2. Trains ML models using existing pipeline
3. Uploads model artifacts to S3
4. Auto-terminates on completion (for cost savings)

Usage:
    python training_job.py --symbol BTCUSDT --date-range 2025-01-01:2025-01-07
    python training_job.py --all-symbols --days 7

Environment variables:
    FIDEL_PROCESSED_BUCKET: S3 bucket with processed features
    FIDEL_MODELS_BUCKET: S3 bucket for model artifacts
    AWS_REGION: AWS region
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("training")

# AWS clients
s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
ec2 = boto3.client("ec2", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def get_instance_id() -> Optional[str]:
    """Get current EC2 instance ID from metadata."""
    try:
        import urllib.request
        with urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id",
            timeout=2
        ) as response:
            return response.read().decode()
    except Exception:
        return None


def download_features(
    bucket: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    local_dir: Path,
) -> list[Path]:
    """Download feature files from S3."""
    logger.info(f"Downloading features for {symbol} from {start_date.date()} to {end_date.date()}")

    downloaded = []
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        prefix = f"features/{symbol}/{date_str}/"

        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

            for obj in response.get("Contents", []):
                key = obj["Key"]
                local_path = local_dir / key

                # Create directory
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Download
                s3.download_file(bucket, key, str(local_path))
                downloaded.append(local_path)
                logger.debug(f"Downloaded: {key}")

        except ClientError as e:
            logger.warning(f"Error listing {prefix}: {e}")

        current += timedelta(days=1)

    logger.info(f"Downloaded {len(downloaded)} feature files")
    return downloaded


def load_features(files: list[Path]) -> Any:
    """Load and concatenate feature files."""
    try:
        import polars as pl
    except ImportError:
        logger.error("polars not installed")
        return None

    dfs = []
    for f in files:
        try:
            df = pl.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")

    if not dfs:
        return None

    combined = pl.concat(dfs)
    logger.info(f"Loaded {len(combined)} rows with {len(combined.columns)} columns")
    return combined


def train_model(
    df: Any,
    model_type: str = "lightgbm",
    target_col: str = "return_1",
    **kwargs,
) -> dict:
    """Train ML model on features."""
    import numpy as np

    try:
        import polars as pl
    except ImportError:
        raise ImportError("polars required for training")

    logger.info(f"Training {model_type} model...")

    # Prepare features and target
    # Drop non-feature columns
    exclude_cols = [
        "timestamp", "processed_at", "symbol",
        target_col,
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Create binary classification target (price up/down)
    df = df.with_columns([
        (pl.col(target_col).shift(-1) > 0).cast(pl.Int32).alias("target")
    ]).drop_nulls()

    # Convert to numpy
    X = df.select(feature_cols).to_numpy()
    y = df.select("target").to_numpy().ravel()

    # Handle inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

    if model_type == "lightgbm":
        try:
            import lightgbm as lgb

            params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "n_estimators": 100,
                **kwargs,
            }

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
            )

            # Evaluate
            from sklearn.metrics import accuracy_score, roc_auc_score

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "auc": float(roc_auc_score(y_test, y_prob)),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": len(feature_cols),
            }

            logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test AUC: {metrics['auc']:.4f}")

            return {
                "model": model,
                "model_type": model_type,
                "feature_cols": feature_cols,
                "metrics": metrics,
            }

        except ImportError:
            logger.warning("lightgbm not installed, falling back to sklearn")
            model_type = "random_forest"

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
            **kwargs,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_prob)),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(feature_cols),
        }

        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test AUC: {metrics['auc']:.4f}")

        return {
            "model": model,
            "model_type": model_type,
            "feature_cols": feature_cols,
            "metrics": metrics,
        }

    raise ValueError(f"Unknown model type: {model_type}")


def save_model(
    result: dict,
    symbol: str,
    bucket: str,
    local_dir: Path,
) -> str:
    """Save model artifacts to S3."""
    import joblib

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = f"{symbol}_{result['model_type']}_{timestamp}"

    # Create model directory
    model_dir = local_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.joblib"
    joblib.dump(result["model"], model_path)

    # Save metadata
    metadata = {
        "symbol": symbol,
        "model_type": result["model_type"],
        "feature_cols": result["feature_cols"],
        "metrics": result["metrics"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Upload to S3
    s3_prefix = f"checkpoints/{symbol}/{model_name}"

    for file_path in model_dir.iterdir():
        s3_key = f"{s3_prefix}/{file_path.name}"
        s3.upload_file(str(file_path), bucket, s3_key)
        logger.info(f"Uploaded: s3://{bucket}/{s3_key}")

    return s3_prefix


def terminate_instance():
    """Terminate the current EC2 instance."""
    instance_id = get_instance_id()
    if instance_id:
        logger.info(f"Terminating instance {instance_id}...")
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
        except Exception as e:
            logger.error(f"Failed to terminate instance: {e}")
    else:
        logger.info("Not running on EC2, skipping termination")


def main():
    parser = argparse.ArgumentParser(description="FiDel ML Training Job")
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Train on all available symbols",
    )
    parser.add_argument(
        "--date-range",
        help="Date range in format YYYY-MM-DD:YYYY-MM-DD",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to train on (default: 7)",
    )
    parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "random_forest"],
        help="Model type (default: lightgbm)",
    )
    parser.add_argument(
        "--processed-bucket",
        default=os.environ.get("FIDEL_PROCESSED_BUCKET", ""),
        help="S3 bucket with processed features",
    )
    parser.add_argument(
        "--models-bucket",
        default=os.environ.get("FIDEL_MODELS_BUCKET", ""),
        help="S3 bucket for model artifacts",
    )
    parser.add_argument(
        "--no-terminate",
        action="store_true",
        help="Don't terminate instance after training",
    )
    args = parser.parse_args()

    if not args.processed_bucket or not args.models_bucket:
        logger.error("S3 buckets not specified")
        sys.exit(1)

    # Parse date range
    if args.date_range:
        start_str, end_str = args.date_range.split(":")
        start_date = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)

    logger.info("=" * 50)
    logger.info("FiDel ML Training Job")
    logger.info("=" * 50)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Model type: {args.model_type}")
    logger.info("")

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Download features
            files = download_features(
                bucket=args.processed_bucket,
                symbol=args.symbol,
                start_date=start_date,
                end_date=end_date,
                local_dir=temp_path / "features",
            )

            if not files:
                logger.error("No feature files found!")
                sys.exit(1)

            # Load features
            df = load_features(files)
            if df is None:
                logger.error("Failed to load features!")
                sys.exit(1)

            # Train model
            result = train_model(
                df=df,
                model_type=args.model_type,
            )

            # Save model
            s3_path = save_model(
                result=result,
                symbol=args.symbol,
                bucket=args.models_bucket,
                local_dir=temp_path / "models",
            )

            logger.info("")
            logger.info("=" * 50)
            logger.info("Training Complete!")
            logger.info("=" * 50)
            logger.info(f"Model saved to: s3://{args.models_bucket}/{s3_path}")
            logger.info(f"Accuracy: {result['metrics']['accuracy']:.4f}")
            logger.info(f"AUC: {result['metrics']['auc']:.4f}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

    # Terminate instance if requested
    if not args.no_terminate:
        terminate_instance()


if __name__ == "__main__":
    main()
