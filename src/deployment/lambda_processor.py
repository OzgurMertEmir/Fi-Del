"""
lambda_processor.py - AWS Lambda function for feature engineering

This Lambda function is triggered by S3 events when new raw data arrives.
It:
1. Loads the raw data from S3
2. Applies feature engineering
3. Saves processed data back to S3 as parquet

Environment variables:
    PROCESSED_BUCKET: S3 bucket for processed data
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3 = boto3.client("s3")

# Configuration from environment
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET", "")


def parse_store_file(content: str) -> list[dict]:
    """Parse .store CSV file from Java Binance streamer.

    Format: symbol,timestamp,bidPrice,bidQuantity,askPrice,askQuantity,dailyTotalVolume
    """
    records = []
    for line in content.strip().split('\n'):
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 6:
            try:
                records.append({
                    "symbol": parts[0],
                    "timestamp": int(parts[1]),
                    "bid_price": float(parts[2]),
                    "bid_qty": float(parts[3]),
                    "ask_price": float(parts[4]),
                    "ask_qty": float(parts[5]),
                    "volume": float(parts[6]) if len(parts) > 6 else 0.0,
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed line: {line[:50]}... - {e}")
    return records


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features from raw trade data using pandas.

    This is a simplified version of the feature engineering pipeline
    suitable for Lambda execution.
    """
    # Handle .store format (from Java streamer)
    if "bid_price" in df.columns:
        # Convert timestamp from epoch ms to datetime
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
        # Use mid price as the main price
        df["price"] = (df["bid_price"] + df["ask_price"]) / 2
        df["qty"] = (df["bid_qty"] + df["ask_qty"]) / 2
        df["spread"] = df["ask_price"] - df["bid_price"]

    # Handle JSON format (from Python ingestor)
    elif "timestamp" not in df.columns and "T" in df.columns:
        df["timestamp"] = df["T"]
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    elif "timestamp" not in df.columns and "t" in df.columns:
        df["timestamp"] = df["t"]
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Extract price and quantity
    if "price" not in df.columns:
        if "p" in df.columns:
            df["price"] = pd.to_numeric(df["p"], errors="coerce")
        else:
            logger.warning(f"Price column not found. Available: {df.columns.tolist()}")
            return df

    if "qty" not in df.columns:
        if "q" in df.columns:
            df["qty"] = pd.to_numeric(df["q"], errors="coerce")
        else:
            df["qty"] = 1.0

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic features
    df["return_1"] = df["price"].pct_change()
    df["log_price"] = np.log(df["price"])
    df["dollar_volume"] = df["price"] * df["qty"]

    # Trade direction (based on 'm' flag for maker side)
    if "m" in df.columns:
        df["trade_direction"] = np.where(df["m"] is True, -1, 1)
    else:
        df["trade_direction"] = 1

    # Rolling features
    for window in [5, 10, 30]:
        df[f"price_ma_{window}"] = df["price"].rolling(window=window, min_periods=1).mean()
        df[f"price_std_{window}"] = df["price"].rolling(window=window, min_periods=1).std()
        df[f"volume_sum_{window}"] = df["qty"].rolling(window=window, min_periods=1).sum()
        df[f"return_ma_{window}"] = df["return_1"].rolling(window=window, min_periods=1).mean()

    # Lag features
    for lag in [1, 5, 10]:
        df[f"price_lag_{lag}"] = df["price"].shift(lag)
        df[f"return_lag_{lag}"] = df["return_1"].shift(lag)

    # Order flow imbalance
    for window in [10, 30]:
        df[f"order_flow_{window}"] = (
            (df["trade_direction"] * df["qty"])
            .rolling(window=window, min_periods=1)
            .sum()
        )

    # Add processing metadata
    df["processed_at"] = datetime.now(timezone.utc).isoformat()

    return df


def handler(event: dict, context: Any) -> dict:
    """
    Lambda handler for S3 trigger events.

    Args:
        event: S3 event notification
        context: Lambda context

    Returns:
        Processing result
    """
    logger.info(f"Received event: {json.dumps(event)}")

    if not PROCESSED_BUCKET:
        raise ValueError("PROCESSED_BUCKET environment variable not set")

    results = []

    for record in event.get("Records", []):
        # Get S3 object info
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        logger.info(f"Processing s3://{bucket}/{key}")

        try:
            # Download raw data
            response = s3.get_object(Bucket=bucket, Key=key)
            body = response["Body"].read()

            # Decompress if gzipped
            if key.endswith(".gz"):
                body = gzip.decompress(body)

            content = body.decode("utf-8")

            # Parse based on file type
            if key.endswith(".store"):
                # CSV format from Java Binance streamer
                data = parse_store_file(content)
                logger.info(f"Loaded {len(data)} records from .store file")
            else:
                # JSON format from Python ingestor
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
                logger.info(f"Loaded {len(data)} records from JSON")

            if len(data) == 0:
                logger.warning(f"No data parsed from {key}")
                results.append({
                    "source_key": key,
                    "status": "skipped",
                    "reason": "no data",
                })
                continue

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Compute features
            df_features = compute_features(df)

            logger.info(f"Computed {len(df_features.columns)} features for {len(df_features)} rows")

            # Generate output key
            # Input: trades/BTCUSDT/2025-01-01/12/20250101_120000_123456.json.gz or .store
            # Output: features/BTCUSDT/2025-01-01/12/20250101_120000_123456.parquet
            output_key = key.replace("trades/", "features/")
            output_key = output_key.replace(".json.gz", ".parquet")
            output_key = output_key.replace(".json", ".parquet")
            output_key = output_key.replace(".store", ".parquet")

            # Save to parquet
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                # Convert to pyarrow table and write parquet
                table = pa.Table.from_pandas(df_features)
                pq.write_table(table, f.name)

                # Upload to S3
                s3.upload_file(
                    f.name,
                    PROCESSED_BUCKET,
                    output_key,
                    ExtraArgs={
                        "ContentType": "application/octet-stream",
                        "Metadata": {
                            "source_bucket": bucket,
                            "source_key": key,
                            "record_count": str(len(df_features)),
                            "feature_count": str(len(df_features.columns)),
                        },
                    },
                )

                # Cleanup temp file
                os.unlink(f.name)

            logger.info(f"Saved features to s3://{PROCESSED_BUCKET}/{output_key}")

            result = {
                "source_key": key,
                "output_key": output_key,
                "record_count": len(df_features),
                "feature_count": len(df_features.columns),
                "status": "success",
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {key}: {e}")
            results.append({
                "source_key": key,
                "status": "error",
                "error": str(e),
            })

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": f"Processed {len(results)} files",
            "results": results,
        }),
    }


# For local testing
if __name__ == "__main__":
    # Simulate S3 event
    test_event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "test-bucket"},
                    "object": {"key": "trades/BTCUSDT/2025-01-01/12/test.json"},
                }
            }
        ]
    }

    # Create test data
    test_data = [
        {"T": 1704110400000, "p": "42000.50", "q": "0.1", "m": False},
        {"T": 1704110401000, "p": "42001.00", "q": "0.2", "m": True},
        {"T": 1704110402000, "p": "42000.75", "q": "0.15", "m": False},
    ]

    df = pd.DataFrame(test_data)
    features = compute_features(df)
    print("Test features:")
    print(features)
