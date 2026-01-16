#!/usr/bin/env python3
"""
RunPod Training Script for FiDel ML Pipeline.

This script is designed to run on RunPod GPU instances and:
1. Downloads training data from AWS S3
2. Runs the training pipeline
3. Uploads model artifacts back to S3

Usage on RunPod:
    # Set AWS credentials as environment variables first
    export AWS_ACCESS_KEY_ID=your_key
    export AWS_SECRET_ACCESS_KEY=your_secret
    export AWS_DEFAULT_REGION=us-east-1

    # Run training
    python runpod_train.py --model transformer --days 14 --epochs 100

    # Or with specific dates
    python runpod_train.py --model lstm --start-date 2026-01-01 --end-date 2026-01-14
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Optional imports - installed on RunPod
try:
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError:
    np = None
    torch = None
    nn = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            logger.warning("No CUDA GPU available, using CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed, GPU check skipped")
        return False


def check_aws_credentials():
    """Verify AWS credentials are configured."""
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError

    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        logger.info(f"AWS Account: {identity['Account']}")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return False
    except ClientError as e:
        logger.error(f"AWS credentials invalid: {e}")
        return False


def get_bucket_names():
    """Get S3 bucket names from environment or discover them."""
    import boto3

    # Check environment variables first
    ml_bucket = os.environ.get("FIDEL_ML_BUCKET")
    models_bucket = os.environ.get("FIDEL_MODELS_BUCKET")

    if ml_bucket and models_bucket:
        return ml_bucket, models_bucket

    # Auto-discover buckets
    s3 = boto3.client("s3")
    buckets = s3.list_buckets()["Buckets"]

    for bucket in buckets:
        name = bucket["Name"]
        if "fidel-ml-ready" in name:
            ml_bucket = name
        elif "fidel-models" in name:
            models_bucket = name

    if not ml_bucket or not models_bucket:
        raise ValueError(
            "Could not find FiDel buckets. Set FIDEL_ML_BUCKET and FIDEL_MODELS_BUCKET"
        )

    logger.info(f"ML Data bucket: {ml_bucket}")
    logger.info(f"Models bucket: {models_bucket}")

    return ml_bucket, models_bucket


def install_dependencies():
    """Install required Python packages."""
    import subprocess

    packages = [
        "boto3",
        "pandas",
        "pyarrow",
        "torch",
        "scikit-learn",
        "xgboost",
        "lightgbm",
    ]

    logger.info("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", *packages
    ])
    logger.info("Dependencies installed")


def download_training_data(
    ml_bucket: str,
    symbols: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> list[Path]:
    """Download parquet files from S3."""
    import boto3

    s3 = boto3.client("s3")
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    for symbol in symbols:
        prefix = f"features/{symbol}/"
        logger.info(f"Listing files for {symbol} from {start_date} to {end_date}...")

        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=ml_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                # Filter by date range
                # Expected format: features/BTCUSDT/2026-01-15/14/xxx.parquet
                parts = key.split("/")
                if len(parts) >= 3:
                    date_str = parts[2]
                    if start_date <= date_str <= end_date:
                        local_path = output_dir / key.replace("/", "_")

                        if not local_path.exists():
                            logger.info(f"Downloading {key}...")
                            s3.download_file(ml_bucket, key, str(local_path))

                        downloaded_files.append(local_path)

    logger.info(f"Downloaded {len(downloaded_files)} files")
    return downloaded_files


def load_data(files: list[Path]):
    """Load and combine parquet files."""
    import pandas as pd

    if not files:
        raise ValueError("No data files found")

    logger.info(f"Loading {len(files)} parquet files...")

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        raise ValueError("No valid data files loaded")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Loaded {len(combined):,} rows with {len(combined.columns)} columns")
    return combined


def train_model(
    df,
    model_type: str,
    label_type: str = "direction",
    horizon: int = 30,
    epochs: int = 100,
    batch_size: int = 256,
    sequence_length: int = 60,
    hidden_size: int = 128,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    val_fraction: float = 0.2,
):
    """Train the specified model."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Training {model_type} model...")
    logger.info(f"Label type: {label_type}, Horizon: {horizon}")

    # Prepare features - only numeric columns
    exclude_cols = ["timestamp", "symbol", "date", "hour"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Filter to numeric columns only
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    logger.info(f"Using {len(numeric_cols)} numeric features out of {len(feature_cols)} total columns")
    feature_cols = numeric_cols

    X = df[feature_cols].values.astype(np.float32)

    # Create labels
    if "mid_price" in df.columns:
        price_col = "mid_price"
    elif "price" in df.columns:
        price_col = "price"
    else:
        # Use first numeric column as price proxy
        price_col = feature_cols[0]

    prices = df[price_col].values.astype(np.float32)

    if label_type == "direction":
        # Binary classification: up (1) or down (0)
        future_returns = np.roll(prices, -horizon) / prices - 1
        y = (future_returns > 0).astype(np.int64)
    elif label_type == "regression":
        # Predict actual return
        y = (np.roll(prices, -horizon) / prices - 1).astype(np.float32)
    else:
        future_returns = np.roll(prices, -horizon) / prices - 1
        y = (future_returns > 0).astype(np.int64)

    # Remove last horizon rows (no valid labels)
    X = X[:-horizon]
    y = y[:-horizon]

    # Handle NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    logger.info(f"Training data: {X.shape[0]:,} samples, {X.shape[1]} features")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/val split (time-based)
    split_idx = int(len(X_scaled) * (1 - val_fraction))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Train based on model type
    if model_type in ["xgboost", "lightgbm", "random_forest", "logistic", "ridge"]:
        model, metrics = train_classical(model_type, X_train, y_train, X_val, y_val, label_type)
    else:
        model, metrics = train_neural(
            model_type, X_train, y_train, X_val, y_val,
            label_type, epochs, batch_size, sequence_length,
            hidden_size, num_layers, learning_rate
        )

    return model, scaler, metrics, feature_cols


def train_classical(model_type, X_train, y_train, X_val, y_val, label_type):
    """Train classical ML models."""
    from sklearn.metrics import accuracy_score, mean_squared_error

    if model_type == "xgboost":
        import xgboost as xgb
        if label_type == "regression":
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
        else:
            model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)

    elif model_type == "lightgbm":
        import lightgbm as lgb
        if label_type == "regression":
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
        else:
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if label_type == "regression":
            model = RandomForestRegressor(n_estimators=100, max_depth=10)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=10)

    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)

    elif model_type == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)

    logger.info(f"Training {model_type}...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)

    if label_type == "regression":
        mse = mean_squared_error(y_val, y_pred)
        metrics = {"mse": mse, "rmse": mse ** 0.5}
        logger.info(f"Validation RMSE: {metrics['rmse']:.6f}")
    else:
        acc = accuracy_score(y_val, y_pred)
        metrics = {"accuracy": acc}
        logger.info(f"Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    return model, metrics


def train_neural(
    model_type, X_train, y_train, X_val, y_val,
    label_type, epochs, batch_size, sequence_length,
    hidden_size, num_layers, learning_rate
):
    """Train neural network models (LSTM, GRU, Transformer)."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create sequences
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

    logger.info(f"Sequence shape: {X_train_seq.shape}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_seq).to(device)
    y_train_t = torch.FloatTensor(y_train_seq).to(device)
    X_val_t = torch.FloatTensor(X_val_seq).to(device)
    y_val_t = torch.FloatTensor(y_val_seq).to(device)

    if label_type != "regression":
        y_train_t = y_train_t.long()
        y_val_t = y_val_t.long()

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train_seq.shape[2]
    output_size = 1 if label_type == "regression" else 2

    # Create model
    if model_type == "lstm":
        model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_type == "gru":
        model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
    elif model_type == "transformer":
        model = TransformerModel(input_size, hidden_size, num_layers, output_size, sequence_length).to(device)

    # Loss and optimizer
    if label_type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)

            if label_type == "regression":
                loss = criterion(output.squeeze(), y_batch)
            else:
                loss = criterion(output, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            if label_type == "regression":
                val_loss = criterion(val_output.squeeze(), y_val_t).item()
            else:
                val_loss = criterion(val_output, y_val_t).item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        if label_type == "regression":
            mse = nn.MSELoss()(val_output.squeeze(), y_val_t).item()
            metrics = {"mse": mse, "rmse": mse ** 0.5}
            logger.info(f"Final Validation RMSE: {metrics['rmse']:.6f}")
        else:
            _, predicted = torch.max(val_output, 1)
            acc = (predicted == y_val_t).float().mean().item()
            metrics = {"accuracy": acc}
            logger.info(f"Final Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    return model, metrics


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out


def save_model(model, scaler, metrics, feature_cols, model_type, models_bucket, symbol):
    """Save model artifacts to S3."""
    import boto3
    import pickle
    import json

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"models/{symbol}/{model_type}/{timestamp}"

    s3 = boto3.client("s3")

    # Save model
    if hasattr(model, "state_dict"):
        # PyTorch model
        import torch
        model_path = f"/tmp/{model_type}_model.pt"
        torch.save(model.state_dict(), model_path)
        s3.upload_file(model_path, models_bucket, f"{prefix}/model.pt")
    else:
        # Sklearn/XGBoost model
        model_path = f"/tmp/{model_type}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        s3.upload_file(model_path, models_bucket, f"{prefix}/model.pkl")

    # Save scaler
    scaler_path = "/tmp/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    s3.upload_file(scaler_path, models_bucket, f"{prefix}/scaler.pkl")

    # Save metadata
    metadata = {
        "model_type": model_type,
        "symbol": symbol,
        "timestamp": timestamp,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "trained_on": "runpod",
    }
    metadata_path = "/tmp/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    s3.upload_file(metadata_path, models_bucket, f"{prefix}/metadata.json")

    s3_path = f"s3://{models_bucket}/{prefix}/"
    logger.info(f"Model saved to {s3_path}")

    return s3_path


def main():
    parser = argparse.ArgumentParser(description="RunPod Training for FiDel")

    # Data arguments
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Trading symbols")
    parser.add_argument("--days", type=int, default=7, help="Days of data")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")

    # Model arguments
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "random_forest", "logistic", "ridge", "lstm", "gru", "transformer"], default="xgboost")
    parser.add_argument("--label-type", choices=["direction", "regression"], default="direction")
    parser.add_argument("--horizon", type=int, default=30, help="Prediction horizon")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # Other
    parser.add_argument("--skip-upload", action="store_true", help="Don't upload model to S3")

    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 50)
    logger.info("FiDel RunPod Training")
    logger.info("=" * 50)

    # Check GPU
    check_gpu()

    # Check AWS credentials
    if not check_aws_credentials():
        sys.exit(1)

    # Get bucket names
    ml_bucket, models_bucket = get_bucket_names()

    # Calculate date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Model: {args.model}")

    # Download data
    data_dir = Path("/tmp/fidel_data")
    files = download_training_data(ml_bucket, args.symbols, start_date, end_date, data_dir)

    if not files:
        logger.error("No data files found!")
        sys.exit(1)

    # Load data
    df = load_data(files)

    # Train model
    model, scaler, metrics, feature_cols = train_model(
        df,
        model_type=args.model,
        label_type=args.label_type,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
    )

    # Save model
    if not args.skip_upload:
        s3_path = save_model(model, scaler, metrics, feature_cols, args.model, models_bucket, args.symbols[0])
        logger.info(f"Model saved to: {s3_path}")

    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Training completed in {elapsed/60:.1f} minutes")
    logger.info(f"Metrics: {metrics}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
