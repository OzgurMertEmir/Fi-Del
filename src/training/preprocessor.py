"""
Feature preprocessor for training pipeline.

Handles:
- Label generation (direction, regression, volatility)
- Feature scaling (standard, minmax, robust)
- Sequence creation for sequential models (LSTM, Transformer)
- Train/validation splitting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

if TYPE_CHECKING:
    from training.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedData:
    """Container for processed training data."""

    # Features and labels
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray

    # Metadata
    feature_names: list[str]
    n_features: int
    n_classes: int | None  # None for regression

    # For sequential models
    sequence_length: int | None = None

    # Scaler for inference
    scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None

    # Original timestamps for analysis
    train_timestamps: np.ndarray | None = None
    val_timestamps: np.ndarray | None = None


class FeaturePreprocessor:
    """Preprocess features for ML training."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = None
        self.feature_names = []

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate prediction labels based on config."""
        from training.config import LabelType

        horizon = self.config.label_horizon

        if self.config.label_type == LabelType.DIRECTION:
            # Binary: 1 if price goes up, 0 if down
            future_return = df["price"].shift(-horizon) / df["price"] - 1
            df["label"] = (future_return > self.config.label_threshold).astype(int)
            logger.info(f"Generated direction labels (horizon={horizon})")

        elif self.config.label_type == LabelType.DIRECTION_3:
            # 3-class: 0=down, 1=flat, 2=up
            future_return = df["price"].shift(-horizon) / df["price"] - 1
            df["label"] = np.where(
                future_return > self.config.label_threshold,
                2,  # Up
                np.where(future_return < -self.config.label_threshold, 0, 1),  # Down or Flat
            )
            logger.info(f"Generated 3-class direction labels (horizon={horizon})")

        elif self.config.label_type == LabelType.REGRESSION:
            # Continuous return
            df["label"] = df["price"].shift(-horizon) / df["price"] - 1
            logger.info(f"Generated regression labels (horizon={horizon})")

        elif self.config.label_type == LabelType.VOLATILITY:
            # Future volatility (rolling std of returns)
            future_vol = df["return_1"].shift(-horizon).rolling(horizon).std()
            df["label"] = future_vol
            logger.info(f"Generated volatility labels (horizon={horizon})")

        return df

    def select_features(self, df: pd.DataFrame) -> list[str]:
        """Select feature columns for training."""
        if self.config.feature_cols:
            # Use explicitly specified features
            features = [c for c in self.config.feature_cols if c in df.columns]
        else:
            # Use all numeric columns except excluded ones
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [c for c in numeric_cols if c not in self.config.exclude_cols and c != "label"]

        logger.info(f"Selected {len(features)} features: {features[:10]}...")
        return features

    def create_scaler(self) -> StandardScaler | MinMaxScaler | RobustScaler | None:
        """Create scaler based on config."""
        from training.config import ScalerType

        if self.config.scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        elif self.config.scaler_type == ScalerType.MINMAX:
            return MinMaxScaler()
        elif self.config.scaler_type == ScalerType.ROBUST:
            return RobustScaler()
        return None

    def split_data(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation sets."""
        # Remove rows with NaN labels
        df_clean = df.dropna(subset=["label"] + feature_cols)
        logger.info(f"After dropping NaN: {len(df_clean)} rows (dropped {len(df) - len(df_clean)})")

        if self.config.time_based_split:
            # Time-based split: earlier data for training, later for validation
            split_idx = int(len(df_clean) * (1 - self.config.val_fraction))
            train_df = df_clean.iloc[:split_idx]
            val_df = df_clean.iloc[split_idx:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split

            train_df, val_df = train_test_split(
                df_clean,
                test_size=self.config.val_fraction,
                random_state=42,
            )

        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        y_train = train_df["label"].values
        y_val = val_df["label"].values

        # Get timestamps if available
        if "timestamp" in train_df.columns:
            train_ts = train_df["timestamp"].values
            val_ts = val_df["timestamp"].values
        else:
            train_ts = np.arange(len(train_df))
            val_ts = np.arange(len(val_df))

        logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
        return X_train, X_val, y_train, y_val, train_ts, val_ts

    def process(self, df: pd.DataFrame) -> ProcessedData:
        """Full preprocessing pipeline."""
        from training.config import LabelType

        # Generate labels
        df = self.generate_labels(df)

        # Select features
        self.feature_names = self.select_features(df)

        # Split data
        X_train, X_val, y_train, y_val, train_ts, val_ts = self.split_data(df, self.feature_names)

        # Scale features
        self.scaler = self.create_scaler()
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            logger.info(f"Applied {self.config.scaler_type.value} scaling")

        # Determine number of classes
        if self.config.label_type in [LabelType.DIRECTION]:
            n_classes = 2
        elif self.config.label_type == LabelType.DIRECTION_3:
            n_classes = 3
        else:
            n_classes = None  # Regression

        return ProcessedData(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            feature_names=self.feature_names,
            n_features=len(self.feature_names),
            n_classes=n_classes,
            scaler=self.scaler,
            train_timestamps=train_ts,
            val_timestamps=val_ts,
        )


class SequencePreprocessor(FeaturePreprocessor):
    """Preprocessor that creates sequences for LSTM/Transformer models."""

    def __init__(self, config: DataConfig, sequence_length: int = 60):
        super().__init__(config)
        self.sequence_length = sequence_length

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert flat features into sequences for sequential models.

        Input shape: (n_samples, n_features)
        Output shape: (n_samples - seq_len, seq_len, n_features)
        """
        sequences = []
        labels = []

        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i - self.sequence_length : i])
            labels.append(y[i])

        return np.array(sequences), np.array(labels)

    def process(self, df: pd.DataFrame) -> ProcessedData:
        """Process with sequence creation."""
        # First do standard preprocessing
        data = super().process(df)

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(data.X_train, data.y_train)
        X_val_seq, y_val_seq = self.create_sequences(data.X_val, data.y_val)

        logger.info(
            f"Created sequences: train={X_train_seq.shape}, val={X_val_seq.shape}"
        )

        return ProcessedData(
            X_train=X_train_seq,
            X_val=X_val_seq,
            y_train=y_train_seq,
            y_val=y_val_seq,
            feature_names=data.feature_names,
            n_features=data.n_features,
            n_classes=data.n_classes,
            sequence_length=self.sequence_length,
            scaler=data.scaler,
        )


def create_preprocessor(
    config: DataConfig, sequence_length: int | None = None
) -> FeaturePreprocessor | SequencePreprocessor:
    """Factory function to create appropriate preprocessor."""
    if sequence_length is not None:
        return SequencePreprocessor(config, sequence_length)
    return FeaturePreprocessor(config)
