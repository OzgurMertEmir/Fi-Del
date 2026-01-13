"""
Training configuration classes.

Defines all configurable parameters for the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ModelType(str, Enum):
    """Supported model types."""

    # Classical models (sklearn-compatible)
    LOGISTIC = "logistic"
    RIDGE = "ridge"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"

    # Sequential models (PyTorch)
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"


class LabelType(str, Enum):
    """Type of prediction target."""

    DIRECTION = "direction"  # Up/Down classification
    DIRECTION_3 = "direction_3"  # Up/Flat/Down (3-class)
    REGRESSION = "regression"  # Continuous return prediction
    VOLATILITY = "volatility"  # Volatility prediction


class ScalerType(str, Enum):
    """Feature scaling methods."""

    NONE = "none"
    STANDARD = "standard"  # Zero mean, unit variance
    MINMAX = "minmax"  # Scale to [0, 1]
    ROBUST = "robust"  # Median and IQR based


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # S3 settings
    bucket: str = ""  # Will be filled from environment
    prefix: str = "features/"

    # Symbol selection
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])

    # Time range
    start_date: str | None = None  # Format: "2026-01-01"
    end_date: str | None = None
    days_back: int = 7  # If no dates specified, use last N days

    # Sampling
    sample_rate: float = 1.0  # 1.0 = all data, 0.1 = 10% sample
    max_rows: int | None = None  # Cap total rows

    # Train/validation split
    val_fraction: float = 0.2
    time_based_split: bool = True  # Split by time, not random

    # Label generation
    label_type: LabelType = LabelType.DIRECTION
    label_horizon: int = 30  # Predict N steps ahead
    label_threshold: float = 0.0001  # For direction: min return to count as up/down

    # Feature selection
    feature_cols: list[str] | None = None  # None = use all numeric
    exclude_cols: list[str] = field(
        default_factory=lambda: [
            "symbol",
            "timestamp",
            "timestamp_dt",
            "processed_at",
        ]
    )

    # Scaling
    scaler_type: ScalerType = ScalerType.STANDARD


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""

    model_type: ModelType = ModelType.XGBOOST

    # Classical model hyperparameters
    classical_params: dict[str, Any] = field(default_factory=dict)

    # Sequential model hyperparameters
    sequence_length: int = 60  # Lookback window for LSTM/Transformer
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    num_heads: int = 4  # For Transformer

    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10

    # Regularization
    weight_decay: float = 0.0001

    def get_classical_params(self) -> dict[str, Any]:
        """Get model-specific default parameters."""
        defaults = {
            ModelType.LOGISTIC: {
                "C": 1.0,
                "max_iter": 1000,
                "class_weight": "balanced",
            },
            ModelType.RIDGE: {
                "alpha": 1.0,
            },
            ModelType.XGBOOST: {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "early_stopping_rounds": 10,
            },
            ModelType.LIGHTGBM: {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary",
                "metric": "binary_logloss",
            },
            ModelType.RANDOM_FOREST: {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "class_weight": "balanced",
            },
        }

        base_params = defaults.get(self.model_type, {})
        base_params.update(self.classical_params)
        return base_params


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Output settings
    output_bucket: str = ""  # S3 bucket for models
    output_prefix: str = "models/"
    local_output_dir: str = "artifacts/training"

    # Experiment tracking
    experiment_name: str = ""
    run_name: str = ""

    # Misc
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    verbose: bool = True

    def __post_init__(self):
        """Set defaults based on other fields."""
        if not self.experiment_name:
            self.experiment_name = f"{self.model.model_type.value}_{self.data.label_type.value}"

        if not self.run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols = "_".join(self.data.symbols[:2])
            self.run_name = f"{symbols}_{timestamp}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Create config from dictionary."""
        data_cfg = DataConfig(**d.get("data", {}))
        model_cfg = ModelConfig(**d.get("model", {}))

        return cls(
            data=data_cfg,
            model=model_cfg,
            output_bucket=d.get("output_bucket", ""),
            output_prefix=d.get("output_prefix", "models/"),
            experiment_name=d.get("experiment_name", ""),
            run_name=d.get("run_name", ""),
            seed=d.get("seed", 42),
            device=d.get("device", "auto"),
            verbose=d.get("verbose", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "data": {
                "bucket": self.data.bucket,
                "symbols": self.data.symbols,
                "start_date": self.data.start_date,
                "end_date": self.data.end_date,
                "label_type": self.data.label_type.value,
                "label_horizon": self.data.label_horizon,
                "val_fraction": self.data.val_fraction,
                "scaler_type": self.data.scaler_type.value,
            },
            "model": {
                "model_type": self.model.model_type.value,
                "sequence_length": self.model.sequence_length,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "learning_rate": self.model.learning_rate,
                "batch_size": self.model.batch_size,
                "epochs": self.model.epochs,
            },
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "seed": self.seed,
        }
