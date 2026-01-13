"""
Trainer class for model training and evaluation.

Handles:
- Training orchestration
- Evaluation metrics computation
- Model checkpointing
- Experiment logging
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from training.config import TrainingConfig
    from training.models.base import BaseModel
    from training.preprocessor import ProcessedData

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    # Classification metrics
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc_roc: float | None = None
    log_loss: float | None = None

    # Regression metrics
    mse: float | None = None
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None

    # Trading-specific metrics
    directional_accuracy: float | None = None  # % of correct direction predictions
    sharpe_proxy: float | None = None  # Simple Sharpe approximation

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class Trainer:
    """Orchestrates model training and evaluation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model: BaseModel | None = None
        self.train_history: dict[str, Any] = {}
        self.eval_metrics: EvaluationMetrics | None = None

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> EvaluationMetrics:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics = EvaluationMetrics()

        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics.f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    metrics.auc_roc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics.auc_roc = roc_auc_score(y_true, y_proba, multi_class="ovr")
                metrics.log_loss = log_loss(y_true, y_proba)
            except ValueError:
                pass  # AUC not defined for single class

        return metrics

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> EvaluationMetrics:
        """Compute regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = EvaluationMetrics()

        metrics.mse = mean_squared_error(y_true, y_pred)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.mae = mean_absolute_error(y_true, y_pred)
        metrics.r2 = r2_score(y_true, y_pred)

        # Directional accuracy (did we predict the sign correctly?)
        actual_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        metrics.directional_accuracy = np.mean(actual_direction == pred_direction)

        return metrics

    def evaluate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> EvaluationMetrics:
        """Evaluate model performance."""
        y_pred = model.predict(X)

        if model.is_classification:
            y_proba = model.predict_proba(X)
            metrics = self.evaluate_classification(y, y_pred, y_proba)
        else:
            metrics = self.evaluate_regression(y, y_pred)

        return metrics

    def train(
        self,
        model: BaseModel,
        data: ProcessedData,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Train model on processed data.

        Args:
            model: Model to train
            data: Processed training data

        Returns:
            Tuple of (trained model, training history)
        """
        self.model = model

        # Training parameters
        train_kwargs = {
            "epochs": self.config.model.epochs,
            "batch_size": self.config.model.batch_size,
            "learning_rate": self.config.model.learning_rate,
            "early_stopping_patience": self.config.model.early_stopping_patience,
            "weight_decay": self.config.model.weight_decay,
        }

        # Train
        logger.info(f"Starting training: {model}")
        self.train_history = model.fit(
            data.X_train,
            data.y_train,
            data.X_val,
            data.y_val,
            **train_kwargs,
        )

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        self.eval_metrics = self.evaluate(model, data.X_val, data.y_val)

        logger.info(f"Validation metrics: {self.eval_metrics.to_dict()}")

        return model, self.train_history

    def get_feature_importance(
        self,
        feature_names: list[str],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Get top feature importances."""
        if self.model is None:
            return []

        importance = self.model.get_feature_importance()
        if importance is None:
            return []

        # Pair with names and sort
        pairs = list(zip(feature_names, importance))
        pairs.sort(key=lambda x: x[1], reverse=True)

        return pairs[:top_n]

    def save_results(
        self,
        output_dir: str | Path,
        data: ProcessedData,
    ) -> dict[str, str]:
        """Save model and training results to disk.

        Args:
            output_dir: Directory to save results
            data: Processed data (for saving scaler)

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save model
        model_path = output_dir / "model.joblib"
        if self.model:
            self.model.save(model_path)
            saved_files["model"] = str(model_path)
            logger.info(f"Saved model to {model_path}")

        # Save scaler
        if data.scaler:
            import joblib

            scaler_path = output_dir / "scaler.joblib"
            joblib.dump(data.scaler, scaler_path)
            saved_files["scaler"] = str(scaler_path)

        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        saved_files["config"] = str(config_path)

        # Save metrics
        if self.eval_metrics:
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "validation": self.eval_metrics.to_dict(),
                        "training_history": {
                            k: [float(x) for x in v] if isinstance(v, list) else v
                            for k, v in self.train_history.items()
                        },
                    },
                    f,
                    indent=2,
                )
            saved_files["metrics"] = str(metrics_path)

        # Save feature importance
        if self.model:
            importance = self.get_feature_importance(data.feature_names)
            if importance:
                importance_path = output_dir / "feature_importance.json"
                with open(importance_path, "w") as f:
                    json.dump(importance, f, indent=2)
                saved_files["feature_importance"] = str(importance_path)

        # Save feature names
        features_path = output_dir / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump(data.feature_names, f)
        saved_files["feature_names"] = str(features_path)

        return saved_files

    def upload_to_s3(
        self,
        local_dir: str | Path,
        s3_prefix: str | None = None,
    ) -> str:
        """Upload training results to S3.

        Args:
            local_dir: Local directory with results
            s3_prefix: S3 prefix (default: models/{experiment_name}/{run_name}/)

        Returns:
            S3 URI of uploaded results
        """
        import boto3

        if not self.config.output_bucket:
            logger.warning("No output bucket configured, skipping S3 upload")
            return ""

        s3 = boto3.client("s3")
        local_dir = Path(local_dir)

        if s3_prefix is None:
            s3_prefix = f"{self.config.output_prefix}{self.config.experiment_name}/{self.config.run_name}/"

        # Upload all files in directory
        for file_path in local_dir.glob("*"):
            if file_path.is_file():
                key = f"{s3_prefix}{file_path.name}"
                s3.upload_file(str(file_path), self.config.output_bucket, key)
                logger.info(f"Uploaded {file_path.name} to s3://{self.config.output_bucket}/{key}")

        s3_uri = f"s3://{self.config.output_bucket}/{s3_prefix}"
        logger.info(f"Results uploaded to {s3_uri}")
        return s3_uri
