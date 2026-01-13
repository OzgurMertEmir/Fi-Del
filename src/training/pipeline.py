"""
Training pipeline that orchestrates the full training workflow.

Usage:
    from training import TrainingPipeline, TrainingConfig

    config = TrainingConfig(...)
    pipeline = TrainingPipeline(config)
    results = pipeline.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from training.config import TrainingConfig
from training.data_loader import create_data_loader
from training.models import create_model, is_sequential_model
from training.preprocessor import create_preprocessor
from training.trainer import Trainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_loader = None
        self.preprocessor = None
        self.trainer = None
        self.model = None
        self.processed_data = None

    def setup(self) -> None:
        """Initialize pipeline components."""
        logger.info("Setting up training pipeline...")

        # Configure logging
        if self.config.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Set random seeds
        import numpy as np

        np.random.seed(self.config.seed)

        try:
            import torch

            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
        except ImportError:
            pass

        # Initialize data loader
        self.data_loader = create_data_loader(self.config.data)

        # Initialize preprocessor
        if is_sequential_model(self.config.model.model_type):
            self.preprocessor = create_preprocessor(
                self.config.data,
                sequence_length=self.config.model.sequence_length,
            )
        else:
            self.preprocessor = create_preprocessor(self.config.data)

        # Initialize trainer
        self.trainer = Trainer(self.config)

        logger.info("Pipeline setup complete")

    def load_data(self):
        """Load and preprocess data."""
        logger.info("Loading data...")

        # Load from S3
        df = self.data_loader.load_all()
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Preprocess
        self.processed_data = self.preprocessor.process(df)
        logger.info(
            f"Preprocessed: {self.processed_data.n_features} features, "
            f"{self.processed_data.n_classes} classes"
        )

        return self.processed_data

    def create_model(self):
        """Create model instance."""
        logger.info(f"Creating model: {self.config.model.model_type.value}")

        self.model = create_model(
            self.config.model,
            n_features=self.processed_data.n_features,
            n_classes=self.processed_data.n_classes,
        )

        return self.model

    def train(self):
        """Train the model."""
        logger.info("Starting training...")

        self.model, history = self.trainer.train(self.model, self.processed_data)

        return self.model, history

    def save(self) -> dict[str, str]:
        """Save results locally and to S3."""
        # Local save
        local_dir = Path(self.config.local_output_dir) / self.config.experiment_name / self.config.run_name
        saved_files = self.trainer.save_results(local_dir, self.processed_data)

        # S3 upload
        if self.config.output_bucket:
            s3_uri = self.trainer.upload_to_s3(local_dir)
            saved_files["s3_uri"] = s3_uri

        return saved_files

    def run(self) -> dict[str, Any]:
        """Execute full training pipeline.

        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info(f"Starting training pipeline: {self.config.experiment_name}")
        logger.info(f"Run name: {self.config.run_name}")
        logger.info("=" * 60)

        # Setup
        self.setup()

        # Load and preprocess data
        self.load_data()

        # Create model
        self.create_model()

        # Train
        self.train()

        # Save results
        saved_files = self.save()

        # Summary
        results = {
            "experiment_name": self.config.experiment_name,
            "run_name": self.config.run_name,
            "model_type": self.config.model.model_type.value,
            "n_features": self.processed_data.n_features,
            "n_train": len(self.processed_data.X_train),
            "n_val": len(self.processed_data.X_val),
            "metrics": self.trainer.eval_metrics.to_dict() if self.trainer.eval_metrics else {},
            "saved_files": saved_files,
        }

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Validation metrics: {results['metrics']}")
        logger.info("=" * 60)

        return results


def run_quick_experiment(
    symbols: list[str] | None = None,
    model_type: str = "xgboost",
    days_back: int = 3,
    label_horizon: int = 30,
) -> dict[str, Any]:
    """Run a quick experiment with sensible defaults.

    Args:
        symbols: List of symbols (default: ["BTCUSDT"])
        model_type: Model type (default: "xgboost")
        days_back: Days of data to use (default: 3)
        label_horizon: Prediction horizon (default: 30)

    Returns:
        Training results dictionary
    """
    from training.config import DataConfig, LabelType, ModelConfig, ModelType, TrainingConfig

    if symbols is None:
        symbols = ["BTCUSDT"]

    config = TrainingConfig(
        data=DataConfig(
            symbols=symbols,
            days_back=days_back,
            label_horizon=label_horizon,
            label_type=LabelType.DIRECTION,
            val_fraction=0.2,
        ),
        model=ModelConfig(
            model_type=ModelType(model_type),
            epochs=50,
            batch_size=256,
        ),
    )

    pipeline = TrainingPipeline(config)
    return pipeline.run()
