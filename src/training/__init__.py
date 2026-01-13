"""
FiDel Training Pipeline

A modular ML training framework supporting multiple model types:
- Classical: Logistic Regression, XGBoost, LightGBM
- Sequential: LSTM, GRU, Transformer

Usage:
    from training import TrainingPipeline, TrainingConfig

    config = TrainingConfig(
        model_type="xgboost",
        symbols=["BTCUSDT"],
        label_type="direction",  # or "regression"
    )

    pipeline = TrainingPipeline(config)
    pipeline.run()
"""

from training.config import TrainingConfig, DataConfig, ModelConfig
from training.data_loader import S3DataLoader
from training.preprocessor import FeaturePreprocessor
from training.trainer import Trainer
from training.pipeline import TrainingPipeline

__all__ = [
    "TrainingConfig",
    "DataConfig",
    "ModelConfig",
    "S3DataLoader",
    "FeaturePreprocessor",
    "Trainer",
    "TrainingPipeline",
]
