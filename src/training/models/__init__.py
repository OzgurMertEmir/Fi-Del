"""
Model registry for training pipeline.

Provides a unified interface to create different model types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from training.models.base import BaseModel
from training.models.classical import (
    LightGBMModel,
    LogisticModel,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
)
from training.models.sequential import GRUModel, LSTMModel, TransformerModel

if TYPE_CHECKING:
    from training.config import ModelConfig, ModelType

# Model registry
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "logistic": LogisticModel,
    "ridge": RidgeModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "random_forest": RandomForestModel,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "transformer": TransformerModel,
}

# Models that require sequences
SEQUENTIAL_MODELS = {"lstm", "gru", "transformer"}


def create_model(config: ModelConfig, n_features: int, n_classes: int | None = 2) -> BaseModel:
    """Factory function to create a model from config.

    Args:
        config: Model configuration
        n_features: Number of input features
        n_classes: Number of output classes (None for regression)

    Returns:
        Initialized model
    """
    model_type = config.model_type.value

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_type]

    # Sequential models need additional parameters
    if model_type in SEQUENTIAL_MODELS:
        return model_class(
            n_features=n_features,
            n_classes=n_classes,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_heads=config.num_heads,
            sequence_length=config.sequence_length,
        )

    # Classical models
    params = config.get_classical_params()
    return model_class(n_features=n_features, n_classes=n_classes, **params)


def is_sequential_model(model_type: str | ModelType) -> bool:
    """Check if model type requires sequence input."""
    if hasattr(model_type, "value"):
        model_type = model_type.value
    return model_type in SEQUENTIAL_MODELS


__all__ = [
    "BaseModel",
    "LogisticModel",
    "RidgeModel",
    "XGBoostModel",
    "LightGBMModel",
    "RandomForestModel",
    "LSTMModel",
    "GRUModel",
    "TransformerModel",
    "create_model",
    "is_sequential_model",
    "MODEL_REGISTRY",
    "SEQUENTIAL_MODELS",
]
