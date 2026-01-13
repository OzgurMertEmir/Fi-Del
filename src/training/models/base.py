"""
Base model interface for training pipeline.

All models inherit from BaseModel and implement the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, n_features: int, n_classes: int | None = 2, **kwargs):
        """Initialize model.

        Args:
            n_features: Number of input features
            n_classes: Number of output classes (None for regression)
            **kwargs: Model-specific parameters
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.is_classification = n_classes is not None
        self.model = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted labels or values
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Predict class probabilities (for classification).

        Args:
            X: Input features

        Returns:
            Class probabilities or None if not applicable
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from
        """
        pass

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance scores if available.

        Returns:
            Array of importance scores or None
        """
        return None

    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "is_classification": self.is_classification,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_features={self.n_features}, n_classes={self.n_classes})"
