"""Simple baseline models for the prototype pipelines."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_price_move_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a multinomial logistic regression baseline."""
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            n_jobs=4,
            class_weight="balanced",
        ),
    )
    model.fit(X_train, y_train)
    return model


def train_execution_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a binary logistic regression for execution fill probability."""
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            max_iter=1000,
            n_jobs=4,
            class_weight="balanced",
        ),
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """Compute simple metrics for classification tasks."""
    y_pred = model.predict(X_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, average="weighted")),
    }
    # AUC only for binary labels
    try:
        if len(np.unique(y_val)) == 2:
            y_prob = model.predict_proba(X_val)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_val, y_prob))
    except Exception:
        pass
    return metrics
