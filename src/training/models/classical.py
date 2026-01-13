"""
Classical ML models (sklearn-compatible).

Includes: Logistic Regression, Ridge, XGBoost, LightGBM, Random Forest
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from training.models.base import BaseModel

logger = logging.getLogger(__name__)


class LogisticModel(BaseModel):
    """Logistic Regression for classification."""

    def __init__(self, n_features: int, n_classes: int | None = 2, **kwargs):
        super().__init__(n_features, n_classes)
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(**kwargs)
        self.params = kwargs

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        logger.info("Training Logistic Regression...")
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        history = {"train_score": self.model.score(X_train, y_train)}
        if X_val is not None:
            history["val_score"] = self.model.score(X_val, y_val)

        logger.info(f"Training complete. Score: {history}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        self._is_fitted = True

    def get_feature_importance(self) -> np.ndarray:
        return np.abs(self.model.coef_[0])


class RidgeModel(BaseModel):
    """Ridge Regression for continuous prediction."""

    def __init__(self, n_features: int, n_classes: int | None = None, **kwargs):
        super().__init__(n_features, None)  # Always regression
        from sklearn.linear_model import Ridge

        self.model = Ridge(**kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        logger.info("Training Ridge Regression...")
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        history = {"train_r2": self.model.score(X_train, y_train)}
        if X_val is not None:
            history["val_r2"] = self.model.score(X_val, y_val)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> None:
        return None  # Regression doesn't have probabilities

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        self._is_fitted = True

    def get_feature_importance(self) -> np.ndarray:
        return np.abs(self.model.coef_)


class XGBoostModel(BaseModel):
    """XGBoost for classification or regression."""

    def __init__(self, n_features: int, n_classes: int | None = 2, **kwargs):
        super().__init__(n_features, n_classes)

        # Adjust objective based on task
        if self.is_classification:
            if n_classes == 2:
                kwargs.setdefault("objective", "binary:logistic")
            else:
                kwargs.setdefault("objective", "multi:softprob")
                kwargs["num_class"] = n_classes
        else:
            kwargs.setdefault("objective", "reg:squarederror")

        self.params = kwargs
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        import xgboost as xgb

        logger.info("Training XGBoost...")

        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, "train")]
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        # Extract training params
        num_rounds = self.params.pop("n_estimators", 100)
        early_stopping = self.params.pop("early_stopping_rounds", None)

        # Train
        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        self._is_fitted = True

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        return evals_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)

        if self.is_classification and self.n_classes == 2:
            return (preds > 0.5).astype(int)
        elif self.is_classification:
            return np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self.is_classification:
            return None

        import xgboost as xgb

        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)

        if self.n_classes == 2:
            return np.column_stack([1 - preds, preds])
        return preds

    def save(self, path: str | Path) -> None:
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        import xgboost as xgb

        self.model = xgb.Booster()
        self.model.load_model(str(path))
        self._is_fitted = True

    def get_feature_importance(self) -> np.ndarray:
        importance = self.model.get_score(importance_type="gain")
        # Convert to array
        n_features = self.n_features
        result = np.zeros(n_features)
        for k, v in importance.items():
            idx = int(k.replace("f", ""))
            if idx < n_features:
                result[idx] = v
        return result


class LightGBMModel(BaseModel):
    """LightGBM for classification or regression."""

    def __init__(self, n_features: int, n_classes: int | None = 2, **kwargs):
        super().__init__(n_features, n_classes)

        if self.is_classification:
            kwargs.setdefault("objective", "binary" if n_classes == 2 else "multiclass")
            if n_classes and n_classes > 2:
                kwargs["num_class"] = n_classes
        else:
            kwargs.setdefault("objective", "regression")

        kwargs.setdefault("verbose", -1)
        self.params = kwargs
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        import lightgbm as lgb

        logger.info("Training LightGBM...")

        dtrain = lgb.Dataset(X_train, label=y_train)
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(dval)
            valid_names.append("val")

        num_rounds = self.params.pop("n_estimators", 100)

        evals_result = {}
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.record_evaluation(evals_result)],
        )
        self._is_fitted = True

        logger.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        return evals_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)

        if self.is_classification:
            if self.n_classes == 2:
                return (preds > 0.5).astype(int)
            return np.argmax(preds, axis=1)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self.is_classification:
            return None

        preds = self.model.predict(X)
        if self.n_classes == 2:
            return np.column_stack([1 - preds, preds])
        return preds

    def save(self, path: str | Path) -> None:
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(path))
        self._is_fitted = True

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importance(importance_type="gain")


class RandomForestModel(BaseModel):
    """Random Forest for classification or regression."""

    def __init__(self, n_features: int, n_classes: int | None = 2, **kwargs):
        super().__init__(n_features, n_classes)

        if self.is_classification:
            from sklearn.ensemble import RandomForestClassifier

            self.model = RandomForestClassifier(**kwargs)
        else:
            from sklearn.ensemble import RandomForestRegressor

            self.model = RandomForestRegressor(**kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        logger.info("Training Random Forest...")
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        history = {"train_score": self.model.score(X_train, y_train)}
        if X_val is not None:
            history["val_score"] = self.model.score(X_val, y_val)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not self.is_classification:
            return None
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        self._is_fitted = True

    def get_feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_
