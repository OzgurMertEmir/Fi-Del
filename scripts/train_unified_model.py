#!/usr/bin/env python
# ruff: noqa: E402
"""
train_unified_model.py

Train and evaluate models on unified multi-source datasets.

This script loads datasets built by build_unified_dataset.py and trains
classifiers to predict fills or price direction.

Example:
    python scripts/train_unified_model.py --symbol BTCUSDT
    python scripts/train_unified_model.py --all-symbols
    python scripts/train_unified_model.py --symbol BTCUSDT --model xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from myproj.ml.models import train_price_move_model, evaluate_classifier


def load_dataset(unified_dir: Path, symbol: str) -> dict:
    """Load a pre-built unified dataset."""
    arrays_path = unified_dir / f"unified_{symbol}_arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Dataset not found: {arrays_path}")

    data = np.load(arrays_path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "X_val": data["X_val"],
        "y_train": data["y_train"],
        "y_val": data["y_val"],
        "feature_cols": data["feature_cols"].tolist(),
        "timestamps_train": data["timestamps_train"],
        "timestamps_val": data["timestamps_val"],
    }


def get_available_symbols(unified_dir: Path) -> list[str]:
    """Get list of symbols with built datasets."""
    symbols = []
    for path in unified_dir.glob("unified_*_arrays.npz"):
        symbol = path.stem.replace("unified_", "").replace("_arrays", "")
        symbols.append(symbol)
    return sorted(symbols)


def train_and_evaluate(
    unified_dir: Path,
    symbol: str,
    model_type: str = "logistic",
    output_dir: Path = None,
) -> dict:
    """Train and evaluate a model on unified data.

    Args:
        unified_dir: Directory with built datasets
        symbol: Symbol to train on
        model_type: Model type ("logistic", "xgboost", "random_forest")
        output_dir: Directory to save results

    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print(f"Training model for {symbol}")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset(unified_dir, symbol)
    X_train = dataset["X_train"]
    X_val = dataset["X_val"]
    y_train = dataset["y_train"]
    y_val = dataset["y_val"]
    feature_cols = dataset["feature_cols"]

    print(f"Train: {len(X_train)} samples, {len(feature_cols)} features")
    print(f"Val: {len(X_val)} samples")
    print(f"Label distribution (train): {np.bincount(y_train.astype(int))}")

    # Check for trivial case (all same label)
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        print(f"Warning: Only one label class ({unique_labels[0]}) in training data. Skipping.")
        return {
            "symbol": symbol,
            "error": "single_class",
            "train_size": len(X_train),
            "val_size": len(X_val),
        }

    # Train model
    print(f"\nTraining {model_type} model...")

    if model_type == "logistic":
        model = train_price_move_model(X_train, y_train)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=1,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train)
        except ImportError:
            print("XGBoost not installed, falling back to logistic regression")
            model = train_price_move_model(X_train, y_train)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate
    print("\nEvaluating on validation set...")
    metrics = evaluate_classifier(model, X_val, y_val)

    print("\nValidation Results:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    # Feature importance
    importance = get_feature_importance(model, feature_cols)
    if importance:
        print("\nTop 10 Features:")
        for name, imp in importance[:10]:
            print(f"  {name}: {imp:.4f}")

    # Save results
    results = {
        "symbol": symbol,
        "model_type": model_type,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "n_features": len(feature_cols),
        "metrics": metrics,
        "top_features": importance[:20] if importance else [],
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"model_results_{symbol}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

    return results


def get_feature_importance(model, feature_cols: list[str]) -> list[tuple[str, float]]:
    """Extract feature importance from model."""
    importance = []

    # Try different methods to get feature importance
    if hasattr(model, "feature_importances_"):
        # XGBoost, Random Forest
        imp = model.feature_importances_
        importance = list(zip(feature_cols, imp))
    elif hasattr(model, "named_steps"):
        # Pipeline (e.g., StandardScaler + LogisticRegression)
        lr = model.named_steps.get("logisticregression")
        if lr is not None and hasattr(lr, "coef_"):
            coef = np.abs(lr.coef_.ravel())
            importance = list(zip(feature_cols, coef))
    elif hasattr(model, "coef_"):
        # Direct LogisticRegression
        coef = np.abs(model.coef_.ravel())
        importance = list(zip(feature_cols, coef))

    # Sort by importance
    if importance:
        importance = sorted(importance, key=lambda x: -x[1])

    return importance


def main():
    parser = argparse.ArgumentParser(description="Train models on unified datasets")
    parser.add_argument(
        "--unified-dir",
        default="data/unified",
        help="Directory with built datasets",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol to train on",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Train on all available symbols",
    )
    parser.add_argument(
        "--model",
        default="logistic",
        choices=["logistic", "xgboost", "random_forest"],
        help="Model type (default: logistic)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/unified_models",
        help="Output directory for results",
    )
    args = parser.parse_args()

    unified_dir = Path(args.unified_dir)

    if not unified_dir.exists():
        print(f"Error: Unified data directory not found: {unified_dir}")
        print("Run build_unified_dataset.py first to create datasets.")
        return

    # Get symbols
    available = get_available_symbols(unified_dir)
    print(f"Available symbols: {available}")

    if args.all_symbols:
        symbols = available
    elif args.symbol:
        if args.symbol not in available:
            print(f"Error: Symbol {args.symbol} not found in unified datasets")
            return
        symbols = [args.symbol]
    else:
        # Default to first symbol
        symbols = [available[0]] if available else []

    if not symbols:
        print("No symbols to process")
        return

    # Train models
    all_results = {}
    for symbol in symbols:
        try:
            results = train_and_evaluate(
                unified_dir,
                symbol,
                model_type=args.model,
                output_dir=args.output_dir,
            )
            all_results[symbol] = results
        except Exception as e:
            print(f"Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Symbol':<12} {'Train':>8} {'Val':>6} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 60)

    for symbol, res in all_results.items():
        if "error" in res:
            print(f"{symbol:<12} {res['train_size']:>8} {res['val_size']:>6} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
        else:
            m = res["metrics"]
            print(f"{symbol:<12} {res['train_size']:>8} {res['val_size']:>6} "
                  f"{m['accuracy']:>8.2%} {m['f1']:>8.2%} {m.get('auc', 0):>8.2%}")

    # Save overall summary
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nOverall summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
