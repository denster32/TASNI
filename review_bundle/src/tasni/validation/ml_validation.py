#!/usr/bin/env python3
"""
ML Validation Module for TASNI

Provides proper machine learning validation without circular labels.
Uses the expanded brown dwarf catalog as ground truth for:
- Cross-validation with held-out test sets
- Model comparison (XGBoost, LightGBM, RandomForest)
- Feature importance analysis
- Calibration curves

This replaces the problematic circular validation in ml_scoring.py:
  # OLD (circular):
  y_proxy = (df[existing_score_col] > df[existing_score_col].quantile(0.8)).astype(int)

  # NEW (proper):
  y = ground_truth_labels_from_known_bds
"""

import logging
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from tasni.core.config import OUTPUT_DIR

from .expanded_bd_catalog import load_expanded_brown_dwarf_catalog
from .rigorous_validation import RigorousValidator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class MLValidator:
    """
    ML Model Validation with proper ground truth.

    This class provides:
    1. Proper train/test splits using known brown dwarfs
    2. Model comparison across multiple algorithms
    3. Feature importance analysis
    4. Calibration assessment
    """

    def __init__(
        self, match_radius_arcsec: float = 3.0, test_size: float = 0.2, random_state: int = 42
    ):
        self.match_radius = match_radius_arcsec
        self.test_size = test_size
        self.random_state = random_state
        self.validator = RigorousValidator(match_radius_arcsec=match_radius_arcsec)
        self.known_bds = load_expanded_brown_dwarf_catalog()
        self.scaler = StandardScaler()
        self.feature_cols = []

    def prepare_training_data(
        self, candidates: pd.DataFrame, feature_cols: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with proper labels.

        Args:
            candidates: DataFrame with candidate sources
            feature_cols: List of feature column names

        Returns:
            Tuple of (X, y) arrays
        """
        # Cross-match with known brown dwarfs
        candidates = self.validator.crossmatch_with_catalog(candidates)

        # Determine feature columns
        if feature_cols is None:
            # Auto-detect numeric feature columns
            feature_cols = [
                col
                for col in candidates.columns
                if candidates[col].dtype in ["float64", "float32", "int64", "int32"]
                and col not in ["ra", "dec", "is_known_bd", "match_separation_arcsec"]
                and not col.startswith("matched_")
            ]

        self.feature_cols = feature_cols
        log.info(f"Using {len(feature_cols)} features")

        # Prepare X
        X = candidates[feature_cols].fillna(0).values

        # Prepare y (ground truth from known BDs)
        y = candidates["is_known_bd"].astype(int).values

        # Log class distribution
        n_positive = y.sum()
        n_negative = len(y) - n_positive
        log.info(f"Class distribution: {n_positive} positive, {n_negative} negative")
        log.info(f"Positive rate: {n_positive / len(y):.4%}")

        return X, y

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost",
        model_params: dict | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """
        Train a model with proper train/test split.

        Args:
            X: Feature matrix
            y: Labels
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'gradient_boosting')
            model_params: Model hyperparameters

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        if model_params is None:
            model_params = {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if y.sum() > 1 else None,
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create model
        if model_type == "xgboost":
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "eval_metric": "logloss",
            }
            default_params.update(model_params)
            model = xgb.XGBClassifier(**default_params)

        elif model_type == "lightgbm":
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "verbose": -1,
            }
            default_params.update(model_params)
            model = lgb.LGBMClassifier(**default_params)

        elif model_type == "random_forest":
            default_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            default_params.update(model_params)
            model = RandomForestClassifier(**default_params)

        elif model_type == "gradient_boosting":
            default_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": self.random_state,
            }
            default_params.update(model_params)
            model = GradientBoostingClassifier(**default_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train model
        log.info(f"Training {model_type} model...")
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = (
            model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        )

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }

        if y_proba is not None and y_test.sum() > 0:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                metrics["average_precision"] = average_precision_score(y_test, y_proba)
            except ValueError:
                metrics["roc_auc"] = 0.5
                metrics["average_precision"] = 0.0

        log.info(
            f"{model_type} metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, AUC={metrics.get('roc_auc', 0):.4f}"
        )

        return model, metrics

    def cross_validate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost",
        n_splits: int = 5,
        model_params: dict | None = None,
    ) -> dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            X: Feature matrix
            y: Labels
            model_type: Type of model
            n_splits: Number of folds
            model_params: Model hyperparameters

        Returns:
            Dictionary with cross-validation results
        """
        if model_params is None:
            model_params = {}

        np.random.seed(self.random_state)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Use stratified k-fold
        if y.sum() >= n_splits:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = kfold.split(X_scaled, y)
        else:
            # Not enough positive samples for stratified
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = kfold.split(X_scaled)

        results = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "roc_auc": [],
        }

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Skip fold if no positive samples
            if y_train.sum() == 0 or y_val.sum() == 0:
                log.warning(f"Skipping fold {fold}: no positive samples")
                continue

            # Train model
            model, metrics = self.train_model(
                X_train, y_train, model_type=model_type, model_params=model_params
            )

            # Predict on validation
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

            # Store metrics
            results["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            results["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            results["f1_score"].append(f1_score(y_val, y_pred, zero_division=0))

            if y_proba is not None:
                try:
                    results["roc_auc"].append(roc_auc_score(y_val, y_proba))
                except ValueError:
                    results["roc_auc"].append(0.5)

            log.info(
                f"Fold {fold + 1}: P={results['precision'][-1]:.4f}, "
                f"R={results['recall'][-1]:.4f}, F1={results['f1_score'][-1]:.4f}"
            )

        # Calculate summary
        summary = {
            "precision_mean": np.mean(results["precision"]) if results["precision"] else 0,
            "precision_std": np.std(results["precision"]) if results["precision"] else 0,
            "recall_mean": np.mean(results["recall"]) if results["recall"] else 0,
            "recall_std": np.std(results["recall"]) if results["recall"] else 0,
            "f1_mean": np.mean(results["f1_score"]) if results["f1_score"] else 0,
            "f1_std": np.std(results["f1_score"]) if results["f1_score"] else 0,
            "roc_auc_mean": np.mean(results["roc_auc"]) if results["roc_auc"] else 0.5,
            "roc_auc_std": np.std(results["roc_auc"]) if results["roc_auc"] else 0,
        }

        return {"per_fold": results, "summary": summary}

    def compare_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_types: list[str] = ["xgboost", "lightgbm", "random_forest"],
    ) -> pd.DataFrame:
        """
        Compare multiple model types.

        Args:
            X: Feature matrix
            y: Labels
            model_types: List of model types to compare

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_type in model_types:
            log.info(f"\n{'='*50}")
            log.info(f"Training {model_type}...")
            log.info(f"{'='*50}")

            try:
                model, metrics = self.train_model(X, y, model_type=model_type)
                metrics["model_type"] = model_type
                results.append(metrics)
            except Exception as e:
                log.error(f"Failed to train {model_type}: {e}")

        return pd.DataFrame(results)

    def get_feature_importance(self, model, feature_cols: list[str] | None = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            model: Trained model
            feature_cols: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if feature_cols is None:
            feature_cols = self.feature_cols

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_).flatten()
        else:
            log.warning("Model does not have feature_importances_ or coef_ attribute")
            return pd.DataFrame()

        df = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values(
            "importance", ascending=False
        )

        return df


def cross_validate_model(
    X: np.ndarray, y: np.ndarray, model_type: str = "xgboost", n_splits: int = 5
) -> dict[str, Any]:
    """
    Convenience function for cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of model
        n_splits: Number of folds

    Returns:
        Dictionary with cross-validation results
    """
    validator = MLValidator()
    return validator.cross_validate_model(X, y, model_type, n_splits)


def create_synthetic_labels(
    candidates: pd.DataFrame, score_col: str = "composite_score", threshold_percentile: float = 95
) -> np.ndarray:
    """
    Create synthetic positive labels for semi-supervised learning.

    WARNING: This is still proxy-based labeling. Use with caution.
    Prefer ground truth labels from known brown dwarfs when available.

    Args:
        candidates: DataFrame with candidates
        score_col: Score column to use
        threshold_percentile: Percentile threshold for positive class

    Returns:
        Binary label array
    """
    threshold = np.percentile(candidates[score_col].dropna(), threshold_percentile)
    return (candidates[score_col] >= threshold).astype(int).values


def main():
    """Test the ML validation framework."""
    print("=" * 70)
    print("TASNI ML Validation Framework Test")
    print("=" * 70)

    # Load golden targets if available
    golden_file = OUTPUT_DIR / "golden_targets.csv"
    if not golden_file.exists():
        golden_file = OUTPUT_DIR / "final" / "golden_targets.csv"

    if golden_file.exists():
        print(f"\nLoading golden targets from {golden_file}")
        golden = pd.read_csv(golden_file)
        print(f"Loaded {len(golden)} candidates")

        # Initialize validator
        validator = MLValidator()

        # Prepare data
        X, y = validator.prepare_training_data(golden)

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Labels: {y.sum()} positive, {len(y) - y.sum()} negative")

        if y.sum() > 0:
            # Compare models
            print("\nComparing models...")
            comparison = validator.compare_models(X, y)
            print("\nModel Comparison Results:")
            print(comparison.to_string())

            # Cross-validation
            print("\nPerforming 5-fold cross-validation with XGBoost...")
            cv_results = validator.cross_validate_model(X, y, model_type="xgboost", n_splits=5)
            print("\nCross-Validation Summary:")
            for key, value in cv_results["summary"].items():
                print(f"  {key}: {value:.4f}")
        else:
            print("\nNo positive samples found in cross-match.")
            print("This could mean:")
            print("  1. Known brown dwarfs are not in this candidate set")
            print("  2. Match radius is too small")
            print("  3. Coordinate columns are incorrect")
    else:
        print(f"\nGolden targets file not found: {golden_file}")
        print("Run the full pipeline first to generate candidates.")


if __name__ == "__main__":
    main()
