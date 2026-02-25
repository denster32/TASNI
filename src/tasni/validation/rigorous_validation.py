#!/usr/bin/env python3
"""
Rigorous Validation Framework for TASNI

Provides proper validation with:
- K-fold cross-validation
- Precision/Recall/F1 metrics
- ROC-AUC scoring
- Bootstrap confidence intervals
- Recovery rate calculations

This addresses the circular validation issue in the original ml_scoring.py
which used existing scores as proxy labels.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tasni.core.config import OUTPUT_DIR

from .expanded_bd_catalog import load_expanded_brown_dwarf_catalog

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""

    precision: float
    recall: float
    f1_score: float
    accuracy: float
    roc_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    recovery_rate: float
    precision_at_10: float
    precision_at_100: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "recovery_rate": self.recovery_rate,
            "precision_at_10": self.precision_at_10,
            "precision_at_100": self.precision_at_100,
        }

    def __str__(self) -> str:
        return (
            f"ValidationMetrics(\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  F1 Score: {self.f1_score:.4f}\n"
            f"  ROC-AUC: {self.roc_auc:.4f}\n"
            f"  Recovery Rate: {self.recovery_rate:.4f}\n"
            f"  Precision@10: {self.precision_at_10:.4f}\n"
            f"  Precision@100: {self.precision_at_100:.4f}\n"
            f")"
        )


class RigorousValidator:
    """
    Rigorous validation of TASNI pipeline using known brown dwarfs.

    This class provides:
    1. Ground truth cross-matching with expanded brown dwarf catalog
    2. K-fold cross-validation for ML models
    3. Bootstrap confidence intervals for all metrics
    4. Precision@K metrics for ranking evaluation
    """

    def __init__(
        self, match_radius_arcsec: float = 3.0, n_bootstrap: int = 1000, random_state: int = 42
    ):
        self.match_radius = match_radius_arcsec
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.known_bds = load_expanded_brown_dwarf_catalog()

    def crossmatch_with_catalog(
        self, candidates: pd.DataFrame, ra_col: str = "ra", dec_col: str = "dec"
    ) -> pd.DataFrame:
        """
        Cross-match candidates with known brown dwarf catalog.

        Args:
            candidates: DataFrame with candidate sources
            ra_col: Name of RA column
            dec_col: Name of Dec column

        Returns:
            DataFrame with 'is_known_bd' column added
        """
        log.info(
            f"Cross-matching {len(candidates)} candidates with {len(self.known_bds)} known BDs..."
        )

        # Create coordinate arrays
        cand_coords = SkyCoord(
            ra=candidates[ra_col].values * u.degree, dec=candidates[dec_col].values * u.degree
        )
        bd_coords = SkyCoord(
            ra=self.known_bds["ra"].values * u.degree, dec=self.known_bds["dec"].values * u.degree
        )

        # Find matches
        idx, sep, _ = match_coordinates_sky(cand_coords, bd_coords)

        # Create match flag
        candidates = candidates.copy()
        candidates["is_known_bd"] = sep.arcsec < self.match_radius
        candidates["matched_bd_name"] = None
        candidates["matched_bd_spectral_type"] = None
        candidates["match_separation_arcsec"] = sep.arcsec

        # Fill in matched BD info
        match_mask = candidates["is_known_bd"]
        candidates.loc[match_mask, "matched_bd_name"] = self.known_bds.iloc[idx[match_mask]][
            "name"
        ].values
        candidates.loc[match_mask, "matched_bd_spectral_type"] = self.known_bds.iloc[
            idx[match_mask]
        ]["spectral_type"].values

        n_matched = match_mask.sum()
        log.info(f"Matched {n_matched}/{len(self.known_bds)} known brown dwarfs")

        return candidates

    def calculate_recovery_rate(
        self, candidates: pd.DataFrame, spectral_classes: list[str] = ["Y", "T"]
    ) -> tuple[float, int, int]:
        """
        Calculate recovery rate for known brown dwarfs.

        Args:
            candidates: DataFrame with 'is_known_bd' and 'matched_bd_spectral_type' columns
            spectral_classes: Spectral classes to include

        Returns:
            Tuple of (recovery_rate, n_recovered, n_total)
        """
        # Get subset of known BDs by spectral class
        bd_subset = self.known_bds[self.known_bds["bd_class"].isin(spectral_classes)]
        n_total = len(bd_subset)

        # Count recovered
        if "matched_bd_spectral_type" in candidates.columns:
            recovered = (
                candidates[candidates["is_known_bd"]]["matched_bd_spectral_type"]
                .apply(
                    lambda x: (
                        any(cls in str(x) for cls in spectral_classes) if pd.notna(x) else False
                    )
                )
                .sum()
            )
        else:
            recovered = candidates["is_known_bd"].sum()

        recovery_rate = recovered / n_total if n_total > 0 else 0.0

        return recovery_rate, int(recovered), n_total

    def calculate_precision_recall(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Calculate precision, recall, and related metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = 0.5

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"] = tp
        metrics["false_positives"] = fp
        metrics["true_negatives"] = tn
        metrics["false_negatives"] = fn

        return metrics

    def calculate_precision_at_k(self, y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
        """
        Calculate Precision@K.

        Args:
            y_true: True labels
            scores: Scores (higher = better)
            k: Number of top results to consider

        Returns:
            Precision@K
        """
        # Get indices of top k scores
        top_k_idx = np.argsort(scores)[-k:]

        # Calculate precision
        n_relevant = y_true[top_k_idx].sum()

        return n_relevant / k

    def bootstrap_confidence_interval(
        self, metric_func, y_true: np.ndarray, y_pred: np.ndarray, confidence: float = 0.95
    ) -> tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a metric.

        Args:
            metric_func: Function that calculates metric from (y_true, y_pred)
            y_true: True labels
            y_pred: Predicted labels
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        np.random.seed(self.random_state)

        n_samples = len(y_true)
        bootstrap_scores = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            score = metric_func(y_true[idx], y_pred[idx])
            bootstrap_scores.append(score)

        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

        return lower, upper

    def validate_ranking(
        self, candidates: pd.DataFrame, score_col: str = "improved_composite_score"
    ) -> ValidationMetrics:
        """
        Validate a ranking of candidates.

        Args:
            candidates: DataFrame with 'is_known_bd' and score columns
            score_col: Name of score column to use for ranking

        Returns:
            ValidationMetrics object
        """
        # Ensure cross-match has been done
        if "is_known_bd" not in candidates.columns:
            candidates = self.crossmatch_with_catalog(candidates)

        # Sort by score
        candidates = candidates.sort_values(score_col, ascending=False)

        # Create binary labels
        y_true = candidates["is_known_bd"].astype(int).values
        scores = candidates[score_col].values

        # Calculate precision@K
        precision_at_10 = self.calculate_precision_at_k(y_true, scores, 10)
        precision_at_100 = self.calculate_precision_at_k(y_true, scores, 100)

        # Calculate recovery rate
        recovery_rate, n_recovered, n_total = self.calculate_recovery_rate(candidates)

        # Create predictions based on top percentile
        threshold = np.percentile(scores, 99)  # Top 1%
        y_pred = (scores >= threshold).astype(int)

        # Calculate metrics
        metrics = self.calculate_precision_recall(y_true, y_pred, scores)

        return ValidationMetrics(
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            accuracy=metrics["accuracy"],
            roc_auc=metrics.get("roc_auc", 0.5),
            true_positives=metrics["true_positives"],
            false_positives=metrics["false_positives"],
            true_negatives=metrics["true_negatives"],
            false_negatives=metrics["false_negatives"],
            recovery_rate=recovery_rate,
            precision_at_10=precision_at_10,
            precision_at_100=precision_at_100,
        )


def validate_with_kfold(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    n_splits: int = 5,
    model_params: dict | None = None,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """
    Perform k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        model_class: Model class (e.g., RandomForestClassifier)
        n_splits: Number of folds
        model_params: Model hyperparameters
        random_state: Random seed

    Returns:
        Dictionary of metrics per fold
    """
    if model_params is None:
        model_params = {}

    np.random.seed(random_state)

    # Use stratified k-fold for imbalanced data
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        model = model_class(random_state=random_state, **model_params)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_val_scaled)
        y_proba = (
            model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        )

        # Calculate metrics
        results["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        results["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        results["f1_score"].append(f1_score(y_val, y_pred, zero_division=0))

        if y_proba is not None:
            try:
                results["roc_auc"].append(roc_auc_score(y_val, y_proba))
            except ValueError:
                results["roc_auc"].append(0.5)

        log.info(
            f"Fold {fold + 1}/{n_splits}: "
            f"P={results['precision'][-1]:.4f}, "
            f"R={results['recall'][-1]:.4f}, "
            f"F1={results['f1_score'][-1]:.4f}"
        )

    # Calculate mean and std
    summary = {
        "precision_mean": np.mean(results["precision"]),
        "precision_std": np.std(results["precision"]),
        "recall_mean": np.mean(results["recall"]),
        "recall_std": np.std(results["recall"]),
        "f1_mean": np.mean(results["f1_score"]),
        "f1_std": np.std(results["f1_score"]),
    }

    if results["roc_auc"]:
        summary["roc_auc_mean"] = np.mean(results["roc_auc"])
        summary["roc_auc_std"] = np.std(results["roc_auc"])

    log.info("\n" + "=" * 50)
    log.info("Cross-Validation Summary")
    log.info("=" * 50)
    for metric, value in summary.items():
        log.info(f"  {metric}: {value:.4f}")

    return {"per_fold": results, "summary": summary}


def calculate_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Convenience function for precision/recall."""
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
    )


def calculate_roc_metrics(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve metrics.

    Returns:
        Tuple of (auc, fpr, tpr)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return auc, fpr, tpr


def main():
    """Test the validation framework."""
    print("=" * 70)
    print("TASNI Rigorous Validation Framework Test")
    print("=" * 70)

    # Load known brown dwarfs
    validator = RigorousValidator()
    print(f"\nLoaded {len(validator.known_bds)} known brown dwarfs")

    # Print summary by class
    print("\nKnown BDs by class:")
    for cls in ["Y", "T", "L"]:
        count = (validator.known_bds["bd_class"] == cls).sum()
        print(f"  {cls} dwarfs: {count}")

    # Test with golden targets if available
    golden_file = OUTPUT_DIR / "golden_targets.csv"
    if golden_file.exists():
        print(f"\nTesting with golden targets from {golden_file}")
        golden = pd.read_csv(golden_file)

        # Cross-match
        golden = validator.crossmatch_with_catalog(golden)

        # Calculate metrics
        metrics = validator.validate_ranking(golden)
        print(f"\n{metrics}")

        # Recovery rate
        recovery_rate, n_recovered, n_total = validator.calculate_recovery_rate(golden)
        print(f"\nRecovery: {n_recovered}/{n_total} ({recovery_rate:.2%})")
    else:
        print(f"\nGolden targets file not found: {golden_file}")
        print("Run the full pipeline first to generate candidates.")


if __name__ == "__main__":
    main()
