#!/usr/bin/env python3
"""
Enhanced ML Ensemble for TASNI

Implements:
1. XGBoost + LightGBM + RandomForest ensemble
2. Proper cross-validation with ground truth
3. Feature importance analysis
4. Anomaly score calibration
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from tasni.validation.rigorous_validation import RigorousValidator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class EnhancedEnsembleML:
    """
    Enhanced ML ensemble with proper ground truth validation.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        use_cross_validation: bool = True,
        n_folds: int = 5,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = None

    def prepare_features(
        self, df: pd.DataFrame, feature_cols: list[str] | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """Prepare feature matrix."""
        if feature_cols is None:
            # Auto-select numeric features
            feature_cols = [
                col
                for col in df.columns
                if df[col].dtype in ["float64", "float32", "int64", "int32"]
                and (not col.endswith("_value") or col in ["w1_w2_color", "pm_total"])
                and "score" not in col.lower()
                and "rank" not in col.lower()
            ]
            # Deduplicate
            feature_cols = list(set(feature_cols))

        # Filter to existing columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols].fillna(0).values
        return X, feature_cols

    def train_ensemble(
        self, X: np.ndarray, y: np.ndarray, feature_cols: list[str]
    ) -> dict[str, Any]:
        """Train ensemble of models."""
        results = {}

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        # 1. Isolation Forest (unsupervised anomaly detection)
        log.info("Training Isolation Forest...")
        if_model = IsolationForest(contamination=0.1, random_state=self.random_state, n_jobs=-1)
        if_model.fit(X_scaled_df)
        if_scores = -if_model.decision_function(X_scaled_df)  # Higher = more anomalous
        self.models["isolation_forest"] = if_model
        results["if_scores"] = if_scores

        # 2. XGBoost (if we have labels)
        if y is not None and y.sum() > 0:
            log.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss",
            )
            xgb_model.fit(X_scaled_df, y)
            xgb_scores = xgb_model.predict_proba(X_scaled_df)[:, 1]
            self.models["xgboost"] = xgb_model
            results["xgb_scores"] = xgb_scores

            # 3. LightGBM
            log.info("Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
            )
            lgb_model.fit(X_scaled_df, y)
            lgb_scores = lgb_model.predict_proba(X_scaled_df)[:, 1]
            self.models["lightgbm"] = lgb_model
            results["lgb_scores"] = lgb_scores

            # 4. Random Forest
            log.info("Training Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf_model.fit(X_scaled_df, y)
            rf_scores = rf_model.predict_proba(X_scaled_df)[:, 1]
            self.models["random_forest"] = rf_model
            results["rf_scores"] = rf_scores

            # Cross-validation metrics
            if self.use_cross_validation:
                log.info("Running cross-validation...")
                cv = StratifiedKFold(
                    n_splits=self.n_folds, shuffle=True, random_state=self.random_state
                )
                cv_preds = cross_val_predict(
                    xgb_model, X_scaled_df, y, cv=cv, method="predict_proba"
                )[:, 1]
                results["cv_scores"] = cv_preds
                results["cv_roc_auc"] = roc_auc_score(y, cv_preds) if y.sum() > 0 else 0.5
        else:
            # No labels - use unsupervised only
            log.warning("No positive labels - using unsupervised models only")
            results["xgb_scores"] = np.zeros(len(X))
            results["lgb_scores"] = np.zeros(len(X))
            results["rf_scores"] = np.zeros(len(X))
            results["cv_scores"] = np.zeros(len(X))
            results["cv_roc_auc"] = 0.5

        # Ensemble scores
        scores = np.column_stack(
            [
                results["if_scores"],
                results["xgb_scores"],
                results["lgb_scores"],
                results["rf_scores"],
            ]
        )

        # Normalize each column
        scores_norm = (scores - scores.min(axis=0)) / (
            scores.max(axis=0) - scores.min(axis=0) + 1e-8
        )

        # Weighted ensemble (give more weight to supervised models if available)
        if y is not None and y.sum() > 0:
            weights = [0.1, 0.3, 0.3, 0.3]  # IF, XGB, LGB, RF
        else:
            weights = [1.0, 0.0, 0.0, 0.0]  # Only IF

        ensemble_score = np.average(scores_norm, axis=1, weights=weights)
        results["ensemble_score"] = ensemble_score

        # Feature importance
        if "xgboost" in self.models:
            self.feature_importance = pd.DataFrame(
                {"feature": feature_cols, "importance": self.models["xgboost"].feature_importances_}
            ).sort_values("importance", ascending=False)

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from ensemble."""
        return self.feature_importance


def run_enhanced_ml_pipeline(
    features_path: str, output_path: str, use_ground_truth: bool = True
) -> pd.DataFrame:
    """
    Run the enhanced ML pipeline.

    Args:
        features_path: Path to features parquet
        output_path: Path for output ranked candidates
        use_ground_truth: Whether to use known BDs as ground truth

    Returns:
        DataFrame with ranked candidates
    """
    # Load features
    log.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df)} sources with {len(df.columns)} columns")

    # Check for designation column
    if "designation" not in df.columns:
        df = df.reset_index()
        if "designation" not in df.columns:
            df["designation"] = [
                f"J{ra:.0f}{dec:+.0f}"
                for ra, dec in zip(
                    df.get("ra_value", [0] * len(df)),
                    df.get("dec_value", [0] * len(df)),
                    strict=False,
                )
            ]

    # Prepare features
    ml = EnhancedEnsembleML(n_estimators=200, n_folds=5)
    X, feature_cols = ml.prepare_features(df)
    log.info(f"Using {len(feature_cols)} features")

    # Create ground truth labels
    y = None
    if use_ground_truth:
        # Cross-match with known brown dwarfs
        validator = RigorousValidator()

        # Need ra/dec columns
        if "ra_value" in df.columns and "dec_value" in df.columns:
            coords_df = df[["ra_value", "dec_value"]].copy()
            coords_df.columns = ["ra", "dec"]
            coords_df["designation"] = df["designation"]

            coords_df = validator.crossmatch_with_catalog(coords_df)
            y = coords_df["is_known_bd"].astype(int).values
            log.info(f"Ground truth: {y.sum()} known BDs matched")
        else:
            log.warning("No coordinate columns found - using unsupervised only")

    # Train ensemble
    results = ml.train_ensemble(X, y, feature_cols)

    # Add scores to DataFrame
    df["if_score"] = results["if_scores"]
    df["xgb_score"] = results.get("xgb_scores", np.zeros(len(df)))
    df["lgb_score"] = results.get("lgb_scores", np.zeros(len(df)))
    df["rf_score"] = results.get("rf_scores", np.zeros(len(df)))
    df["ml_ensemble_score"] = results["ensemble_score"]

    if "cv_scores" in results:
        df["cv_score"] = results["cv_scores"]

    # Rank by ensemble score
    df = df.sort_values("ml_ensemble_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info(f"Saved {len(df)} ranked candidates to {output_path}")

    # Print summary
    log.info("=" * 60)
    log.info("ML Pipeline Summary")
    log.info("=" * 60)
    log.info(f"Sources processed: {len(df)}")
    log.info(f"Features used: {len(feature_cols)}")
    if y is not None:
        log.info(f"Known BDs in sample: {y.sum()}")
    log.info(f"Top score: {df['ml_ensemble_score'].max():.4f}")
    log.info(f"Mean score: {df['ml_ensemble_score'].mean():.4f}")
    log.info(f"CV ROC-AUC: {results.get('cv_roc_auc', 0):.4f}")

    # Feature importance
    if ml.feature_importance is not None:
        log.info("\nTop 10 features:")
        for _, row in ml.feature_importance.head(10).iterrows():
            log.info(f"  {row['feature']}: {row['importance']:.4f}")

    return df


def main():
    """Run enhanced ML pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced ML Pipeline for TASNI")
    parser.add_argument("--input", default="output/features/tier5_features.parquet")
    parser.add_argument("--output", default="output/ml/ranked_candidates_enhanced.parquet")
    parser.add_argument("--no-ground-truth", action="store_true")
    args = parser.parse_args()

    run_enhanced_ml_pipeline(args.input, args.output, use_ground_truth=not args.no_ground_truth)


if __name__ == "__main__":
    main()
