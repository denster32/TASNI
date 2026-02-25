#!/usr/bin/env python3
"""
TASNI Pipeline: Enhanced ML Scoring (Isolation Forest + Ensemble)
=================================================================

Loads tier5_features.parquet and computes anomaly ranking.

Default behavior is strictly unsupervised (Isolation Forest). Supervised models
(XGBoost/LightGBM) are used only when an explicit external label column is
provided to avoid circular training on prior score columns.

Outputs ranked_tier5_improved.parquet with new composite scores.

Usage:
  poetry run python src/tasni/pipeline/ml_scoring.py \
      --input output/features/tier5_features.parquet \
      --output data/processed/ml/ranked_tier5_improved.parquet
"""

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from tasni.core.seeds import DEFAULT_RANDOM_SEED, seed_numpy_and_python

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_v = np.nanmin(values)
    max_v = np.nanmax(values)
    span = max(max_v - min_v, 1e-8)
    return (values - min_v) / span


def load_features(input_path: Path | str) -> tuple[pd.DataFrame, list[str]]:
    """Load tier5 features."""
    df = pd.read_parquet(input_path)
    feature_cols = [
        col
        for col in df.columns
        if col.startswith(("w1_", "w2_", "pm_", "var_", "rms_")) or "score" in col.lower()
    ]
    feature_cols = [col for col in feature_cols if df[col].dtype in ["float64", "float32", "int64"]]
    logger.info(f"Loaded {len(df)} samples, {len(feature_cols)} features: {feature_cols[:10]}...")
    return df, feature_cols


def train_isolation_forest(
    X: np.ndarray, contamination: float = 0.1
) -> tuple[IsolationForest, np.ndarray]:
    """Train Isolation Forest for anomaly detection."""
    model = IsolationForest(
        contamination=contamination, random_state=DEFAULT_RANDOM_SEED, n_jobs=-1
    )
    model.fit(X)
    scores = model.decision_function(X)  # Higher = more normal, invert for anomalies
    anomaly_scores = -scores  # Higher = more anomalous
    return model, anomaly_scores


def train_xgboost(X: np.ndarray, y: np.ndarray) -> tuple[xgb.Booster, np.ndarray]:
    """Train XGBoost."""
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": DEFAULT_RANDOM_SEED,
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    preds = model.predict(dtrain)
    return model, preds


def train_lightgbm(X: np.ndarray, y: np.ndarray) -> tuple[lgb.Booster, np.ndarray]:
    """Train LightGBM."""
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "random_state": DEFAULT_RANDOM_SEED,
    }
    model = lgb.train(params, train_data, num_boost_round=100)
    preds = model.predict(X)
    return model, preds


def _extract_supervised_labels(df: pd.DataFrame, label_col: str) -> np.ndarray:
    y = df[label_col].to_numpy()
    if y.dtype == bool:
        return y.astype(int)

    # Handle 0/1 numeric labels directly.
    if np.issubdtype(y.dtype, np.number):
        unique = np.unique(y[~pd.isna(y)])
        if len(unique) == 2 and set(unique).issubset({0, 1}):
            return y.astype(int)

    raise ValueError(
        f"Label column '{label_col}' must contain binary labels (0/1 or bool). "
        f"Found dtype={y.dtype}."
    )


def ensemble_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    existing_score_col: str = "composite_score",
    label_col: str | None = None,
    contamination: float = 0.1,
    blend_with_existing: bool = True,
) -> pd.DataFrame:
    """Compute anomaly ranking with optional externally supervised components."""
    X = df[feature_cols].fillna(0).values

    logger.info("Training Isolation Forest...")
    _, if_scores = train_isolation_forest(X, contamination=contamination)
    components = [_normalize_scores(if_scores)]
    supervised_used = False

    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in input dataframe")

        y = _extract_supervised_labels(df, label_col)
        if len(np.unique(y)) < 2:
            raise ValueError(f"Label column '{label_col}' must include both classes")

        logger.info("Training XGBoost with external labels...")
        _, xgb_scores = train_xgboost(X, y)
        logger.info("Training LightGBM with external labels...")
        _, lgb_scores = train_lightgbm(X, y)

        df["xgb_score"] = xgb_scores
        df["lgb_score"] = lgb_scores
        components.append(_normalize_scores(xgb_scores))
        components.append(_normalize_scores(lgb_scores))
        supervised_used = True

        combined_supervised = (_normalize_scores(xgb_scores) + _normalize_scores(lgb_scores)) / 2.0
        try:
            auc = roc_auc_score(y, combined_supervised)
            logger.info("Supervised component AUC (in-sample): %.4f", auc)
        except ValueError:
            logger.warning("Could not compute AUC for supervised component")
    else:
        logger.info("No label column provided; running strictly unsupervised ranking")

    df["if_score"] = if_scores
    df["ml_supervised_used"] = supervised_used

    score_matrix = np.column_stack(components)
    ensemble_score = score_matrix.mean(axis=1)
    df["ml_ensemble_score"] = ensemble_score

    # Blend with historical score for continuity, but never for training labels.
    if blend_with_existing and existing_score_col in df.columns:
        df["improved_composite_score"] = 0.7 * df[existing_score_col] + 0.3 * ensemble_score
    else:
        df["improved_composite_score"] = ensemble_score

    # Rank
    df["rank"] = df["improved_composite_score"].rank(ascending=False).astype(int)
    return df.sort_values("improved_composite_score", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="TASNI Enhanced ML Scoring")
    parser.add_argument("--input", default="output/features/tier5_features.parquet")
    parser.add_argument("--output", default="data/processed/ml/ranked_tier5_improved.parquet")
    parser.add_argument(
        "--label-col",
        default=None,
        help="Optional external binary label column (0/1 or bool) for supervised models",
    )
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument(
        "--no-blend-existing",
        action="store_true",
        help="Disable blending with existing composite score columns",
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    seed_numpy_and_python(DEFAULT_RANDOM_SEED)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return

    df, feature_cols = load_features(input_path)
    if args.test:
        df = df.head(1000)
        logger.info("TEST MODE: Limited to 1000 samples")

    df = ensemble_scores(
        df,
        feature_cols,
        label_col=args.label_col,
        contamination=args.contamination,
        blend_with_existing=not args.no_blend_existing,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(f"Saved {len(df)} ranked candidates to {output_path}")
    logger.info(
        f"Top score: {df['improved_composite_score'].max():.4f} (AUC not computed; labels required)"
    )
    logger.info(f"Top 100: mean score {df.head(100)['improved_composite_score'].mean():.4f}")


if __name__ == "__main__":
    main()
