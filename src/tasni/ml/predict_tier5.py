"""
Predict Scores for All Tier5 Sources

Use trained ML models to predict scores and rank all 810K tier5 sources.

Usage:
    python src/tasni/ml/predict_tier5.py \
        --features data/processed/features/tier5_features.parquet \
        --models data/processed/ml/models/ \
        --output data/processed/ml/ranked_tier5.parquet \
        --top 10000
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from tasni.core.tasni_logging import setup_logging

logger = setup_logging("predict_tier5", level="INFO")


class TASNIPredictor:
    """Predict scores for TASNI sources using trained models"""

    def __init__(self, models_dir: str = "data/processed/ml/models/"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = []
        self.load_models()

    def load_models(self) -> None:
        """Load trained models"""
        logger.info(f"Loading models from {self.models_dir}...")

        for model_file in self.models_dir.glob("*.pkl"):
            if model_file.name == "feature_names.pkl":
                continue

            with open(model_file, "rb") as f:
                self.models[model_file.stem] = pickle.load(f)

            logger.info(f"✓ Loaded {model_file.stem}")

        # Load feature names
        feature_names_path = self.models_dir / "feature_names.pkl"
        if feature_names_path.exists():
            with open(feature_names_path, "rb") as f:
                self.feature_names = pickle.load(f)
            logger.info(f"✓ Loaded {len(self.feature_names)} feature names")

    def predict_supervised(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict using supervised models"""
        logger.info("Predicting with supervised models...")

        predictions = pd.DataFrame(index=X.index)

        # Random Forest
        if "random_forest" in self.models:
            rf = self.models["random_forest"]
            predictions["rf_prob"] = rf.predict_proba(X)[:, 1]
            predictions["rf_pred"] = rf.predict(X)
            logger.info("✓ Random Forest predictions")

        # XGBoost
        if "xgboost" in self.models:
            xgb = self.models["xgboost"]
            predictions["xgb_prob"] = xgb.predict_proba(X)[:, 1]
            predictions["xgb_pred"] = xgb.predict(X)
            logger.info("✓ XGBoost predictions")

        # Neural Network
        if "neural_network" in self.models:
            mlp = self.models["neural_network"]
            predictions["nn_prob"] = mlp.predict_proba(X)[:, 1]
            predictions["nn_pred"] = mlp.predict(X)
            logger.info("✓ Neural Network predictions")

        return predictions

    def predict_unsupervised(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict using unsupervised models"""
        logger.info("Predicting with unsupervised models...")

        predictions = pd.DataFrame(index=X.index)

        # Isolation Forest
        if "isolation_forest" in self.models:
            iso = self.models["isolation_forest"]
            predictions["iso_score"] = iso.decision_function(X)
            predictions["iso_pred"] = iso.predict(X)  # -1 = anomaly, 1 = normal
            # Convert to probability (higher = more anomalous)
            predictions["iso_prob"] = -predictions["iso_score"]
            predictions["iso_prob"] = (predictions["iso_prob"] - predictions["iso_prob"].min()) / (
                predictions["iso_prob"].max() - predictions["iso_prob"].min()
            )
            logger.info("✓ Isolation Forest predictions")

        # K-Means
        if "kmeans" in self.models:
            kmeans = self.models["kmeans"]
            predictions["cluster"] = kmeans.predict(X)
            predictions["distance_to_center"] = np.min(kmeans.transform(X), axis=1)
            logger.info("✓ K-Means predictions")

        return predictions

    def combine_scores(
        self, supervised_df: pd.DataFrame, unsupervised_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine scores from multiple models"""
        logger.info("Combining scores...")

        # Combine all predictions
        all_predictions = pd.concat([supervised_df, unsupervised_df], axis=1)

        # Calculate composite score
        prob_cols = [col for col in all_predictions.columns if col.endswith("_prob")]

        if prob_cols:
            # Ensemble: average of all probabilities
            all_predictions["composite_score"] = all_predictions[prob_cols].mean(axis=1)

            # Weighted ensemble (give more weight to RF and XGBoost)
            weights = {}
            for col in prob_cols:
                if "rf" in col:
                    weights[col] = 1.5
                elif "xgb" in col:
                    weights[col] = 1.5
                elif "nn" in col:
                    weights[col] = 1.0
                elif "iso" in col:
                    weights[col] = 1.0
                else:
                    weights[col] = 1.0

            weighted_probs = [all_predictions[col] * weights.get(col, 1.0) for col in prob_cols]
            all_predictions["weighted_score"] = sum(weighted_probs) / sum(weights.values())

            # Rank
            all_predictions["rank"] = all_predictions["weighted_score"].rank(ascending=False)

            logger.info("✓ Composite score calculated (weighted ensemble)")
            logger.info(f"  Used probability columns: {prob_cols}")
            logger.info(f"  Weights: {weights}")

        return all_predictions

    def predict_all(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Run complete prediction pipeline"""
        logger.info("=" * 70)
        logger.info("TASNI ML Prediction Pipeline")
        logger.info("=" * 70)
        logger.info(f"Sources to predict: {len(features_df)}")
        logger.info(f"Features: {len(features_df.columns)}")

        # Align features with model
        X = features_df[self.feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)

        # Supervised predictions
        supervised = self.predict_supervised(X)

        # Unsupervised predictions
        unsupervised = self.predict_unsupervised(X)

        # Combine scores
        results = self.combine_scores(supervised, unsupervised)

        # Add original features (selected)
        result_cols = [
            "ra",
            "dec",
            "w1mpro",
            "w2mpro",
            "w3mpro",
            "w4mpro",
            "w1_w2_color",
            "w2_w3_color",
            "w3_w4_color",
            "pmra",
            "pmdec",
            "T_eff_K",
        ]
        for col in result_cols:
            if col in features_df.columns:
                results[col] = features_df[col]

        logger.info("=" * 70)
        logger.info("Prediction Complete")
        logger.info("=" * 70)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict scores for tier5 sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features/tier5_features.parquet",
        help="Path to features parquet file",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="data/processed/ml/models/",
        help="Path to trained models directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/ml/ranked_tier5.parquet",
        help="Output path for ranked results",
    )

    parser.add_argument("--top", type=int, default=10000, help="Save top N sources")

    args = parser.parse_args()

    # Load features
    logger.info(f"Loading features from {args.features}...")
    features_df = pd.read_parquet(args.features)
    logger.info(f"Loaded {len(features_df)} sources with {len(features_df.columns)} features")

    # Initialize predictor
    predictor = TASNIPredictor(models_dir=args.models)

    # Predict
    results_df = predictor.predict_all(features_df)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save all results
    logger.info(f"Saving all results to {args.output}...")
    results_df.to_parquet(args.output, compression="snappy")

    # Save top N
    top_output = args.output.replace(".parquet", f"_top{args.top}.csv")
    top_df = results_df.nsmallest(args.top, "rank")
    top_df.to_csv(top_output, index=True)
    logger.info(f"Saved top {args.top} to {top_output}")

    # Print summary
    logger.info("=" * 70)
    logger.info("Prediction Summary")
    logger.info("=" * 70)
    logger.info(f"Total sources predicted: {len(results_df)}")
    logger.info(f"Top {args.top} sources saved")

    if "composite_score" in results_df.columns:
        logger.info(
            f"Composite score range: {results_df['composite_score'].min():.4f} - {results_df['composite_score'].max():.4f}"
        )
        logger.info(f"Mean score: {results_df['composite_score'].mean():.4f}")
        logger.info(f"Std score: {results_df['composite_score'].std():.4f}")

    # Print top 10
    logger.info("\nTop 10 Candidates:")
    if "rank" in results_df.columns:
        top_10 = results_df.nsmallest(10, "rank")
        for i, (idx, row) in enumerate(top_10.iterrows(), 1):
            logger.info(
                f"  {i}. {idx} - Score: {row.get('weighted_score', row.get('composite_score', 0)):.4f} - T_eff: {row.get('T_eff_K', 'N/A')} K"
            )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
