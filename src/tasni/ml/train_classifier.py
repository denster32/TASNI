"""
Machine Learning Classifier for TASNI Sources

Train supervised and unsupervised models to classify tier5 sources.

Usage:
    python src/tasni/ml/train_classifier.py \
        --features data/processed/features/tier5_features.parquet \
        --golden data/processed/final/golden_targets.csv \
        --output data/processed/ml/models/
        --train
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from tasni.core.tasni_logging import setup_logging

logger = setup_logging("ml_classifier", level="INFO")


class TASNIMLClassifier:
    """Complete ML pipeline for TASNI source classification"""

    def __init__(self, output_dir: str = "data/processed/ml/models/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.feature_names = []

    def prepare_data(
        self, features_df: pd.DataFrame, golden_df: pd.DataFrame = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for ML training"""
        logger.info("Preparing data for ML training...")

        # Handle missing values
        features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Remove non-numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols]

        self.feature_names = list(features_df.columns)
        logger.info(f"Using {len(self.feature_names)} numeric features")

        # Create labels (if golden targets provided)
        labels_df = None
        if golden_df is not None:
            labels_df = pd.Series(0, index=features_df.index)

            # Mark golden targets as 1
            golden_designations = golden_df["designation"].values
            labels_df.loc[golden_designations] = 1

            logger.info(f"Golden targets: {np.sum(labels_df == 1)}")
            logger.info(f"Other tier5: {np.sum(labels_df == 0)}")

        return features_df, labels_df

    def train_supervised_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train supervised classification models"""
        logger.info("Training supervised models...")

        models = {}

        # 1. Random Forest
        try:
            from sklearn.ensemble import RandomForestClassifier

            logger.info("Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            rf.fit(X_train, y_train)
            models["random_forest"] = rf
            logger.info("✓ Random Forest trained")

        except ImportError:
            logger.warning("sklearn not available, skipping Random Forest")

        # 2. XGBoost
        try:
            import xgboost as xgb

            logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=10,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42,
                verbosity=1,
            )
            xgb_model.fit(X_train, y_train)
            models["xgboost"] = xgb_model
            logger.info("✓ XGBoost trained")

        except ImportError:
            logger.warning("xgboost not available, skipping XGBoost")

        # 3. Neural Network
        try:
            from sklearn.neural_network import MLPClassifier

            logger.info("Training Neural Network...")
            mlp = MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=500,
                random_state=42,
                verbose=True,
            )
            mlp.fit(X_train, y_train)
            models["neural_network"] = mlp
            logger.info("✓ Neural Network trained")

        except ImportError:
            logger.warning("sklearn not available, skipping Neural Network")

        self.models.update(models)
        return models

    def train_unsupervised_models(self, X: pd.DataFrame) -> dict:
        """Train unsupervised models"""
        logger.info("Training unsupervised models...")

        models = {}

        # 1. Isolation Forest (anomaly detection)
        try:
            from sklearn.ensemble import IsolationForest

            logger.info("Training Isolation Forest...")
            iso_forest = IsolationForest(
                n_estimators=100, contamination=0.01, n_jobs=-1, random_state=42, verbose=1
            )
            iso_forest.fit(X)
            models["isolation_forest"] = iso_forest
            logger.info("✓ Isolation Forest trained")

        except ImportError:
            logger.warning("sklearn not available, skipping Isolation Forest")

        # 2. Local Outlier Factor
        try:
            from sklearn.neighbors import LocalOutlierFactor

            logger.info("Training LOF...")
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, n_jobs=-1)
            # LOF doesn't have fit_predict for new data
            lof.fit(X)
            models["lof"] = lof
            logger.info("✓ LOF trained")

        except ImportError:
            logger.warning("sklearn not available, skipping LOF")

        # 3. K-Means Clustering
        try:
            from sklearn.cluster import KMeans

            logger.info("Training K-Means...")
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10, verbose=1)
            kmeans.fit(X)
            models["kmeans"] = kmeans
            logger.info("✓ K-Means trained")

        except ImportError:
            logger.warning("sklearn not available, skipping K-Means")

        self.models.update(models)
        return models

    def save_models(self) -> None:
        """Save trained models"""
        logger.info(f"Saving models to {self.output_dir}...")

        for name, model in self.models.items():
            model_path = self.output_dir / f"{name}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"✓ Saved {name} to {model_path}")

        # Save feature names
        feature_names_path = self.output_dir / "feature_names.pkl"
        with open(feature_names_path, "wb") as f:
            pickle.dump(self.feature_names, f)

        logger.info(f"✓ Saved feature names to {feature_names_path}")

    def load_models(self) -> None:
        """Load trained models"""
        logger.info(f"Loading models from {self.output_dir}...")

        self.models = {}

        for model_file in self.output_dir.glob("*.pkl"):
            if model_file.name == "feature_names.pkl":
                continue

            with open(model_file, "rb") as f:
                self.models[model_file.stem] = pickle.load(f)

            logger.info(f"✓ Loaded {model_file.stem}")

        # Load feature names
        feature_names_path = self.output_dir / "feature_names.pkl"
        if feature_names_path.exists():
            with open(feature_names_path, "rb") as f:
                self.feature_names = pickle.load(f)
            logger.info(f"✓ Loaded {len(self.feature_names)} feature names")


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for TASNI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features/tier5_features.parquet",
        help="Path to features parquet file",
    )

    parser.add_argument(
        "--golden",
        type=str,
        default="data/processed/final/golden_targets.csv",
        help="Path to golden targets CSV file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/ml/models/",
        help="Output directory for models",
    )

    parser.add_argument("--train", action="store_true", help="Train models")

    parser.add_argument(
        "--test-only", action="store_true", help="Only test models (requires pre-trained models)"
    )

    args = parser.parse_args()

    # Load features
    logger.info(f"Loading features from {args.features}...")
    features_df = pd.read_parquet(args.features)
    logger.info(f"Loaded {len(features_df)} sources with {len(features_df.columns)} features")

    # Load golden targets
    golden_df = None
    if args.golden and Path(args.golden).exists():
        logger.info(f"Loading golden targets from {args.golden}...")
        golden_df = pd.read_csv(args.golden)
        logger.info(f"Loaded {len(golden_df)} golden targets")

    # Initialize classifier
    classifier = TASNIMLClassifier(output_dir=args.output)

    # Prepare data
    X, y = classifier.prepare_data(features_df, golden_df)

    if args.train:
        # Train supervised models
        if y is not None:
            logger.info("Training supervised models...")
            classifier.train_supervised_models(X, y)

        # Train unsupervised models
        logger.info("Training unsupervised models...")
        classifier.train_unsupervised_models(X)

        # Save models
        classifier.save_models()

        logger.info("=" * 70)
        logger.info("Training Complete")
        logger.info("=" * 70)
        logger.info(f"Models trained: {len(classifier.models)}")
        logger.info(f"Saved to: {args.output}")

    if args.test_only:
        # Load models
        classifier.load_models()

        # Test models
        logger.info("Testing models...")
        for name, model in classifier.models.items():
            logger.info(f"  {name}: {type(model).__name__}")


if __name__ == "__main__":
    main()
