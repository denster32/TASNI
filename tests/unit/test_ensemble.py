"""Unit tests for enhanced_ensemble.py ML pipeline."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame for ML testing."""
    np.random.seed(42)
    n_samples = 200

    df = pd.DataFrame(
        {
            "designation": [f"J{180+i:06.2f}{-30+i*0.1:+06.2f}" for i in range(n_samples)],
            "ra_value": np.random.uniform(0, 360, n_samples),
            "dec_value": np.random.uniform(-90, 90, n_samples),
            "w1_w2_color": np.random.uniform(0.5, 3.0, n_samples),
            "pm_total": np.random.uniform(0, 500, n_samples),
            "w1mpro": np.random.uniform(10, 16, n_samples),
            "w2mpro": np.random.uniform(10, 16, n_samples),
            "isolation_score": np.random.uniform(0, 1, n_samples),
            "weirdness_score": np.random.uniform(0, 10, n_samples),
        }
    )

    return df


@pytest.fixture
def sample_features_with_labels(sample_features):
    """Add ground truth labels to sample features."""
    df = sample_features.copy()

    # Mark ~10% as known brown dwarfs (positive class)
    n_positive = 20
    positive_indices = np.random.choice(len(df), n_positive, replace=False)
    df["is_known_bd"] = False
    df.loc[positive_indices, "is_known_bd"] = True

    # Make positive samples have more extreme features
    df.loc[df["is_known_bd"], "w1_w2_color"] = np.random.uniform(2.0, 3.5, n_positive)
    df.loc[df["is_known_bd"], "pm_total"] = np.random.uniform(200, 500, n_positive)

    return df


def test_enhanced_ensemble_import():
    """Test that the ensemble module can be imported."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML

        assert EnhancedEnsembleML is not None
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")


def test_ensemble_initialization():
    """Test EnhancedEnsembleML initialization."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML(n_estimators=100, random_state=42, use_cross_validation=True, n_folds=5)

    assert ml.n_estimators == 100
    assert ml.random_state == 42
    assert ml.use_cross_validation is True
    assert ml.n_folds == 5


def test_prepare_features(sample_features):
    """Test feature preparation."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML()
    X, feature_cols = ml.prepare_features(sample_features)

    # Check that we got a numpy array
    assert isinstance(X, np.ndarray)

    # Check that we have the right number of samples
    assert X.shape[0] == len(sample_features)

    # Check that feature columns don't include designation
    assert "designation" not in feature_cols


def test_train_ensemble_unsupervised(sample_features):
    """Test ensemble training without labels (unsupervised)."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML(n_estimators=50, use_cross_validation=False)
    X, feature_cols = ml.prepare_features(sample_features)

    # Train without labels
    results = ml.train_ensemble(X, None, feature_cols)

    # Should have Isolation Forest scores
    assert "if_scores" in results
    assert len(results["if_scores"]) == len(sample_features)

    # Should have ensemble score
    assert "ensemble_score" in results

    # Scores should be normalized between 0 and 1
    assert results["ensemble_score"].min() >= 0
    assert results["ensemble_score"].max() <= 1


def test_train_ensemble_supervised(sample_features_with_labels):
    """Test ensemble training with labels (supervised)."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML(n_estimators=50, use_cross_validation=False)
    X, feature_cols = ml.prepare_features(sample_features_with_labels)
    y = sample_features_with_labels["is_known_bd"].astype(int).values

    # Train with labels
    results = ml.train_ensemble(X, y, feature_cols)

    # Should have all model scores
    assert "if_scores" in results
    assert "xgb_scores" in results
    assert "lgb_scores" in results
    assert "rf_scores" in results
    assert "ensemble_score" in results

    # Models should be stored
    assert "isolation_forest" in ml.models
    assert "xgboost" in ml.models
    assert "lightgbm" in ml.models
    assert "random_forest" in ml.models


def test_feature_importance(sample_features_with_labels):
    """Test feature importance extraction."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML(n_estimators=50, use_cross_validation=False)
    X, feature_cols = ml.prepare_features(sample_features_with_labels)
    y = sample_features_with_labels["is_known_bd"].astype(int).values

    ml.train_ensemble(X, y, feature_cols)

    importance_df = ml.get_feature_importance()

    # Should be a DataFrame with feature and importance columns
    assert isinstance(importance_df, pd.DataFrame)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns

    # Should have importance for all features
    assert len(importance_df) == len(feature_cols)

    # Importance values should be non-negative
    assert (importance_df["importance"] >= 0).all()


def test_cross_validation(sample_features_with_labels):
    """Test cross-validation functionality."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML(n_estimators=50, use_cross_validation=True, n_folds=3)
    X, feature_cols = ml.prepare_features(sample_features_with_labels)
    y = sample_features_with_labels["is_known_bd"].astype(int).values

    results = ml.train_ensemble(X, y, feature_cols)

    # Should have CV scores
    assert "cv_scores" in results
    assert len(results["cv_scores"]) == len(sample_features_with_labels)

    # Should have CV ROC-AUC
    assert "cv_roc_auc" in results
    assert 0 <= results["cv_roc_auc"] <= 1


def test_random_state_reproducibility(sample_features):
    """Test that random state produces reproducible results."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml1 = EnhancedEnsembleML(n_estimators=50, random_state=42, use_cross_validation=False)
    ml2 = EnhancedEnsembleML(n_estimators=50, random_state=42, use_cross_validation=False)

    X, feature_cols = ml1.prepare_features(sample_features)

    results1 = ml1.train_ensemble(X, None, feature_cols)
    results2 = ml2.train_ensemble(X, None, feature_cols)

    # Results should be identical with same random state
    np.testing.assert_array_almost_equal(results1["ensemble_score"], results2["ensemble_score"])


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML()
    empty_df = pd.DataFrame()

    X, feature_cols = ml.prepare_features(empty_df)

    # Should handle gracefully
    assert X.shape[0] == 0


def test_missing_values_handling():
    """Test handling of missing values in features."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        }
    )

    # Add some NaN values
    df.loc[0:10, "feature1"] = np.nan
    df.loc[5:15, "feature2"] = np.nan

    ml = EnhancedEnsembleML()
    X, feature_cols = ml.prepare_features(df)

    # Should not raise an error
    assert X.shape[0] == n_samples


@pytest.mark.parametrize(
    "n_estimators,expected_min_models",
    [
        (10, 1),  # At least Isolation Forest
        (100, 1),
        (200, 1),
    ],
)
def test_different_n_estimators(n_estimators, expected_min_models, sample_features):
    """Test with different numbers of estimators."""
    try:
        from tasni.ml.enhanced_ensemble import EnhancedEnsembleML
    except ImportError:
        pytest.skip("enhanced_ensemble module not available")

    ml = EnhancedEnsembleML(n_estimators=n_estimators, use_cross_validation=False)
    X, feature_cols = ml.prepare_features(sample_features)

    results = ml.train_ensemble(X, None, feature_cols)

    # Should have at least Isolation Forest
    assert len(ml.models) >= expected_min_models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
