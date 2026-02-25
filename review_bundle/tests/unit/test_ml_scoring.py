"""Unit tests for ml_scoring.py pipeline module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_parquet_df():
    """Synthetic DataFrame mimicking tier5_features.parquet schema."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "designation": [f"J{180+i:06.2f}+00.00" for i in range(n)],
            "ra": np.random.uniform(0, 360, n),
            "dec": np.random.uniform(-90, 90, n),
            "w1_mag": np.random.uniform(10, 16, n),
            "w2_mag": np.random.uniform(10, 16, n),
            "w1_w2_color": np.random.uniform(0.5, 3.0, n),
            "pm_total": np.random.uniform(0, 500, n),
            "var_chi2": np.random.uniform(0, 10, n),
            "rms_w1": np.random.uniform(0.01, 0.1, n),
            "composite_score": np.random.uniform(0, 1, n),
        }
    )


@pytest.fixture
def feature_cols():
    return ["w1_mag", "w2_mag", "w1_w2_color", "pm_total", "var_chi2", "rms_w1", "composite_score"]


def test_load_features_returns_dataframe_and_columns(tmp_path, sample_parquet_df):
    parquet_path = tmp_path / "features.parquet"
    sample_parquet_df.to_parquet(parquet_path, index=False)

    from tasni.pipeline.ml_scoring import load_features

    df, cols = load_features(parquet_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_parquet_df)
    assert isinstance(cols, list)
    assert "designation" not in cols
    assert "ra" not in cols


def test_load_features_selects_correct_prefixes(tmp_path):
    df = pd.DataFrame(
        {
            "w1_flux": [1.0, 2.0],
            "w2_flux": [1.5, 2.5],
            "pm_ra": [10.0, 20.0],
            "var_amp": [0.1, 0.2],
            "rms_total": [0.05, 0.06],
            "anomaly_score": [0.8, 0.9],
            "designation": ["A", "B"],
            "notes": ["x", "y"],
        }
    )
    path = tmp_path / "test.parquet"
    df.to_parquet(path, index=False)

    from tasni.pipeline.ml_scoring import load_features

    _, cols = load_features(path)
    assert "w1_flux" in cols
    assert "anomaly_score" in cols
    assert "designation" not in cols
    assert "notes" not in cols


def test_ensemble_scores_unsupervised_only(sample_parquet_df, feature_cols):
    from tasni.pipeline.ml_scoring import ensemble_scores

    result = ensemble_scores(sample_parquet_df.copy(), feature_cols, label_col=None)

    assert "if_score" in result.columns
    assert "ml_ensemble_score" in result.columns
    assert "improved_composite_score" in result.columns
    assert "rank" in result.columns
    assert "ml_supervised_used" in result.columns
    assert not result["ml_supervised_used"].any()
    assert "xgb_score" not in result.columns
    assert "lgb_score" not in result.columns
    assert result["ml_ensemble_score"].between(0, 1).all()


@patch("tasni.pipeline.ml_scoring.lgb")
@patch("tasni.pipeline.ml_scoring.xgb")
def test_ensemble_scores_with_external_labels(mock_xgb, mock_lgb, sample_parquet_df, feature_cols):
    sample_parquet_df = sample_parquet_df.copy()
    sample_parquet_df["external_label"] = [0, 1] * (len(sample_parquet_df) // 2)

    mock_xgb.DMatrix.return_value = MagicMock()
    xgb_model = MagicMock()
    xgb_model.predict.return_value = np.random.uniform(0, 1, len(sample_parquet_df))
    mock_xgb.train.return_value = xgb_model

    mock_lgb.Dataset.return_value = MagicMock()
    lgb_model = MagicMock()
    lgb_model.predict.return_value = np.random.uniform(0, 1, len(sample_parquet_df))
    mock_lgb.train.return_value = lgb_model

    from tasni.pipeline.ml_scoring import ensemble_scores

    result = ensemble_scores(sample_parquet_df, feature_cols, label_col="external_label")
    assert "xgb_score" in result.columns
    assert "lgb_score" in result.columns
    assert result["ml_supervised_used"].all()


def test_ensemble_scores_missing_label_column_raises(sample_parquet_df, feature_cols):
    from tasni.pipeline.ml_scoring import ensemble_scores

    with pytest.raises(ValueError):
        ensemble_scores(sample_parquet_df, feature_cols, label_col="does_not_exist")


def test_train_isolation_forest():
    from tasni.pipeline.ml_scoring import train_isolation_forest

    np.random.seed(42)
    X = np.random.randn(100, 5)
    model, scores = train_isolation_forest(X, contamination=0.1)
    assert len(scores) == 100
    assert model is not None
    assert scores.std() > 0


def test_ensemble_scores_sorted_by_score(sample_parquet_df, feature_cols):
    from tasni.pipeline.ml_scoring import ensemble_scores

    result = ensemble_scores(sample_parquet_df.copy(), feature_cols)
    scores = result["improved_composite_score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
