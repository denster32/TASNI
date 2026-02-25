"""Unit tests for bayesian_selection.py pipeline module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Skip entire module if pymc is not installed
pymc = pytest.importorskip("pymc")


@pytest.fixture
def sample_df_with_parallax():
    """DataFrame with parallax and teff columns for error propagation."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame(
        {
            "designation": [f"J{i:04d}" for i in range(n)],
            "parallax": np.random.uniform(1.0, 50.0, n),
            "parallax_error": np.random.uniform(0.1, 2.0, n),
            "teff_estimate": np.random.uniform(300, 1500, n),
            "improved_composite_score": np.random.uniform(0, 1, n),
        }
    )


@pytest.fixture
def sample_df_missing_columns():
    """DataFrame missing expected columns."""
    return pd.DataFrame(
        {
            "designation": ["J0001", "J0002"],
            "ra": [180.0, 181.0],
        }
    )


def test_propagate_errors_distance_positive(sample_df_with_parallax):
    """Distance posteriors should always be positive for positive parallax."""
    from tasni.pipeline.bayesian_selection import propagate_errors

    result = propagate_errors(
        sample_df_with_parallax.copy(),
        parallax_col="parallax",
        teff_col="teff_estimate",
    )

    assert "dist_posterior_mean" in result.columns
    assert "dist_posterior_std" in result.columns
    assert "dist_posterior_p16" in result.columns
    assert "dist_posterior_p84" in result.columns
    assert (result["dist_posterior_mean"] > 0).all()
    assert (result["dist_posterior_std"] > 0).all()


def test_propagate_errors_known_values():
    """Check error propagation with known input values."""
    from tasni.pipeline.bayesian_selection import propagate_errors

    df = pd.DataFrame(
        {
            "parallax": [10.0, 50.0],  # mas
            "parallax_error": [1.0, 5.0],  # mas
            "teff_estimate": [400.0, 800.0],  # K
        }
    )

    result = propagate_errors(df.copy(), parallax_col="parallax", teff_col="teff_estimate")

    # Means should be close to inverse parallax estimates, but Monte Carlo allows small offsets.
    assert result["dist_posterior_mean"].iloc[0] == pytest.approx(100.0, abs=3.0)
    assert result["dist_posterior_mean"].iloc[1] == pytest.approx(20.0, abs=1.0)
    assert result["dist_posterior_std"].iloc[0] > 0
    assert result["dist_posterior_std"].iloc[1] > 0
    assert result["dist_posterior_p84"].iloc[0] > result["dist_posterior_p16"].iloc[0]

    # Teff posterior should be input value with 50 K assumed error
    np.testing.assert_almost_equal(result["teff_posterior_mean"].iloc[0], 400.0)
    np.testing.assert_almost_equal(result["teff_posterior_std"].iloc[0], 50.0)


def test_propagate_errors_missing_parallax_error():
    """When parallax_error column is missing, should use 20% fallback."""
    from tasni.pipeline.bayesian_selection import propagate_errors

    df = pd.DataFrame(
        {
            "parallax": [10.0],
            "teff_estimate": [500.0],
        }
    )

    result = propagate_errors(df.copy(), parallax_col="parallax", teff_col="teff_estimate")

    assert "dist_posterior_std" in result.columns
    assert result["dist_posterior_mean"].iloc[0] == pytest.approx(100.0, abs=6.0)
    assert result["dist_posterior_std"].iloc[0] > 0


def test_propagate_errors_missing_teff_column(sample_df_with_parallax):
    """Should raise KeyError when teff column is missing."""
    from tasni.pipeline.bayesian_selection import propagate_errors

    df = sample_df_with_parallax.drop(columns=["teff_estimate"])
    with pytest.raises(KeyError):
        propagate_errors(df, parallax_col="parallax", teff_col="teff_estimate")


def test_propagate_errors_missing_parallax_column(sample_df_with_parallax):
    """Should raise KeyError when parallax column is missing."""
    from tasni.pipeline.bayesian_selection import propagate_errors

    df = sample_df_with_parallax.drop(columns=["parallax"])
    with pytest.raises(KeyError):
        propagate_errors(df, parallax_col="parallax", teff_col="teff_estimate")


def test_propagate_errors_negative_parallax_returns_nan():
    from tasni.pipeline.bayesian_selection import propagate_errors

    df = pd.DataFrame(
        {
            "parallax": [-1.0, -0.2],
            "parallax_error": [0.5, 0.3],
            "teff_estimate": [400.0, 450.0],
        }
    )
    result = propagate_errors(df.copy())
    assert result["dist_posterior_mean"].isna().all()
    assert result["dist_posterior_std"].isna().all()


@patch("tasni.pipeline.bayesian_selection.pm.sample")
@patch("tasni.pipeline.bayesian_selection.az.summary")
def test_bayesian_fp_model_mocked(mock_summary, mock_sample, sample_df_with_parallax):
    """Test bayesian_fp_model with mocked PyMC sampling."""
    # Mock arviz summary to return a simple result
    mock_summary_result = MagicMock()
    mock_summary_result.__getitem__ = MagicMock(return_value=MagicMock(mean=0.08))
    mock_summary.return_value = mock_summary_result

    mock_trace = MagicMock()
    mock_sample.return_value = mock_trace

    from tasni.pipeline.bayesian_selection import bayesian_fp_model

    fp_mean, trace = bayesian_fp_model(sample_df_with_parallax)

    assert isinstance(fp_mean, float) or hasattr(fp_mean, "__float__")
    assert trace is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
