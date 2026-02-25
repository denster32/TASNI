"""Unit tests for fig_gen.py pipeline module."""

import matplotlib

matplotlib.use("Agg")


import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ml_df():
    """Synthetic ML-scored DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "improved_composite_score": np.random.uniform(0, 1, n),
            "designation": [f"J{i:04d}" for i in range(n)],
        }
    )


@pytest.fixture
def sample_synth_df():
    """Synthetic population synthesis DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "fading_w1_mmag_yr": np.random.exponential(20, n),
            "teff": np.random.normal(350, 50, n),
        }
    )


@pytest.fixture
def sample_obs_df():
    """Synthetic observed golden DataFrame."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "trend_w1": np.random.exponential(15, n),
            "designation": [f"J{i:04d}" for i in range(n)],
        }
    )


def test_plot_ml_scores_creates_file(tmp_path, sample_ml_df):
    """plot_ml_scores should create a PNG file."""
    input_path = tmp_path / "ml_input.parquet"
    sample_ml_df.to_parquet(input_path, index=False)
    output_path = tmp_path / "ml_scores.png"

    from tasni.pipeline.fig_gen import plot_ml_scores

    plot_ml_scores(str(input_path), str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_pop_synth_creates_file(tmp_path, sample_synth_df, sample_obs_df):
    """plot_pop_synth should create a PNG file."""
    synth_path = tmp_path / "synth.parquet"
    sample_synth_df.to_parquet(synth_path, index=False)

    obs_path = tmp_path / "golden.csv"
    sample_obs_df.to_csv(obs_path, index=False)

    output_path = tmp_path / "pop_synth.png"

    from tasni.pipeline.fig_gen import plot_pop_synth

    plot_pop_synth(str(synth_path), str(obs_path), str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_pop_synth_no_trend_column(tmp_path, sample_synth_df):
    """plot_pop_synth should handle missing trend_w1 column gracefully."""
    synth_path = tmp_path / "synth.parquet"
    sample_synth_df.to_parquet(synth_path, index=False)

    obs_df = pd.DataFrame({"designation": ["J0001"], "ra": [180.0]})
    obs_path = tmp_path / "golden_no_trend.csv"
    obs_df.to_csv(obs_path, index=False)

    output_path = tmp_path / "pop_synth_notread.png"

    from tasni.pipeline.fig_gen import plot_pop_synth

    plot_pop_synth(str(synth_path), str(obs_path), str(output_path))

    assert output_path.exists()


def test_plot_ml_scores_histogram_range(tmp_path):
    """Scores outside [0,1] should still produce a valid plot."""
    df = pd.DataFrame({"improved_composite_score": [-0.5, 0.0, 0.5, 1.0, 1.5]})
    input_path = tmp_path / "edge.parquet"
    df.to_parquet(input_path, index=False)
    output_path = tmp_path / "edge_plot.png"

    from tasni.pipeline.fig_gen import plot_ml_scores

    plot_ml_scores(str(input_path), str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
