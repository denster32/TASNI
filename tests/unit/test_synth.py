"""Unit tests for synth.py pipeline module."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def default_synth():
    """Generate a synthetic catalog with fixed seed for reproducibility."""
    np.random.seed(42)
    from tasni.pipeline.synth import generate_synth_catalog

    return generate_synth_catalog(n_samples=5000)


def test_generate_synth_catalog_returns_dataframe():
    """generate_synth_catalog should return a DataFrame."""
    np.random.seed(42)
    from tasni.pipeline.synth import generate_synth_catalog

    result = generate_synth_catalog(n_samples=100)
    assert isinstance(result, pd.DataFrame)


def test_expected_columns(default_synth):
    """Synthetic catalog must contain required columns."""
    expected = [
        "ra",
        "dec",
        "teff",
        "logg",
        "dist_pc",
        "pm_tot",
        "fading_w1_mmag_yr",
        "fading_w2_mmag_yr",
        "w1_mag",
        "w2_mag",
        "w1_w2",
        "rms_w1",
    ]
    for col in expected:
        assert col in default_synth.columns, f"Missing column: {col}"


def test_teff_range_clipped(default_synth):
    """Teff values should be clipped to [200, 500] K."""
    assert default_synth["teff"].min() >= 200
    assert default_synth["teff"].max() <= 500


def test_logg_range_clipped(default_synth):
    """logg values should be clipped to [4.5, 5.5]."""
    assert default_synth["logg"].min() >= 4.5
    assert default_synth["logg"].max() <= 5.5


def test_row_count_within_range():
    """After clipping, row count should be less than n_samples but > 0."""
    np.random.seed(42)
    from tasni.pipeline.synth import generate_synth_catalog

    result = generate_synth_catalog(n_samples=10000)
    # Clipping removes some rows, but should retain a meaningful fraction
    assert len(result) > 0
    assert len(result) <= 10000


def test_ra_dec_ranges(default_synth):
    """RA should be [0,360], Dec should be [-90,90]."""
    assert default_synth["ra"].min() >= 0
    assert default_synth["ra"].max() <= 360
    assert default_synth["dec"].min() >= -90
    assert default_synth["dec"].max() <= 90


def test_fading_rates_positive(default_synth):
    """Fading rates (exponential) should be non-negative."""
    assert (default_synth["fading_w1_mmag_yr"] >= 0).all()
    assert (default_synth["fading_w2_mmag_yr"].notna()).all()


def test_reproducibility_with_seed():
    """Same random seed should produce identical catalogs."""
    from tasni.pipeline.synth import generate_synth_catalog

    np.random.seed(123)
    cat1 = generate_synth_catalog(n_samples=500)

    np.random.seed(123)
    cat2 = generate_synth_catalog(n_samples=500)

    pd.testing.assert_frame_equal(cat1, cat2)


def test_small_sample():
    """Should work with very small n_samples."""
    np.random.seed(42)
    from tasni.pipeline.synth import generate_synth_catalog

    result = generate_synth_catalog(n_samples=50)
    # Might lose some to clipping, but should not error
    assert isinstance(result, pd.DataFrame)


def test_distance_positive(default_synth):
    """Distances should be positive (lognormal)."""
    assert (default_synth["dist_pc"] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
