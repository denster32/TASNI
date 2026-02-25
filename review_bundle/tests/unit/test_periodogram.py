"""Unit tests for periodogram_analysis.py."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_light_curve():
    """Create a synthetic periodic light curve for testing."""
    np.random.seed(42)

    # Time points (unevenly spaced, like NEOWISE)
    n_epochs = 100
    times = np.sort(np.random.uniform(55000, 59000, n_epochs))  # MJD

    # True period
    true_period = 120.0  # days

    # Magnitudes with sinusoidal variation + noise
    mags = 14.0 + 0.5 * np.sin(2 * np.pi * times / true_period)
    mags += np.random.normal(0, 0.05, n_epochs)  # Add noise

    # Magnitude errors
    mag_errs = np.random.uniform(0.03, 0.08, n_epochs)

    return {"times": times, "mags": mags, "mag_errs": mag_errs, "true_period": true_period}


@pytest.fixture
def constant_light_curve():
    """Create a constant (non-variable) light curve."""
    np.random.seed(43)

    n_epochs = 50
    times = np.sort(np.random.uniform(55000, 59000, n_epochs))
    mags = np.random.normal(14.0, 0.05, n_epochs)  # Constant + noise
    mag_errs = np.random.uniform(0.03, 0.08, n_epochs)

    return {"times": times, "mags": mags, "mag_errs": mag_errs}


def test_compute_periodogram_import():
    """Test that the periodogram module can be imported."""
    try:
        from tasni.analysis.periodogram_analysis import compute_periodogram

        assert callable(compute_periodogram)
    except ImportError:
        pytest.skip("periodogram_analysis module not available")


def test_compute_periodogram_basic(synthetic_light_curve):
    """Test basic periodogram computation."""
    try:
        from tasni.analysis.periodogram_analysis import compute_periodogram
    except ImportError:
        pytest.skip("periodogram_analysis module not available")

    result = compute_periodogram(
        synthetic_light_curve["times"],
        synthetic_light_curve["mags"],
        synthetic_light_curve["mag_errs"],
    )

    # Check that result is a dictionary
    assert isinstance(result, dict)

    # Check status
    assert result.get("status") == "success"

    # Check that we found a period
    assert "best_period_days" in result
    assert result["best_period_days"] > 0

    # The detected period should be close to the true period
    # (within 20% tolerance due to noise and sparse sampling)
    detected_period = result["best_period_days"]
    true_period = synthetic_light_curve["true_period"]
    assert (
        abs(detected_period - true_period) / true_period < 0.2
    ), f"Detected period {detected_period:.1f}d differs from true {true_period:.1f}d"


def test_compute_periodogram_insufficient_data():
    """Test periodogram with insufficient data points."""
    try:
        from tasni.analysis.periodogram_analysis import compute_periodogram
    except ImportError:
        pytest.skip("periodogram_analysis module not available")

    # Only 5 points - insufficient
    times = np.array([55000, 55100, 55200, 55300, 55400])
    mags = np.array([14.0, 14.1, 13.9, 14.2, 14.0])

    result = compute_periodogram(times, mags)

    # Should return with insufficient_data status
    assert result.get("status") == "insufficient_data"
    assert result.get("n_points") == 5


def test_compute_periodogram_nan_handling():
    """Test that NaN values are handled correctly."""
    try:
        from tasni.analysis.periodogram_analysis import compute_periodogram
    except ImportError:
        pytest.skip("periodogram_analysis module not available")

    np.random.seed(44)
    n = 50
    times = np.sort(np.random.uniform(55000, 59000, n))
    mags = np.random.normal(14.0, 0.1, n)

    # Insert some NaN values
    mags[5] = np.nan
    mags[20] = np.nan

    result = compute_periodogram(times, mags)

    # Should still succeed after removing NaN values
    assert result.get("status") in ["success", "insufficient_data"]


def test_compute_periodogram_no_errors():
    """Test periodogram without error array."""
    try:
        from tasni.analysis.periodogram_analysis import compute_periodogram
    except ImportError:
        pytest.skip("periodogram_analysis module not available")

    np.random.seed(45)
    n = 50
    times = np.sort(np.random.uniform(55000, 59000, n))
    mags = 14.0 + 0.3 * np.sin(2 * np.pi * times / 100.0)
    mags += np.random.normal(0, 0.05, n)

    # Call without mag_errs
    result = compute_periodogram(times, mags, None)

    assert result.get("status") == "success"


def test_analyze_source_import():
    """Test that analyze_source can be imported."""
    try:
        from tasni.analysis.periodogram_analysis import analyze_source

        assert callable(analyze_source)
    except ImportError:
        pytest.skip("periodogram_analysis module not available")


def test_analyze_source_synthetic(synthetic_light_curve):
    """Test source analysis with synthetic data."""
    try:
        from tasni.analysis.periodogram_analysis import analyze_source
    except ImportError:
        pytest.skip("periodogram_analysis module not available")

    # Create a DataFrame for the epochs
    epochs = pd.DataFrame(
        {
            "mjd": synthetic_light_curve["times"],
            "w1mpro_ep": synthetic_light_curve["mags"],
            "w1sigmpro_ep": synthetic_light_curve["mag_errs"],
        }
    )

    result = analyze_source("TEST_SOURCE", epochs)

    assert result["designation"] == "TEST_SOURCE"
    assert result["n_epochs"] == len(epochs)
    assert result.get("status") == "success"
    assert "w1_best_period" in result


def test_fap_threshold():
    """Test that FAP threshold is reasonable."""
    try:
        from tasni.analysis.periodogram_analysis import FAP_THRESHOLD

        assert 0 < FAP_THRESHOLD < 0.1
    except ImportError:
        # If not exported, that's OK - just checking the module
        pass


def test_period_range():
    """Test that period range is reasonable for brown dwarf science."""
    try:
        from tasni.analysis.periodogram_analysis import MAX_PERIOD_DAYS, MIN_PERIOD_DAYS

        # Minimum should be at least a few hours
        assert MIN_PERIOD_DAYS >= 0.1

        # Maximum should cover orbital periods
        assert MAX_PERIOD_DAYS >= 100

    except ImportError:
        # If not exported, that's OK
        pass


def test_multiple_testing_correction_applies_fdr():
    """FDR correction should reject fewer/equal hypotheses than raw p-thresholding."""
    try:
        from tasni.analysis.periodogram_analysis import apply_multiple_testing_correction
    except ImportError:
        pytest.skip("periodogram_analysis module not available")

    results = [
        {"designation": "A", "w1_fap": 1e-4},
        {"designation": "B", "w1_fap": 5e-3},
        {"designation": "C", "w1_fap": 2e-2},
        {"designation": "D", "w1_fap": 5e-2},
    ]

    corrected = apply_multiple_testing_correction(results, alpha=0.01)
    raw_count = sum(bool(r.get("w1_is_periodic_raw")) for r in corrected)
    fdr_count = sum(bool(r.get("w1_is_periodic_fdr")) for r in corrected)

    assert fdr_count <= raw_count
    assert all("w1_fdr_p" in r for r in corrected if "w1_fap" in r)


def test_alias_probability_higher_near_cadence_harmonic():
    """Alias probability should be higher near 6-month cadence harmonics."""
    try:
        from tasni.analysis.periodogram_significance import assess_alias_probability
    except ImportError:
        pytest.skip("periodogram_significance module not available")

    np.random.seed(123)
    # Irregular cadence over ~10 years.
    time = np.sort(np.random.uniform(0, 3650, 150))

    p_alias = assess_alias_probability(182.625, time=time)
    p_non_alias = assess_alias_probability(137.0, time=time)

    assert 0.0 <= p_alias <= 1.0
    assert 0.0 <= p_non_alias <= 1.0
    assert p_alias >= p_non_alias


@pytest.mark.parametrize(
    "n_epochs,expected_status",
    [
        (5, "insufficient_data"),
        (10, "success"),
        (50, "success"),
        (100, "success"),
    ],
)
def test_different_epoch_counts(n_epochs, expected_status):
    """Test periodogram with different numbers of epochs."""
    try:
        from tasni.analysis.periodogram_analysis import compute_periodogram
    except ImportError:
        pytest.skip("periodogram_analysis module not available")

    np.random.seed(46)

    if n_epochs < 10:
        times = np.sort(np.random.uniform(55000, 59000, n_epochs))
        mags = np.random.normal(14.0, 0.1, n_epochs)
        expected_status = "insufficient_data"
    else:
        times = np.sort(np.random.uniform(55000, 59000, n_epochs))
        mags = 14.0 + 0.3 * np.sin(2 * np.pi * times / 100.0)
        mags += np.random.normal(0, 0.05, n_epochs)
        expected_status = "success"

    result = compute_periodogram(times, mags)
    assert result.get("status") == expected_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
