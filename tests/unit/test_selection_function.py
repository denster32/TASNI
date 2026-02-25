"""Unit tests for selection_function.py."""

import numpy as np
import pandas as pd


def test_catalog_completeness_bounds_and_shape():
    from tasni.analysis.selection_function import (
        calculate_2mass_completeness,
        calculate_gaia_completeness,
        calculate_wise_completeness,
    )

    mags = np.array([14.0, 15.5, 17.0, 18.5])
    for func in (
        calculate_wise_completeness,
        calculate_gaia_completeness,
        calculate_2mass_completeness,
    ):
        out_mags, comp = func(mags)
        assert len(out_mags) == len(mags)
        assert len(comp) == len(mags)
        assert np.all(comp >= 0.0)
        assert np.all(comp <= 1.0)


def test_combined_selection_function_has_uncertainties():
    from tasni.analysis.selection_function import calculate_combined_selection_function

    df = pd.DataFrame(
        {
            "designation": ["A", "B", "C"],
            "w1mpro": [14.0, 15.5, 16.5],
            "gaia_g_mag": [18.0, np.nan, 20.0],
            "twomass_k_mag": [13.5, 14.2, np.nan],
        }
    )
    sf = calculate_combined_selection_function(df)
    required = {
        "designation",
        "c_wise",
        "c_gaia",
        "c_2mass",
        "c_combined",
        "c_combined_err",
    }
    assert required.issubset(set(sf.columns))
    assert sf["c_combined"].between(0.001, 1.0).all()
    assert (sf["c_combined_err"] >= 0.0).all()


def test_corrected_density_not_lower_than_raw():
    from tasni.analysis.selection_function import calculate_corrected_space_density

    df = pd.DataFrame(
        {
            "designation": [f"S{i}" for i in range(50)],
            "w1mpro": np.linspace(14.0, 17.0, 50),
            "gaia_g_mag": np.linspace(17.0, 21.0, 50),
            "twomass_k_mag": np.linspace(13.0, 16.0, 50),
        }
    )
    result = calculate_corrected_space_density(df, max_distance=50.0, n_bootstrap=200)

    assert "density" in result
    assert "raw_density" in result
    assert "lower_95ci" in result
    assert "upper_95ci" in result
    assert result["density"] >= result["raw_density"]
    assert result["lower_95ci"] <= result["density"] <= result["upper_95ci"]
