"""
Sample data fixtures for TASNI tests
"""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord


@pytest.fixture
def sample_wise_sources(n_sources=100):
    """
    Generate sample WISE catalog data
    """
    np.random.seed(42)

    # Random positions in a small region
    ra = np.random.uniform(180.0, 180.5, n_sources)
    dec = np.random.uniform(-30.0, -29.5, n_sources)

    # Random magnitudes
    w1mag = np.random.uniform(10.0, 16.0, n_sources)
    w2mag = w1mag + np.random.uniform(0.1, 1.5, n_sources)

    # Quality flags
    ph_qual_options = ["A", "B", "C", "A", "A", "B", "U"]
    ph_qual = np.random.choice(ph_qual_options, n_sources)

    # Some sources with contamination
    cc_flags = np.random.choice(["00", "00", "00", "p0", "0p"], n_sources)

    # Variability flags
    var_flag = np.random.choice([0, 0, 0, 1], n_sources)

    # Designations
    designations = [f"WISEA J{ra[i]:07.3f}{dec[i]:+07.3f}" for i in range(n_sources)]

    df = pd.DataFrame(
        {
            "designation": designations,
            "ra": ra,
            "dec": dec,
            "w1mpro": w1mag,
            "w2mpro": w2mag,
            "w1snr": np.random.uniform(10, 100, n_sources),
            "w2snr": np.random.uniform(10, 100, n_sources),
            "ph_qual": ph_qual,
            "cc_flags": cc_flags,
            "var_flag": var_flag,
            "ext_flg": np.random.choice([0, 1], n_sources, p=[0.9, 0.1]),
        }
    )

    return df


@pytest.fixture
def sample_gaia_sources(n_sources=50):
    """
    Generate sample Gaia catalog data
    """
    np.random.seed(43)

    # Positions - some overlapping with WISE
    ra = np.random.uniform(180.0, 180.5, n_sources)
    dec = np.random.uniform(-30.0, -29.5, n_sources)

    # Gaia magnitudes
    g_mag = np.random.uniform(12.0, 19.0, n_sources)
    bp_mag = g_mag + np.random.uniform(0.0, 1.0, n_sources)
    rp_mag = g_mag - np.random.uniform(0.0, 1.0, n_sources)

    # Parallax (mostly distant sources)
    parallax = np.random.exponential(2.0, n_sources)
    parallax_err = np.random.uniform(0.1, 0.5, n_sources)

    # Proper motion
    pmra = np.random.normal(0, 5, n_sources)
    pmdec = np.random.normal(0, 5, n_sources)

    df = pd.DataFrame(
        {
            "source_id": np.arange(4000000000000000000, 4000000000000000000 + n_sources),
            "ra": ra,
            "dec": dec,
            "phot_g_mean_mag": g_mag,
            "phot_bp_mean_mag": bp_mag,
            "phot_rp_mean_mag": rp_mag,
            "parallax": parallax,
            "parallax_error": parallax_err,
            "pmra": pmra,
            "pmdec": pmdec,
            "pmra_error": np.random.uniform(0.1, 1.0, n_sources),
            "pmdec_error": np.random.uniform(0.1, 1.0, n_sources),
        }
    )

    return df


@pytest.fixture
def sample_crossmatch_result():
    """
    Sample crossmatch results
    """
    data = {
        "wise_ra": [180.1, 180.2, 180.3, 180.4, 180.5],
        "wise_dec": [-30.1, -30.2, -30.3, -30.4, -30.5],
        "gaia_ra": [180.1001, 180.2001, None, None, None],
        "gaia_dec": [-30.1001, -30.2001, None, None, None],
        "separation_arcsec": [0.3, 0.25, None, None, None],
        "w1mag": [12.5, 14.2, 15.8, 13.1, 14.5],
        "w2mag": [12.8, 14.9, 16.5, 13.5, 15.0],
        "g_mag": [14.0, 15.5, None, None, None],
        "matched": [True, True, False, False, False],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_anomalies():
    """
    Sample filtered anomalies
    """
    data = {
        "wise_ra": [180.15, 180.25, 180.35, 180.45],
        "wise_dec": [-30.15, -30.25, -30.35, -30.45],
        "w1mag": [12.5, 14.2, 15.8, 13.1],
        "w2mag": [13.5, 15.2, 16.8, 13.9],
        "w1_w2_color": [1.0, 1.0, 1.0, 0.8],
        "isolation_score": [0.95, 0.88, 0.92, 0.75],
        "weirdness_score": [8.5, 7.2, 8.1, 6.3],
        "quality_flag": [True, True, True, False],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_neowise_epochs():
    """
    Sample NEOWISE time series data
    """
    n_epochs = 20
    mjd = np.arange(58000, 58000 + n_epochs * 0.5, 0.5)
    mjd_err = np.random.uniform(0.1, 0.5, n_epochs)
    w1mpro = 12.5 + np.random.normal(0, 0.1, n_epochs)
    w1sigmpro = np.random.uniform(0.02, 0.05, n_epochs)
    w2mpro = 13.5 + np.random.normal(0, 0.1, n_epochs)
    w2sigmpro = np.random.uniform(0.02, 0.05, n_epochs)

    # Simulate fading source
    w1mpro -= np.linspace(0, 0.5, n_epochs)

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "mjd_error": mjd_err,
            "w1mpro": w1mpro,
            "w1sigmpro": w1sigmpro,
            "w2mpro": w2mpro,
            "w2sigmpro": w2sigmpro,
            "ph_qual": np.random.choice(["A", "B"], n_epochs),
            "cc_flags": np.random.choice(["00", "00", "00", "p0"], n_epochs),
        }
    )

    return df


@pytest.fixture
def sample_coordinates():
    """
    Sample astropy SkyCoord object
    """
    ra = np.array([180.0, 180.1, 180.2, 180.3])
    dec = np.array([-30.0, -30.1, -30.2, -30.3])
    return SkyCoord(ra=ra * u.degree, dec=dec * u.degree)


@pytest.fixture
def sample_spectroscopic_targets():
    """
    Sample spectroscopic target list
    """
    data = {
        "designation": [
            "WISEA J180100.0-300500.0",
            "WISEA J180150.0-301000.0",
            "WISEA J180200.0-301500.0",
        ],
        "ra": [180.1, 180.15, 180.2],
        "dec": [-30.5, -30.6, -30.7],
        "priority": ["HIGH", "MEDIUM", "LOW"],
        "w1mag": [12.5, 13.5, 14.5],
        "fade_rate": [-0.1, -0.05, 0.0],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_healpix_indices():
    """
    Sample HEALPix tile indices
    """
    return {
        "order_5": np.arange(12288),
        "order_6": np.arange(49152),
        "sample_indices": [0, 1234, 5678, 9999, 12287],
    }


@pytest.fixture
def sample_filtering_results():
    """
    Sample filtering pipeline results
    """
    return {
        "total_sources": 747000000,
        "after_wise_filter": 747000000,
        "after_gaia_veto": 405000000,
        "after_w1w2_filter": 4050000,
        "after_2mass_veto": 243000,
        "after_ps1_veto": 151000,
        "after_legacy_veto": 150000,
        "after_nvss_veto": 16500,
        "after_temp_filter": 8200,
        "after_lamost_filter": 8100,
        "final_candidates": 100,
    }


@pytest.fixture
def sample_variability_metrics():
    """
    Sample variability metrics
    """
    return {
        "rms": [0.05, 0.03, 0.10, 0.02],
        "chi2": [15.2, 8.5, 45.3, 5.1],
        "stetson_j": [2.3, 1.5, 5.8, 0.8],
        "fade_rate": [-0.01, -0.005, -0.02, 0.0],
        "is_variable": [True, True, True, False],
    }
