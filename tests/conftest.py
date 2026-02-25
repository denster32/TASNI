"""
Pytest fixtures for TASNI test suite
"""

import tempfile
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord


# Test configuration
@pytest.fixture
def test_config() -> dict[str, Any]:
    """Test configuration with temporary paths"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        yield {
            "wise_dir": tmp_path / "wise",
            "gaia_dir": tmp_path / "gaia",
            "output_dir": tmp_path / "output",
            "tmp_dir": tmp_path / "temp",
            "match_radius_arcsec": 3.0,
            "w1_w2_threshold": 0.5,
            "isolation_radius_deg": 0.5,
        }


@pytest.fixture
def temp_dirs(test_config):
    """Create and cleanup temporary directories"""
    dirs = [
        test_config["wise_dir"],
        test_config["gaia_dir"],
        test_config["output_dir"],
        test_config["tmp_dir"],
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    yield dirs

    # Cleanup happens automatically with tempfile


# Mock data fixtures


@pytest.fixture
def sample_wise_data() -> pd.DataFrame:
    """
    Create synthetic WISE catalog data for testing
    20 sources with various properties
    """
    np.random.seed(42)

    n_sources = 20

    # Random positions in a small region
    ra = np.random.uniform(180.0, 180.5, n_sources)
    dec = np.random.uniform(-30.0, -29.5, n_sources)

    # Random magnitudes (WISE bands)
    # Some sources will be bright, some faint
    w1mpro = np.random.uniform(10.0, 16.0, n_sources)
    w2mpro = w1mpro + np.random.uniform(0.1, 1.5, n_sources)  # W2 typically redder

    # Quality flags (ph_qual: A/B/C = good, U = upper limit)
    ph_qual_options = ["A", "B", "C", "A", "A", "B", "U"]  # Weight toward good quality
    ph_qual = np.random.choice(ph_qual_options, n_sources)

    # Some sources with cc_flags (contamination)
    cc_flags = np.random.choice(["00", "00", "00", "p0", "0p"], n_sources)

    # Variability flags
    var_flag = np.random.choice([0, 0, 0, 1], n_sources)

    # Create DataFrame
    # Simplified designation generation to avoid syntax errors
    designations = []
    for i in range(n_sources):
        designations.append(f"WISEA J{ra[i]:07.3f}{dec[i]:+07.3f}")

    df = pd.DataFrame(
        {
            "designation": designations,
            "ra": ra,
            "dec": dec,
            "w1mpro": w1mpro,
            "w2mpro": w2mpro,
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
def sample_gaia_data() -> pd.DataFrame:
    """
    Create synthetic Gaia catalog data for testing
    15 sources, some overlapping with WISE
    """
    np.random.seed(43)

    n_sources = 15

    # Positions - some overlapping with WISE, some not
    ra = np.random.uniform(180.0, 180.5, n_sources)
    dec = np.random.uniform(-30.0, -29.5, n_sources)

    # Gaia magnitudes
    g_mag = np.random.uniform(12.0, 19.0, n_sources)
    bp_mag = g_mag + np.random.uniform(0.0, 1.0, n_sources)
    rp_mag = g_mag - np.random.uniform(0.0, 1.0, n_sources)

    # Parallax (mostly distant sources)
    parallax = np.random.exponential(2.0, n_sources)  # mas
    parallax_err = np.random.uniform(0.1, 0.5, n_sources)

    # Proper motion (mas/yr)
    pmra = np.random.normal(0, 5, n_sources)
    pmdec = np.random.normal(0, 5, n_sources)
    pmra_err = np.random.uniform(0.1, 1.0, n_sources)
    pmdec_err = np.random.uniform(0.1, 1.0, n_sources)

    # Create DataFrame
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
            "pmra_error": pmra_err,
            "pmdec_error": pmdec_err,
        }
    )

    return df


@pytest.fixture
def sample_crossmatch_result() -> pd.DataFrame:
    """
    Sample cross-match results showing matched and unmatched sources
    """
    # 10 WISE sources, 6 matched to Gaia, 4 orphans
    data = {
        "wise_ra": [180.1, 180.2, 180.3, 180.4, 180.5, 180.6, 180.7, 180.8, 180.9, 181.0],
        "wise_dec": [-30.1, -30.2, -30.3, -30.4, -30.5, -30.6, -30.7, -30.8, -30.9, -31.0],
        "gaia_ra": [180.1001, 180.2001, 180.3001, None, None, 180.6001, None, None, None, None],
        "gaia_dec": [-30.1001, -30.2001, -30.3001, None, None, -30.6001, None, None, None, None],
        "separation_arcsec": [0.3, 0.25, 0.35, None, None, 0.4, None, None, None, None],
        "w1mpro": [12.5, 14.2, 15.8, 13.1, 14.5, 16.0, 12.8, 13.5, 14.9, 15.2],
        "w2mpro": [12.8, 14.9, 16.5, 13.5, 15.0, 16.7, 13.2, 14.0, 15.5, 15.8],
        "g_mag": [14.0, 15.5, 17.0, None, None, 18.2, None, None, None, None],
        "matched": [True, True, True, False, False, True, False, False, False, False],
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_anomalies() -> pd.DataFrame:
    """
    Sample filtered anomalies with various weirdness scores
    """
    data = {
        "wise_ra": [180.15, 180.25, 180.35, 180.45],
        "wise_dec": [-30.15, -30.25, -30.35, -30.45],
        "w1mpro": [12.5, 14.2, 15.8, 13.1],
        "w2mpro": [13.5, 15.2, 16.8, 13.9],
        "w1_w2_color": [1.0, 1.0, 1.0, 0.8],
        "isolation_score": [0.95, 0.88, 0.92, 0.75],
        "weirdness_score": [8.5, 7.2, 8.1, 6.3],
        "quality_flag": True,
    }

    return pd.DataFrame(data)


# Coordinate fixtures


@pytest.fixture
def sample_coordinates() -> SkyCoord:
    """Sample astropy SkyCoord object for coordinate matching tests (array of coords)"""
    ra = np.array([180.0, 180.1, 180.2, 180.3])
    dec = np.array([-30.0, -30.1, -30.2, -30.3])
    return SkyCoord(ra=ra * u.degree, dec=dec * u.degree)


@pytest.fixture
def sample_coord() -> SkyCoord:
    """Sample single astropy SkyCoord object for catalog query tests"""
    return SkyCoord(ra=180.0 * u.degree, dec=-30.0 * u.degree)


@pytest.fixture
def healpix_tiles() -> dict[str, int]:
    """
    Sample HEALPix tile configurations for different orders
    Order 5: 12,288 tiles
    Order 6: 49,152 tiles
    """
    return {
        "order_5": 12288,
        "order_6": 49152,
        "order_7": 196608,
    }


# Mock response fixtures


@pytest.fixture
def mock_tap_response():
    """Mock TAP service response structure"""

    class MockResponse:
        def __init__(self, data: pd.DataFrame):
            self.data = data.to_records(index=False)

        def to_table(self):
            from astropy.table import Table

            return Table.from_pandas(self.data)

    return MockResponse


@pytest.fixture
def mock_wise_tap_url() -> str:
    """Mock WISE TAP service URL"""
    return "http://irsa.ipac.caltech.edu/TAP"


@pytest.fixture
def mock_gaia_tap_url() -> str:
    """Mock Gaia TAP service URL"""
    return "http://gea.esac.esa.int/tap-server/tap"


# Test data file paths


@pytest.fixture
def wise_fixture_path(test_config) -> Path:
    """Path to WISE fixture data file"""
    return test_config["wise_dir"] / "wise_sample.csv"


@pytest.fixture
def gaia_fixture_path(test_config) -> Path:
    """Path to Gaia fixture data file"""
    return test_config["gaia_dir"] / "gaia_sample.csv"


@pytest.fixture
def create_fixture_files(sample_wise_data, sample_gaia_data, wise_fixture_path, gaia_fixture_path):
    """
    Create fixture CSV files for testing
    This fixture can be used in tests that need actual files
    """
    wise_fixture_path.parent.mkdir(parents=True, exist_ok=True)
    gaia_fixture_path.parent.mkdir(parents=True, exist_ok=True)

    sample_wise_data.to_csv(wise_fixture_path, index=False)
    sample_gaia_data.to_csv(gaia_fixture_path, index=False)

    return {
        "wise": wise_fixture_path,
        "gaia": gaia_fixture_path,
    }


# Pytest markers configuration


def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, may use files)")
    config.addinivalue_line("markers", "slow: Slow tests (actual network calls or large datasets)")
    config.addinivalue_line("markers", "postgres: Tests requiring PostgreSQL database")
