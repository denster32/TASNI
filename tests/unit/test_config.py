"""
Unit tests for configuration management
"""

import os
from pathlib import Path


def test_config_paths_exist():
    """Test that configured paths exist or can be created"""
    from tasni.core.config import DATA_ROOT, OUTPUT_DIR

    # These paths should exist or their parent should exist
    assert DATA_ROOT.parent.exists(), f"DATA_ROOT parent does not exist: {DATA_ROOT.parent}"
    assert OUTPUT_DIR.parent.exists(), f"OUTPUT_DIR parent does not exist: {OUTPUT_DIR.parent}"


def test_config_healpix_settings():
    """Test HEALPix configuration"""
    from tasni.core.config import HEALPIX_NSIDE

    assert HEALPIX_NSIDE == 32, f"HEALPIX_NSIDE should be 32, got {HEALPIX_NSIDE}"
    # NPIXELS = 12 * NSIDE^2
    expected_npixels = 12 * HEALPIX_NSIDE**2
    assert expected_npixels == 12288


def test_config_tap_urls():
    """Test TAP service URLs"""
    from tasni.core.config import GAIA_TAP_URL, WISE_TAP_URL

    assert WISE_TAP_URL.startswith("http"), f"WISE_TAP_URL invalid: {WISE_TAP_URL}"
    assert GAIA_TAP_URL.startswith("http"), f"GAIA_TAP_URL invalid: {GAIA_TAP_URL}"


def test_config_catalog_names():
    """Test catalog names"""
    from tasni.core.config import WISE_CATALOG

    assert isinstance(WISE_CATALOG, str), "WISE_CATALOG should be string"


def test_config_filtering_thresholds():
    """Test filtering thresholds"""
    from tasni.core.config import MATCH_RADIUS_ARCSEC, MIN_SNR_W1, MIN_SNR_W2

    assert MATCH_RADIUS_ARCSEC > 0, "MATCH_RADIUS_ARCSEC should be positive"
    assert MIN_SNR_W1 > 0, "MIN_SNR_W1 should be positive"
    assert MIN_SNR_W2 > 0, "MIN_SNR_W2 should be positive"


def test_config_parallelization():
    """Test parallelization settings"""
    from tasni.core.config import CHUNK_SIZE, N_WORKERS

    assert N_WORKERS > 0, "N_WORKERS should be positive"
    assert CHUNK_SIZE > 0, "CHUNK_SIZE should be positive"


def test_config_env_variables():
    """Test environment variable loading"""
    from tasni.core.config_env import get_bool, get_float, get_int, get_path

    # Test get_int
    assert get_int("NONEXISTENT_INT", 42) == 42
    os.environ["TEST_INT"] = "123"
    assert get_int("TEST_INT", 42) == 123
    del os.environ["TEST_INT"]

    # Test get_float
    assert get_float("NONEXISTENT_FLOAT", 3.14) == 3.14
    os.environ["TEST_FLOAT"] = "2.718"
    assert get_float("TEST_FLOAT", 3.14) == 2.718
    del os.environ["TEST_FLOAT"]

    # Test get_bool
    assert get_bool("NONEXISTENT_BOOL", True) == True
    assert get_bool("NONEXISTENT_BOOL", False) == False

    os.environ["TEST_BOOL_TRUE"] = "true"
    assert get_bool("TEST_BOOL_TRUE", False) == True
    del os.environ["TEST_BOOL_TRUE"]

    os.environ["TEST_BOOL_FALSE"] = "false"
    assert get_bool("TEST_BOOL_FALSE", True) == False
    del os.environ["TEST_BOOL_FALSE"]

    # Test get_path
    test_path = Path("/tmp/test")
    assert get_path("NONEXISTENT_PATH", test_path) == test_path


def test_config_logging():
    """Test logging-related config"""
    from tasni.core.config import LOG_DIR

    assert isinstance(LOG_DIR, Path), "LOG_DIR should be a Path"


def test_config_gpu_detection():
    """Test GPU detection (can be auto or manual)"""
    from tasni.core.config_env import USE_CUDA, USE_XPU

    # Just verify these are boolean
    assert isinstance(USE_CUDA, bool), f"USE_CUDA should be bool, got {type(USE_CUDA)}"
    assert isinstance(USE_XPU, bool), f"USE_XPU should be bool, got {type(USE_XPU)}"


def test_config_scoring_weights():
    """Test scoring weight configuration"""
    from tasni.core.config import (
        COLOR_BONUS_WEIGHT,
        IR_BRIGHTNESS_WEIGHT,
        ISOLATION_BONUS_WEIGHT,
        OPTICAL_PENALTY_WEIGHT,
    )

    assert isinstance(IR_BRIGHTNESS_WEIGHT, (int, float))
    assert isinstance(COLOR_BONUS_WEIGHT, (int, float))
    assert isinstance(OPTICAL_PENALTY_WEIGHT, (int, float))
    assert isinstance(ISOLATION_BONUS_WEIGHT, (int, float))


def test_config_network_settings():
    """Test network configuration"""
    from tasni.core.config import MAX_CONNECTIONS, MAX_RETRIES, REQUEST_TIMEOUT

    assert MAX_RETRIES > 0
    assert REQUEST_TIMEOUT > 0
    assert MAX_CONNECTIONS > 0


def test_config_variability_settings():
    """Test variability analysis settings"""
    from tasni.core.config import (
        CHI2_VARIABILITY_THRESHOLD,
        MIN_BASELINE_YEARS,
        MIN_EPOCHS_VARIABILITY,
    )

    assert MIN_EPOCHS_VARIABILITY > 0
    assert MIN_BASELINE_YEARS > 0
    assert CHI2_VARIABILITY_THRESHOLD > 0


def test_tasni_version():
    """Test that TASNI version is accessible"""
    import tasni

    assert hasattr(tasni, "__version__")
    assert tasni.__version__ == "1.0.0"
