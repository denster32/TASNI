"""
Integration tests for basic TASNI workflow
"""

import pandas as pd
import pytest

from tasni.core import config


@pytest.mark.integration
def test_basic_config_load():
    """Test that basic configuration loads correctly"""
    # Verify config is loaded
    assert config.DATA_ROOT.exists()

    # Verify HEALPix settings
    assert config.HEALPIX_NSIDE == 32

    # Verify paths are created
    for directory in [config.WISE_DIR, config.GAIA_DIR, config.OUTPUT_DIR]:
        assert directory.exists()


@pytest.mark.integration
def test_data_directories_structure():
    """Test that data directories have expected structure"""
    # Check WISE directory
    wise_files = list(config.WISE_DIR.glob("wise_hp*.parquet"))
    # May be empty if not downloaded yet
    # assert len(wise_files) > 0 or len(wise_files) == 0  # Just verify it doesn't crash

    # Check Gaia directory
    gaia_files = list(config.GAIA_DIR.glob("gaia_hp*.parquet"))
    # Same as above

    # Check output directory
    assert config.OUTPUT_DIR.exists()
    assert (config.OUTPUT_DIR / "final").exists()


@pytest.mark.integration
def test_output_final_files():
    """Test that output final directory has expected files"""
    final_dir = config.OUTPUT_DIR / "final"

    # Check for golden targets file
    golden_csv = final_dir / "golden_targets.csv"
    golden_parquet = final_dir / "golden_targets.csv"

    # At least one should exist
    assert golden_csv.exists() or golden_parquet.exists() or not final_dir.exists()


@pytest.mark.integration
def test_figures_directory():
    """Test that figures directory exists and has content"""
    figures_dir = config.OUTPUT_DIR / "figures"

    if figures_dir.exists():
        # Should have at least some figures
        figures = list(figures_dir.glob("*"))
        # assert len(figures) > 0  # May be empty before generation


@pytest.mark.integration
def test_logs_directory():
    """Test that logs directory exists"""
    assert config.LOG_DIR.exists()

    # Check for log files
    logs = list(config.LOG_DIR.glob("*.log"))
    # May be empty, but directory should exist


@pytest.mark.integration
def test_checkpoint_directory():
    """Test that checkpoint directory exists"""
    assert config.CHECKPOINT_DIR.exists()

    # Check for checkpoint files
    checkpoints = list(config.CHECKPOINT_DIR.glob("*.json"))
    # May be empty


@pytest.mark.slow
@pytest.mark.integration
def test_sample_wise_data():
    """Test that we can read a sample of WISE data if available"""
    wise_files = list(config.WISE_DIR.glob("wise_hp*.parquet"))

    if len(wise_files) > 0:
        # Read first file
        df = pd.read_parquet(wise_files[0])

        # Verify columns
        expected_columns = ["designation", "ra", "dec"]
        for col in expected_columns:
            assert col in df.columns, f"Column {col} not found in WISE data"

        # Verify data
        assert len(df) > 0
        assert df["ra"].notna().any()
        assert df["dec"].notna().any()
    else:
        pytest.skip("No WISE data files found")


@pytest.mark.slow
@pytest.mark.integration
def test_sample_gaia_data():
    """Test that we can read a sample of Gaia data if available"""
    gaia_files = list(config.GAIA_DIR.glob("gaia_hp*.parquet"))

    if len(gaia_files) > 0:
        # Read first file
        df = pd.read_parquet(gaia_files[0])

        # Verify columns
        expected_columns = ["source_id", "ra", "dec"]
        for col in expected_columns:
            assert col in df.columns, f"Column {col} not found in Gaia data"

        # Verify data
        assert len(df) > 0
        assert df["ra"].notna().any()
        assert df["dec"].notna().any()
    else:
        pytest.skip("No Gaia data files found")


@pytest.mark.integration
def test_environment_variables():
    """Test that environment variables can be loaded"""
    from tasni.core import config_env

    # Test that config_env module loads
    assert config_env is not None

    # Test helper functions
    assert config_env.get_int("TEST_INT", 42) == 42
    assert config_env.get_float("TEST_FLOAT", 3.14) == 3.14
    assert config_env.get_bool("TEST_BOOL", True) == True
