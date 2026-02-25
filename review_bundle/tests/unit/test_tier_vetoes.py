"""Unit tests for tier_vetoes.py."""

from unittest.mock import patch

import pandas as pd
import pytest
from astropy import units as u

from tasni.pipeline.tier_vetoes import VetoConfig, apply_veto, query_ukidss


@pytest.fixture
def sample_df():
    """Sample DataFrame with RA/Dec."""
    return pd.DataFrame(
        {"ra": [180.0, 181.0], "dec": [30.0, 31.0], "source_id": ["test1", "test2"]}
    )


@pytest.fixture
def veto_config():
    """Default VetoConfig."""
    return VetoConfig(radius_arcsec=3.0, batch_size=50)


def test_veto_config_validation():
    """Test Pydantic config validation."""
    config = VetoConfig(radius_arcsec=2.5, batch_size=20)
    assert config.radius_arcsec == 2.5
    with pytest.raises(ValueError):
        VetoConfig(batch_size=15)  # Not multiple of 10


@patch("tasni.pipeline.tier_vetoes.query_catalog")
def test_query_ukidss(mock_query_catalog, sample_coord):
    """Test UKIDSS query wrapper."""
    mock_df = pd.DataFrame({"_r": [1.0]})
    mock_query_catalog.return_value = mock_df
    result = query_ukidss(sample_coord, 3 * u.arcsec)
    assert result is not None
    assert len(result) == 1
    mock_query_catalog.assert_called_once()


@pytest.mark.parametrize(
    "has_match,expected_has_ukidss",
    [
        (True, True),
        (False, False),
    ],
)
@patch("tasni.pipeline.tier_vetoes.query_ukidss")
@patch("tasni.pipeline.tier_vetoes.query_vhs")
@patch("tasni.pipeline.tier_vetoes.query_catwise")
def test_apply_veto(
    mock_catwise, mock_vhs, mock_ukidss, sample_df, veto_config, has_match, expected_has_ukidss
):
    """Parametrized test for apply_veto."""
    mock_result = pd.DataFrame({"_r": [1.0]}) if has_match else None

    mock_ukidss.return_value = mock_result
    mock_vhs.return_value = None
    mock_catwise.return_value = None

    result_df = apply_veto(sample_df, veto_config)

    assert result_df["has_ukidss"].iloc[0] == expected_has_ukidss
    assert "tier1_improved" in result_df.columns
    assert len(result_df) == len(sample_df)


def test_run_tier_vetoes(tmp_path, sample_df):
    """Test runner with temp files (mocks external deps)."""
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    sample_df.to_parquet(input_path)

    config = VetoConfig()

    with (
        patch("tasni.pipeline.tier_vetoes.query_ukidss", return_value=None),
        patch("tasni.pipeline.tier_vetoes.query_vhs", return_value=None),
        patch("tasni.pipeline.tier_vetoes.query_catwise", return_value=None),
    ):
        from tasni.pipeline.tier_vetoes import run_tier_vetoes

        run_tier_vetoes(input_path, output_path, config)

    assert output_path.exists()
    loaded = pd.read_parquet(output_path)
    assert len(loaded) == len(sample_df)
