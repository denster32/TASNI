"""Tests for tier_vetoes.py pipeline module."""

from unittest.mock import patch

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from tasni.pipeline.tier_vetoes import (
    VetoConfig,
    apply_veto,
    query_catwise,
    query_ukidss,
    query_vhs,
)


@pytest.fixture
def sample_candidates():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "ra": np.random.uniform(0, 360, 10),
            "dec": np.random.uniform(-90, 90, 10),
            "designation": [f"J{i:08d}" for i in range(10)],
        }
    )


@pytest.fixture
def sample_coord():
    return SkyCoord(ra=180 * u.deg, dec=0 * u.deg)


def test_query_ukidss(sample_coord):
    """Test UKIDSS query with correct SkyCoord + Quantity signature."""
    with patch("tasni.pipeline.tier_vetoes.query_catalog") as mock_qc:
        mock_qc.return_value = pd.DataFrame({"_r": [2.5]})
        result = query_ukidss(sample_coord, 3 * u.arcsec)
        assert result is not None
        assert len(result) > 0
        mock_qc.assert_called_once()


def test_apply_veto(sample_candidates):
    """Test apply_veto with correct (DataFrame, VetoConfig) signature."""
    config = VetoConfig(radius_arcsec=2.0, batch_size=10)
    with (
        patch("tasni.pipeline.tier_vetoes.query_ukidss", return_value=None),
        patch("tasni.pipeline.tier_vetoes.query_vhs", return_value=None),
        patch("tasni.pipeline.tier_vetoes.query_catwise", return_value=None),
    ):
        df = apply_veto(sample_candidates, config)
    assert "has_ukidss" in df.columns
    assert "tier1_improved" in df.columns
    assert df["tier1_improved"].sum() >= 0


def test_vhs_catwise(sample_coord):
    """Test VHS and CatWISE queries with correct SkyCoord + Quantity signature."""
    with patch("tasni.pipeline.tier_vetoes.query_catalog", return_value=None):
        result_vhs = query_vhs(sample_coord, 3 * u.arcsec)
        result_catwise = query_catwise(sample_coord, 3 * u.arcsec)
    assert result_vhs is None
    assert result_catwise is None


if __name__ == "__main__":
    pytest.main(["-v"])
