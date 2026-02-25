import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path so tasni is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasni.filtering.filter_anomalies_full import (
    compute_thermal_profile,
    compute_weirdness_score,
    filter_quality,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "ra": [180.0, 180.1, 180.2],
            "cc_flags": ["0000", "p000", "00p0"],
            "ext_flg": [0, 0, 0],
            "ph_qual": ["AAA", "AAA", "AAA"],
            "w1mpro": [12.0, 13.0, 14.0],
            "w2mpro": [12.5, 13.5, 14.5],
            "w3mpro": [11.0, 12.0, 13.0],
            "w4mpro": [10.0, 11.0, 12.0],
            "nearest_gaia_sep_arcsec": [10.0, 10.0, 10.0],
        }
    )


class TestFilter:
    def test_filter_quality(self, sample_df):
        result = filter_quality(sample_df)
        assert len(result) >= 1

    def test_thermal_profile(self, sample_df):
        result = compute_thermal_profile(sample_df)
        assert "w1_w2" in result.columns

    def test_weirdness(self, sample_df):
        result = compute_weirdness_score(sample_df)
        assert "weirdness" in result.columns
