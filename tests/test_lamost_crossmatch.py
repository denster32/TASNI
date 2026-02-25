"""
Tests for LAMOST cross-matching functionality

Tests the integration with LAMOST spectroscopy data for:
1. Temperature estimation from IR colors
2. Spectral subclass classification
3. LAMOST scoring logic
4. Cross-match functionality
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tasni.crossmatch.crossmatch_lamost import (
    classify_lamost_subclass,
    compute_ir_temperature,
    compute_lamost_scores,
)


class TestIRTemperature:
    """Tests for IR temperature estimation from WISE colors"""

    def test_hot_star_returns_none(self):
        """Hot stars (small W1-W2) should return None"""
        teff = compute_ir_temperature(w1=12.0, w2=11.8)  # W1-W2 = 0.2
        assert teff is None

    def test_l_dwarf_temperature(self):
        """L dwarfs should return ~1300-2000K"""
        teff = compute_ir_temperature(w1=14.0, w2=12.5)  # W1-W2 = 1.5
        assert teff is not None
        assert 1000 <= teff <= 2500

    def test_t_dwarf_temperature(self):
        """T dwarfs should return ~600-1300K"""
        teff = compute_ir_temperature(w1=16.0, w2=13.5)  # W1-W2 = 2.5
        assert teff is not None
        assert 500 <= teff <= 1500

    def test_y_dwarf_temperature(self):
        """Y dwarfs should return <600K"""
        teff = compute_ir_temperature(w1=18.0, w2=13.5)  # W1-W2 = 4.5
        assert teff is not None
        assert teff < 700

    def test_invalid_magnitudes_return_none(self):
        """Invalid magnitudes should return None"""
        assert compute_ir_temperature(w1=99, w2=12.0) is None
        assert compute_ir_temperature(w1=np.nan, w2=12.0) is None
        assert compute_ir_temperature(w1=12.0, w2=np.nan) is None

    def test_temperature_floor(self):
        """Extreme Y dwarfs should not go below floor temperature"""
        teff = compute_ir_temperature(w1=20.0, w2=13.0)  # W1-W2 = 7.0
        assert teff is not None
        assert teff >= 250  # Floor value


class TestSubclassClassification:
    """Tests for LAMOST subclass classification"""

    def test_m_dwarf_is_known_ir(self):
        """M dwarfs should be classified as known IR types"""
        category, is_known = classify_lamost_subclass("M5V")
        assert category == "COOL_DWARF"
        assert is_known is True

    def test_l_dwarf_is_known_ir(self):
        """L dwarfs should be classified as known IR types"""
        category, is_known = classify_lamost_subclass("L3")
        assert category == "COOL_DWARF"
        assert is_known is True

    def test_t_dwarf_is_known_ir(self):
        """T dwarfs should be classified as known IR types"""
        category, is_known = classify_lamost_subclass("T5")
        assert category == "COOL_DWARF"
        assert is_known is True

    def test_carbon_star_is_known_ir(self):
        """Carbon stars should be classified as known IR types"""
        category, is_known = classify_lamost_subclass("C-N4")
        assert category == "CARBON_STAR"
        assert is_known is True

    def test_s_type_is_known_ir(self):
        """S-type stars should be classified as known IR types"""
        category, is_known = classify_lamost_subclass("S3")
        assert category == "S_TYPE"
        assert is_known is True

    def test_normal_star_not_known_ir(self):
        """Normal stars should NOT be classified as known IR types"""
        category, is_known = classify_lamost_subclass("G2V")
        assert category == "NORMAL_STAR"
        assert is_known is False

    def test_unknown_returns_unknown(self):
        """Unknown subclass should return UNKNOWN category"""
        category, is_known = classify_lamost_subclass("Unknown")
        assert category == "UNKNOWN"
        assert is_known is False

        category, is_known = classify_lamost_subclass("")
        assert category == "UNKNOWN"
        assert is_known is False

        category, is_known = classify_lamost_subclass(np.nan)
        assert category == "UNKNOWN"
        assert is_known is False


class TestLAMOSTScoring:
    """Tests for LAMOST-based anomaly scoring"""

    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate data with LAMOST matches"""
        return pd.DataFrame(
            {
                "designation": ["TEST_001", "TEST_002", "TEST_003", "TEST_004", "TEST_005"],
                "ra": [180.0, 180.1, 180.2, 180.3, 180.4],
                "dec": [-30.0, -30.1, -30.2, -30.3, -30.4],
                "w1mpro": [14.0, 15.0, 16.0, 14.5, 13.0],
                "w2mpro": [12.5, 13.5, 14.0, 13.0, 11.5],
                "lamost_match": [True, True, True, False, True],
                "lamost_subclass": ["M5", "Unknown", "G2V", None, "L3"],
                "lamost_teff": [3000, None, 5800, None, 1500],
            }
        )

    def test_known_ir_type_penalized(self, sample_candidates):
        """Known IR types should receive penalty"""
        result = compute_lamost_scores(sample_candidates)

        # M5 dwarf (index 0) should be penalized
        m_dwarf_score = result.loc[0, "lamost_score"]
        assert m_dwarf_score < 0
        assert result.loc[0, "lamost_is_known_ir"] == True

    def test_unknown_type_bonus(self, sample_candidates):
        """Unknown spectral type should receive bonus"""
        result = compute_lamost_scores(sample_candidates)

        # Unknown type (index 1) should get bonus
        unknown_score = result.loc[1, "lamost_score"]
        assert unknown_score > 0

    def test_no_match_neutral(self, sample_candidates):
        """No LAMOST match should have neutral score"""
        result = compute_lamost_scores(sample_candidates)

        # No match (index 3) should have score of 0
        no_match_score = result.loc[3, "lamost_score"]
        assert no_match_score == 0

    def test_ir_temperature_computed(self, sample_candidates):
        """IR temperature should be computed from WISE colors"""
        result = compute_lamost_scores(sample_candidates)

        # All rows should have ir_teff computed
        assert "ir_teff" in result.columns
        assert result["ir_teff"].notna().any()

    def test_temp_mismatch_detected(self):
        """Temperature mismatch should be detected when IR and spectral Teff disagree"""
        df = pd.DataFrame(
            {
                "designation": ["MISMATCH_TEST"],
                "ra": [180.0],
                "dec": [-30.0],
                "w1mpro": [15.0],  # Implies cool IR (~1000K)
                "w2mpro": [12.5],  # W1-W2 = 2.5
                "lamost_match": [True],
                "lamost_subclass": ["G2V"],
                "lamost_teff": [5800],  # Hot spectral temp
            }
        )

        result = compute_lamost_scores(df)

        # Should detect mismatch (5800K spectral vs ~1000K IR)
        assert result.loc[0, "lamost_temp_mismatch"] == True

    def test_category_assigned(self, sample_candidates):
        """LAMOST category should be assigned to all sources"""
        result = compute_lamost_scores(sample_candidates)

        assert "lamost_category" in result.columns
        assert result["lamost_category"].notna().all()


class TestScoringConstants:
    """Test that scoring constants are imported correctly"""

    def test_penalty_is_negative(self):
        """Known type penalty should be negative"""
        from tasni.core.config import LAMOST_KNOWN_TYPE_PENALTY

        assert LAMOST_KNOWN_TYPE_PENALTY < 0

    def test_unknown_bonus_is_positive(self):
        """Unknown type bonus should be positive"""
        from tasni.core.config import LAMOST_UNKNOWN_BONUS

        assert LAMOST_UNKNOWN_BONUS > 0

    def test_temp_mismatch_bonus_is_positive(self):
        """Temperature mismatch bonus should be positive"""
        from tasni.core.config import LAMOST_TEMP_MISMATCH_BONUS

        assert LAMOST_TEMP_MISMATCH_BONUS > 0


# Skip network-dependent tests by default
@pytest.mark.slow
class TestLAMOSTTAP:
    """Tests that require network access to LAMOST TAP service"""

    def test_tap_connection(self):
        """Test that LAMOST TAP service is reachable"""
        try:
            import pyvo as vo

            from tasni.core.config import LAMOST_TAP_URL

            tap = vo.dal.TAPService(LAMOST_TAP_URL)
            # Just check we can create the service
            assert tap is not None
        except ImportError:
            pytest.skip("pyvo not installed")
        except Exception as e:
            pytest.skip(f"LAMOST TAP not reachable: {e}")
