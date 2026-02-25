from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class TestOutputs:
    def test_golden_targets_exists(self):
        golden = DATA_DIR / "tasni_golden_targets.csv"
        if not golden.exists():
            pytest.skip(f"{golden} not found")
        df = pd.read_csv(golden)
        assert len(df) > 0, "golden_targets.csv is empty"
        assert "designation" in df.columns
        assert "ra" in df.columns
        assert "dec" in df.columns

    def test_tier5_exists(self):
        tier5 = DATA_DIR / "final" / "golden_improved.csv"
        if not tier5.exists():
            pytest.skip(f"{tier5} not found")
        df = pd.read_csv(tier5)
        assert len(df) > 0, "golden_improved.csv is empty"

    def test_tier4_exists(self):
        pytest.skip("tier4_final.parquet not part of current golden sample pipeline")


class TestSchema:
    def test_golden_targets_columns(self):
        golden = DATA_DIR / "tasni_golden_targets.csv"
        if not golden.exists():
            pytest.skip(f"{golden} not found")
        df = pd.read_csv(golden)
        required = ["designation", "ra", "dec"]
        for col in required:
            assert col in df.columns, f"Missing: {col}"
