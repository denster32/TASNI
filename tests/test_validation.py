"""Tests for golden sample validation."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def golden_path():
    """Resolve golden_improved path relative to project root."""
    # Try multiple known locations
    candidates = [
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "final"
        / "golden_improved.csv",
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "final"
        / "golden_improved.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    pytest.skip("Golden sample file not found; skipping validation test")


def test_golden_improved(golden_path):
    """Validate golden sample has expected shape and scores."""
    if golden_path.suffix == ".parquet":
        df = pd.read_parquet(golden_path)
    else:
        df = pd.read_csv(golden_path)
    assert len(df) >= 100, f"Golden candidates: {len(df)} (expected >=100)"
    if "improved_composite_score" in df.columns:
        assert df["improved_composite_score"].max() > 0.5
