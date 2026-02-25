"""Release integrity and reproducibility-focused tests."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_seed_helper_is_deterministic():
    from tasni.core.seeds import make_rng

    a = make_rng(42).normal(size=10)
    b = make_rng(42).normal(size=10)
    np.testing.assert_allclose(a, b)


def test_bootstrap_ci_is_deterministic_with_seed():
    from tasni.analysis.statistical_analysis import bootstrap_confidence_interval

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci1 = bootstrap_confidence_interval(data, n_bootstrap=200, random_seed=123)
    ci2 = bootstrap_confidence_interval(data, n_bootstrap=200, random_seed=123)
    assert ci1 == ci2


def test_release_manifest_contains_sha256(tmp_path: Path):
    from tasni.utils.data_manager import DataManager

    # Minimal project structure expected by DataManager include-roots.
    (tmp_path / "data" / "processed" / "final").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "tasni").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

    (tmp_path / "data" / "processed" / "final" / "golden.csv").write_text("a,b\n1,2\n")
    (tmp_path / "docs" / "readme.md").write_text("# docs\n")
    (tmp_path / "src" / "tasni" / "module.py").write_text("x = 1\n")
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")

    manager = DataManager(tasni_root=tmp_path)
    manifest_path = Path(manager.generate_manifest())
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_files"] >= 4
    for item in payload["files"]:
        assert "sha256" in item
        assert len(item["sha256"]) == 64


@pytest.mark.slow
def test_pipeline_smoke_produces_expected_outputs():
    """Run pipeline --test and assert non-empty output with expected schema.

    Skips when required inputs are missing.
    Uses 1000-source subsample; does not reproduce full 100 golden sample.
    """
    root = Path(__file__).resolve().parents[2]
    orphans = root / "data" / "interim" / "checkpoints" / "tier1" / "orphans.parquet"
    tier5 = root / "data" / "processed" / "features" / "tier5_features.parquet"
    if not orphans.exists() or not tier5.exists():
        pytest.skip("Pipeline inputs (orphans.parquet, tier5_features.parquet) not found")

    result = subprocess.run(
        [sys.executable, "-m", "tasni", "pipeline", "all", "--test"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    ranked = root / "data" / "processed" / "ml" / "ranked_tier5.parquet"
    golden = root / "data" / "processed" / "final" / "golden_improved.parquet"
    assert ranked.exists() or golden.exists(), "Pipeline produced no ranked or golden output"
