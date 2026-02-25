# TASNI Reproducibility Quickstart

> **See also:** [docs/reproducibility.md](reproducibility.md) for the detailed reproducibility guide covering full pipeline reproduction from raw data.

**Version:** 1.0.0
**Last Updated:** 2026-02-15

This guide provides step-by-step instructions to reproduce all TASNI results from scratch.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12 |
| RAM | 16 GB | 64 GB |
| Storage | 500 GB | 2 TB |
| CPU | 4 cores | 16+ cores |
| GPU | Optional | NVIDIA with CUDA 12.x |

---

## Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/dpalucki/tasni.git
cd tasni

# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --no-interaction

# Verify installation
poetry run python -c "import tasni; print(f'TASNI v{tasni.__version__}')"
```

**Expected Output:**
```
TASNI v1.0.0
```

---

## Step 2: Verify Data Availability

```bash
# Check for required data files
ls -la data/processed/final/

# Expected files:
# golden_improved.parquet
# golden_improved_parallax.parquet
# golden_improved_kinematics.parquet
# golden_improved_erosita.parquet
# golden_improved_bayesian.parquet
```

If data files are missing, download from Zenodo:
```bash
# Download data release from Zenodo
wget https://zenodo.org/record/18717105/files/tasni_data_release.zip
unzip tasni_data_release.zip -d data/
```

---

## Step 3: Run Test Suite

```bash
# Run unit tests (fast)
poetry run pytest tests/unit/ -v

# Expected: All tests pass
# Runtime: ~30 seconds

# Run full test suite with coverage
poetry run pytest tests/ -v --cov=src/tasni --cov-report=term-missing

# Expected: Coverage >= 40% (current CI gate)
# Runtime: ~5 minutes
```

---

## Step 4: Reproduce Key Results

### 4.1 Golden Sample Statistics

```bash
poetry run python -c "
import pandas as pd
golden = pd.read_parquet('data/processed/final/golden_improved.parquet')
print(f'Total candidates: {len(golden)}')
print(f'Mean T_eff: {golden[\"T_eff_K\"].mean():.1f} K')
print(f'Mean proper motion: {golden[\"pm_total\"].mean():.1f} mas/yr')
print(f'Fading sources: {(golden[\"variability_flag\"] == \"FADING\").sum()}')
"
```

**Expected Output:**
```
Total candidates: 100
Mean T_eff: 272.3 K
Mean proper motion: 187.4 mas/yr
Fading sources: 4
```

### 4.2 Periodogram Analysis

```bash
poetry run python src/tasni/analysis/periodogram_analysis.py --fading-only

# Output: reports/figures/periodogram/fading_periodograms.png
# Runtime: ~2 minutes
```

### 4.3 Generate Publication Figures

```bash
poetry run python src/tasni/generation/generate_publication_figures.py

# Output: reports/figures/*.png and *.pdf
# Runtime: ~5 minutes
```

---

## Step 5: Reproduce Pipeline (Optional)

To reproduce the full pipeline from raw data:

```bash
# WARNING: Requires ~500GB of catalog data and 8+ hours

# Phase 1: Tier1 vetoes
poetry run tasni tier1-vetoes \
    --input data/interim/checkpoints/tier1/orphans.parquet \
    --output data/interim/checkpoints/tier1_improved/tier1_vetoes.parquet

# Phase 2: ML scoring
poetry run tasni ml-scoring \
    --input data/processed/features/tier5_features.parquet \
    --output data/processed/ml/ranked_tier5.parquet

# Phase 3: Validation
poetry run tasni validate \
    --input data/processed/ml/ranked_tier5.parquet \
    --output-dir data/processed/final
```

---

## Step 6: Verify Reproducibility

Run the validation script:

```bash
poetry run python tests/test_validation.py

# Expected: All assertions pass
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'tasni'`

**Solution:**
```bash
poetry install --no-interaction
poetry shell  # Activate virtual environment
```

### Issue: Memory Error during ML scoring

**Solution:**
```bash
# Use test mode for smaller datasets
poetry run tasni ml-scoring --test
```

### Issue: Vizier API rate limiting

**Solution:**
```bash
# Reduce batch size and add pauses
poetry run tasni tier1-vetoes --batch-size 10
```

---

## File Checksums

Verify data integrity:

```bash
sha256sum -c data/processed/final/checksums.txt
```

---

## Expected Runtime Summary

| Step | Runtime | Disk Usage |
|------|---------|------------|
| Environment setup | 5 min | 2 GB |
| Test suite | 5 min | 100 MB |
| Figure generation | 5 min | 50 MB |
| Full pipeline | 8+ hours | 500 GB |

---

## Docker Alternative

For guaranteed reproducibility:

```bash
# Build container
docker build -t tasni:latest .

# Run container
docker run -v $(pwd)/data:/data tasni:latest tasni info

# Run full pipeline
docker run -v $(pwd)/data:/data tasni:latest tasni pipeline all
```

---

## Citation

If you reproduce these results, please cite:

```bibtex
@software{tasni2026,
    author = {{Palucki, Dennis}},
    title = {TASNI: Thermal Anomaly Search for Non-communicating Intelligence},
    version = {1.0.0},
    year = {2026},
    url = {https://github.com/dpalucki/tasni}
}
```

---

## Contact

For reproduction issues, open a GitHub issue or contact: paluckide@yahoo.com
