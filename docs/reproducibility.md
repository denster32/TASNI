# TASNI Reproducibility Guide

> **See also:** [docs/REPRODUCIBILITY_QUICKSTART.md](REPRODUCIBILITY_QUICKSTART.md) for a concise quickstart guide to reproduce key results.

**Date:** February 4, 2026
**Purpose:** Provide instructions for reproducing the TASNI analysis
**Status:** Phase 7 - Publication Readiness

## Overview

This guide provides step-by-step instructions for reproducing the TASNI analysis from raw data to final results. Following this guide will allow researchers to verify the results and extend the analysis.

## System Requirements

### Hardware

- **CPU:** Multi-core processor (8+ cores recommended)
- **RAM:** 32 GB minimum, 64 GB recommended
- **Storage:** 100 GB free space
- **GPU:** NVIDIA GPU with CUDA support (optional, for acceleration)

### Software

- **Operating System:** Linux (Ubuntu 20.04+ recommended) or macOS
- **Python:** 3.9+
- **CUDA:** 11.0+ (if using GPU)

### Python Packages

The project uses **Poetry** for dependency management. See [`pyproject.toml`](../pyproject.toml).

**For a concise quickstart**, see [REPRODUCIBILITY_QUICKSTART.md](REPRODUCIBILITY_QUICKSTART.md).

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/dpalucki/tasni.git
cd tasni
```

### Step 2: Install with Poetry (Recommended)

```bash
poetry install --no-interaction
poetry run python -c "import tasni; print(f'TASNI v{tasni.__version__}')"
```

### Alternative: pip / venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
poetry run python -c "import numpy; import pandas; import astropy; print('Installation successful!')"
```

## Data Download

### Step 1: Download Public Survey Data

Download the following public survey data:

```bash
# WISE AllWISE catalog
wget https://irsa.ipac.caltech.edu/TAP/sync?query=SELECT+*+FROM+wise_allwise_p3as_psd&format=csv

# Gaia DR3 catalog
wget https://gea.esac.esa.int/data-server/gaia_source.csv

# 2MASS catalog
wget https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?project=planck&mission=irsa2mass
```

**Note:** These are example commands. Refer to the survey documentation for actual download instructions.

### Step 2: Place Data in Correct Directories

Place downloaded data in the following directories:

```
data/
  wise/
    allwise.csv
  gaia/
    dr3.csv
  2mass/
    2mass.csv
```

### Step 3: Download TASNI Data Products

Download TASNI data products from the GitHub repository:

```bash
# Golden sample (primary data product)
wget https://github.com/dpalucki/tasni/raw/main/data/processed/final/golden_improved.csv
wget https://github.com/dpalucki/tasni/raw/main/data/processed/final/golden_improved.parquet

# Parallax measurements
wget https://github.com/dpalucki/tasni/raw/main/data/processed/final/golden_improved_parallax.csv
```

Or download the full data release from Zenodo (DOI: 10.5281/zenodo.18774271).

## Reproducing the Analysis

### Step 1: Run the Pipeline

```bash
poetry run python -m tasni.pipeline.full_allwise_pipeline
```

Or use the Makefile: `make pipeline`

This will:
1. Load the WISE catalog (Tier 1)
2. Cross-match with Gaia DR3 (Tier 2)
3. Cross-match with secondary catalogs (Tier 3)
4. Apply quality filters (Tier 4)
5. Generate the tier 5 catalog (Tier 5)

**Expected runtime:** ~2-4 hours (depending on hardware)

### Step 2: Extract Features

```bash
poetry run python -m tasni.ml.extract_features
```

This will extract features for all tier 5 sources.

**Expected runtime:** ~1-2 hours

### Step 3: Train ML Models

```bash
poetry run python -m tasni.ml.train_classifier
```

This will train ML models on the training set.

**Expected runtime:** ~30 minutes

### Step 4: Generate Predictions

```bash
poetry run python -m tasni.ml.predict_tier5
```

This will generate predictions for all tier 5 sources.

**Expected runtime:** ~10 minutes

### Step 5: Rank Candidates

```bash
poetry run python -m tasni.generation.generate_golden_list
```

This will rank candidates by composite score and generate the golden sample.

**Expected runtime:** ~5 minutes

### Step 6: Fit Parallaxes

```bash
poetry run python -m tasni.analysis.extract_neowise_parallax
```

This will fit parallaxes for the golden sample.

**Expected runtime:** ~30 minutes

### Step 7: Analyze Variability

```bash
poetry run python -m tasni.analysis.compute_ir_variability_tier5
```

This will analyze variability for the golden sample.

**Expected runtime:** ~1 hour

### Step 8: Generate Figures

```bash
poetry run python -m tasni.generation.generate_publication_figures
```

This will generate all figures for the paper.

**Expected runtime:** ~30 minutes

### Step 9: Compile the Paper

```bash
cd tasni_paper_final
latexmk -pdf manuscript.tex
```

This will compile the LaTeX paper to PDF.

**Expected runtime:** ~1 minute

## Verification

### Step 1: Check Golden Sample

Verify that the golden sample contains 100 sources:

```python
import pandas as pd

golden = pd.read_parquet('data/processed/final/golden_improved.parquet')
print(f"Golden sample: {len(golden)}")
assert len(golden) == 100
```

### Step 2: Check Parallax Measurements

Verify that parallax measurements are reasonable:

```python
import pandas as pd

parallax = pd.read_csv('data/processed/final/golden_improved_parallax.csv')
pi_col = 'neowise_parallax_mas'
print(f"Parallax range: {parallax[pi_col].min():.2f} - {parallax[pi_col].max():.2f} mas")
assert parallax[pi_col].min() > 0
assert parallax[pi_col].max() < 200
```

### Step 3: Check Variability Classification

Verify that the golden sample has variability flags:

```python
import pandas as pd

golden = pd.read_parquet('data/processed/final/golden_improved.parquet')
print(golden['variability_flag'].value_counts())
assert 'NORMAL' in golden['variability_flag'].values or 'FADING' in golden['variability_flag'].values
```

## Troubleshooting

### Common Issues

#### Issue: Out of Memory

**Solution:** Reduce the batch size in the pipeline:

```bash
# Use environment override (or set in .env)
export BATCH_SIZE=10000
```

#### Issue: CUDA Out of Memory

**Solution:** Disable GPU acceleration:

```bash
# Use environment overrides (or set in .env)
export USE_CUDA=false
export USE_XPU=false
```

#### Issue: Missing Data Files

**Solution:** Ensure all data files are in the correct directories:

```bash
ls data/wise/
ls data/gaia/
ls data/2mass/
```

#### Issue: Import Errors

**Solution:** Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Extending the Analysis

### Adding New Features

To add new features:

1. Define the feature extraction function in `src/tasni/ml/extract_features.py`
2. Add the feature to the feature list in `src/tasni/ml/config.py`
3. Re-run the feature extraction and model training

### Adding New Models

To add new ML models:

1. Define the model in `src/tasni/ml/models.py`
2. Add the model to the model list in `src/tasni/ml/config.py`
3. Re-run the model training

### Adding New Selection Criteria

To add new selection criteria:

1. Define the criteria in `src/tasni/pipeline/filters.py`
2. Add the criteria to the active pipeline module (for example `src/tasni/pipeline/full_allwise_pipeline.py`)
3. Re-run the pipeline

## Performance Optimization

### GPU Acceleration

To enable GPU acceleration:

```bash
export USE_CUDA=true
# Optional: export USE_XPU=true
```

### Parallel Processing

To enable parallel processing:

```bash
export N_WORKERS=8
```

### Memory Optimization

To reduce memory usage:

```bash
export BATCH_SIZE=5000
```

## Documentation

### Code Documentation

See [`docs/api_reference.md`](api_reference.md) for detailed API documentation.

### Pipeline Documentation

See [`docs/pipeline.md`](pipeline.md) for detailed pipeline documentation.

### ML Pipeline Documentation

See [`docs/ML_PIPELINE_REVISED.md`](ML_PIPELINE_REVISED.md) for detailed ML pipeline documentation.

## Contact

For questions about reproducing the TASNI analysis, please contact:

- **Email:** paluckide@yahoo.com
- **GitHub Issues:** https://github.com/dpalucki/tasni/issues

## Cross-Reference

See also:
- [`docs/DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) - Data availability statement
- [`docs/PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) - Publication checklist
- [`README.md`](../README.md) - Project README

---

**Document Version:** 1.0
**Last Updated:** February 4, 2026
**Status:** Ready for Review
