# TASNI Pipeline Guide

Complete guide to running the TASNI pipeline from start to finish.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Pipeline Stages](#pipeline-stages)
4. [Running the Pipeline](#running-the-pipeline)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Quick Start

```bash
# Run full pipeline with GPU acceleration
python src/tasni/optimized_pipeline.py --phase all --workers 16 --gpu

# Or use Makefile
make run-pipeline
```

## Prerequisites

### Hardware
| Component | Minimum | Recommended |
|-----------|----------|-------------|
| CPU | 8 cores | 16+ cores |
| RAM | 16GB | 32GB+ |
| Storage | 500GB SSD | 1TB+ NVMe SSD |
| GPU | None | NVIDIA RTX 3060+ (12GB VRAM) |
| XPU | None | Intel Arc A770+ (16GB VRAM) |

### Software
- Python 3.10+
- CUDA 12.x (for NVIDIA GPU)
- Git (for version control)

## Pipeline Stages

### Stage 0: Status Check

Check current pipeline status:

```bash
python src/tasni/optimized_pipeline.py --status
```

Output:
```
TASNI Pipeline Status
=====================

Data Status:
  WISE tiles: 12,288 / 12,288 (100%)
  Gaia tiles: 7,239 / 12,288 (59%)
  Crossmatch: 7,239 / 12,288 (59%)

Output Status:
  Golden targets: Generated (100 sources)
  Variability analysis: Complete
  NEOWISE epochs: Available

Recommendations:
  - Complete Gaia download
  - Run crossmatch on remaining tiles
  - Update golden targets with new data
```

### Stage 1: Download Catalogs

#### WISE (35GB, 747M sources)

```bash
python src/tasni/download/download_wise_full.py

# Or for specific HEALPix tile
python src/tasni/download/download_wise_full.py --hpix 1234
```

**Progress tracking:** `logs/wise_download_*.log`

**Expected time:** 6-8 hours (network-limited)

**Output:**
- `data/wise/wise_hp00000.parquet`
- `data/wise/wise_hp00001.parquet`
- ...

#### Gaia DR3 (28GB, 1.8B sources)

```bash
python src/tasni/download/download_gaia_full.py

# Parallel download (recommended)
python src/tasni/download/download_gaia_full.py --parallel --workers 8
```

**Progress tracking:** `logs/gaia_download_*.log`

**Expected time:** 10-12 hours (network-limited)

**Output:**
- `data/gaia/gaia_hp00000.parquet`
- `data/gaia/gaia_hp00001.parquet`
- ...

#### Secondary Catalogs

```bash
# 2MASS (NIR, 470M sources)
python src/tasni/download/download_secondary_catalogs.py --catalog 2mass

# Spitzer (Mid-IR, 300K sources)
python src/tasni/download/download_secondary_catalogs.py --catalog spitzer

# LAMOST (Spectroscopy, 10M sources)
python src/tasni/download/download_lamost.py

# Legacy Survey (Deep Optical)
python src/tasni/download/download_legacy_survey.py
```

**Progress tracking:** `logs/*_download_*.log`

### Stage 2: Crossmatching

#### CPU Crossmatch

```bash
python src/tasni/crossmatch/crossmatch_full.py --workers 16

# For specific HEALPix tile
python src/tasni/crossmatch/crossmatch_full.py --hpix 1234 --workers 8
```

**Expected time:** 24 hours (for 7,239 tiles)

**Output:**
- `data/crossmatch/orphans_hp00000.parquet`
- `data/crossmatch/orphans_hp00001.parquet`
- ...

#### GPU Crossmatch (Recommended)

```bash
python src/tasni/crossmatch/gpu_crossmatch.py --batch-size 100000

# Or use distributed orchestrator
python src/tasni/misc/distributed_orchestrator.py --mode crossmatch --workers 16
```

**Expected time:** 2 hours (12x speedup)

**Output:** Same as CPU crossmatch

#### Optimized Crossmatch

```bash
python src/tasni/optimized/optimized_crossmatch.py --workers 16 --method balltree
```

**Methods:**
- `balltree`: scikit-learn, 4x speedup
- `ckdtree`: SciPy, 4x speedup
- `gpu`: cuDF, 100x speedup (requires NVIDIA GPU)

### Stage 3: Filtering

#### Basic Filtering

```bash
python src/tasni/filtering/filter_anomalies_full.py
```

**Filters applied:**
1. Thermal colors: W1-W2 > 0.5 mag
2. Optical veto: No Gaia match within 3"
3. NIR veto: No 2MASS match within 3"
4. Deep optical: No Pan-STARRS match
5. Legacy Survey: No Legacy DR10 match
6. Radio veto: No NVSS match within 30"
7. Temperature: T_eff < 500 K

**Expected time:** 2 hours

**Output:**
- `output/anomalies_filtered.parquet`
- `output/anomalies_ranked.parquet`

#### Multi-Wavelength Scoring

```bash
python src/tasni/filtering/multi_wavelength_scoring.py
```

**Scoring factors:**
- LAMOST cross-check: Known IR types penalized
- Legacy Survey: Deep optical detection penalized
- Proper motion: High PM penalized (nearby objects)
- Galactic latitude: High latitude rewarded (less confusion)

**Expected time:** 30 minutes

**Output:**
- `output/tier5_radio_silent.parquet` (810K candidates)

#### Tier 5 Refiltering

```bash
python src/tasni/filtering/refilter_tier5.py
```

**Additional filters:**
- Solar system exclusion
- Known object cross-check
- Image clustering analysis

**Expected time:** 15 minutes

**Output:**
- `data/processed/final/tier5_cleaned.parquet`

### Stage 4: Variability Analysis

#### IR Variability

```bash
python src/tasni/analysis/compute_ir_variability.py
```

**Metrics computed:**
- RMS scatter
- Chi-squared statistic
- Stetson J index (for multi-band)
- Linear trend (fade/brighten rate)
- Periodogram peaks

**Expected time:** 48 hours (CPU) or 4 hours (GPU)

**Output:**
- `output/golden_variability.csv`
- `output/golden_variability.parquet`

#### Tier 5 Variability

```bash
python src/tasni/filtering/run_tier5_variability.py

# Or use optimized version
python src/tasni/filtering/refilter_tier5.py --run-variability
```

**Expected time:** 2 hours (for 810K sources)

**Output:**
- `data/processed/final/tier5_variability.parquet`
- `data/interim/checkpoints/tier5_variability_checkpoint.json`

#### NEOWISE Queries

```bash
# Async queries (recommended)
python src/tasni/download/async_neowise_query.py --sources data/processed/final/tier5_cleaned.parquet

# Legacy sequential queries
python src/tasni/analysis/query_neowise_variability.py
```

**Progress tracking:** `logs/async_neowise.log`

**Expected time:** 10-50 hours (depends on rate limits)

**Output:**
- `output/neowise_epochs.parquet`
- `output/neowise_epochs.summary.csv`

#### Periodogram Analysis

```bash
python src/tasni/analysis/periodogram_analysis.py
```

**Analysis:**
- Lomb-Scargle periodogram
- Peak detection
- Period classification
- Light curve generation

**Expected time:** 4 hours

**Output:**
- `reports/figures/periodogram/periodogram_results.csv`
- `reports/figures/periodogram/J*_periodogram.png`
- `reports/figures/periodogram/fading_periodograms.png`

### Stage 5: Output Generation

#### Golden Targets

```bash
python src/tasni/generation/generate_golden_list.py
```

**Selection criteria:**
- Composite score ranking
- Variability classification
- Kinematic analysis
- Visual inspection priority

**Expected time:** 5 minutes

**Output:**
- `data/processed/final/golden_targets.csv` (100 top candidates)
- `data/processed/final/golden_kinematics.csv`
- `data/processed/final/golden_variability.csv`

#### Publication Figures

```bash
python src/tasni/generation/generate_publication_figures.py
```

**Figures generated:**
- Sky distribution map
- Color-magnitude diagram
- Temperature distribution
- Variability histogram
- Fading light curves
- Pipeline flowchart

**Expected time:** 30 minutes

**Output:**
- `reports/figures/fig1_sky_distribution.pdf`
- `reports/figures/fig2_color_magnitude.pdf`
- ...

#### Spectroscopy Targets

```bash
python src/tasni/generation/prepare_spectroscopy_targets.py
```

**Target prioritization:**
- Golden sample first
- Fading orphans high priority
- High proper motion candidates

**Expected time:** 10 minutes

**Output:**
- `data/processed/spectroscopy/spectroscopy_targets.txt`
- `data/processed/spectroscopy/spectroscopy_targets.tex`
- `data/processed/spectroscopy/proposal_summary.txt`

## Running the Pipeline

### Full Pipeline (All Stages)

```bash
# CPU only
python src/tasni/optimized/optimized_pipeline.py --phase all --workers 16

# With GPU acceleration
python src/tasni/optimized/optimized_pipeline.py --phase all --workers 16 --gpu

# Streaming mode (lower memory)
python src/tasni/optimized/optimized_pipeline.py --phase all --streaming

# Benchmark mode
python src/tasni/optimized/optimized_pipeline.py --phase all --benchmark
```

### Individual Stages

```bash
# Download only
python src/tasni/optimized/optimized_pipeline.py --phase download

# Crossmatch only
python src/tasni/optimized/optimized_pipeline.py --phase crossmatch

# Filtering only
python src/tasni/optimized/optimized_pipeline.py --phase filter

# Variability only
python src/tasni/optimized/optimized_pipeline.py --phase variability

# Output only
python src/tasni/optimized/optimized_pipeline.py --phase generate
```

### Checkpoint/Resume

```bash
# Resume from last checkpoint
python src/tasni/optimized/optimized_pipeline.py --phase all --resume

# Check checkpoint status
python src/tasni/optimized/optimized_pipeline.py --checkpoints
```

## Configuration

### Path Configuration

Edit `src/tasni/core/config.py`:

```python
# Path configuration
DATA_ROOT = Path("/mnt/data/tasni")
WISE_DIR = DATA_ROOT / "data" / "wise"
GAIA_DIR = DATA_ROOT / "data" / "gaia"
OUTPUT_DIR = DATA_ROOT / "output"
```

Or use environment variables:

```bash
export TASNI_DATA_ROOT=/path/to/data
export TASNI_WISE_DIR=/path/to/wise
export TASNI_GAIA_DIR=/path/to/gaia
```

### TAP Service URLs

```python
# In src/tasni/core/config.py
WISE_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LAMOST_TAP_URL = "http://www.lamost.org/db/v2/tap"
```

### Processing Parameters

```python
# HEALPix configuration
HEALPIX_NSIDE = 32  # 12,288 tiles

# Matching configuration
MATCH_RADIUS_ARCSEC = 3.0

# Parallelization
N_WORKERS = 16

# Filtering thresholds
W1_W2_THRESHOLD = 0.5  # mag
TEMP_THRESHOLD = 500  # K
```

## Troubleshooting

### Download Issues

**Problem:** TAP query fails with timeout
```bash
# Solution: Reduce batch size
python src/tasni/download/download_wise_full.py --batch-size 1000

# Or use checkpoint resume
python src/tasni/download/download_wise_full.py --resume
```

**Problem:** "Rate limit exceeded"
```bash
# Solution: Add delay between requests
python src/tasni/download/download_wise_full.py --delay 1.0

# Or use async downloader with rate limiting
python src/tasni/download/async_neowise_query.py --max-concurrent 5
```

### Crossmatch Issues

**Problem:** Out of memory
```bash
# Solution: Reduce batch size
python src/tasni/crossmatch/gpu_crossmatch.py --batch-size 10000

# Or process by tile
python src/tasni/crossmatch/crossmatch_full.py --hpix 1234 --workers 4
```

**Problem:** Very slow
```bash
# Solution: Use GPU if available
python src/tasni/crossmatch/gpu_crossmatch.py

# Or use optimized methods
python src/tasni/optimized/optimized_crossmatch.py --method balltree
```

### Filtering Issues

**Problem:** No sources pass filters
```bash
# Solution: Check filter thresholds
python src/tasni/filtering/filter_anomalies_full.py --dry-run

# Or relax thresholds
# Edit src/tasni/core/config.py
W1_W2_THRESHOLD = 0.3  # More permissive
```

### Variability Issues

**Problem:** NEOWISE queries timeout
```bash
# Solution: Use async downloader
python src/tasni/download/async_neowise_query.py --timeout 300

# Or process in batches
python src/tasni/analysis/query_neowise_variability.py --batch-size 100
```

**Problem:** No variability detected
```bash
# Solution: Check epoch count
# Need at least 20 epochs for robust variability detection

# Or lower detection threshold
# Edit src/tasni/analysis/compute_ir_variability.py
VARIABILITY_THRESHOLD = 2.0  # sigma
```

### General Issues

**Problem:** Import errors
```bash
# Solution: Reinstall in development mode
pip install -e .

# Or check PYTHONPATH
export PYTHONPATH=/mnt/data/tasni/scripts:$PYTHONPATH
```

**Problem:** File not found errors
```bash
# Solution: Check data directory structure
ls -lh data/wise/
ls -lh data/gaia/
ls -lh data/processed/final/

# Or regenerate with correct paths
# Edit src/tasni/core/config.py
```

## Advanced Usage

### Custom Filtering

```python
# Create custom filter script
import pandas as pd

def custom_filter(df):
    # Add your custom logic here
    mask = (
        (df['w1_w2_color'] > 0.5) &
        (df['w1mag'] < 15.0) &
        (df['proper_motion'] > 100)
    )
    return df[mask]

# Load data
df = pd.read_parquet('output/anomalies_ranked.parquet')

# Apply filter
filtered = custom_filter(df)

# Save results
filtered.to_parquet('output/custom_filtered.parquet')
```

### Custom Scoring

```python
# Add custom scoring metrics
def custom_score(row):
    score = 0.0

    # Your custom scoring logic
    if row['fading_rate'] < 0:
        score += 10.0  # Reward fading

    if row['proper_motion'] > 200:
        score += 5.0  # Reward high PM

    return score
```

### Batch Processing

```bash
# Process HEALPix tiles in parallel
for hpix in {0..127}; do
    python src/tasni/optimized/optimized_pipeline.py \
        --phase crossmatch \
        --hpix $hpix \
        --workers 8 &
done

wait  # Wait for all jobs to complete
```

### Cloud Deployment

```bash
# Using Docker
docker build -t tasni:latest .
docker run -v /data:/mnt/data tasni:latest \
    python src/tasni/optimized/optimized_pipeline.py --phase all

# Or use GPU
docker run --gpus all -v /data:/mnt/data tasni:latest \
    python src/tasni/optimized/optimized_pipeline.py --phase all --gpu
```

## Performance Tips

### Memory Optimization

- Use `--streaming` flag for large catalogs
- Process HEALPix tiles individually
- Close unused DataFrame objects
- Use `del` and `gc.collect()` in custom scripts

### CPU Optimization

- Set `N_WORKERS = (CPU cores / 2)` for best performance
- Use `--workers` flag appropriately
- Avoid oversubscription

### GPU Optimization

- Use cuDF for crossmatch (100x speedup)
- Use Numba-JIT for variability (12-60x speedup)
- Monitor GPU memory with `nvidia-smi`
- Adjust batch size to fit in GPU memory

### I/O Optimization

- Use parquet format (faster than CSV)
- Use PyArrow for reading (faster than Pandas)
- Enable compression (snappy recommended)
- Use SSD for data directory

---

**For more information:**
- Architecture: `docs/ARCHITECTURE.md`
- Data sources: `docs/DATA_SOURCES.md`
- Optimizations: `docs/OPTIMIZATIONS.md`
