# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TASNI (Thermal Anomaly Search for Non-communicating Intelligence) is an astronomical pipeline that identifies mid-infrared sources lacking optical/NIR/radio counterparts. Key discovery: 4 "fading thermal orphans" - room-temperature objects dimming over 10 years.

## Commands

### Installation
```bash
make install          # Production dependencies
make install-dev      # Dev dependencies + pre-commit hooks
```

### Testing
```bash
make test             # All tests with coverage
make test-unit        # Unit tests only
make test-integration # Integration tests only
```

### Code Quality
```bash
make lint             # black, isort, flake8
make format           # Auto-format code
make pre-commit       # All pre-commit checks
```

### Pipeline Operations
```bash
make pipeline-status  # Check status
make run-pipeline     # Full pipeline (CPU)
make run-pipeline-gpu # Full pipeline (GPU)
make golden-targets   # Generate golden list
make figures          # Publication figures
```

### ML Pipeline
```bash
make ml-features      # Extract 500+ features
make ml-train         # Train 5 models
make ml-predict       # Score 810K candidates
make ml-all           # Complete ML pipeline
```

### Running Individual Scripts
```bash
python src/tasni/optimized/optimized_pipeline.py --status
python src/tasni/optimized/optimized_pipeline.py --phase all --workers 16 --gpu
python src/tasni/generation/generate_golden_list.py
python src/tasni/ml/train_classifier.py --features <path> --golden <path> --output <path>
```

## Architecture

### Data Flow (7 Stages)
```
INPUT (WISE 747M, Gaia 1.8B, 2MASS, Spitzer, NEOWISE)
    ↓
DOWNLOAD (HEALPix NSIDE=32 → 12,288 tiles, TAP async, parquet)
    ↓
CROSSMATCH (BallTree/GPU, 3-arcsec radius)
    ↓
FILTER (Optical veto, thermal colors W1-W2>0.5, NIR veto, radio veto)
    ↓
SCORE (Multi-wavelength anomaly, proper motion, isolation)
    ↓
VARIABILITY (NEOWISE 10-year, periodogram, Stetson J, fade rates)
    ↓
OUTPUT (Golden 100, Tier5 810K, figures, spectroscopy plans)
```

### Key Directories
- `src/tasni/core/` - Config (`config.py`), logging, environment
- `src/tasni/download/` - Catalog downloads (WISE, Gaia, NEOWISE)
- `src/tasni/crossmatch/` - Spatial matching (CPU/GPU)
- `src/tasni/filtering/` - Multi-wavelength veto system
- `src/tasni/analysis/` - Variability, kinematics, periodogram
- `src/tasni/ml/` - Feature extraction, training, prediction
- `src/tasni/generation/` - Outputs (figures, target lists)
- `src/tasni/optimized/` - Performance-optimized pipeline
- `data/processed/final/` - Golden targets, tier5, variability data

### Central Configuration
All paths and settings in `src/tasni/core/config.py`:
- DATA_ROOT: `/mnt/data/tasni`
- HEALPix: NSIDE=32 (12,288 tiles)
- Match radius: 3 arcsec
- Thermal threshold: W1-W2 > 0.5

## Hardware

| GPU | VRAM | Environment |
|-----|------|-------------|
| Intel Arc A770 | 16GB | `source /home/server/xpu-env/activate-xpu.sh` |
| NVIDIA RTX 3060 | 12GB | `source /home/server/compute-env/bin/activate` |

CPU: i9-10850K (20 threads), RAM: 32GB, Storage: 746GB at /mnt/data/tasni

## Code Style

- Line length: 100 characters
- Formatting: black, isort (black profile)
- Linting: flake8
- Pre-commit hooks configured (`.pre-commit-config.yaml`)

## Guidelines

- Don't over-engineer
- Don't add ML models without asking
- Be direct, report findings
- Check logs in `logs/`
- Null results are results

## Path Note

`~/tasni/` is a symlink to `/mnt/data/tasni/`
