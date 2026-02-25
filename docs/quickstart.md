# TASNI Quick Start Guide

Get up and running with TASNI (Thermal Anomaly Search for Non-communicating Intelligence) in minutes.

## Prerequisites

- **Python 3.10+**
- **500GB+ free storage** (for WISE/Gaia catalogs)
- **RAM**: 16GB recommended (32GB for full pipeline)
- **Optional**: NVIDIA GPU with CUDA 12.x (for acceleration)
- **Optional**: Intel Arc GPU (for XPU classification)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/tasni.git
cd tasni
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For GPU acceleration (NVIDIA):**
```bash
pip install cudf-cu12 cupy-cuda12x
```

**For XPU support (Intel Arc):**
```bash
pip install torch intel-extension-for-pytorch
```

### 4. Verify Installation

```bash
python -c "import astropy, pandas, scipy, healpy; print('✓ Core dependencies OK')"

# Optional: GPU check
python -c "import cupy; print('✓ NVIDIA GPU OK')" 2>/dev/null || echo "⚠ CUDA not available"
python -c "import torch; torch.xpu.is_available() and print('✓ Intel XPU OK')" 2>/dev/null || echo "⚠ XPU not available"
```

## Quick Run

### Check Pipeline Status

```bash
python src/tasni/optimized_pipeline.py --status
```

### Generate Golden Targets

```bash
python src/tasni/generation/generate_golden_list.py
```

Output: `data/processed/final/golden_targets.csv`

### Run Full Pipeline

```bash
# CPU-only
python src/tasni/optimized_pipeline.py --phase all --workers 16

# GPU-accelerated
python src/tasni/optimized_pipeline.py --phase all --workers 16 --gpu

# With Makefile
make run-pipeline
```

## Output Locations

| Output | Location | Description |
|---------|-----------|-------------|
| Golden targets | `data/processed/final/golden_targets.csv` | Top 100 candidates |
| Variability | `data/processed/final/golden_variability.csv` | Time-series analysis |
| Kinematics | `data/processed/final/golden_kinematics.csv` | Proper motion data |
| Figures | `reports/figures/` | Publication-ready plots |
| Light curves | `reports/figures/periodogram/` | Periodograms & light curves |
| Paper | `paper/tasni_paper.pdf` | Complete publication |

## Common Workflows

### 1. Download Fresh Catalogs

```bash
# WISE (35GB)
python src/tasni/download/download_wise_full.py

# Gaia (28GB)
python src/tasni/download/download_gaia_full.py

# Secondary catalogs
python src/tasni/download/download_secondary_catalogs.py --catalog 2mass
python src/tasni/download/download_secondary_catalogs.py --catalog spitzer
```

### 2. Run Crossmatch

```bash
# CPU
python src/tasni/crossmatch/crossmatch_full.py --workers 16

# GPU
python src/tasni/crossmatch/gpu_crossmatch.py --batch-size 100000
```

### 3. Filter Anomalies

```bash
python src/tasni/filtering/filter_anomalies_full.py
```

### 4. Analyze Variability

```bash
python src/tasni/analysis/compute_ir_variability.py
```

### 5. Generate Figures

```bash
python src/tasni/generation/generate_publication_figures.py
```

## Configuration

Edit `src/tasni/core/config.py` for:
- Path configurations
- TAP service URLs
- Processing parameters
- Scoring thresholds

Or use environment variables:
```bash
export TASNI_DATA_ROOT=/path/to/data
export HEALPIX_NSIDE=32
export N_WORKERS=16
```

## Troubleshooting

### Out of Memory

- Reduce workers: `--workers 8`
- Use streaming: `optimized_pipeline.py --streaming`
- Process by HEALPix tile manually

### Slow Performance

- Enable GPU: `--gpu`
- Increase batch size: `--batch-size 50000`
- Use optimized versions in `src/tasni/optimized/`

### Import Errors

```bash
# Reinstall in development mode
pip install -e .
```

### TAP Query Failures

```bash
# Check service status
python -c "from astroquery.utils.tap.core import TAPService; TAPService('https://irsa.ipac.caltech.edu/TAP').query('SELECT TOP 1 * FROM allwise_p3as_psd')"
```

## Next Steps

1. **Read Architecture**: `docs/ARCHITECTURE.md` - System overview
2. **Review Pipeline**: `docs/pipeline.md` - Detailed pipeline guide
3. **Explore Data**: `docs/DATA_SOURCES.md` - Catalog descriptions
4. **Check Optimization**: `docs/OPTIMIZATIONS.md` - Performance tips
5. **View Results**: `paper/tasni_paper.pdf` - Scientific findings

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/tasni/issues)
- **Documentation**: See `docs/` directory
- **Examples**: Check `notebooks/` directory for Jupyter examples

## Citation

If you use TASNI in your research, please cite:

```bibtex
@article{tasni2024,
  title={The Thermal Anomaly Search for Non-communicating Intelligence (TASNI): Discovery of Four Fading Thermal Orphans in the AllWISE Catalog},
  author={...},
  journal={...},
  year={2024}
}
```

**Happy hunting for thermal anomalies!**
