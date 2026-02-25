# TASNI - Thermal Anomaly Search for Non-communicating Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18774271.svg)](https://doi.org/10.5281/zenodo.18774271)

**Version:** 1.0.0 (Released 2026-02-15)
**Status:** Publication Ready
**DOI:** [10.5281/zenodo.18774271](https://doi.org/10.5281/zenodo.18774271)

---

## Project Overview

TASNI identifies "fading thermal orphans"—sources detectable only in the mid-infrared (WISE W1/W2) that systematically dim over a decade. We discovered **3 confirmed candidates** with unprecedented combinations of cold temperatures (251-293 K), high proper motions, and monotonic fading, plus one additional LMC member.

---

## Key Scientific Results

### The Three Confirmed Fading Thermal Orphans

| Designation | T_eff (K) | Distance (pc) | Period (days) | FAP |
|-------------|-----------|---------------|---------------|-----|
| J143046.35-025927.8 | 293 +/- 47 | 17.4 +3.0/-2.6 | 116.3 +/- 5.0 | 2.1e-61 |
| J231029.40-060547.3 | 258 +/- 38 | 32.6 +13.3/-8.0 | 178.6 +/- 7.0 | 6.7e-46 |
| J193547.43+601201.5 | 251 +/- 35 | --- | 92.6 +/- 4.0 | 2.2e-11 |

> **Note:** J044024.40-731441.6 was initially identified as a fading candidate but is located within the Large Magellanic Cloud (~50 kpc) and is likely an LMC member rather than a nearby brown dwarf.

### Key Findings

1. **Monotonic Fading** - All three confirmed sources show sustained dimming (17-53 mmag/yr) over the 10-year NEOWISE baseline. Apparent periodicities at 90-180 days are likely NEOWISE cadence aliases rather than astrophysical signals.

2. **Nearby Room-Temperature Objects** - J1430 at 17.4 pc is one of the nearest room-temperature objects known (Teff = 293 K).

3. **X-ray Quiet** - All 59 sources within the eROSITA DR1 footprint (western Galactic hemisphere) have no X-ray detection, ruling out AGN or stellar activity.

4. **100 Golden Candidates** - Full catalog of high-priority thermal anomalies for follow-up.

---

## Data Products

| Category | File | Description |
|----------|------|-------------|
| **Golden Sample** | `data/processed/final/golden_improved.parquet` | 100 top candidates |
| **Parallax** | `data/processed/final/golden_improved_parallax.parquet` | 67 distance measurements (parallax > 5 mas; 44 with SNR > 3) |
| **Kinematics** | `data/processed/final/golden_improved_kinematics.parquet` | Proper motion data |
| **X-ray** | `data/processed/final/golden_improved_erosita.parquet` | eROSITA constraints |
| **Figures** | `reports/figures/*.png` | 25+ publication plots |
| **Paper** | `tasni_paper_final/manuscript.tex` | AASTeX manuscript |

---

## Installation

```bash
# Clone repository
git clone https://github.com/denster32/TASNI.git
cd tasni

# Install with Poetry (recommended)
poetry install

# Or with pip
pip install -e .
```

Verify installation:
```bash
python -c "import tasni; print(f'TASNI v{tasni.__version__}')"
```

---

## Quick Start

```bash
# Show project info
tasni info

# Generate publication figures
python src/tasni/generation/generate_publication_figures.py

# Run periodogram analysis
python src/tasni/analysis/periodogram_analysis.py --fading-only
```

---

## CLI Commands

```bash
tasni --help                    # Show all commands
tasni info                      # Project information
tasni tier1-vetoes --help       # Cross-match vetoes
tasni ml-scoring --help         # ML ensemble scoring
tasni validate --help           # Golden candidate selection
tasni pipeline all              # Run full pipeline
```

---

## Citation

If you use TASNI data or code, please cite:

```bibtex
@article{tasni2026,
    author = {{Palucki, Dennis}},
    title = "{TASNI: Thermal Anomaly Search for Non-communicating Intelligence}",
    journal = {The Astrophysical Journal},
    year = {2026},
    volume = {},
    pages = {},
    doi = {}
}
```

---

## Documentation

- [Manuscript](tasni_paper_final/manuscript.tex) - Full manuscript
- [Reproducibility Guide](docs/REPRODUCIBILITY_QUICKSTART.md) - Step-by-step reproduction
- [API Reference](docs/api_reference.md) - Module documentation
- [Architecture](docs/architecture.md) - System design

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This research has made use of data from:
- WISE/NEOWISE (NASA/IPAC Infrared Science Archive)
- Gaia DR3 (European Space Agency)
- LAMOST DR12 (National Astronomical Data Center, China)
- Legacy Survey DR10 (NOIRLab)
- eROSITA DR1 (Max Planck Institute for Extraterrestrial Physics)
- Sonora Cholla Models (Marley et al. 2021)
