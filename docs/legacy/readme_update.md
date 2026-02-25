# TASNI - Thermal Anomaly Search for Non-communicating Intelligence

**Version:** 1.2 (Updated 2026-02-02)
**Status:** Publication Ready

---

## Project Overview
TASNI identifies "fading thermal orphans"—sources detectable only in the mid-infrared (WISE W1/W2) that systematically dim over a decade. We discovered **4 extreme candidates** with unprecedented combinations of cold temperatures (205-466 K), high proper motions, and monotonic fading.

---

## Key Findings (2026 Update)

### 1. Fading Mechanism is **Not Cooling**
- **Evidence:** Periodogram analysis (Lomb-Scargle) reveals significant periodicity (P=40-400 days) in all 4 fading orphans.
- **Conclusion:** Rotational modulation or eclipsing binaries, not secular cooling.
- **File:** `reports/figures/periodogram/fading_periodograms.png`

### 2. Distances Updated
- **Significant Parallax Detections:** 75/100 sources (up from 58).
- **Fading Sources:**
  - J143046.35-025927.8: **17.4 pc** (Teff=293 K) - One of the nearest room-temperature objects known.
  - J044024.40-731441.6: **30.5 pc** (Teff=466 K).
- **File:** `data/processed/final/golden_parallax.csv`

### 3. Spectroscopy Targets Ready
- **Top 4 Fading Orphans:** Finding charts, LaTeX tables, visibility plots generated.
- **Facility Recommendations:** Keck/NIRES, VLT/KMOS.
- **Directory:** `data/processed/spectroscopy/`

### 4. Atmospheric Modeling
- **Models:** Sonora Cholla (2021) grid loaded (423 MB).
- **Comparison:** 500 K models (grid min) plotted vs. 250 K targets.
- **Result:** Targets are consistent with Y dwarfs (cooler than models).
- **Plot:** `reports/figures/models/sonora_cholla_fading_orphans.png`

### 5. X-ray Constraints
- **eROSITA DR1:** 95% of golden sample is X-ray quiet.
- **Implication:** Rules out AGN or stellar coronal activity.
- **File:** `data/processed/final/golden_erosita.parquet`

---

## Data Products

| Category | File | Description |
|-----------|--------|-------------|
| **Golden Targets** | `data/processed/final/golden_targets.csv` | 100 top candidates |
| **Parallax** | `data/processed/final/golden_parallax.csv` | 75 distance measurements |
| **Variability** | `data/processed/final/golden_variability.csv` | NEOWISE metrics |
| **ML Predictions** | `data/processed/ml/ranked_tier5_top10000.csv` | Ranked 4,137 candidates |
| **Figures** | `reports/figures/*.png` | 25 publication-ready plots |
| **Spectroscopy** | `data/processed/spectroscopy/*` | Proposal materials |

---

## Installation
```bash
make install
```

---

## Quick Start
```bash
# View fading targets
python src/tasni/analysis/periodogram_analysis.py

# Generate publication figures
python src/tasni/generation/generate_publication_figures.py
```

---

## Citation
Please cite the upcoming TASNI publication (ApJ/AJ, 2026).

---

## Acknowledgments
This research has made use of data from:
- WISE/NEOWISE (NASA/IPAC)
- Gaia DR3 (ESA)
- LAMOST DR12 (NADC)
- Legacy Survey DR10 (NOIRLab)
- eROSITA DR1 (MPE)
- Sonora Cholla Models (Marley et al. 2021)
