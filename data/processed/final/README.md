# TASNI Data Products - Final Release

**Version:** 1.0.0
**Release Date:** 2026-02-15
**DOI:** 10.5281/zenodo.18717105

Upon acceptance, the following supplementary files will be released on Zenodo with a persistent DOI alongside the manuscript:
- **Full 100-source parallax catalog**: `golden_improved.csv` (or equivalent `golden_sample_parallax_100.csv`) including all parallax columns (`neowise_parallax_mas`, `neowise_parallax_err_mas`, `parallax_snr`, `distance_pc`, `distance_err_pc`).
- **Variability data**: Full variability parquet (e.g. `golden_variability.parquet` and/or `tier5_variability.parquet` for the golden set) as produced by `compute_ir_variability.py` / tier5 pipeline.

---

## Overview

This directory contains the final data products from the TASNI (Thermal Anomaly Search for Non-communicating Intelligence) pipeline. These files represent the culmination of processing 747 million WISE sources to identify 100 high-priority thermal anomaly candidates.

---

## File Inventory

### Golden Sample Files

| File | Format | Description | Records |
|------|--------|-------------|---------|
| `golden_improved.csv` | CSV | Main golden sample (human-readable) | 100 |
| `golden_improved.parquet` | Parquet | Main golden sample (efficient) | 100 |
| `golden_improved_parallax.csv` | CSV | Sources with significant NEOWISE parallax (>5 mas) | 67 |
| `golden_improved_parallax.parquet` | Parquet | Sources with significant parallax (efficient) | 67 |
| `golden_improved_kinematics.csv` | CSV | Sources with high proper motion | varies |
| `golden_improved_kinematics.parquet` | Parquet | Kinematics data (efficient) | varies |
| `golden_improved_erosita.csv` | CSV | eROSITA cross-match results | varies |
| `golden_improved_erosita.parquet` | Parquet | eROSITA data (efficient) | varies |
| `golden_improved_bayesian.csv` | CSV | Bayesian FP probability results (10 sources with lowest p_false_positive_proxy) | 10 |
| `golden_improved_bayesian.parquet` | Parquet | Bayesian data (efficient) | varies |

> **Note on Bayesian FP Subset:** Contains the 10 sources from the full golden sample
> with the lowest `p_false_positive_proxy` scores (ranked by composite anomaly metric).
> The remaining 90 sources have proxy values ranging from ~0.06 to ~1.0.

### Column Definitions (58 columns)

#### `golden_improved.csv` / `golden_improved.parquet`

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `designation` | string | --- | WISE designation (JHHMMSS.ss+DDMMSS.s) |
| `ra` | float | deg | Right Ascension (J2000) |
| `dec` | float | deg | Declination (J2000) |
| `w1mpro` | float | mag | W1 Vega magnitude (3.4 μm) |
| `w2mpro` | float | mag | W2 Vega magnitude (4.6 μm) |
| `w3mpro` | float | mag | W3 Vega magnitude (12 μm) |
| `w4mpro` | float | mag | W4 Vega magnitude (22 μm) |
| `w1sigmpro` | float | mag | W1 magnitude uncertainty |
| `w2sigmpro` | float | mag | W2 magnitude uncertainty |
| `w3sigmpro` | float | mag | W3 magnitude uncertainty |
| `w4sigmpro` | float | mag | W4 magnitude uncertainty |
| `w1snr` | float | --- | W1 signal-to-noise ratio |
| `w2snr` | float | --- | W2 signal-to-noise ratio |
| `w3snr` | float | --- | W3 signal-to-noise ratio |
| `w4snr` | float | --- | W4 signal-to-noise ratio |
| `w1_w2_color` | float | mag | W1 - W2 color index |
| `T_eff_K` | float | K | Effective temperature from Planck SED fitting |
| `pmra_value` | float | mas/yr | Proper motion in RA |
| `pmdec_value` | float | mas/yr | Proper motion in Dec |
| `pm_total` | float | mas/yr | Total proper motion |
| `pm_angle` | float | deg | Proper motion position angle |
| `pm_class` | string | --- | Proper motion classification |
| `n_epochs` | int | --- | Number of NEOWISE epochs |
| `baseline_years` | float | yr | NEOWISE temporal baseline |
| `rms_w1` | float | mag | W1 RMS variability |
| `rms_w2` | float | mag | W2 RMS variability |
| `chi2_w1` | float | --- | W1 chi-squared variability statistic |
| `chi2_w2` | float | --- | W2 chi-squared variability statistic |
| `trend_w1` | float | mag/yr | W1 linear brightness trend |
| `trend_w2` | float | mag/yr | W2 linear brightness trend |
| `is_variable` | bool | --- | Significant variability flag |
| `is_fading` | bool | --- | Monotonic fading flag |
| `variability_flag` | string | --- | FADING, VARIABLE, or NORMAL |
| `variability_score` | float | --- | Composite variability metric |
| `ps1_detected` | bool | --- | Pan-STARRS1 detection flag |
| `rosat_detected` | bool | --- | ROSAT X-ray detection flag |
| `detection_count` | int | --- | Number of survey detections |
| `detection_fraction` | float | --- | Fraction of surveys with detection |
| `mag_mean` | float | mag | Mean magnitude across epochs |
| `mag_std` | float | mag | Standard deviation of magnitudes |
| `mag_min` | float | mag | Minimum (brightest) magnitude |
| `mag_max` | float | mag | Maximum (faintest) magnitude |
| `mag_range` | float | mag | Magnitude range (max - min) |
| `color_mean` | float | mag | Mean W1-W2 color across epochs |
| `color_std` | float | mag | Standard deviation of W1-W2 color |
| `if_score` | float | --- | Isolation Forest anomaly score |
| `xgb_score` | float | --- | XGBoost classification score |
| `lgb_score` | float | --- | LightGBM classification score |
| `ml_ensemble_score` | float | --- | ML ensemble score (mean of IF, XGB, LGB) |
| `improved_composite_score` | float | --- | Final composite ranking score (0-1) |
| `rank` | int | --- | Overall ranking (1 = best candidate) |
| `neowise_parallax_mas` | float | mas | NEOWISE astrometric parallax from 5-parameter fit |
| `neowise_parallax_err_mas` | float | mas | Formal parallax uncertainty |
| `parallax_snr` | float | --- | Parallax signal-to-noise ratio |
| `distance_pc` | float | pc | Distance from NEOWISE parallax (1000/π); matches manuscript Table 3 |
| `distance_err_pc` | float | pc | Distance uncertainty (MCMC-derived for high-SNR; symmetric for low-SNR) |
| `p_false_positive_proxy` | float | --- | False-positive probability proxy |
| `golden_flag` | bool | --- | Golden sample membership flag |

---

## Fading Thermal Orphans

Three confirmed fading thermal orphans plus one LMC candidate, exhibiting monotonic fading:

| Designation | T_eff (K) | Distance (pc) | Period (days) | Status |
|-------------|-----------|---------------|---------------|--------|
| J143046.35-025927.8 | 293 +/- 47 | 17.4 +3.0/-2.6 | 116.3 +/- 5.0 | **Nearest** |
| J044024.40-731441.6 | 466 +/- 52 | 30.5 +1.3/-1.2 | --- | LMC region |
| J231029.40-060547.3 | 258 +/- 38 | 32.6 +13.3/-8.0 | 178.6 +/- 7.0 | High PM |
| J193547.43+601201.5 | 251 +/- 35 | --- | 92.6 +/- 4.0 | Cold |

---

## Data Quality Notes

### Completeness
- **Parallax**: 67/100 (67%) sources have significant (>5 mas) NEOWISE astrometric parallax measurements (from `extract_neowise_parallax.py`)
- **Proper motion**: 100/100 (100%) sources have proper motion estimates
- **Variability**: 100/100 (100%) sources have NEOWISE variability metrics

### Known Issues
1. **LMC Contamination**: J044024.40-731441.6 is near the LMC and may be contaminated
2. **W1 Saturation**: A few bright sources may have W1 saturation issues
3. **Edge Effects**: Sources near Ecliptic poles have reduced NEOWISE coverage

---

## Usage Examples

### Python (pandas)

```python
import pandas as pd

# Load golden sample
golden = pd.read_parquet('golden_improved.parquet')

# Filter to cold sources
cold = golden[golden['T_eff_K'] < 350]

# Get fading sources
fading = golden[golden['variability_flag'] == 'FADING']
```

### TOPCAT

1. Open TOPCAT
2. File → Load Table → Select `golden_improved.csv`
3. Use the "Sort" function on `improved_composite_score`
4. Use the "Plot" window for color-magnitude diagrams

---

## Checksums

See `checksums.txt` for SHA256 hashes of all 10 data files (CSV + Parquet).

Verify with: `sha256sum -c checksums.txt`

---

## Citation

If you use these data, please cite:

```bibtex
@article{tasni2026,
    author = {{Palucki, Dennis}},
    title = "{TASNI: Thermal Anomaly Search for Non-communicating Intelligence}",
    journal = {ApJ},
    year = {2026},
    volume = {},
    pages = {},
    doi = {}
}
```

---

## Contact

For questions about these data, contact: paluckide@yahoo.com

---

**Last Updated:** 2026-02-15
