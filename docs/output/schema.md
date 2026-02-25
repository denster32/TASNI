# TASNI Output Schema Documentation

## Pipeline Overview

```
747M WISE sources
    ↓ Cross-match Gaia
406M WISE orphans (no optical counterpart)
    ↓ Quality filters (SNR, contamination flags)
2.37M high-quality IR sources
    ↓ ROSAT X-ray filter
2.37M X-ray quiet sources
    ↓ 2MASS near-IR filter
62,856 no NIR detection
    ↓ Pan-STARRS optical filter
39,188 Ultra-Stealth candidates (Tier 3)
    ↓ Physics + Scoring + Radio + Solar system
100 Golden Targets
```

## File Manifest

### Final Outputs (Top Priority)

| File | Count | Description | Key Columns |
|------|-------|-------------|-------------|
| `golden_targets.csv` | 100 | Top-ranked for follow-up | `prime_score`, `T_eff_K`, `cluster`, `score` |
| `extreme_anomalies.csv` | 100K | Most extreme thermal profiles | `anomaly_score`, `w1_w2_color` |

### Tier Files (Progressive Filtering)

| File | Count | Description | Key Columns |
|------|-------|-------------|-------------|
| `tier2_top100k.csv` | 62,856 | Gaia orphans + quality filters | `tier2_stealth_score`, `has_rosat`, `has_2mass` |
| `tier3_ultrastealth.csv` | 39,188 | After ROSAT + 2MASS + PS1 | `tier3_stealth_score`, `has_ps1` |
| `tier4_final_ultrastealth.csv` | 39,188 | Final with all constraints | `final_stealth_score`, `invisibility_count` |
| `tier4_prime.csv` | 4,624 | Room temp candidates (200-400K) | `prime_score`, `pm_total` |
| `xray_quiet_top100k.csv` | 100K | X-ray quiet with anomaly scores | `stealth_score`, `w1_w2_color_zscore` |

### Filtered Subsets

| File | Count | Description | Key Columns |
|------|-------|-------------|-------------|
| `tier4_solar_filtered.csv` | 35,631 | Solar system objects removed | `final_stealth_score`, `pm_total` |
| `tier4_solar_filtered_with_flags.csv` | 39,188 | With SSO/pm flags | `flag_sso`, `flag_pm_outlier` |

### Summary Files

| File | Description |
|------|-------------|
| `phase3_summary.csv` | Phase 3 filtering statistics |
| `phase4_final_summary.csv` | Final phase statistics |

### Documentation Files

| File | Description |
|------|-------------|
| `SOLAR_SYSTEM_COMPLETE.md` | Complete solar system object analysis |
| `solar_system_report.md` | Solar system object check report |
| `visual_inspection_report.md` | Visual inspection of candidates |

---

## Column Definitions

### Designations
- `designation`: WISE designation (e.g., J050319.34-652057.4 = RA_DEC format)

### Positions
- `ra`, `dec`: J2000 coordinates in degrees
- `pmra`, `pmdec`: Proper motion in mas/yr (Gaia-derived)
- `pm_total`: Total proper motion = sqrt(pmra² + pmdec²)

### WISE Photometry
- `w1mpro` - `w4mpro`: Profile-fit magnitudes in W1(3.4μm), W2(4.6μm), W3(12μm), W4(22μm)
- `w1sigmpro` - `w4sigmpro`: Uncertainties
- `w1snr` - `w4snr`: Signal-to-noise ratios
- `w1_w2_color` = w1mpro - w2mpro (color index; redder = higher value)
- `w3mpro_1`, `w1mpro_1`: Alternative magnitude columns

### Quality Flags
- `ph_qual`: Photometric quality (AAA = best, UAA = unusable)

### Multi-Wavelength Flags
- `has_rosat`: Has ROSAT X-ray detection (True/False)
- `has_2mass`: Has 2MASS near-IR detection (True/False)
- `has_ps1`: Has Pan-STARRS optical detection (True/False)
- `has_nvss_radio`: Has NVSS radio detection (True/False)
- `nvss_separation`: Separation from NVSS source in arcsec

### Anomaly Scoring
- `anomaly_score`: Raw anomaly score from thermal profile
- `snr_w1w2`: Signal-to-noise of W1-W2 color
- `w1_w2_zscore`: Standardized W1-W2 color
- `w3_zscore`, `w1_zscore`: Standardized W3/W1 magnitudes

### Stealth Scores (Tiered)
- `tier2_stealth_score`: Stealthiness after Tier 2 (quality + Gaia orphan)
- `tier3_stealth_score`: Stealthiness after Tier 3 (+ ROSAT + 2MASS + PS1)
- `final_stealth_score`: Final stealth score with all constraints
- `stealth_score`: Alternative stealth scoring (xray_quiet_top100k.csv)

### Invisibility Metrics
- `invisibility_count`: Number of wavelengths where source is invisible
- `no_gaia_optical`: Invisible to Gaia (1=yes, 0=no)
- `no_rosat_xray`: Invisible to ROSAT X-ray
- `no_2mass_nir`: Invisible to 2MASS near-IR
- `no_ps1_deep_optical`: Invisible to Pan-STARRS deep optical
- `spitzer_coverage`: Spitzer coverage flag

### Physics
- `T_eff_K`: Estimated effective temperature in Kelvin (blackbody fit)
- `SolidAngle_scale`: Solid angle scale factor from SED fitting

### Visualization & Clustering
- `cluster`: DBSCAN cluster assignment
- `umap_x`, `umap_y`: UMAP 2D embedding coordinates
- `vote`: Ensemble vote from clustering models
- `filename`: Image cutout filename
- `score`: Combined visualization score

### Filtering Flags
- `flag_sso`: Solar system object flag
- `flag_pm_outlier`: Proper motion outlier flag
- `flag_remove`: Flagged for removal

---

## Data Quality Notes

### Golden Targets (100)
- Filtered for solar system objects
- Room temperature candidates (200-400K)
- Highest combined anomaly + stealth scores
- Cross-matched with NVSS radio catalog

### Tier 4 Prime (4,624)
- Room temperature range (200-400K)
- All multi-wavelength stealth criteria met
- High proper motion (likely nearby)

### Ultra-Stealth (39,188)
- Invisible at all wavelengths except mid-IR
- No optical (Gaia), no X-ray (ROSAT), no near-IR (2MASS), no deep optical (PS1)
- Largest anomaly catalog for statistical analysis

### Extreme Anomalies (100K)
- Highest anomaly scores across all WISE orphans
- Useful for understanding thermal profile distribution
- Includes some sources that may have optical counterparts

---

## Usage Examples

```python
import pandas as pd

# Load golden targets (priority follow-up targets)
golden = pd.read_csv("golden_targets.csv")

# Load full ultra-stealth catalog
ultrastealth = pd.read_csv("tier3_ultrastealth.csv")

# Load room temperature candidates
prime = pd.read_csv("tier4_prime.csv")

# Get temperature statistics for golden targets
print(golden['T_eff_K'].describe())

# Find highest scoring targets
top_targets = golden.nlargest(10, 'prime_score')
```

## File Locations

- **Root TASNI directory**: `golden_targets.csv` (100 targets with clustering)
- **output/ directory**: All tier files, summaries, and reports
- **cutouts/ directory**: Image cutouts (currently empty)
