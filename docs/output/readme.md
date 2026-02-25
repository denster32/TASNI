# TASNI Output Directory

This directory contains all pipeline outputs from the Thermal Anomaly Search for Non-Communicating Intelligence (TASNI).

## Quick Start

```python
import pandas as pd

# Load golden targets (priority follow-up targets)
golden = pd.read_csv("../golden_targets.csv")

# Load full ultra-stealth catalog (39K candidates)
ultrastealth = pd.read_csv("tier3_ultrastealth.csv")

# Load room temperature candidates
prime = pd.read_csv("tier4_prime.csv")

# Load extreme anomalies for statistical analysis
extreme = pd.read_csv("extreme_anomalies.csv")
```

## Pipeline Summary

| Phase | Sources | Filter | Output File |
|-------|---------|--------|-------------|
| Phase 1 | 747,000,000 | WISE catalog | - |
| Phase 2 | 406,387,755 | No Gaia match | tier2_top100k.csv |
| Phase 3 | 39,188 | Multi-wavelength veto | tier3_ultrastealth.csv |
| Phase 4 | 4,624 | Room temp (200-400K) | tier4_prime.csv |
| Phase 5 | 100 | Golden targets | ../golden_targets.csv |

## File Guide

### Start Here
- **`golden_targets.csv`** (root) - 100 top-ranked targets for follow-up observation
- **`extreme_anomalies.csv`** - 100K most extreme thermal profiles

### Full Catalogs
- **`tier3_ultrastealth.csv`** - 39,188 sources invisible at all wavelengths except mid-IR
- **`tier4_prime.csv`** - 4,624 room temperature candidates (200-400K)

### Intermediate Files
- **`tier2_top100k.csv`** - Gaia orphans after quality filtering
- **`xray_quiet_top100k.csv`** - X-ray quiet with anomaly scores
- **`tier4_solar_filtered.csv`** - SSO-removed subset
- **`tier4_solar_filtered_with_flags.csv`** - With filtering flags

### Summaries
- **`phase3_summary.csv`** - Phase 3 filtering statistics
- **`phase4_final_summary.csv`** - Final phase statistics

### Reports
- **`SOLAR_SYSTEM_COMPLETE.md`** - Solar system object analysis
- **`visual_inspection_report.md`** - Visual inspection results

## Key Columns

| Column | Description |
|--------|-------------|
| `designation` | WISE source name (JHHMMSS.ss+DDMMSS.s) |
| `ra`, `dec` | J2000 coordinates (degrees) |
| `w1mpro` - `w4mpro` | WISE magnitudes (3.4, 4.6, 12, 22 μm) |
| `w1_w2_color` | W1 - W2 color (redder = higher) |
| `anomaly_score` | Thermal profile anomaly score |
| `prime_score` | Final combined score (golden targets) |
| `T_eff_K` | Blackbody temperature (Kelvin) |
| `has_rosat` | Has X-ray detection? |
| `has_2mass` | Has near-IR detection? |
| `has_ps1` | Has optical detection? |
| `cluster` | DBSCAN cluster assignment |
| `umap_x`, `umap_y` | UMAP embedding coordinates |

## Temperature Ranges

| Category | T_eff_K | Interpretation |
|----------|---------|----------------|
| Cold debris | < 100K | Distant/cool debris disks |
| Brown dwarf transition | 100-300K | Y-dwarfs, coldest brown dwarfs |
| **Room temperature** | **200-400K** | **Golden target range** |
| Young hot Jupiters | 400-1000K | Self-luminous planets |
| Main sequence M-dwarfs | > 1000K | Stellar sources |

## Citation

If you use these results, please cite:

```
Palucki, D. (2026). TASNI: Thermal Anomaly Search for Non-Communicating Intelligence.
A systematic search for infrared sources with no optical counterpart.
```

## Related Files

- **`../gbt_golden_targets.cat`** - Green Bank Telescope observation catalog
- **`../README.md`** - Main project documentation
- **`../docs/THESIS.md`** - Full thesis document
- **`../docs/DEVLOG.md`** - Development log

## Contact

For questions about the pipeline or results, refer to the main TASNI project documentation.
