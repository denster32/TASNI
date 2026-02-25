# TASNI Data Verification Audit

**Date**: 2026-02-20
**Auditor**: Data-Verifier Agent (Claude Opus 4.6)
**Scope**: Computational verification of every number in the manuscript against actual data files

---

## 1. Data File Inventory and Integrity

### 1.1 Golden Sample CSV
- **File**: `data/processed/final/golden_improved.csv`
- **Shape**: 100 rows x 58 columns
- **NaN audit**: 21 NaN in `neowise_parallax_mas`, 21 in `neowise_parallax_err_mas`, 20 in `parallax_snr`, 30 in `distance_pc`, 30 in `distance_err_pc`. All other columns have 0 NaN.
- **Result**: PASS

### 1.2 SHA-256 Checksum Verification
All 10 data files verified against `data/processed/final/checksums.txt`:

| File | Status |
|------|--------|
| golden_improved.csv | MATCH |
| golden_improved.parquet | MATCH |
| golden_improved_parallax.csv | MATCH |
| golden_improved_parallax.parquet | MATCH |
| golden_improved_kinematics.csv | MATCH |
| golden_improved_kinematics.parquet | MATCH |
| golden_improved_erosita.csv | MATCH |
| golden_improved_erosita.parquet | MATCH |
| golden_improved_bayesian.csv | MATCH |
| golden_improved_bayesian.parquet | MATCH |

**Result**: ALL 10 CHECKSUMS MATCH

---

## 2. Fading Orphan Property Verification (Task 3)

### 2.1 J143046.35-025927.8 (Rank 4)

| Property | CSV Value | Manuscript (Table 3) | Status |
|----------|-----------|---------------------|--------|
| T_eff_K | 293.23 | 293 +/- 47 | MATCH (rounds to 293) |
| pm_total | 55.23 mas/yr | 55 +/- 5 | MATCH |
| trend_w1 | -0.0255 mag/yr | 0.026 +/- 0.003 | MATCH (abs rounds to 0.026) |
| neowise_parallax_mas | 57.58 | 57.6 +/- 9.9 | MATCH |
| neowise_parallax_err | 9.90 | 9.9 | MATCH |
| parallax_snr | 5.82 | 5.8 | MATCH |
| distance_pc | 17.37 | 17.4 | MATCH |
| variability_flag | FADING | (confirmed fading) | MATCH |

### 2.2 J231029.40-060547.3 (Rank 86)

| Property | CSV Value | Manuscript (Table 3) | Status |
|----------|-----------|---------------------|--------|
| T_eff_K | 258.13 | 258 +/- 38 | MATCH |
| pm_total | 165.42 mas/yr | 165 +/- 17 | MATCH |
| trend_w1 | -0.0526 mag/yr | 0.053 +/- 0.006 | MATCH |
| neowise_parallax_mas | 30.66 | 30.7 +/- 12.5 | MATCH |
| neowise_parallax_err | 12.53 | 12.5 | MATCH |
| parallax_snr | 2.45 | 2.4 | MATCH |
| distance_pc | 32.61 | 32.6 | MATCH |
| variability_flag | FADING | (confirmed fading) | MATCH |

### 2.3 J193547.43+601201.5 (Rank 94)

| Property | CSV Value | Manuscript (Table 3) | Status |
|----------|-----------|---------------------|--------|
| T_eff_K | 250.51 | 251 +/- 35 | MATCH (rounds to 251) |
| pm_total | 306.47 mas/yr | 306 +/- 31 | MATCH |
| trend_w1 | -0.0229 mag/yr | 0.023 +/- 0.003 | MATCH |
| neowise_parallax_mas | -0.96 | (no detection) | MATCH |
| distance_pc | NaN | --- | MATCH |
| variability_flag | FADING | (confirmed fading) | MATCH |

### 2.4 LMC Source: J044024.40-731441.6 (Rank 7)

| Property | CSV Value | Manuscript | Status |
|----------|-----------|------------|--------|
| T_eff_K | 465.85 | (not in Table 3) | N/A |
| pm_total | 165.47 | 165 mas/yr | MATCH |
| distance_pc | 30.53 | ~30.5 pc | MATCH |
| variability_flag | FADING | (LMC member excluded) | MATCH |
| Galactic l | 285.7 deg | 285.7 deg | MATCH (astropy verified) |
| Galactic b | -35.1 deg | -35.1 deg | MATCH (astropy verified) |
| Sep from LMC | 4.9 deg | 4.9 deg | MATCH (astropy verified) |

### 2.5 Total FADING Sources
- CSV: 4 sources with variability_flag == "FADING"
- Paper: 3 confirmed + 1 LMC member = 4 total
- **Result**: MATCH

---

## 3. J1430 Temperature Investigation (Task 4)

**Question**: MEMORY.md states "CSV shows T_eff_K = 309.84, manuscript says 293+/-47". Is there a discrepancy?

**Finding**: The 309.84 K value belongs to **J161308.63-513447.6** (Rank 1), NOT to J1430. The CSV confirms J1430 has T_eff_K = 293.23, which rounds to 293 K and matches the manuscript exactly.

**4th Fading Source**: There are exactly 4 FADING sources in the CSV. The 4th is J044024.40-731441.6 (the LMC member), which the paper correctly identifies and excludes from the confirmed sample.

**Result**: NO DISCREPANCY. The MEMORY.md reference was a confusion between J161308 (rank 1) and J1430 (rank 4).

---

## 4. Pipeline Statistics Verification (Task 5, Table 2)

| Phase | Manuscript Count | Manuscript Reduction | Computed Reduction | Status |
|-------|-----------------|---------------------|-------------------|--------|
| AllWISE catalog | 747,634,026 | --- | --- | External claim |
| No Gaia match | 406,387,755 | 54.6% | 54.4% | **MINOR DISCREPANCY** |
| Quality filters | 2,371,667 | 99.4% | 99.4% | MATCH |
| No 2MASS | 62,856 | 97.3% | 97.3% | MATCH |
| No Legacy Survey | 39,188 | 37.6% | 37.7% | MATCH |
| MW quiet | 4,137 | 89.5% | 89.4% | MATCH |
| Golden sample | 100 | 97.6% | 97.6% | MATCH |
| Fading orphans | 3 | 97.0% | 97.0% | MATCH |

**Note on Gaia percentage**: The "Reduction" column for the Gaia step shows 54.6%, but 406,387,755 / 747,634,026 = 54.4%. For all other rows, "Reduction" means percentage removed (e.g., 99.4% = fraction removed by quality filters). For the Gaia step, the column appears to show the fraction *retained* rather than removed (54.4% retained vs 45.6% removed). This is an inconsistency in column interpretation. The absolute count (406M) is what matters.

**Severity**: MINOR

---

## 5. CDS Table Verification (Task 6)

### 5.1 Value Comparison
- **100 data rows** parsed from `tasni_paper_final/golden_sample_cds.txt`
- Every row compared against `golden_improved.csv` on: T_eff, pm_total, rank, variability_flag, W1, W2, parallax
- **Result**: ZERO discrepancies found

### 5.2 Byte-by-Byte Format Compliance
- All 100 lines pass designation, RA, Dec, and Rank format checks
- All designations start with 'J'
- RA values in [0, 360], Dec values in [-90, 90]
- Ranks 1-100, all integers
- Column separators present at correct byte positions
- **Result**: ALL FORMAT CHECKS PASSED

### 5.3 FADING Source Flags
- CDS note states: "4 sources are classified as FADING (3 confirmed + 1 LMC member)"
- Actual FADING count in CDS: 4
- **Result**: MATCH

---

## 6. eROSITA Verification (Task 8)

From `golden_improved_erosita.csv` (100 rows, 60 columns):

| Category | Count | Manuscript Claim | Status |
|----------|-------|-----------------|--------|
| erosita_coverage = True | 59 | 59 | MATCH |
| erosita_coverage = False | 41 | 41 | MATCH |
| erosita_flag = NO_DETECTION | 59 | "none have X-ray counterparts" | MATCH |
| erosita_flag = OUTSIDE_COVERAGE | 41 | "41 sources lie in eastern hemisphere" | MATCH |

**Verification method**: Astropy galactic coordinate computation confirms 59 sources have l > 180 deg (western hemisphere) and 41 have l <= 180 deg (eastern hemisphere).

**Result**: ALL eROSITA CLAIMS VERIFIED

---

## 7. Parallax Verification (Task 9)

### 7.1 Parallax Counts
| Metric | Data Value | Manuscript Claim | Status |
|--------|-----------|-----------------|--------|
| Sources with parallax > 5 mas | 67 | "67 of 100" | MATCH |
| Parallax CSV rows | 67 | 67 | MATCH |
| Sources with any parallax | 79 | (not claimed) | N/A |
| Sources with SNR > 2 | 58 | (not claimed) | N/A |

### 7.2 Distance Error Bars

**DISCREPANCY FOUND**: The manuscript Table 3 claims asymmetric distance errors, and the table comments state: "Distance uncertainties are asymmetric due to the non-linear parallax-to-distance transformation (d = 1000/pi); we propagate the formal parallax uncertainty through this transformation."

However, the computed values do not match this description:

#### J143046.35-025927.8 (pi = 57.58 +/- 9.90 mas)

| Method | Upper Error | Lower Error |
|--------|------------|-------------|
| Manuscript | +3.0 | -2.6 |
| Naive asymmetric (1000/(pi-sig) - d) | +3.6 | -2.5 |
| Symmetric (d * sig/pi) | +/-3.0 | +/-3.0 |
| CSV distance_err_pc | 2.99 | 2.99 |

The upper error (+3.0) matches the symmetric method. The lower error (-2.6) matches neither method.

#### J231029.40-060547.3 (pi = 30.66 +/- 12.53 mas)

| Method | Upper Error | Lower Error |
|--------|------------|-------------|
| Manuscript | +13.3 | -8.0 |
| Naive asymmetric (1000/(pi-sig) - d) | +22.5 | -9.5 |
| Symmetric (d * sig/pi) | +/-13.3 | +/-13.3 |
| CSV distance_err_pc | 13.32 | 13.32 |

The upper error (+13.3) matches the symmetric method exactly. The lower error (-8.0) matches neither method.

**Analysis**: The code in `extract_neowise_parallax.py` line 250-254 computes `distance_err_pc = distance_pc * (parallax_err / parallax_fit)` — this is the symmetric first-order approximation. An `asymmetric_distance_errors()` function exists at line 54-73 that computes correct asymmetric errors, but it is not called in the main pipeline.

The table comments claim the errors are from "non-linear parallax-to-distance transformation" but the actual upper error matches the symmetric method, while the lower error appears to be an ad-hoc value that matches neither computation.

**Severity**: MODERATE for J1430 (diff ~0.6 pc in upper, ~0.1 in lower), SIGNIFICANT for J2310 (upper should be +22.5 not +13.3 per the stated method, lower should be -9.5 not -8.0).

---

## 8. Comprehensive Manuscript Number Sweep

### 8.1 Abstract Claims

| Claim | Verification | Status |
|-------|-------------|--------|
| "100 high-priority candidates" | len(CSV) = 100 | MATCH |
| "T_eff = 251 +/- 35 to 293 +/- 47 K" | min=250.5 (rounds 251), max=293.2 (rounds 293) | MATCH |
| "mu = 55--306 mas/yr" | min=55.2 (rounds 55), max=306.5 (rounds 306) | MATCH |
| "17.4+3.0/-2.6 pc" | CSV distance_pc=17.37 (rounds 17.4) | MATCH (distance), see Sec 7.2 for errors |
| "59 sources within eROSITA DR1 footprint" | 59 with l > 180 deg | MATCH |

### 8.2 Section 2.2 (Pipeline)

| Claim | Verification | Status |
|-------|-------------|--------|
| "747,634,026 AllWISE sources" | External catalog, not verifiable from data | N/A |
| "406M WISE orphans" | Table 2 value | N/A |

### 8.3 Section 3.2 (Fading Orphans)

| Claim | Verification | Status |
|-------|-------------|--------|
| "initially identified five sources" | 4 FADING + J0605 (VARIABLE, dropped) | CONSISTENT |
| J0605 footnote: W1-W2=2.00 | CSV: 2.00 | MATCH |
| J0605 footnote: T_eff~253 | CSV: 253.1 | MATCH |
| J0605 footnote: mu=359 | CSV: 358.7 (rounds 359) | MATCH |
| J0605 footnote: fade=17.9 mmag/yr | CSV: |trend_w1|*1000=17.9 | MATCH |
| J0605 variability_flag | VARIABLE (not FADING) | CONSISTENT (dropped below threshold) |

### 8.4 Section 3.4 (Parallax)

| Claim | Verification | Status |
|-------|-------------|--------|
| "67 of 100 golden sample sources" with plx > 5 mas | (plx > 5).sum() = 67 | MATCH |
| J1430: "57.6 +/- 9.9 mas (SNR=5.8)" | CSV: 57.58/9.90/5.82 | MATCH |
| J2310: "30.7 +/- 12.5 mas (SNR=2.4)" | CSV: 30.66/12.53/2.45 | MATCH |
| J1935: "lacks significant parallax" | CSV: plx=-0.96, SNR=0.19 | MATCH |

### 8.5 Section 4.2 (Fading Nature)

| Claim | Verification | Status |
|-------|-------------|--------|
| "23--53 mmag/yr" | |trend_w1|*1000: min=22.9, max=52.6 | MATCH (rounds to 23--53) |
| "55--306 mas/yr" | pm_total: 55.23--306.47 | MATCH |

### 8.6 Section 4.3 (LMC Source)

| Claim | Verification | Status |
|-------|-------------|--------|
| "(l,b) = (285.7, -35.1)" | Astropy: l=285.7, b=-35.1 | MATCH |
| "4.9 deg from LMC center" | Astropy separation: 4.9 deg | MATCH |
| "distance ~30.5 pc" | CSV: 30.53 | MATCH |
| "proper motion of 165 mas/yr" | CSV: 165.47 | MATCH |

### 8.7 Table 5 (Y Dwarf Comparison)

| Source | Property | Table 5 | CSV | Status |
|--------|----------|---------|-----|--------|
| J1430 | T_eff | 293 | 293.23 | MATCH |
| J1430 | Dist | 17.4 | 17.37 | MATCH |
| J1430 | mu | 55 | 55.23 | MATCH |
| J2310 | T_eff | 258 | 258.13 | MATCH |
| J2310 | Dist | 32.6 | 32.61 | MATCH |
| J2310 | mu | 165 | 165.42 | MATCH |
| J1935 | T_eff | 251 | 250.51 | MATCH |
| J1935 | mu | 306 | 306.47 | MATCH |

### 8.8 Figure 3 Caption

**Claim**: "The three confirmed fading thermal orphans (red stars) occupy the reddest colors"

**Finding**: Only J1430 (W1-W2 = 3.37, 98th percentile) is among the reddest sources. J2310 (W1-W2 = 1.75, 25th percentile) and J1935 (W1-W2 = 1.53, 4th percentile) are NOT among the reddest.

The top 5 reddest sources are:
1. J043338.57-731619.4: W1-W2=3.67 (NORMAL)
2. J161308.63-513447.6: W1-W2=3.55 (NORMAL)
3. J143046.35-025927.8: W1-W2=3.37 (FADING)
4. J054235.56-713535.8: W1-W2=3.14 (VARIABLE)
5. J055348.26-573136.0: W1-W2=2.82 (NORMAL)

**Severity**: MODERATE. The caption overstates the color properties. Only 1 of 3 fading orphans is among the reddest. The caption should say something like "J1430 occupies one of the reddest colors" or qualify the statement.

---

## 9. Validation Code Verification

### 9.1 validation.py Golden Selection
- Selects top 100 by `improved_composite_score` (line 53): **VERIFIED** (100 rows in output)
- Drops stale `est_parallax_mas` column (line 68): **VERIFIED** (column absent from CSV)
- Merges NEOWISE parallax from `golden_parallax.csv` (lines 71-79): **VERIFIED** (parallax columns present)

### 9.2 Kinematics Filter
- Filter: `pm_total > 100` (line 20)
- J1430 (pm=55.23) correctly excluded from kinematics CSV
- Kinematics CSV: 87 rows (from full tier5, not just golden 100)

### 9.3 eROSITA Coverage
- Uses astropy SkyCoord → galactic, `l > 180 deg` (lines 28-31)
- Independently verified: 59 western, 41 eastern

### 9.4 Parallax Filter
- Filter: `neowise_parallax_mas > 5 and not null` (line 43)
- Parallax CSV: 67 rows -- **MATCHES** manuscript claim

### 9.5 MCMC Comparison Script
- Uses SYNTHETIC data (not real NEOWISE astrometry)
- TRUE_PLX_MAS set to 57.6 (matching J1430 value)
- LS result on synthetic data: 97.22 mas (does NOT match CSV 57.58 -- expected for different data)
- Purpose: appendix illustration only, does not affect Table 3 values

---

## 10. Summary of Discrepancies

### CRITICAL
None found.

### SIGNIFICANT
1. **Distance error bars (J2310)**: Manuscript claims +13.3/-8.0 pc. The stated methodology ("asymmetric from non-linear transformation") would yield +22.5/-9.5. The actual upper error (+13.3) matches the *symmetric* first-order approximation, contradicting the table comment. The lower error (-8.0) matches neither method. The `asymmetric_distance_errors()` function in `extract_neowise_parallax.py` is defined but never called by the pipeline.

### MODERATE
2. **Distance error bars (J1430)**: Paper claims +3.0/-2.6. Naive asymmetric gives +3.6/-2.5; symmetric gives +/-3.0. The upper matches symmetric, the lower (-2.6) is between the two methods.

3. **Figure 3 caption overstatement**: "fading orphans occupy the reddest colors" is only true for J1430 (98th percentile in W1-W2). J2310 (25th percentile) and J1935 (4th percentile) are among the bluest W1-W2 colors in the sample.

### MINOR
4. **Table 2 Gaia percentage**: Listed as 54.6% but computes to 54.4% (406,387,755 / 747,634,026). The "Reduction" column also changes meaning between the Gaia row (appears to show fraction retained) and all other rows (shows fraction removed). Does not affect scientific conclusions.

5. **Table 2 Legacy Survey percentage**: Listed as 37.6% but computes to 37.7% (1 - 39,188/62,856). Rounding difference only.

---

## 11. Recommendations

1. **Fix distance error methodology** (Sec 3.4, Table 3): Either (a) call `asymmetric_distance_errors()` to produce correct asymmetric errors and update the table, or (b) report symmetric errors with +/- notation and update the table comment to not claim non-linear propagation.

2. **Fix Figure 3 caption**: Change "The three confirmed fading thermal orphans (red stars) occupy the reddest colors" to "The fading thermal orphans are highlighted as red stars, with J1430 occupying one of the reddest W1-W2 colors in the sample."

3. **Clarify Table 2 "Reduction" column**: Make consistent whether it means % removed or % retained. Currently the Gaia row shows retained (54.4%) while all other rows show removed.

---

## 12. Verification Summary

| Category | Items Checked | Pass | Fail | Discrepancies |
|----------|--------------|------|------|---------------|
| Checksums | 10 files | 10 | 0 | 0 |
| Fading orphan properties | 3 sources x 8 properties | 24 | 0 | 0 |
| LMC source | 5 properties | 5 | 0 | 0 |
| Table 2 pipeline stats | 8 rows | 6 | 0 | 2 minor |
| CDS vs CSV | 100 sources x 7 fields | 700 | 0 | 0 |
| CDS format | 100 lines | 100 | 0 | 0 |
| eROSITA counts | 4 values | 4 | 0 | 0 |
| Parallax counts | 2 values | 2 | 0 | 0 |
| Distance error bars | 2 sources | 0 | 2 | 2 (1 significant + 1 moderate) |
| Abstract numbers | 5 claims | 5 | 0 | 0 |
| Section text numbers | ~15 claims | 15 | 0 | 0 |
| Table 5 values | 8 values | 8 | 0 | 0 |
| Figure caption accuracy | 1 claim | 0 | 1 | 1 moderate |
| **TOTAL** | **~875** | **~869** | **3** | **5** |

**Overall assessment**: The data files and manuscript are highly consistent. All 100 CDS entries match the CSV exactly. All fading orphan properties in Table 3 match to rounding precision. The three discrepancies requiring attention are: (1) the distance error bar methodology mismatch (significant for J2310), (2) Figure 3 caption overstatement about color properties, and (3) a minor Table 2 percentage ambiguity.
