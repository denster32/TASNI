# Phase 3: Code Pipeline Audit - Notes

## 3.1 ML Circularity Analysis

### CRITICAL FINDING: ML Circularity in ml_scoring.py

**The Issue:**
The `ml_scoring.py` pipeline has **textbook data leakage**:

| Model | Features Include Scores? | Labels Source |
|-------|-------------------------|---------------|
| Isolation Forest | YES (includes 'score' columns) | None (unsupervised) |
| XGBoost | YES (same features) | Top 20% of composite_score |
| LightGBM | YES (same features) | Top 20% of composite_score |

**Code Evidence:**
```python
# Features include score columns (line 37-38)
feature_cols = [col for col in df.columns if col.startswith(('w1_', 'w2_', 'pm_', 'var_', 'rms_')) or 'score' in col.lower()]

# Labels derived from existing score (line 70-74)
y_proxy = (df[existing_score_col] > df[existing_score_col].quantile(0.8)).astype(int)
```

**Severity: HIGH** - Models reinforce existing rankings rather than discovering new candidates.

**Mitigation Exists:** `enhanced_ensemble.py` excludes score columns and uses known BD catalogs as labels.

---

## 3.2 Temperature Estimation (SED Fitting)

### VERIFIED: Implementation is Correct

**Planck Function:** Correctly implemented
```python
B_ν(T) = (2hν³/c²) × 1/(exp(hν/kT) - 1)
```

**WISE Zero Points:** Uses Jarrett et al. 2011 values (correct, updated from Wright 2010)

| Band | Code Value | Wright 2010 | Difference |
|------|------------|-------------|------------|
| W1 | 309.540 Jy | 306.682 Jy | +0.93% |
| W2 | 171.787 Jy | 170.663 Jy | +0.66% |

**Temperature Verification:**

| Source | Claimed T | Fitted T | Status |
|--------|-----------|----------|--------|
| J143046.35-025927.8 | 293 K | 293.23 K | PASS |
| J231029.40-060547.3 | 258 K | 258.13 K | PASS |
| J193547.43+601201.5 | 251 K | 250.51 K | PASS |

---

## 3.3 Variability Analysis

### Fade Rate Calculation: VERIFIED

| Source | Data Value | Manuscript | Status |
|--------|------------|------------|--------|
| J1430 | 25.5 mmag/yr | 26 mmag/yr | PASS |
| J2310 | 52.6 mmag/yr | 53 mmag/yr | PASS |
| J1935 | 22.9 mmag/yr | 23 mmag/yr | PASS |

### 3-Sigma Threshold Definition
- `p_value < 0.01` for slope ≠ 0
- Minimum fade rate: 15-20 mmag/yr
- Uses `scipy.stats.linregress()` on magnitude vs. time

### MAJOR CONCERN: Lomb-Scargle Not Appropriate for Secular Trends

**Issue:** Lomb-Scargle is designed for periodic signals, not monotonic fading.

**Consequences:**
1. Detected "periods" (93-179 days) are NEOWISE cadence aliases
2. Extremely small FAP values (10^-61, 10^-46, 10^-11) are misleading
3. Manuscript acknowledges this but may give false impression of periodicity

**Manuscript statement (correct):**
> "these periods are closely related to the ~182-day NEOWISE observing cadence... We therefore conclude these are likely sampling aliases rather than astrophysical signals."

---

## 3.4 Parallax and Distance Verification

### Method: 5-Parameter Linear Astrometric Model
```
RA(t)  = RA₀ + μα* × Δt + π × Pα(t)
Dec(t) = Dec₀ + μδ × Δt + π × Pδ(t)
```

### CRITICAL DISCREPANCY: Distance Values

| Source | Manuscript Distance | Parallax File Range | Issue |
|--------|--------------------|--------------------|-------|
| J1430 | 17.4 pc | 50-199 pc | NOT FOUND in file |

**Mathematical Check:**
- d = 17.4 pc requires π = 1000/17.4 = **57.5 mas**
- Parallax file shows range 5-20 mas
- J1430 with 17.4 pc distance is NOT in the parallax file

### PSF Limitation: NOT ADDRESSED
- NEOWISE PSF ≈ 6 arcsec
- 57.5 mas parallax signal is **0.9%** of PSF width
- Code uses uniform weighting, no position errors

### Asymmetric Errors: NOT CALCULATED
- Manuscript claims +3.0/-2.6 pc for J1430
- Code only produces symmetric errors
- Asymmetric values are **hardcoded in figures**, not derived

---

## 3.5 Cross-Match Verification

### All Claims VERIFIED:

| Claim | Status | Evidence |
|-------|--------|----------|
| Gaia radius = 3 arcsec | PASS | `MATCH_RADIUS_ARCSEC = 3.0` |
| WISE orphan = no Gaia match | PASS | `sep2d.arcsec > MATCH_RADIUS_ARCSEC` |
| UKIDSS veto | PASS | `query_ukidss()` in tier_vetoes.py |
| VHS veto | PASS | `query_vhs()` in tier_vetoes.py |
| CatWISE veto | PASS | `query_catwise()` in tier_vetoes.py |
| Legacy Survey veto | PASS | `crossmatch_legacy.py` |
| Source counts | PASS | 747M→406M→2.37M→62K→39K→4137→100→3 |

---

## Summary of Phase 3 Findings

| Category | Status | Key Issues |
|----------|--------|------------|
| ML Circularity | CRITICAL | Score-as-feature + score-as-label in ml_scoring.py |
| SED Fitting | VERIFIED | Temperatures correct (293, 258, 251 K) |
| Variability | VERIFIED | Fade rates correct (23, 26, 53 mmag/yr) |
| Lomb-Scargle | MAJOR | Not appropriate for secular trends |
| Parallax | CRITICAL | Distance discrepancy (17.4 pc vs 50-199 pc) |
| Cross-match | VERIFIED | All implementation correct |
