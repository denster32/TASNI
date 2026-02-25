# TASNI Error Budget

**Date:** February 4, 2026
**Purpose:** Document all sources of uncertainty in the TASNI pipeline and their contributions
**Status:** Phase 6 - Clarity and Precision Enhancement

## Overview

This document provides a comprehensive error budget for the TASNI pipeline, identifying all sources of uncertainty and their contributions to the final results. Understanding the error budget is essential for interpreting results correctly and for planning future observations.

## Error Budget Summary

| Source | Uncertainty | Contribution | Mitigation |
|--------|-------------|--------------|------------|
| Photometric calibration | 2% | Major | Use standard stars |
| Parallax fitting | 0.1-1.0 mas | Major | Longer time baseline |
| Temperature estimation | ±50 K | Moderate | Use SED fitting |
| Variability detection | ±0.1 mag | Moderate | More epochs |
| Selection function | ±10% | Moderate | Completeness analysis |
| Cross-matching | 1-2 arcsec | Minor | Use proper motion |

## Detailed Error Sources

### 1. Photometric Errors

#### 1.1 WISE Photometric Calibration

**Source:** WISE All-Sky Release Explanatory Supplement

**Uncertainty:** 2% (systematic)

**Impact:**
- Affects all magnitudes and colors
- Propagates to temperature estimates
- Affects flux measurements for variability

**Mitigation:**
- Use standard star calibrations
- Cross-check with 2MASS
- Use ensemble calibration methods

**Reference:** Wright et al. (2010), AJ, 140, 1868

#### 1.2 NEOWISE Photometric Repeatability

**Source:** NEOWISE Data Release

**Uncertainty:** 0.03 mag (random), 0.05 mag (systematic)

**Impact:**
- Affects variability detection
- Limits ability to detect small amplitude changes
- Affects light curve fitting

**Mitigation:**
- Use median of multiple epochs
- Apply photometric corrections
- Use source-specific calibrations

**Reference:** Mainzer et al. (2011), ApJ, 731, 53

#### 1.3 2MASS Photometric Calibration

**Source:** 2MASS All-Sky Catalog of Point Sources

**Uncertainty:** 2% (J), 2% (H), 2% (Ks)

**Impact:**
- Affects J-H and H-Ks colors
- Affects SED fitting
- Affects temperature estimates

**Mitigation:**
- Use standard star calibrations
- Cross-check with WISE
- Use ensemble calibration methods

**Reference:** Skrutskie et al. (2006), AJ, 131, 1163

### 2. Astrometric Errors

#### 2.1 Parallax Fitting

**Source:** Five-parameter linear model

**Uncertainty:** 0.1-1.0 mas (depending on source brightness)

**Impact:**
- Affects distance estimates
- Affects luminosity calculations
- Affects tangential velocity estimates

**Mitigation:**
- Longer time baseline
- More epochs
- Use Gaia DR3 as reference

**Reference:** [`docs/PARALLAX_METHODS.md`](PARALLAX_METHODS.md)

#### 2.2 Proper Motion Uncertainty

**Source:** NEOWISE proper motion catalog

**Uncertainty:** 10-20 mas/yr (depending on source brightness)

**Impact:**
- Affects tangential velocity estimates
- Affects kinematic analysis
- Affects cross-matching

**Mitigation:**
- Use Gaia DR3 proper motions
- Longer time baseline
- More epochs

**Reference:** Kirkpatrick et al. (2019), ApJS, 240, 19

### 3. Temperature Estimation Errors

#### 3.1 W1-W2 Color-Temperature Relation

**Source:** Empirical relation from known Y dwarfs

**Uncertainty:** ±50 K

**Impact:**
- Affects classification as Y dwarf
- Affects comparison to models
- Affects evolutionary stage estimates

**Mitigation:**
- Use SED fitting
- Use Bayesian inference
- Incorporate multiple color indices

**Reference:** [`docs/TEMPERATURE_ESTIMATION.md`](TEMPERATURE_ESTIMATION.md)

#### 3.2 Model Atmosphere Uncertainties

**Source:** Sonora Cholla atmospheric models

**Uncertainty:** ±100 K (systematic)

**Impact:**
- Affects temperature estimates
- Affects luminosity estimates
- Affects evolutionary stage estimates

**Mitigation:**
- Use multiple model grids
- Compare to observations
- Use empirical calibrations

**Reference:** Marley et al. (2021), ApJ, 916, 89

### 4. Variability Detection Errors

#### 4.1 Photometric Noise

**Source:** NEOWISE photometric noise

**Uncertainty:** ±0.1 mag per epoch

**Impact:**
- Limits detection of small amplitude variations
- Affects periodogram significance
- Affects variability classification

**Mitigation:**
- Use median of multiple epochs
- Apply photometric corrections
- Use source-specific calibrations

**Reference:** Metchev et al. (2015), ApJ, 799, 154

#### 4.2 Periodogram False Alarm Probability

**Source:** Lomb-Scargle periodogram

**Uncertainty:** FAP < 0.01 for significant periods

**Impact:**
- Affects detection of periodic signals
- Affects identification of rotational modulation
- Affects interpretation of variability

**Mitigation:**
- Use Monte Carlo simulations
- Use multiple periodogram methods
- Assess 6-month aliasing

**Reference:** [`src/tasni/analysis/periodogram_significance.py`](../src/tasni/analysis/periodogram_significance.py)

### 5. Selection Function Errors

#### 5.1 Completeness

**Source:** Selection criteria and survey coverage

**Uncertainty:** ±10%

**Impact:**
- Affects space density calculations
- Affects population comparisons
- Affects comparison to models

**Mitigation:**
- Completeness analysis
- Injection tests
- Selection function modeling

**Reference:** [`src/tasni/analysis/selection_function.py`](../src/tasni/analysis/selection_function.py)

#### 5.2 Cross-Matching Errors

**Source:** Positional uncertainties and proper motion

**Uncertainty:** 1-2 arcsec

**Impact:**
- Affects source association
- Affects multi-wavelength analysis
- Affects kinematic analysis

**Mitigation:**
- Use proper motion corrections
- Use larger matching radii
- Visual inspection

**Reference:** [`docs/pipeline.md`](pipeline.md)

### 6. Machine Learning Errors

#### 6.1 Feature Importance Uncertainty

**Source:** Random Forest feature importance

**Uncertainty:** ±0.01 (normalized)

**Impact:**
- Affects interpretability of ML models
- Affects ranking of candidates
- Affects physical interpretation

**Mitigation:**
- Use multiple models
- Use permutation importance
- Use SHAP values

**Reference:** [`docs/ML_PIPELINE_REVISED.md`](ML_PIPELINE_REVISED.md)

#### 6.2 Composite Score Uncertainty

**Source:** Weighted combination of model outputs

**Uncertainty:** ±0.05 (normalized)

**Impact:**
- Affects ranking of candidates
- Affects selection of golden sample
- Affects follow-up target selection

**Mitigation:**
- Use cross-validation
- Use multiple weighting schemes
- Use ensemble methods

**Reference:** [`docs/ML_PIPELINE_REVISED.md`](ML_PIPELINE_REVISED.md)

## Error Propagation

### Temperature to Luminosity

The luminosity is calculated as:

```
L = 4πd²F
```

where:
- d is distance (from parallax)
- F is flux (from magnitude)

The uncertainty in luminosity is:

```
σ_L/L = 2σ_d/d + σ_F/F
```

### Temperature to Luminosity

The luminosity-temperature relation is:

```
L/L☉ = (R/R☉)²(T/T☉)⁴
```

The uncertainty in luminosity is:

```
σ_L/L = 2σ_R/R + 4σ_T/T
```

### Tangential Velocity

The tangential velocity is calculated as:

```
v_tan = 4.74µd
```

where:
- µ is proper motion (in mas/yr)
- d is distance (in parsecs)

The uncertainty in tangential velocity is:

```
σ_vtan = 4.74√(σ_µ²d² + µ²σ_d²)
```

## Error Budget Tables

### Photometric Error Budget

| Source | Uncertainty | Type | Contribution |
|--------|-------------|------|-------------|
| WISE calibration | 2% | Systematic | Major |
| NEOWISE repeatability | 0.03 mag | Random | Moderate |
| 2MASS calibration | 2% | Systematic | Moderate |
| Total | ~3% | Combined | Major |

### Astrometric Error Budget

| Source | Uncertainty | Type | Contribution |
|--------|-------------|------|-------------|
| Parallax fitting | 0.1-1.0 mas | Random | Major |
| Proper motion | 10-20 mas/yr | Random | Moderate |
| Cross-matching | 1-2 arcsec | Systematic | Minor |
| Total | ~0.5 mas | Combined | Major |

### Temperature Error Budget

| Source | Uncertainty | Type | Contribution |
|--------|-------------|------|-------------|
| W1-W2 color | ±0.05 mag | Random | Moderate |
| Color-temperature relation | ±50 K | Systematic | Major |
| Model uncertainties | ±100 K | Systematic | Major |
| Total | ±120 K | Combined | Major |

### Variability Error Budget

| Source | Uncertainty | Type | Contribution |
|--------|-------------|------|-------------|
| Photometric noise | ±0.1 mag | Random | Major |
| Periodogram FAP | <0.01 | Systematic | Moderate |
| Epoch coverage | Variable | Systematic | Moderate |
| Total | ±0.15 mag | Combined | Major |

## Recommendations

1. **Prioritize reducing systematic errors** - Systematic errors are often larger than random errors
2. **Use multiple independent methods** - Cross-check results with different methods
3. **Document all assumptions** - Clearly state all assumptions and their impact
4. **Propagate uncertainties** - Always propagate uncertainties through calculations
5. **Report uncertainties with results** - Always report uncertainties with all results

## Cross-Reference

See also:
- [`docs/PARALLAX_METHODS.md`](PARALLAX_METHODS.md) - Parallax fitting methodology
- [`docs/TEMPERATURE_ESTIMATION.md`](TEMPERATURE_ESTIMATION.md) - Temperature estimation methods
- [`docs/STATISTICAL_METHODS.md`](STATISTICAL_METHODS.md) - Statistical methods
- [`src/tasni/analysis/selection_function.py`](../src/tasni/analysis/selection_function.py) - Selection function analysis

---

**Document Version:** 1.0
**Last Updated:** February 4, 2026
**Status:** Ready for Review
