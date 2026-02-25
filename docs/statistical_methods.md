# Statistical Methods for TASNI

**Date:** February 4, 2026
**Purpose:** Document statistical methods used in TASNI analysis
**Status:** Phase 3 - Statistical Analysis Enhancement

## Overview

This document describes the statistical methods used to analyze TASNI results, including p-value calculations, confidence intervals, periodogram significance, and selection function analysis.

## P-Value Calculations

### Fading Significance Test

We calculate the probability of observing three or more fading sources by chance using Monte Carlo simulations. (Note: the fourth candidate, J044024, was excluded as an LMC member.)

**Method:**
1. Simulate 10,000 random light curves with NEOWISE noise properties
2. Apply same fading detection criterion (trend > 15 mmag/yr)
3. Count fading sources in each simulation
4. Calculate p-value: fraction of simulations with >=3 fading sources

**Results:**
- Expected fading: 0.5 +/- 0.7 sources (mean +/- std)
- Observed fading: 3 confirmed sources (excluding LMC member)
- P-value: ~0.01 (1%)

**Conclusion:** The observation of three fading sources is statistically significant at the ~1% level, though this should be interpreted cautiously given the multiple comparisons inherent in a survey of 100 sources.

## Confidence Intervals

### Gaussian Error Propagation

For measurements with Gaussian errors, we calculate 95% confidence intervals:

CI = value ± 1.96 × σ

Where σ is the standard error of the measurement.

### Bootstrap Confidence Intervals

For distributions without Gaussian errors, we use bootstrap resampling:

1. Resample data with replacement (10,000 iterations)
2. Calculate statistic of interest for each resample
3. Take 2.5th and 97.5th percentiles as 95% CI

**Applied to:**
- Space density calculations
- Temperature distributions
- Proper motion distributions

## Periodogram Significance

### False Alarm Probability (FAP)

We calculate FAP using two methods:

#### 1. Analytical Method (Horne & Baliunas 1986)

FAP = 1 - (1 - exp(-z))^N

Where:
- z = power / mean(power)
- N = number of independent frequencies

#### 2. Bootstrap Method

1. Resample light curve with replacement (1,000 iterations)
2. Calculate periodogram for each resample
3. FAP = fraction of bootstrap power exceeding observed power

### Significance Thresholds

| Significance | FAP | σ-equivalent |
|--------------|-----|---------------|
| 3σ | 0.0027 | High confidence |
| 2σ | 0.0455 | Moderate confidence |
| 1σ | 0.3173 | Low confidence |

**Publication threshold:** FAP < 0.01 (3.9σ) for claiming significant periodicity.

## Selection Function Analysis

### Completeness Calculations

We calculate completeness for each survey as a function of magnitude:

| Survey | Completeness Function | Reference |
|---------|---------------------|-----------|
| WISE | C(W1) = 0.98 × (1 - Φ((W1-16)/1)) | Wright et al. 2010 |
| Gaia DR3 | C(G) = 0.95 × (1 - Φ((G-18)/1.5)) | Lindegren et al. 2021 |
| 2MASS | C(K) = 0.97 × (1 - Φ((K-14.5)/0.8)) | Skrutskie et al. 2006 |

Where Φ is the standard normal CDF.

### Combined Selection Function

The combined selection function is the product of individual completenesses:

C_total = C_WISE × C_Gaia × C_2MASS × ...

### Volume Calculation

For each source, the effective survey volume is:

V = (4/3) × π × d_max³ × C_total

Where d_max is the maximum distance for which the source would be detectable.

## Variability Classification

### Quantitative Criteria

We classify sources using the following quantitative criteria:

| Classification | chi^2/nu | |trend| (mmag/yr) | Direction | p-value |
|---------------|----------|------------------|-----------|----------|
| NORMAL | < 3 | < 5 | any | - |
| VARIABLE | > 3 | < 15 | any | - |
| FADING | > 3 | >= 15 | negative | < 0.01 |
| BRIGHTENING | > 3 | >= 15 | positive | < 0.01 |

**Justification:**
- χ²/ν < 3: Consistent with measurement noise
- |trend| < 5 mmag/yr: Below typical brown dwarf variability
- p < 0.01: Statistically significant fading

### Cross-Correlation Analysis

We calculate Pearson correlation between W1 and W2 light curves to assess chromatic variability:

- r > 0.8: Strong correlation (similar behavior in both bands)
- r < 0.5: Weak correlation (different behavior)

## References

1. Horne, J. H., & Baliunas, S. L. (1986). "A periodogram analysis algorithm." *ApJ*, 312, 513.
2. Press, W. H., et al. (2007). *Numerical Recipes*, 3rd ed. Cambridge Univ. Press.
3. Lomb, N. R. (1976). "Least-squares frequency analysis of unequally spaced data." *Ap&SS*, 39, 447.
4. Fisher, R. A. (1935). "The Design of Experiments." 8th ed. Oliver & Boyd.
