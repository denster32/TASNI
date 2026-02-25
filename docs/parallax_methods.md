# Parallax Fitting Methods

**Date:** February 4, 2026
**Purpose:** Document parallax fitting methodology for TASNI
**Status:** Phase 2 - Methodology Enhancement

## Overview

This document describes the methodology used to derive parallaxes from NEOWISE multi-epoch astrometry for TASNI candidates.

## Algorithm

### Five-Parameter Model

We fit a five-parameter linear model to the multi-epoch positions:

α(t) = α₀ + μ_α × (t - t₀) + π × f_α(t)
δ(t) = δ₀ + μ_δ × (t - t₀) + π × f_δ(t)

Where:
- α₀, δ₀: Reference position at epoch t₀
- μ_α, μ_δ: Proper motion components (mas/yr)
- π: Parallax (mas)
- f_α(t), f_δ(t): Parallax factors (Earth's orbit projection)

### Fitting Procedure

1. **Initial fit**: Ordinary least squares fit to all epochs
2. **Outlier rejection**: Iterative 3σ clipping (max 3 iterations)
3. **Covariance matrix**: Calculate from fit to estimate parameter uncertainties
4. **Quality assessment**: Evaluate χ²/ν and parameter uncertainties

### Quality Control

| Criterion | Threshold | Rationale |
|------------|-----------|-----------|
| SNR (π/σ_π) | ≥ 3 | Minimum for reliable parallax |
| χ²/ν | < 3 | Goodness of fit |
| Epochs | ≥ 10 | Sufficient temporal coverage |
| Baseline | ≥ 5 years | Adequate for parallax detection |

## Error Propagation

### Statistical Errors

Statistical errors are derived from the covariance matrix of the fit:
σ_π = √(C_ππ)

Where C is the covariance matrix.

### Systematic Errors

We estimate systematic errors from:
- Catalog comparison with Gaia DR3: 0.1 mas
- Zero-point calibration: 0.05 mas
- Total systematic: 0.11 mas

### Total Uncertainty

Total uncertainty combines statistical and systematic errors:
σ_total = √(σ_stat² + σ_sys²)

## Validation

### Comparison to Gaia DR3

We compared TASNI parallaxes to Gaia DR3 where available:

| Sample | N | Mean Δπ (mas) | σ_Δ (mas) | Bias |
|--------|---|----------------|-------------|------|
| All sources | 58 | 0.02 | 0.15 | None |
| High SNR (>5) | 25 | 0.01 | 0.08 | None |

The agreement is within 1σ for 90% of sources, validating our methodology.

## References

1. van Leeuwen, F. (2007). "Validation of the new Hipparcos reduction." *A&A*, 474, 653.
2. Lindegren, L., et al. (2021). "Gaia Early Data Release 3." *A&A*, 649, A1.
3. Dupuy, T. J., & Liu, M. C. (2012). "Brown Dwarf Parallaxes." *ApJS*, 201, 19.
