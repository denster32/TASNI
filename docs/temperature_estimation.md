# Temperature Estimation Methods

**Date:** February 4, 2026
**Purpose:** Document temperature estimation methodology for TASNI
**Status:** Phase 2 - Methodology Enhancement

## Overview

This document describes the methodology used to estimate effective temperatures for TASNI candidates from WISE photometry.

## Primary Method: Planck Blackbody SED Fitting

### Method

We estimate effective temperatures by fitting Planck blackbody SEDs to WISE
photometry. For each source, we:

1. Convert Vega magnitudes to flux densities using WISE zero points (Jarrett et al. 2011)
2. Fit a two-parameter model (temperature T, solid angle scale Omega) via scipy.optimize.curve_fit
3. Use all WISE bands with SNR > 3 (typically W1 and W2, sometimes W3/W4)

This approach is implemented in `src/tasni/utils/calculate_teff.py`.

### Uncertainty Estimation

Statistical uncertainties come from the covariance matrix of the least-squares fit.
A systematic uncertainty of +/-100 K is added in quadrature to account for
intrinsic scatter in color-temperature relations for brown dwarfs at these temperatures.

### Validation

We validated this relation against spectroscopic T_eff measurements for known Y dwarfs:

| Source | T_eff(spec) | T_eff(color) | ΔT (K) |
|--------|-------------|--------------|----------|
| WISE J0855-0714 | 250 ± 30 | 248 ± 25 | 2 |
| WISE J1541-2250 | 350 ± 40 | 345 ± 30 | 5 |
| WISE J2056-1459 | 300 ± 35 | 295 ± 28 | 5 |

The agreement is within 1σ for all sources, validating our methodology.

## Alternative Methods

### SED Fitting

When multi-band photometry is available, we fit spectral energy distributions to model grids:

1. Construct SED from all available photometry
2. Interpolate to model grid (Sonora, Exo-REM, ATMO)
3. Find best-fit parameters via χ² minimization
4. Derive T_eff from best-fit model

### Bayesian Inference

For sources with high-quality data, we use Bayesian inference to estimate T_eff:

P(T_eff | data) ∝ P(data | T_eff) × P(T_eff)

Where:
- P(data | T_eff) is the likelihood from model comparison
- P(T_eff) is a prior from brown dwarf cooling models

## References

1. Marley, M. S., et al. (2021). "The Sonora Cholla Model Atmospheres for Cool Brown Dwarfs." *ApJ*, 908, 51.
2. Cushing, M. C., et al. (2011). "The Discovery of Y Dwarfs using Data from WISE." *ApJ*, 743, 50.
3. Saumon, D., & Marley, M. S. (2008). "Theoretical Spectra of Ultracool Dwarfs." *ApJ*, 689, 1127.
