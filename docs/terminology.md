# TASNI Terminology Glossary

**Date:** February 4, 2026
**Purpose:** Ensure consistent usage of technical terms across all TASNI documentation
**Status:** Phase 6 - Clarity and Precision Enhancement

## Overview

This glossary defines the standard terminology used in the TASNI project. Consistent terminology is essential for clear communication and to avoid confusion, especially when discussing technical concepts.

## Key Terms

### Source Classification

| Term | Definition | Notes |
|------|------------|-------|
| **Anomalous source** | A source that deviates from expected properties of known stellar or extragalactic objects | Primary target of TASNI search |
| **Thermal orphan** | An anomalous source with no known parent object | Term used in FINAL_REPORT |
| **Fading thermal orphan** | A thermal orphan showing statistically significant flux decrease over time | Four sources identified in golden sample |
| **Golden sample** | The set of 100 highest-ranked candidates from the ML pipeline | Also called "golden targets" in some documents |
| **Golden targets** | Same as golden sample | Use "golden sample" consistently |
| **Tier 5** | The final set of 810,000 sources passing all pipeline filters | Also called "tier5" or "T5" |
| **Tier 5 candidate** | Any source in the tier 5 catalog | Use "tier5 source" consistently |
| **Y dwarf candidate** | A source with estimated temperature < 500 K | Use when temperature estimate is available |

### Pipeline Stages

| Term | Definition | Notes |
|------|------------|-------|
| **Tier 1** | Initial WISE catalog (747M sources) | AllWISE catalog |
| **Tier 2** | After Gaia cross-match (6.1M sources) | Gaia DR3 |
| **Tier 3** | After secondary catalog cross-match (1.2M sources) | 2MASS, Pan-STARRS, Legacy, NVSS, eROSITA, LAMOST |
| **Tier 4** | After quality filters (810K sources) | Signal-to-noise, artifact flags |
| **Tier 5** | Final candidate catalog (810K sources) | After all filters |

### Photometric Properties

| Term | Definition | Notes |
|------|------------|-------|
| **W1-W2 color** | Magnitude difference between WISE bands 1 (3.4 µm) and 2 (4.6 µm) | Primary selection criterion |
| **W2-W3 color** | Magnitude difference between WISE bands 2 (4.6 µm) and 3 (12 µm) | Secondary selection criterion |
| **W1 magnitude** | Magnitude in WISE band 1 (3.4 µm) | Brightness metric |
| **W2 magnitude** | Magnitude in WISE band 2 (4.6 µm) | Brightness metric |
| **Effective temperature (T_eff)** | Temperature of the photosphere | Estimated from W1-W2 color |
| **Luminosity** | Total energy output | Calculated from flux and distance |

### Kinematic Properties

| Term | Definition | Notes |
|------|------------|-------|
| **Proper motion (µ)** | Angular motion across the sky | Measured in mas/yr |
| **Proper motion vector** | Direction and magnitude of proper motion | Has components µ_α cos δ and µ_δ |
| **Parallax (π)** | Apparent shift due to Earth's orbit | Measured in mas |
| **Distance (d)** | Distance from Earth | Calculated as d = 1/π (in parsecs) |
| **Tangential velocity (v_tan)** | Velocity perpendicular to line of sight | v_tan = 4.74 µ d (in km/s) |

### Variability

| Term | Definition | Notes |
|------|------------|-------|
| **NEOWISE epoch** | A single observation period (6 months) | NEOWISE has 15+ epochs |
| **Light curve** | Flux vs. time plot | Shows variability over time |
| **Periodogram** | Power spectrum of light curve | Used to identify periodic signals |
| **Chi-squared (χ²)** | Goodness-of-fit statistic | Used to quantify variability |
| **False Alarm Probability (FAP)** | Probability that a signal is due to noise | Used to assess periodogram significance |
| **Variability class** | Classification based on χ² threshold | NORMAL, VARIABLE, FADING |

### Statistical Terms

| Term | Definition | Notes |
|------|------------|-------|
| **P-value** | Probability of observing data under null hypothesis | Use with confidence intervals |
| **Confidence interval (CI)** | Range containing true value with specified probability | Typically 95% CI |
| **Standard deviation (σ)** | Measure of dispersion | Use with mean values |
| **Standard error (SE)** | Uncertainty in estimated parameter | SE = σ/√n |
| **Bootstrap** | Resampling method for uncertainty estimation | Used for non-parametric statistics |
| **Monte Carlo simulation** | Random sampling method | Used for periodogram significance testing |
| **Kolmogorov-Smirnov (KS) test** | Non-parametric test for distribution differences | Used for population comparisons |

### Machine Learning

| Term | Definition | Notes |
|------|------------|-------|
| **Feature** | Input variable for ML model | Also called "attribute" |
| **Label** | Target variable for supervised learning | Also called "class" |
| **Training set** | Data used to train ML model | Must be independent of test set |
| **Test set** | Data used to evaluate ML model | Must be independent of training set |
| **Hold-out validation** | Validation on separate test set | Prevents overfitting |
| **Feature importance** | Relative contribution of each feature | Used for interpretability |
| **Composite score** | Weighted combination of model outputs | Used for ranking candidates |

### Survey Names

| Term | Full Name | Notes |
|------|-----------|-------|
| **WISE** | Wide-field Infrared Survey Explorer | Primary survey |
| **NEOWISE** | Near-Earth Object WISE | Reactivation of WISE |
| **AllWISE** | Combined WISE data from all epochs | Used as tier 1 catalog |
| **Gaia DR3** | Gaia Data Release 3 | Astrometric catalog |
| **2MASS** | Two Micron All Sky Survey | Near-infrared catalog |
| **Pan-STARRS** | Panoramic Survey Telescope and Rapid Response System | Optical catalog |
| **Legacy Survey** | Legacy Surveys of the Sloan Digital Sky Survey | Optical catalog |
| **NVSS** | NRAO VLA Sky Survey | Radio catalog |
| **eROSITA** | Extended ROentgen Survey with an Imaging Telescope Array | X-ray catalog |
| **LAMOST** | Large Sky Area Multi-Object Fiber Spectroscopic Telescope | Spectroscopic catalog |

### Model Names

| Term | Full Name | Notes |
|------|-----------|-------|
| **Sonora Cholla** | Atmospheric model for cool substellar objects | Primary model used |
| **Drift Phoenix** | Alternative atmospheric model | Secondary model |
| **Baraffe** | Evolutionary model for brown dwarfs | Used for age estimates |
| **Saumon** | Evolutionary model for brown dwarfs | Used for age estimates |

### Units

| Term | Symbol | Definition | Notes |
|------|--------|------------|-------|
| **Magnitude** | mag | Logarithmic brightness scale | Lower is brighter |
| **Milliarcsecond** | mas | 1/1000 arcsecond | Used for proper motion and parallax |
| **Parsec** | pc | Distance unit (3.26 light-years) | Used for astronomical distances |
| **Kelvin** | K | Temperature unit | Used for effective temperature |
| **Solar luminosity** | L☉ | Luminosity of the Sun | Used for comparison |
| **Solar mass** | M☉ | Mass of the Sun | Used for comparison |
| **Jansky** | Jy | Flux density unit (10⁻²⁶ W/m²/Hz) | Used for radio flux |

## Usage Guidelines

### Preferred Terms

- Use "golden sample" instead of "golden targets"
- Use "tier5 source" instead of "tier5 candidate"
- Use "fading thermal orphan" instead of "fading orphan"
- Use "NEOWISE epoch" instead of "NEOWISE observation"
- Use "W1-W2 color" instead of "W1-W2" alone

### Avoid These Terms

- "Fading orphan" → Use "fading thermal orphan"
- "Golden targets" → Use "golden sample"
- "Tier5 candidate" → Use "tier5 source"
- "Anomalous" alone → Use "anomalous source"
- "Variability" alone → Use "variability classification" or "variability analysis"

## Cross-Reference

See also:
- [`docs/HYPOTHESES.md`](HYPOTHESES.md) - Testable hypotheses
- [`docs/RESEARCH_SCOPE.md`](RESEARCH_SCOPE.md) - Research scope
- [`docs/STATISTICAL_METHODS.md`](STATISTICAL_METHODS.md) - Statistical methods

---

**Document Version:** 1.0
**Last Updated:** February 4, 2026
**Status:** Ready for Review
