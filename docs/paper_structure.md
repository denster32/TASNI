# TASNI Paper Structure

**Date:** February 4, 2026
**Purpose:** Define new paper structure separating observational results from physical interpretation
**Status:** Phase 5 - Logical Flow Restructuring

## Overview

This document defines the new structure for the TASNI paper, addressing the logical flow issue identified in the comprehensive analysis by separating observational results from physical interpretation.

## New Structure

### 1. Introduction

- Motivation: Anomalous sources and scientific context
- Literature review: Y dwarfs, variability, selection methods
- Research question: Explicit, testable hypotheses
- Paper organization: Overview of sections

### 2. Data and Methods

#### 2.1 Data Sources

- WISE/NEOWISE (747M sources)
- Gaia DR3 (1.8B sources)
- Secondary catalogs: 2MASS, Pan-STARRS, Legacy, NVSS, eROSITA, LAMOST
- Reference formats and versions

#### 2.2 Selection Criteria

- Theoretical justification for W1-W2 threshold
- Multi-wavelength veto strategy
- Sensitivity analysis
- Selection function and completeness

#### 2.3 Pipeline Implementation

- Crossmatching methodology
- GPU-accelerated processing
- Filtering stages (Tiers 1-5)
- Quality control and validation

#### 2.4 Parallax Fitting

- Five-parameter linear model
- Error propagation
- Validation against Gaia DR3
- Quality control criteria

#### 2.5 Temperature Estimation

- W1-W2 color-temperature relation
- Calibration to Sonora Cholla models
- Uncertainty estimation
- Alternative methods (SED fitting, Bayesian inference)

#### 2.6 Variability Analysis

- NEOWISE time-series processing
- Statistical significance testing (p-values, Monte Carlo)
- Periodogram analysis (FAP calculations)
- Quantitative classification criteria

#### 2.7 Statistical Methods

- P-value calculations
- Confidence intervals
- Bootstrap methods
- Selection function analysis
- Space density calculations

### 3. Observational Results

#### 3.1 Pipeline Source Counts

- Table showing source counts at each stage
- Reduction statistics
- Completeness estimates

#### 3.2 Golden Sample Properties

- Photometric properties
- Kinematic properties
- Variability classification
- Statistical summaries with uncertainties

#### 3.3 Fading Thermal Orphans

- Four fading sources with properties
- Light curves
- Periodograms
- Statistical significance

#### 3.4 Parallax Measurements

- 58 sources with significant parallaxes
- Distance distribution
- Validation against Gaia DR3

### 4. Physical Interpretation

#### 4.1 Brown Dwarf Comparison

- Comparison to known Y dwarfs
- Color-color diagrams
- Temperature distribution comparison
- Luminosity-luminosity relation

#### 4.2 Population Analysis

- Space density with selection function corrections
- Comparison to theoretical models
- Implications for substellar mass function

#### 4.3 Fading Mechanisms

- Secular cooling vs. rotational modulation
- Atmospheric variability vs. eclipsing binaries
- Assessment of each mechanism

#### 4.4 Alternative Interpretations

- Young brown dwarfs
- Unresolved binaries
- Instrumental artifacts (ruled out)

### 5. Discussion

#### 5.1 Y Dwarf Context

- Place of fading orphans in Y dwarf sequence
- Comparison to Sonora Cholla models
- Spectroscopic confirmation needed

#### 5.2 Limitations

- Selection biases
- Missing surveys (UKIDSS/VISTA)
- Statistical uncertainties
- Small sample size

#### 5.3 Future Work

- Spectroscopic follow-up requirements
- JWST observations
- Extended search of AllWISE catalog

### 6. Conclusions

- Summary of key findings
- Implications for brown dwarf science
- Recommendations for future work

### 7. Acknowledgments

- Data acknowledgments
- Software acknowledgments

### 8. References

- Complete bibliography

---

## Separation of Results and Interpretation

The key principle is to maintain a clear distinction between:

1. **Observational results** (Section 3): What we measured, with uncertainties
2. **Physical interpretation** (Section 4): What the results mean scientifically

This separation prevents overinterpretation and makes the paper logically coherent.

---

**Document Version:** 1.0
**Last Updated:** February 4, 2026
**Status:** Ready for Review
