# Changelog

All notable changes to the TASNI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-23

### Added
- Initial public release of TASNI pipeline
- Complete cross-match pipeline for WISE/Gaia/LAMOST/eROSITA data
- Machine learning classification with ensemble methods (XGBoost, LightGBM, RandomForest)
- Bayesian population inference using PyMC
- Bootstrap confidence interval analysis
- Rigorous validation framework with k-fold cross-validation
- Periodogram analysis for NEOWISE light curves (Lomb-Scargle)
- Spectroscopy planning tools for Keck/VLT/JWST
- Publication-ready figure generation
- AASTeX manuscript (aastex701)

### Scientific Results
- Identified 3 confirmed fading thermal orphans:
  - J143046.35-025927.8: Teff=293 K, distance=17.4 +3.0/-2.6 pc (nearest room-temperature object)
  - J231029.40-060547.3: Teff=258 K, distance=32.6 +13.3/-8.0 pc
  - J193547.43+601201.5: Teff=251 K, no parallax
- J044024.40-731441.6 identified as LMC member (MSX LMC 1152), excluded from confirmed sample
- Apparent periodicities at 90-180 days acknowledged as likely NEOWISE cadence aliases
- 59/100 golden sample sources within eROSITA DR1 footprint (western Galactic hemisphere), all X-ray quiet
- 67 distance measurements with parallax > 5 mas; 44 with SNR > 3
- Golden sample of 100 high-priority thermal anomaly candidates from 747,634,026 WISE sources

### Changed (pre-submission hardening, 2026-02-16 through 2026-02-23)
- Fixed validation pipeline to select top 100 (was incorrectly selecting 150)
- Removed bogus est_parallax_mas column; merged real NEOWISE parallax data
- Fixed eROSITA coverage to correctly compute galactic hemisphere
- Fixed W1/W2 trend sign convention (FADING = positive) across all CSV files
- Added ATMO 2020 SED fitting analysis (Phillips et al. 2020)
- Added Monte Carlo background blend analysis ruling out blend contamination
- Added parallax injection-recovery validation with IRSA TAP
- Added ML ablation study documenting feature importance and information leakage
- Fixed all broken LaTeX cross-references (39 labels, 25 refs, 48 cite keys)
- Removed false MCMC posterior claim for J2310 from Table 3
- Added AI-assisted development disclosure subsection

### Infrastructure
- CLI interface using Typer
- Comprehensive test suite with pytest
- CI/CD pipeline with GitHub Actions (Python 3.11 + 3.12)
- Pre-commit hooks for code quality
- Full documentation suite
- NEOWISE mission decommissioned August 2024 (documented in manuscript)

### Data Products
- Golden targets catalog (100 sources)
- Parallax measurements (67 sources with > 5 mas)
- Variability metrics from NEOWISE
- ML-ranked candidates
- 25+ publication-ready figures
- CDS machine-readable table

## [0.1.0] - 2025-01-01

### Added
- Initial development version
- Basic WISE/Gaia cross-match pipeline
- Simple anomaly scoring algorithm
- Proof-of-concept results

---

## Future Roadmap

### [1.1.0] - Planned
- Spectroscopic follow-up integration
- JWST proposal tools
- Enhanced atmospheric modeling with Sonora Bobcat
- Citizen science interface

### [2.0.0] - Planned
- Multi-survey integration (Euclid, Roman)
- Distributed computing support
- Interactive web dashboard

---

[1.0.0]: https://github.com/dpalucki/tasni/releases/tag/v1.0.0
[0.1.0]: https://github.com/dpalucki/tasni/releases/tag/v0.1.0
