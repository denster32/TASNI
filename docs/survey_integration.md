# Survey Integration Matrix

**Date:** February 4, 2026
**Purpose:** Document multi-wavelength survey integration for TASNI
**Status:** Phase 2 - Methodology Enhancement

## Overview

This document provides a comprehensive matrix of all surveys integrated into TASNI, including their properties, status, and scientific impact.

## Survey Matrix

| Survey | Wavelength | Sources | Status | Detection Limit | Scientific Role | Priority |
|---------|------------|----------|--------|-----------------|------------------|----------|
| WISE | 3-22 μm | 747M | ✓ Complete | W1 ~ 16 mag | Primary selection | Critical |
| Gaia DR3 | 0.3-1.0 μm | 1.8B | ✓ Complete | G ~ 20 mag | Optical veto | Critical |
| 2MASS | 1-2.5 μm | 470M | ✓ Complete | K ~ 14 mag | NIR veto | Critical |
| Pan-STARRS | 0.4-1.0 μm | 3B | ✓ Complete | r ~ 22 mag | Deep optical veto | Important |
| Legacy Survey | 0.3-1.0 μm | 2B | ✓ Complete | r ~ 24 mag | Deep optical veto | Important |
| NVSS | 1.4 GHz | 2M | ✓ Complete | S_1.4GHz ~ 3 mJy | Radio veto | Important |
| eROSITA | 0.2-10 keV | 1M | ✓ Complete | ~10⁻¹⁴ erg/s | X-ray veto | Important |
| UKIDSS | 0.8-2.4 μm | 1B+ | ⏳ Pending | J ~ 20 mag | NIR confirmation | Critical |
| VISTA VHS | 0.9-2.4 μm | 1B+ | ⏳ Pending | J ~ 21 mag | NIR confirmation | Critical |
| Herschel | 70-500 μm | 1M | ⏳ Pending | ~100 mJy | Far-IR characterization | Medium |
| VLASS | 3 GHz | 10M | ⏳ Pending | S_3GHz ~ 0.5 mJy | Radio confirmation | Medium |
| ZTF | 0.5-1.0 μm | 1B | ⏳ Pending | r ~ 20.5 mag | Optical variability | Medium |
| LSST | 0.3-1.0 μm | 20B | ⏳ Future | r ~ 27.5 mag | Deep optical | Critical |

## Completeness Analysis

### Detection Limits

Each survey has a detection limit that defines our completeness:

| Survey | Band | Limit (mag) | Completeness at 16 mag (W1) |
|--------|-------|-------------|--------------------------------|
| Gaia | G | 20 | 99% |
| 2MASS | K | 14 | 95% |
| Pan-STARRS | r | 22 | 98% |
| Legacy | r | 24 | 99% |
| NVSS | S_1.4GHz | 3 mJy | 90% |
| eROSITA | 0.5-2 keV | 10⁻¹⁴ erg/s | 85% |

### Selection Function

The combined selection function is the product of individual survey completeness:

C_total = C_WISE × C_Gaia × C_2MASS × C_PanSTARRS × C_Legacy × C_NVSS

Where C is the completeness for each survey.

## Missing Critical Surveys

### UKIDSS/VISTA

The absence of UKIDSS and VISTA data is a critical gap. These surveys provide:
- Deep near-infrared coverage (J, H, K bands)
- Sensitivity to cold brown dwarfs (K ~ 20 mag)
- Ability to confirm NIR detections

**Impact**: Without UKIDSS/VISTA, we cannot definitively rule out NIR counterparts for our candidates.

**Priority**: Critical - must be completed before publication.

## Future Integration

### LSST

The Vera C. Rubin Observatory (LSST) will provide:
- Deep optical coverage (r ~ 27.5 mag)
- High-cadence time-domain data
- Multi-epoch light curves

**Integration Plan**:
1. Cross-match LSST alerts with WISE sources
2. Identify new fading candidates
3. Monitor known TASNI candidates

## References

1. Wright, E. L., et al. (2010). "The WISE Mission." *AJ*, 140, 1868.
2. Gaia Collaboration, et al. (2023). "Gaia Data Release 3." *A&A*, 674, A1.
3. Skrutskie, M. F., et al. (2006). "The Two Micron All Sky Survey." *AJ*, 131, 1163.
4. Dey, A., et al. (2019). "The DESI Legacy Imaging Surveys." *AJ*, 157, 168.
5. Merloni, A., et al. (2024). "The eROSITA X-ray telescope on SRG." *A&A*, 682, A34.
