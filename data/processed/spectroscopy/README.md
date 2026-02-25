# TASNI Spectroscopy Follow-up Materials

**Version:** 1.0.0
**Date:** 2026-02-15

This directory contains materials for spectroscopic follow-up of the four fading thermal orphans.

---

## Target Summary

| Target | T_eff (K) | Distance (pc) | W1 (mag) | W2 (mag) | Priority |
|--------|-----------|---------------|----------|----------|----------|
| J143046.35-025927.8 | 293 ± 47 | 17.4 +3.0/-2.6 | 14.0 | 10.6 | **1** |
| J044024.40-731441.6 | 466 ± 52 | 30.5 +1.3/-1.2 | 14.0 | 12.2 | 2 |
| J231029.40-060547.3 | 258 ± 38 | 32.6 +13.3/-8.0 | 14.0 | 12.2 | 2 |
| J193547.43+601201.5 | 251 ± 35 | --- | 13.6 | 12.0 | 2 |

---

## Facility Recommendations

### Primary: JWST NIRSpec

**Configuration:**
- Grating: G395H (2.9-5.3 μm, R~2700)
- Detector: NRS1+NRS2
- Slit: S200A1 (0.2" x 3.3")

**Exposure Times (S/N=10 per pixel):**

| Target | Integration Time | Total Time (incl. overhead) |
|--------|------------------|----------------------------|
| J143046.35-025927.8 | 15 min | 22 min |
| J044024.40-731441.6 | 45 min | 52 min |
| J231029.40-060547.3 | 30 min | 37 min |
| J193547.43+601201.5 | 20 min | 27 min |

**Total Request:** 2.3 hours

### Ground-based Alternatives

#### Keck/NIRES (0.95-2.45 μm, R~2700)
- Suitable for J1430 only (brightest target)
- Integration: 2 hours for S/N=5
- Visibility: Feb-Aug

#### VLT/KMOS (2.0-2.45 μm, R~4000)
- Suitable for northern targets
- Integration: 3 hours per target for S/N=5
- Visibility: Oct-Apr

---

## Visibility Windows

### J143046.35-025927.8 (RA=217.7°, Dec=-2.99°)
- **Keck:** Feb-Aug (best: Apr-Jun)
- **VLT:** Sep-Mar (best: Nov-Jan)
- **JWST:** Continuous (zodiacal constraints: Mar-Sep)

### J044024.40-731441.6 (RA=70.1°, Dec=-73.24°)
- **Keck:** Not visible (Dec < -60°)
- **VLT:** Jun-Dec (best: Aug-Oct)
- **JWST:** Continuous

### J231029.40-060547.3 (RA=347.6°, Dec=-6.10°)
- **Keck:** Jun-Dec (best: Aug-Oct)
- **VLT:** Nov-May (best: Jan-Mar)
- **JWST:** Continuous (zodiacal constraints: Jul-Jan)

### J193547.43+601201.5 (RA=293.9°, Dec=+60.2°)
- **Keck:** Mar-Sep (best: May-Jul)
- **VLT:** Not visible (Dec > +50°)
- **JWST:** Continuous (zodiacal constraints: May-Nov)

---

## Expected Spectral Features

For Y dwarfs at 250-500 K, expect:

| Wavelength (μm) | Species | Diagnostic |
|-----------------|---------|------------|
| 2.0-2.5 | H2O | Temperature |
| 3.3 | CH4 | Carbon chemistry |
| 4.0-4.5 | CO2 | Temperature |
| 4.6-5.0 | CO | Temperature gradient |
| 5.0+ | H2O, CH4 | Gravity |

---

## Finding Charts

Finding charts are available in `figures/`:
- `J1430_finding_chart.png`
- `J0440_finding_chart.png`
- `J0855_finding_chart.png`
- `J2248_finding_chart.png`

Each chart shows:
- 2x2 arcmin field
- WISE W1 image background
- Target marked with crosshair
- Nearby sources labeled
- Coordinates and epoch

---

## Phase-folded Light Curves

For timing observations to avoid minima:

| Target | Period (days) | Best Phase | Avoid Phase |
|--------|---------------|------------|-------------|
| J143046.35-025927.8 | 116.3 ± 5.0 | 0.25, 0.75 | 0.0, 0.5 |
| J231029.40-060547.3 | 178.6 ± 7.0 | 0.25, 0.75 | 0.0, 0.5 |
| J193547.43+601201.5 | 92.6 ± 4.0 | 0.25, 0.75 | 0.0, 0.5 |
| J044024.40-731441.6 | --- | --- | --- |

---

## Proposal Materials

JWST Cycle 4 template available in:
- `proposals/jwst_cycle4/`

Contains:
- Scientific justification (2 pages)
- Technical justification (1 page)
- Target list (machine-readable)
- Exposure time calculator outputs

---

## Contact

For spectroscopy coordination: paluckide@yahoo.com
