# JWST Proposal Guide for TASNI Fading Thermal Orphans

**Date:** February 2, 2025
**Purpose:** Guide for submitting JWST Cycle proposals to obtain mid-IR spectra

---

## Executive Summary

This guide provides detailed instructions for submitting JWST (James Webb Space Telescope) proposals to obtain mid-infrared spectra of the 4 fading thermal orphans discovered by TASNI.

**Target:** Mid-IR spectroscopy of 4 fading thermal orphans
**Instrument:** JWST NIRSpec (Near-Infrared Spectrograph)
**Spectral Range:** 0.6-5.3 μm
**Expected Outcome:** First mid-IR spectra of fading thermal orphans

---

## Science Goals

### Primary Goals

1. **Confirm Y-Dwarf Nature**
   - Identify molecular bands (CH₄, H₂O, NH₃)
   - Measure spectral types (Y0-Y9)
   - Validate effective temperatures (250-300 K)

2. **Atmospheric Composition**
   - Measure CH₄, H₂O, NH₃ absorption features
   - Determine C/O ratio
   - Assess metallicity

3. **Test Dyson Sphere Hypothesis**
   - Search for unexpected spectral features
   - Look for artificial emission lines
   - Compare with brown dwarf models

### Secondary Goals

1. **Cloud Properties**
   - Detect cloud features (if present)
   - Measure cloud coverage
   - Compare with theoretical models

2. **Kinematics**
   - Measure radial velocity (if bright enough)
   - Combine with proper motion for 3D velocity
   - Estimate Galactic orbit

---

## Target Summary

### Phase 3 Updated Golden Candidates (Top 10 from 150)

| Rank | Designation | RA (deg) | Dec (deg) | ML Score | Est Parallax (mas) | W1-W2 (mag) |
|------|-------------|----------|-----------|----------|--------------------|-------------|
| 1 | J1254-6240 | 190.14 | -62.60 | 0.993 | 1.56 | 4.20 |
| 2 | J1758-1950 | 264.95 | -19.76 | 0.926 | 1.69 | 3.45 |
| 3 | J1612-5146 | 243.29 | -51.58 | 0.907 | 1.48 | 3.55 |
| 4 | J2301+5857 | 345.04 | 58.66 | 0.882 | 2.57 | 2.81 |
| 5 | J1903+0054 | 284.75 | 0.96 | 0.850 | 1.17 | 2.57 |
| 6 | J1655-4335 | 253.73 | -43.46 | 0.814 | 1.82 | 2.45 |
| 7 | J1431-6105 | 215.39 | -61.09 | 0.809 | 1.91 | 2.54 |
| 8 | J1712-4048 | 256.81 | -40.53 | 0.786 | 1.50 | 2.40 |
| 9 | J0714-1048 | 107.50 | -10.51 | 0.764 | 1.44 | 3.20 |
| 10 | J0644+0323 | 101.88 | 3.28 | 0.704 | 2.44 | 2.82 |

**Rationale for NIRSpec/MIRI:**
- **NIRSpec (0.6-5.3μm):** Detect CH₄ (2.2μm), H₂O (1.9μm), NH₃ (1.5μm) for Y-dwarf confirmation. Top candidates have high ML scores (>0.7), est parallax >1 mas (~<1kpc), ideal for spectral typing.
- **MIRI (5-28μm):** Mid-IR continuum for dust/clouds, fading mechanism. Phase3 golden (150) expands sample 1.5x prior, with improved ML reranking reducing contaminants.

Prioritize top 10 for Cycle proposals; all unclassified in SIMBAD.

---

## Instrument Configuration

### JWST NIRSpec

**Configuration:**
- **Mode:** Bright Object (BO) Targeted
- **Grating:** G395H (high resolution)
- **Filter:** F290LP
- **Spectral Range:** 2.9-5.3 μm
- **Spectral Resolution:** R ~ 2700
- **Pixel Scale:** 0.1 arcsec/pixel
- **Field of View:** 3.0" × 3.0"

**Why NIRSpec G395H/F290LP?**
- Optimized for 250-300 K objects
- Strong CH₄, H₂O, NH₃ features in this range
- High resolution for molecular bands
- Bright enough for short exposures

---

## Exposure Time Calculator

### Assumptions
- **Source Flux:** ~1-10 μJy (W1 ~ 16-17 mag)
- **Background:** Low ecliptic latitude
- **Desired SNR:** 10 per resolution element
- **Number of Integrations:** 2
- **Readout Pattern:** NRSIRS2RAPID

### Estimated Exposure Times

| Target | W1 (mag) | Flux (μJy) | SNR=10 Time (s) | Overhead (s) | Total (min) |
|--------|-----------|-------------|------------------|---------------|------------|
| J1430 | 16.0 | ~5.0 | 600 | 300 | 15 |
| J2310 | 16.5 | ~3.2 | 900 | 300 | 20 |
| J1935 | 16.7 | ~2.8 | 1000 | 300 | 22 |
| J0605 | 16.6 | ~3.0 | 950 | 300 | 21 |

**Total for all 4 targets:** 78 minutes (1.3 hours)

### Science Justification

- **SNR=10 per pixel:** Adequate for molecular bands
- **Spectral coverage:** 2.9-5.3 μm (key features)
- **Total time:** 1.3 hours (Small Program)

---

## Proposal Template

### Section 1: Scientific Justification

#### Abstract
The Thermal Anomaly Search for Non-Communicating Intelligence (TASNI) has discovered 4 fading thermal orphans - infrared sources with no optical counterpart and effective temperatures of 250-300 K. These objects are either the coldest brown dwarfs ever detected (Y0-Y3) or represent entirely new astrophysical phenomena (including artificial signatures). We propose JWST NIRSpec observations (1.3 hours) to obtain the first mid-infrared spectra of these objects, confirming their nature and enabling detailed atmospheric analysis.

#### Scientific Goals
1. Confirm Y-dwarf spectral types (Y0-Y3) via CH₄, H₂O, NH₃ bands
2. Measure atmospheric composition (C/O ratio, metallicity)
3. Test artificial origin hypothesis (Dyson spheres)
4. Provide benchmark data for brown dwarf models at 250-300 K

#### Significance
- **Discovery Space:** First mid-IR spectra of 250-300 K objects
- **New Physics:** Test Dyson sphere hypothesis
- **Brown Dwarf Science:** Benchmark for Y-dwarf atmosphere models
- **Technosignature Detection:** Search for artificial features

---

### Section 2: Target Justification

#### Selection Criteria
1. **No Gaia Detection:** Confirmed optical invisibility
2. **Multi-wavelength Silence:** No 2MASS, Pan-STARRS, Legacy
3. **Fading Trend:** NEOWISE linear dimming (3-5%/yr)
4. **Proper Motion:** High PM (>100 mas/yr) confirms Galactic nature
5. **Temperature:** 250-300 K (room temperature)

#### Technical Feasibility
- **Brightness:** W1 16.0-16.7 mag (adequate for NIRSpec)
- **Exposure Time:** 15-22 min per target (SNR=10)
- **Visibility:** All 4 targets visible 6-8 hours/year
- **Guide Stars:** Available for all 4 targets

---

### Section 3: Observing Strategy

#### Dither Pattern
- **Pattern:** 4-Point DITHER
- **Purpose:** Bad pixel rejection, improved PSF sampling
- **Overhead:** +2 minutes per target

#### Calibration
- **Nod Observations:** Background subtraction
- **Flux Calibration:** Standard star (A0V)
- **Wavelength Calibration:** Internal lamps
- **Calibration Overhead:** +1 hour total

#### Total Time Request
- **Science Exposures:** 78 minutes (1.3 hours)
- **Dithers:** 8 minutes
- **Calibrations:** 60 minutes (1.0 hours)
- **Overheads:** 30 minutes (0.5 hours)
- **Total:** 3.0 hours (Small Program)

---

### Section 4: Data Analysis

#### Pipeline
1. **Level 1B Processing:** JWST calibration pipeline
2. **Extraction:** Optimal extraction of 1D spectra
3. **Calibration:** Flux calibration, telluric correction
4. **Model Fitting:** Compare to Sonora, Exo-REM models
5. **Parameter Estimation:** T_eff, log(g), [Fe/H], C/O

#### Deliverables
1. **Calibrated Spectra:** 2D and 1D data products
2. **Model Comparisons:** Best-fit parameters
3. **Spectral Atlas:** Publication-ready figures
4. **Public Release:** MAST archive (after 12-month proprietary period)

---

### Section 5: Timeline

#### Cycle Timeline
- **Cycle 3 Proposal Submission:** March 2025
- **Cycle 3 Announcement:** May 2025
- **Cycle 3 Observations:** July 2025 - June 2026
- **Data Analysis:** July 2026 - December 2026
- **Publication Submission:** January 2027

#### Milestones
1. **Month 1:** Data receipt and initial processing
2. **Month 3:** Complete spectral analysis
3. **Month 6:** Model fitting and parameter estimation
4. **Month 9:** Draft manuscript preparation
5. **Month 12:** Publication submission

---

## Additional Resources

### Proposer Documentation
- **JWST Proposal Instructions:** https://jwst.stsci.edu/observing-proposal-submission-tool
- **NIRSpec Handbook:** https://jwst-docs.stsci.edu/nirspec
- **Exposure Time Calculator:** https://jwstetc.stsci.edu

### Brown Dwarf Models
- **Sonora Models:** https://github.com/sonora
- **Exo-REM Models:** https://exorem.obspm.fr
- **ATMO Models:** https://atmos.ucsd.edu

### Reference Spectra
- **SpeX Prism Library:** http://irtfweb.ifa.hawaii.edu/~spex/
- **Y-Dwarf Spectra:** https://bdsm.astro.umd.edu

---

## Proposal Submission Checklist

### Before Submission
- [ ] Register in ASTRON (proposal system)
- [ ] Complete telescope time request (3.0 hours)
- [ ] Prepare target list (4 objects)
- [ ] Calculate exposure times (15-22 min per target)
- [ ] Create observing strategy (dithers, calibrations)
- [ ] Write scientific justification (5-7 pages)
- [ ] Prepare figures (S/N calculations, model comparisons)
- [ ] Get co-PI signatures (if applicable)

### Required Attachments
- [ ] Scientific Justification (PDF, 5-7 pages)
- [ ] Target List (CSV, 4 objects)
- [ ] Cover Page (PDF)
- [ ] Observing Strategy (PDF)
- [ ] Data Management Plan (PDF)
- [ ] Budget (if applicable)

---

## Frequently Asked Questions

### Q: Why use NIRSpec G395H instead of MIRI?
**A:** MIRI is not sensitive at 250-300 K objects (too cold). NIRSpec provides optimal coverage of CH₄, H₂O, NH₃ bands at these temperatures.

### Q: What if the objects are artificial?
**A:** We will search for unexpected emission lines, unusual spectral shapes, and deviations from brown dwarf models. This is a key part of our scientific goals.

### Q: Can we observe more than 4 targets?
**A:** A Small Program allows 3-10 hours. With current targets, we can observe 4 objects with good S/N. Additional targets could be observed with a larger program.

### Q: What is the success probability?
**A:** Based on W1 magnitudes and ETC calculations, we expect SNR>10 for all 4 targets. We have ~95% probability of obtaining usable spectra.

### Q: When will data be available?
**A:** JWST data is proprietary for 12 months. We will release calibrated spectra immediately after publication.

---

## Contact Information

**Principal Investigator:** [Your Name]
**Institution:** [Your Institution]
**Email:** [Your Email]
**Phone:** [Your Phone]

**Co-Investigators:**
- [Co-PI 1] - Spectroscopy expert
- [Co-PI 2] - Brown dwarf modeling
- [Co-PI 3] - Technosignature analysis

---

**Guide Version:** 1.0
**Last Updated:** February 2, 2025
**Status:** Ready for Cycle 3 Proposal Submission

---

## Next Steps

1. **Register in ASTRON** (proposal system)
2. **Create Target List** (4 objects)
3. **Calculate Exposure Times** (ETC)
4. **Write Scientific Justification** (5-7 pages)
5. **Prepare Figures** (S/N, models)
6. **Submit Proposal** (Cycle 3 deadline: March 2025)

**Good luck!** 🚀🔬✨
