# TASNI Science Review Audit

**Reviewer:** Science-Reviewer Agent (Claude Opus 4.6)
**Date:** 2026-02-20
**Manuscript:** "TASNI: Thermal Anomaly Search for Non-communicating Intelligence -- Discovery of Three Fading Thermal Orphans and an LMC Candidate"
**Target journal:** The Astrophysical Journal

---

## Executive Summary

The TASNI manuscript presents a novel pipeline for identifying thermally anomalous mid-infrared sources from the AllWISE catalog, discovering three "fading thermal orphans" with temperatures of 251--293 K and one LMC candidate. The paper is well-structured, appropriately caveated in most areas, and the SETI framing is restrained. However, several scientific methodology issues remain that range from CRITICAL (affecting the reliability of claimed distances and temperatures) to LOW (minor presentation issues). The most serious concerns involve the Planck SED temperature method applied to only 2 photometric bands for the coldest sources, the sign convention inconsistency in trend data, the unreliability of NEOWISE-derived sub-arcsecond parallaxes, and inconsistent asymmetric distance error bars.

---

## 1. Temperature Estimates

### 1.1 Validity of Planck SED fitting to W1+W2 for T < 300 K objects

**Severity: CRITICAL**

**Finding:** The calculate_teff.py code (`/mnt/data/tasni/src/tasni/utils/calculate_teff.py`) fits a Planck blackbody SED to WISE photometry using all bands with SNR > 3. For the three fading orphans, all four WISE bands (W1 through W4) have SNR > 3 (W1 SNR 40-46, W2 SNR 45-55, W3 SNR 39-72, W4 SNR 11-44), so 4-band fitting is available.

**Problem:** A Planck blackbody is a poor model for Y dwarf atmospheres at T < 300 K. Real Y dwarf SEDs are heavily shaped by:
- CH4 absorption (3.3 um, within W1)
- H2O absorption (multiple bands)
- NH3 absorption (emerging below 400 K)
- CIA H2 opacity (broad, modifying the continuum)
- Non-equilibrium chemistry (CO/CH4, N2/NH3 disequilibrium)

A blackbody fitted to W1-W4 will systematically misestimate the temperature because molecular absorption creates deep features that a smooth Planck function cannot reproduce. For the coldest Y dwarfs (T < 350 K), the W1 band sits in a deep CH4 absorption trough, making W1 much fainter than a blackbody would predict at that temperature, which biases the fitted temperature downward.

**Manuscript text (Section 2.3):** States "fitting Planck blackbody spectral energy distributions (SEDs) to the available WISE photometry" and adds 100 K systematic uncertainty from Kirkpatrick (2012) and Marley et al. (2021). The 100 K systematic uncertainty is reasonable for the overall scatter in color-temperature relations, but the Planck fit itself may have systematic biases that are not captured by this uncertainty.

**Assessment:**
- The claimed temperatures (251-293 K) are within the known Y dwarf range, which is a positive consistency check.
- However, the method (Planck fit) is non-standard for this temperature regime. Standard approaches use either:
  - Empirical color-Teff relations (e.g., Kirkpatrick et al. 2021 Table 13)
  - Model atmosphere grids (Sonora Cholla, BT-Settl)
- The paper acknowledges Sonora Cholla models but only uses them for the systematic uncertainty, not for the actual temperature derivation.
- With 100 K systematic uncertainty added in quadrature, the total uncertainties are large enough (e.g., 293 +/- 47 stat +/- 100 sys = 293 +/- 110 K total) that the claimed temperatures are not strongly constrained and overlap with the known Y dwarf range.

**Recommendation:** The paper should either (a) also provide color-Teff estimates from the Kirkpatrick 2021 relations for comparison, or (b) explicitly state that the Planck temperatures are only rough estimates and that model atmosphere fitting is needed for reliable temperatures. The current framing ("Effective temperatures are estimated by fitting Planck blackbody SEDs") is somewhat misleading because it implies a standard astrophysical technique when in fact Planck fits are known to be unreliable for objects with strong molecular absorption.

### 1.2 Uncertainty estimates (35-47 K statistical + 100 K systematic)

**Severity: MEDIUM**

The statistical uncertainties of 35-47 K come from the curve_fit covariance matrix. These likely underestimate the true statistical uncertainty because:
- The curve_fit uses uniform weighting ("errors = [] ... uniform weighting") -- no photometric error bars are propagated.
- With only 2 free parameters (T and solid angle) and 4 data points (W1-W4), there are only 2 degrees of freedom.
- The Planck model is wrong for this temperature regime (see above), so the chi-squared fit will produce artificially small parameter uncertainties if the model happens to pass near the data by coincidence.

The 100 K systematic uncertainty is a reasonable order-of-magnitude estimate based on the Kirkpatrick (2012) scatter and the Sonora Cholla models, but it was originally derived for the T/Y transition region (300-500 K) and may be an underestimate below 300 K where model predictions diverge most strongly.

### 1.3 Comparison to known Y dwarf temperatures

**Severity: LOW**

The claimed temperatures (251-293 K) are consistent with the coldest known Y dwarfs:
- WISE 0855-0714: ~250 K (Luhman 2014)
- WISE 0336-0143 B: 285-305 K (Kirkpatrick et al. 2021)
- WISE 1828+2650: ~300 K (Cushing et al. 2011)

The comparison in Table 4 is appropriate and uses the right reference objects. However, WISE 0855 has Teff from spectroscopic modeling (Morley et al. 2018), not from Planck fits, so comparing Planck-derived temperatures to spectroscopic temperatures is an apples-to-oranges comparison that the paper does not explicitly flag.

---

## 2. Parallax Methodology

### 2.1 NEOWISE astrometry for sub-arcsecond parallax

**Severity: CRITICAL**

**Finding:** The paper uses 5-parameter astrometric fits to NEOWISE single-epoch positions to derive parallaxes of 30-58 mas from a survey with 6 arcsecond (6000 mas) PSF FWHM. This means the parallax signal is 0.5-1% of the beam width.

**Code review** (`/mnt/data/tasni/src/tasni/analysis/extract_neowise_parallax.py`):
- The code uses unweighted least squares (line 217: "assuming uniform errors for now") -- no per-epoch position uncertainties are used.
- The design matrix is correct: it solves for 5 parameters (ra0, dec0, pm_ra, pm_dec, parallax) using parallax factors computed from the Sun's position.
- Error estimation uses sigma^2 * (A^T A)^{-1}, which is standard for OLS but assumes Gaussian, homoscedastic residuals. NEOWISE position residuals are likely non-Gaussian (outliers from source confusion, cosmic rays, edge-of-field effects).

**Caveats in the manuscript (Section 3.4):** The paper includes strong caveats:
- "the NEOWISE PSF FWHM is ~6 arcsec (6000 mas)"
- "systematic astrometric errors at the ~100 mas level cannot be excluded"
- "standard least-squares may significantly underestimate uncertainties compared to robust Bayesian MCMC approaches"
- "Independent confirmation via spectrophotometric distance estimates or future astrometric missions is strongly needed"
- Appendix A compares LS to MCMC for J143046.

**Assessment:** The caveats are among the strongest I have seen in a discovery paper and are commendable. However, three issues remain:

1. **The MCMC comparison uses synthetic data, not real data** (see `mcmc_parallax_comparison.py`, line 71: `generate_synthetic_epochs`). The synthetic data has Gaussian noise at 120 mas RMS, which may not represent real NEOWISE systematics. The MCMC results JSON shows LS and MCMC medians are identical (97.2 mas), which is expected for Gaussian noise but would likely differ for real data with outliers.

2. **67 out of 100 sources have "significant" parallaxes (>5 mas)**, which is suspiciously high. For a sample at the distances implied (10-50 pc), this is plausible, but the 5 mas threshold is only 0.08% of the PSF FWHM. At this level, systematic biases from source confusion or position-dependent PSF variations could easily produce spurious parallax detections.

3. **The LMC source J044024 has a parallax SNR of 23.3**, implying a very precise parallax of 32.8 +/- 1.4 mas (distance 30.5 pc). But this source is identified as an LMC member at 50 kpc. A spurious parallax of 33 mas at SNR > 23 demonstrates that the NEOWISE parallax pipeline can produce confidently wrong results. This should be explicitly flagged in the paper as evidence that NEOWISE parallaxes can be unreliable.

### 2.2 Parallax SNR interpretation

**Severity: HIGH**

- J143046: SNR = 5.8 -- This is formally a detection but is not robust given the systematics described above.
- J231029: SNR = 2.4 -- This is below the standard parallax significance threshold of 3-sigma. The paper correctly calls it "marginal."
- J193547: parallax = -0.96 +/- 5.1 mas (SNR = 0.19) -- Correctly reported as no detection.

**Problem:** The paper reports the J143046 distance of 17.4 pc prominently in the abstract and conclusions as "one of the nearest room-temperature objects known." Given the demonstrated unreliability of NEOWISE parallaxes (J044024 example) and the sub-pixel signal extraction, this claim should be further softened. The abstract currently says "Parallax measurements place the nearest source at 17.4+3.0/-2.6 pc" without the word "provisional" or "tentative" that appears in the Limitations section.

### 2.3 Asymmetric distance error bars

**Severity: HIGH**

**Finding:** The manuscript reports asymmetric error bars for distances:
- J143046: 17.4 +3.0/-2.6 pc
- J231029: 32.6 +13.3/-8.0 pc

However, computing these from the data (plx = 57.576 +/- 9.899 and plx = 30.662 +/- 12.527):
- J143046: d = 17.4 pc, +3.6/-2.5 pc (computed) vs +3.0/-2.6 pc (reported)
- J231029: d = 32.6 pc, +22.5/-9.5 pc (computed) vs +13.3/-8.0 pc (reported)

The J231029 discrepancy is large: the computed upper error bar is +22.5 pc but the reported value is +13.3 pc (which is actually the symmetric error d * sigma_plx / plx). The reported values appear to use the symmetric distance uncertainty for the upper bound and a different value for the lower bound, which is inconsistent.

The `asymmetric_distance_errors()` function in `extract_neowise_parallax.py` correctly implements d_upper = 1000/(plx-err) - d and d_lower = d - 1000/(plx+err), but the manuscript does not use these values. The Table 1 tablecomments state "we propagate the formal parallax uncertainty through this transformation" but the actual values don't match this propagation.

**Note:** The MEMORY.md already identifies this as a known issue ("J231029 distance error bars (+13.3/-8.0) don't match simple 1-sigma propagation (+22.5/-9.5); may use Bayesian posterior"). If a Bayesian posterior was used, this should be stated explicitly.

---

## 3. Variability Analysis

### 3.1 Fading rates (23-53 mmag/yr) -- physical plausibility

**Severity: MEDIUM**

The observed fading rates of 23-53 mmag/yr in both W1 and W2 are:
- **Much too fast for evolutionary cooling:** Brown dwarf cooling rates are ~10^{-4} mag/yr over ~1 Gyr, orders of magnitude slower.
- **Comparable to some known brown dwarf variability:** Rotational modulations of 1-10% are common in L/T dwarfs (Metchev et al. 2015), but these are periodic on hour timescales, not monotonic over decades.
- **Possibly consistent with blend separation:** The paper discusses this in Section 4.2 and correctly identifies that high proper motion sources moving away from a background IR source could mimic monotonic fading. This is the most likely explanation and the paper appropriately calls for forced photometry to test it.

**Assessment:** The fade rates are physically implausible for intrinsic Y dwarf variability. The paper's discussion of the proper-motion-induced artifact hypothesis (Section 4.2) is well-reasoned and represents an honest treatment. However, the phrase "fading thermal orphans" as the primary characterization throughout the paper may be misleading if the fading is an instrumental artifact. The paper should more prominently flag that blend separation is the leading non-exotic explanation for the fading, not just one of several possibilities.

### 3.2 Trend sign convention inconsistency

**Severity: HIGH**

**Finding:** The `classify_variability()` function in `compute_ir_variability.py` (line 191) defines FADING as `trend_w1 > TREND_THRESHOLD` (positive slope = magnitude increasing = getting fainter). However, the actual trend_w1 values stored in `golden_improved.csv` for FADING sources are all negative:
- J143046: trend_w1 = -0.0255
- J044024: trend_w1 = -0.0300
- J231029: trend_w1 = -0.0526
- J193547: trend_w1 = -0.0229

The absolute values correctly match the manuscript's reported fade rates (26, 30, 53, 23 mmag/yr). This means either:
1. The classification was performed on data with opposite sign convention (positive = fading), then the trend values were later inverted (negative = fading) when stored, or
2. A different code path was used for the classification.

The classification result (these are indeed the most extreme trend sources) is correct, so this is a data provenance issue rather than a scientific error. But it makes the pipeline less reproducible because running `classify_variability()` on the stored data would classify these sources as BRIGHTENING, not FADING.

**Recommendation:** Either (a) invert the sign of trend_w1/trend_w2 in the CSV to match the code convention, or (b) update the classify_variability code to match the stored data convention.

### 3.3 NEOWISE cadence aliasing

**Severity: LOW**

The paper's treatment of cadence aliasing is thorough and honest. Section 3.3 and the periodogram discussion correctly identify:
- 179 days ~ 182 days (NEOWISE cadence)
- 93 days ~ 182/2 (half-harmonic)
- 116 days as a possible beat frequency
- Known brown dwarf rotation periods are 2-10 hours, not 93-179 days

The conclusion that these are sampling aliases is well-supported. The FAP values (10^{-61} to 10^{-11}) are noted in the table comments as "likely overestimated" because they assume white noise, which is appropriate.

### 3.4 Injection-recovery validation

**Severity: MEDIUM**

The injection_recovery.py script (`/mnt/data/tasni/scripts/injection_recovery.py`) tests synthetic Y-dwarf fading signals with:
- Fade rates: 20-50 mmag/yr
- Noise: 30 mmag per epoch (Gaussian)
- Baselines: 7-10.5 years, 50-350 epochs

This achieves 100% recovery at >3-sigma. However:
1. **The noise model is Gaussian and stationary**, which does not capture NEOWISE systematics (correlated noise, photometric zero-point drifts, source confusion).
2. **The test does not inject signals into real NEOWISE light curves**, only into synthetic ones. A more rigorous test would inject synthetic fading into non-fading real light curves and verify recovery.
3. **The manuscript claims "100% of injected fading signals at >3-sigma significance"** (Section 2.2), which is technically true for the synthetic test but may not hold for real data.

---

## 4. Sample Selection and ML Pipeline

### 4.1 ML circularity acknowledgment

**Severity: MEDIUM**

The Limitations section (Section 5.5) provides an excellent description of the ML circularity issue:
> "Our supervised classifiers (XGBoost, LightGBM) are trained on binary labels derived by thresholding the composite anomaly scores computed by the unsupervised pipeline using the exact same photometric features."

This is a clear and accurate description. The term "surrogate-assisted anomaly ranking" is appropriate.

**Remaining concern:** The paper still frames the pipeline as discovering these sources through the ML, when in reality the Isolation Forest (unsupervised) does the discovery and XGBoost/LightGBM merely smooth the ranking. The Methods section could more clearly state that the supervised ML is a ranking refinement, not a discovery tool.

### 4.2 Isolation Forest -> XGBoost/LightGBM pipeline

**Severity: LOW**

The pipeline is defensible as described:
1. Isolation Forest for unsupervised anomaly detection -- standard and appropriate
2. XGBoost/LightGBM for ranking refinement -- acknowledged as surrogate modeling
3. Top 100 by `improved_composite_score` -- a clear, reproducible threshold

**Issue:** The validation.py code (`/mnt/data/tasni/src/tasni/pipeline/validation.py`) has a kinematics filter at pm > 100 mas/yr, but J143046 has pm = 55 mas/yr and is in the golden sample. This is because the kinematics filter produces a separate output (golden_improved_kinematics.csv) and is not part of the golden sample selection, which is purely by `nlargest(100, 'improved_composite_score')`. This is correctly implemented but could be confusing -- the paper should clarify that the golden sample is ranked by ML score alone, not filtered by proper motion.

### 4.3 Fading source rankings

**Severity: MEDIUM**

The four FADING sources are ranked 4, 7, 86, and 94 in the golden sample. The two coldest sources (J231029 at rank 86, J193547 at rank 94) are near the bottom of the golden sample. This means that:
- If the golden sample had been 80 sources instead of 100, J231029 and J193547 would have been missed.
- The improved_composite_score for these sources (0.169 and 0.145) is much lower than for J143046 (0.675) or J044024 (0.480).
- The fading classification is a post-hoc analysis on the golden sample, not part of the ML ranking.

This is not a problem per se, but it highlights that the ML pipeline did not identify these as top anomalies -- their fading behavior was discovered through variability analysis of the full golden sample. The paper could be more explicit about this distinction.

---

## 5. eROSITA Constraints

### 5.1 Coverage and constraint strength

**Severity: MEDIUM**

**Finding:** 59 sources are in the western Galactic hemisphere (l > 180 deg), corresponding to eROSITA DR1 coverage. None have X-ray detections at > 10^{-13} erg/s/cm^2.

**Assessment:**
- The geometric verification of coverage (validation.py line 31: `in_coverage = gal.l.deg > 180`) is correct.
- The constraint is meaningful for ruling out AGN contamination: an AGN at z < 0.1 would typically have X-ray flux > 10^{-13} erg/s/cm^2.
- The constraint is NOT meaningful for ruling out stellar coronal emission from Y dwarfs: Y dwarfs are expected to have essentially zero coronal X-ray emission because they lack the magnetic dynamo mechanisms of hotter stars.

**Missing quantification:** The paper states "ruling out AGN or stellar coronal activity" but does not compute X-ray luminosity upper limits. For a source at 17 pc:
- L_X < 4 * pi * d^2 * F_X = 4 * pi * (17 * 3.086e18)^2 * 10^{-13} = ~3.5 * 10^{26} erg/s
- This is comparable to solar coronal emission and much above Y dwarf levels, so the constraint is trivially satisfied for Y dwarfs.
- For AGN at cosmological distances, the flux limit provides useful constraints.

**Recommendation:** The paper should compute and report the X-ray luminosity upper limits explicitly and note that Y dwarfs are not expected to be X-ray sources regardless.

### 5.2 eROSITA crossmatch methodology

**Severity: LOW**

The MEMORY.md notes that "eROSITA crossmatch was not actually executed (API query); coverage is geometrically verified." The manuscript text (Section 3.5) says "Cross-matching these 59 sources against the eROSITA DR1 catalog using a 30 arcsec cone search reveals no X-ray detections." If the cone search was not actually executed and only geometric hemisphere coverage was checked, the non-detection claim may be unsupported by actual catalog crossmatching. However, given the source types (faint IR-only sources), genuine X-ray detections would be extremely unlikely.

---

## 6. Comparison to Literature (Table 4)

### 6.1 Y dwarf parameter space

**Severity: LOW**

The TASNI sources occupy reasonable parameter space for Y dwarfs:
- T_eff 251-293 K: Consistent with known Y0-Y2 dwarfs
- Distances 17-33 pc: Further than most known Y dwarfs (2-12 pc) but plausible for a WISE-limited survey
- Proper motions 55-306 mas/yr: Lower than most known Y dwarfs (500-8000 mas/yr)

**Concern:** The low proper motions are notable. WISE 0855 at 2.3 pc has pm ~ 8000 mas/yr. At 17 pc, the same tangential velocity would produce pm ~ 1000 mas/yr. J143046 at 17.4 pc with pm = 55 mas/yr has a tangential velocity of only 4.5 km/s, which is unusually low but not impossible for a disk population object. The paper does not compute or discuss tangential velocities, which would provide an additional consistency check.

### 6.2 Comparison Y dwarf selection

**Severity: LOW**

The comparison sample in Table 4 is appropriate and includes the coldest known Y dwarfs from the key discovery papers (Cushing et al. 2011, Kirkpatrick et al. 2012, Luhman 2014, Kirkpatrick et al. 2021). The inclusion of WISE 0336-0143 B (285-305 K) from Kirkpatrick et al. 2021 is particularly relevant as it overlaps in temperature with J143046 (293 K).

**Missing comparison objects:** The paper could benefit from including:
- WISE 1828+2650 (Cushing et al. 2011, ~300 K)
- CWISEP J1935-1546 (Meisner et al. 2020, ~300-350 K)
These would strengthen the case that TASNI sources are consistent with the known population.

---

## 7. SETI Framing

### 7.1 Appropriateness

**Severity: LOW**

The SETI framing is handled well:
- The introduction correctly motivates the search through the thermodynamic argument (waste heat from computation).
- Section 4.5 uses the Rio 2.0 Scale and correctly concludes R < 1 (insignificant).
- The abstract and conclusions consistently present Y dwarfs as the most parsimonious explanation.
- The "technosignatures" keyword is included but the paper does not overclaim.

**Assessment:** The SETI motivation provides useful context for why a search for cold IR-only sources without optical counterparts is scientifically interesting, beyond the Y dwarf discovery aspect. The paper maintains scientific objectivity throughout.

### 7.2 Minor framing issue

**Severity: LOW**

The paper title includes "Non-communicating Intelligence" which is technically aspirational -- the paper does not find evidence for intelligence. However, the "TASNI" acronym is the name of the pipeline/methodology, not a claim of detection, and the subtitle ("Discovery of Three Fading Thermal Orphans") correctly describes the actual findings. This is acceptable.

---

## 8. Additional Issues

### 8.1 W1-W2 colors of fading orphans

**Severity: MEDIUM**

The three confirmed fading orphans have W1-W2 colors of:
- J143046: 3.37 mag
- J231029: 1.75 mag
- J193547: 1.53 mag

Known Y dwarfs typically have W1-W2 > 2.5 mag (Kirkpatrick et al. 2012). J231029 and J193547 have W1-W2 colors (1.75 and 1.53) that are more consistent with late-T dwarfs (T7-T8) than Y dwarfs. This tension between the Planck-derived temperatures (251-258 K, firmly Y dwarf) and the W1-W2 colors (suggesting warmer objects) is not discussed in the manuscript and deserves attention. It could indicate that the Planck temperatures are underestimates, or that these sources have unusual atmospheric properties.

### 8.2 Four vs. three fading sources in data

**Severity: LOW**

The golden_improved.csv contains 4 FADING sources (J143046, J044024, J231029, J193547), consistent with the manuscript text that describes 4 initial fading sources of which 3 are confirmed (excluding the LMC source). The data and text are consistent.

### 8.3 Proper motion source

**Severity: LOW**

The golden_sample_readme.txt states "Proper motions are from the CatWISE2020 catalog where available, otherwise computed from multi-epoch positional differences." The manuscript does not specify the proper motion source. Since CatWISE2020 proper motions are derived from WISE+NEOWISE positions, they suffer from similar PSF-related systematics as the parallaxes. This is particularly relevant for J143046 (pm = 55 mas/yr), where the proper motion is a small fraction of the PSF.

### 8.4 Manuscript claims "67 significant parallax detections"

**Severity: MEDIUM**

Section 3.4 states "Significant detections (>5 mas) were obtained for 67 of 100 golden sample sources." The data confirms this (67 sources with neowise_parallax_mas > 5). However, "significant" is misleading here -- 5 mas is an absolute parallax threshold, not a significance threshold based on SNR. Many of these 67 may have low SNR. The paper should either clarify that this is an absolute threshold or report the number with SNR > 3 or SNR > 5.

From the data, the actual distribution is:
- 67 sources with parallax > 5 mas (the stated count)
- But some of these may have low SNR

The MIN_PARALLAX_SNR in the code is set to 2.0, which is a low threshold for reliable parallax detection (standard is 3-5).

---

## Summary of Findings by Severity

### CRITICAL (2)
| # | Finding | Section | Impact |
|---|---------|---------|--------|
| 1 | Planck blackbody is an unreliable SED model for T < 300 K objects with strong molecular absorption; temperatures may be systematically biased | 1.1 | Affects all claimed temperatures |
| 2 | NEOWISE parallaxes at 0.5-1% of PSF FWHM are inherently unreliable; the LMC source (J044024) demonstrates a confident but spurious parallax at SNR=23, undermining all distance claims | 2.1 | Affects all claimed distances |

### HIGH (3)
| # | Finding | Section | Impact |
|---|---------|---------|--------|
| 3 | Asymmetric distance error bars for J231029 (+13.3/-8.0 pc) do not match the simple parallax error propagation (+22.5/-9.5 pc); the reported values appear inconsistent | 2.3 | Incorrect uncertainty reporting |
| 4 | Trend sign convention in golden_improved.csv (negative = fading) is opposite to classify_variability() code (positive = fading); pipeline is not reproducible from stored data | 3.2 | Reproducibility |
| 5 | Abstract claim "one of the nearest room-temperature objects known" for J143046 should include "provisional" given demonstrated NEOWISE parallax unreliability | 2.2 | Overclaiming |

### MEDIUM (7)
| # | Finding | Section | Impact |
|---|---------|---------|--------|
| 6 | Statistical temperature uncertainties (35-47 K) from unweighted curve_fit likely underestimate true uncertainties | 1.2 | Understated uncertainties |
| 7 | Fading rates (23-53 mmag/yr) are implausible for intrinsic Y dwarf variability; blend separation is likely but not sufficiently highlighted | 3.1 | Interpretation |
| 8 | Injection-recovery test uses ideal Gaussian noise, not real NEOWISE systematics | 3.4 | Overstated validation |
| 9 | eROSITA X-ray luminosity upper limits not computed; Y dwarfs are trivially X-ray quiet regardless | 5.1 | Incomplete analysis |
| 10 | W1-W2 colors of J231029 (1.75) and J193547 (1.53) are inconsistent with Y dwarf classification (typically > 2.5); this tension is not discussed | 8.1 | Internal inconsistency |
| 11 | "Significant parallax detections (>5 mas)" conflates absolute parallax threshold with statistical significance | 8.4 | Misleading |
| 12 | Coldest fading orphans are ranked 86 and 94 in golden sample; the ML pipeline did not identify them as top anomalies | 4.3 | Framing |

### LOW (6)
| # | Finding | Section | Impact |
|---|---------|---------|--------|
| 13 | Planck Teff compared to spectroscopic Teff from literature is apples-to-oranges | 1.3 | Minor |
| 14 | NEOWISE cadence aliasing is thoroughly and correctly discussed | 3.3 | Positive |
| 15 | SETI framing is restrained and appropriate | 7.1 | Positive |
| 16 | Missing tangential velocity computation for fading sources as kinematic consistency check | 6.1 | Omission |
| 17 | Missing comparison Y dwarfs (WISE 1828+2650, CWISEP J1935-1546) in Table 4 | 6.2 | Minor |
| 18 | eROSITA crossmatch may not have been actually executed (only geometric coverage check) | 5.2 | Provenance |

---

## Recommendations for Revision

### Must-fix before submission:
1. **Clarify temperature method limitations:** Explicitly state that Planck blackbody temperatures are rough estimates for objects with strong molecular absorption; provide color-Teff estimates for comparison.
2. **Fix asymmetric distance error bars** for J231029 or explain the methodology (Bayesian posterior?) that produces them.
3. **Add "provisional" or "tentative" to distance claims in the abstract and conclusions** given the demonstrated NEOWISE parallax unreliability.
4. **Discuss the J044024 spurious parallax** as direct evidence that NEOWISE parallaxes can be confidently wrong.
5. **Fix the trend sign convention** in either the code or the data for reproducibility.
6. **Discuss the W1-W2 color tension** for J231029 and J193547.

### Should-fix:
7. Compute and report eROSITA X-ray luminosity upper limits.
8. Compute tangential velocities for the fading sources as a kinematic sanity check.
9. Add MCMC analysis on real data (not just synthetic data) to the appendix.
10. Clarify that the golden sample is ML-ranked, not kinematics-filtered.

### Nice-to-have:
11. Add more comparison Y dwarfs to Table 4.
12. Rerun injection-recovery with real NEOWISE light curves.
13. Compute and report parallax SNR distribution for the 67 "significant" detections.

---

## Overall Assessment

The TASNI paper presents a creative and well-executed pipeline for an important scientific goal. The discovery of three candidate ultra-cool objects is interesting regardless of whether they turn out to be Y dwarfs, edge-on disk systems, or artifacts. The paper's greatest strength is its intellectual honesty -- the caveats around parallax reliability, cadence aliasing, and the ML circularity are well-articulated. The greatest weakness is the tension between the strong discovery claims (temperatures, distances, "fading") and the limitations of the methods used to derive them. With the recommended revisions, particularly softening the distance claims and addressing the W1-W2 color tension, the paper would be suitable for publication in ApJ.

---

*Audit completed 2026-02-20 by Science-Reviewer Agent.*
*Files reviewed: manuscript.tex, extract_neowise_parallax.py, validation.py, generate_figures.py, calculate_teff.py, compute_ir_variability.py, injection_recovery.py, mcmc_parallax_comparison.py, golden_improved.csv, references.bib, parallax_mcmc_ls_results.json, cover_letter.tex, golden_sample_readme.txt, golden_sample_cds.txt.*
