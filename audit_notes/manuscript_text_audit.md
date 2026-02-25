# Manuscript Text Quality Audit

**Date:** 2026-02-20
**Auditor:** Claude (manuscript-reviewer agent)
**Files reviewed:**
- `/mnt/data/tasni/tasni_paper_final/manuscript.tex` (590 lines)
- `/mnt/data/tasni/tasni_paper_final/cover_letter.tex` (77 lines)
- `/mnt/data/tasni/tasni_paper_final/references.bib` (295 lines)
- `/mnt/data/tasni/data/processed/final/golden_improved.csv` (ground truth)

---

## 1. CRITICAL Issues

### 1.1 Distance Error Bars Do Not Match Stated Propagation Method (Lines 282-284)

**Table 3 tablecomments (line 282-284)** states: "we propagate the formal parallax uncertainty through this transformation to obtain the quoted confidence intervals."

However, simple error propagation of d = 1000/pi gives:
- **J143046**: pi = 57.58 +/- 9.90 mas -> d = 17.4 pc, d_upper = 1000/(57.58-9.90) = 21.0 pc (+3.6), d_lower = 1000/(57.58+9.90) = 14.8 pc (-2.6). Table says **+3.0/-2.6**. The upper error bar (+3.0) does not match (+3.6).
- **J231029**: pi = 30.66 +/- 12.53 mas -> d = 32.6 pc, d_upper = 1000/(30.66-12.53) = 55.2 pc (+22.5), d_lower = 1000/(30.66+12.53) = 23.2 pc (-9.4). Table says **+13.3/-8.0**. Neither error bar matches.

The J231029 discrepancy is especially large (+13.3 vs +22.5). This suggests the errors come from a Bayesian posterior (as noted in project memory) rather than simple propagation. The tablecomments must be updated to accurately describe the method, or the error bars must be recalculated.

**Suggested fix:** Either (a) recalculate distance errors using strict 1/pi propagation and update Table 3, or (b) change the tablecomments to accurately state "Distance uncertainties are derived from the Bayesian MCMC posterior (see Appendix A), which yields tighter constraints than naive parallax inversion." Option (b) is preferred if the MCMC values are indeed more reliable.

### 1.2 `\received{\today}` Should Be a Fixed Date (Line 22)

The `\received{\today}` macro will produce a different date each time the document is compiled, which is inappropriate for a submitted manuscript. ApJ expects the actual submission date.

**Suggested fix:** Replace `\received{\today}` with `\received{2026 February 20}` (or the actual submission date). Alternatively, leave it blank `\received{}` and let the journal fill it in, which is standard practice for initial submissions.

---

## 2. HIGH Issues

### 2.1 Abstract Claims "One of the Nearest Room-Temperature Objects Known" Without Qualification (Lines 41-42)

The abstract states the nearest source at 17.4 pc is "one of the nearest room-temperature objects known." However:
- The distance relies on a provisional NEOWISE parallax with only SNR = 5.8
- The paper itself (Section 3.4, lines 339-348) warns that NEOWISE PSF is 6 arcsec and systematic errors at ~100 mas cannot be excluded
- Section 5.5 explicitly says distances should be "treated as provisional"

Claiming "one of the nearest" in the abstract without any caveat is inconsistent with the body's own caveats.

**Suggested fix:** Add "provisionally" or "pending confirmation": "...provisionally placing it among the nearest room-temperature objects known, pending independent astrometric confirmation."

### 2.2 LMC Section Contains Contradictory Proper Motion Discussion (Lines 414-426)

Section 4.3 (lines 414-426) presents a confusing argument:
1. Line 415: "Its NEOWISE-derived parallax formally implies a distance of ~30.5 pc, and its proper motion of 165 mas yr^{-1} would be consistent with a nearby source."
2. Lines 421-422: "the expected ... proper motion (~1.9 mas yr^{-1}) are far below our measured values"
3. Lines 423-426: "our NEOWISE-derived 165 mas yr^{-1} is therefore inconsistent with LMC kinematics and confirms that the astrometry is spurious."

The logic seems backwards: if the measured PM of 165 mas/yr is inconsistent with LMC kinematics (~1.9 mas/yr), that would suggest the source is NOT an LMC member, yet the paragraph concludes the astrometry is "spurious." The reasoning needs to be clearer: the SIMBAD identification as MSX LMC 1152 is the primary evidence for LMC membership, and the high measured PM/parallax is what is spurious (artifacts of the NEOWISE astrometric pipeline for a faint, distant source), not the LMC identification.

**Suggested fix:** Restructure the paragraph to lead with the SIMBAD identification first, then explain that the measured PM and parallax are inconsistent with LMC membership and therefore must be spurious astrometric artifacts, confirming the source is too distant for NEOWISE astrometry to be reliable.

### 2.3 Running Text Rounds "55--306 mas yr^{-1}" But Includes J044024 PM in Discussion (Lines 93-94, 415)

The introduction (line 94) correctly reports "55--306 mas yr^{-1}" for the three confirmed sources. However, in Section 4.3 (line 415), J044024's PM of 165 mas/yr is stated without clarification that this value is the same as J231029's PM (165 mas/yr), which could confuse readers. The CSV confirms J044024 PM = 165.47 and J231029 PM = 165.42 -- nearly identical values for completely different sources at vastly different distances.

**Suggested fix:** Add a parenthetical noting this coincidence, or ensure Table 3 and the LMC discussion make clear these are different sources with coincidentally similar measured PMs.

### 2.4 Missing Transition Between Sections 3 and 4 (Lines 202-203, 365-366)

Section 3 (Results) ends with the periodogram and parallax subsections, then Section 4 (Discussion) begins abruptly with "What Are These Sources?" There is no bridging sentence summarizing the key results before transitioning to interpretation.

Similarly, Section 3.1 (line 206) jumps directly into filtering statistics without a brief introductory sentence for the Results section as a whole.

**Suggested fix:** Add a brief introductory paragraph at the start of Section 4: "Having established the observational properties of the three fading thermal orphans -- cold temperatures, significant proper motions, monotonic fading, and X-ray non-detections -- we now consider their physical nature."

### 2.5 Injection Recovery Details Are Insufficient (Lines 150-154)

The injection-recovery test is summarized in just three lines (150-154) with "100% of injected fading signals at >3-sigma significance." This is a strong claim with minimal detail:
- How many synthetic sources were injected?
- What was the parameter space sampled (magnitude range, sky position)?
- Is 100% recovery at 3-sigma simply a consequence of the chosen noise level (30 mmag) being much smaller than the signal (20-50 mmag/yr x 10 yr = 200-500 mmag total)?

**Suggested fix:** Either expand to a brief paragraph with the number of injections and parameter ranges, or add a sentence acknowledging the favorable SNR regime: "At the noise levels and baseline considered, the injected signals have cumulative amplitudes of 200--500 mmag, well above the single-epoch noise floor."

---

## 3. MEDIUM Issues

### 3.1 Passive Voice Overuse

Several passages lean heavily on passive voice where active voice would be stronger:

- Line 27: "We present TASNI..." (good, active)
- Line 56: "Traditional searches ... have focused on" (fine)
- Line 159: "Effective temperatures are estimated by fitting..." -> "We estimate effective temperatures by fitting..."
- Line 177: "We analyzed NEOWISE single-epoch photometry using..." (good)
- Line 326: "Parallax measurements were obtained by fitting..." -> "We obtained parallax measurements by fitting..."
- Line 355-356: "We verified the eROSITA DR1 coverage footprint geometrically" (good)

**Suggested fix:** Convert lines 159 and 326 to active voice for consistency with the rest of the paper.

### 3.2 Vague Hedging Language (Lines 468-473)

Section 4.6 (SETI Implications, lines 468-475) is appropriately cautious, but the sentence "While we cannot rule out artificial origins" followed by "suggests astrophysical explanations are more parsimonious" is somewhat redundant with the Rio 2.0 Scale scoring. The paragraph could be tightened.

**Suggested fix:** Combine into: "The combination of temperatures consistent with Y dwarfs, proper motions consistent with nearby stellar populations, and the absence of non-natural signatures yields a Rio 2.0 Scale score of R < 1 (insignificant), consistent with natural astrophysical explanations."

### 3.3 Redundant Statements About NEOWISE Cadence Aliases

The NEOWISE cadence aliasing is explained three times:
1. Abstract (lines 37-40)
2. Section 3.3 Periodogram Results (lines 300-312)
3. Table 3 tablecomments (lines 280-281)

While some repetition is appropriate, the abstract and Section 3.3 descriptions are nearly identical. The abstract version could be shortened.

**Suggested fix:** In the abstract, condense to: "Lomb-Scargle periodograms reveal apparent periodicities at 93--179 days, which we attribute to NEOWISE cadence aliases."

### 3.4 "Fading Thermal Orphans" -- Term Used Without Prior Definition (Line 31)

The abstract (line 31) uses "fading thermal orphans" before defining the concept. While the name is suggestive, a formal first-use definition would help: these are mid-infrared sources without optical counterparts ("orphans") that exhibit thermal spectra ("thermal") and monotonic dimming ("fading").

**Suggested fix:** Add a brief clarification on first use: "...three 'fading thermal orphans' -- mid-infrared sources with no optical counterparts exhibiting cold thermal spectra and monotonic dimming --"

### 3.5 Table 2 Filtering Statistics -- "Reduction" Column Is Ambiguous (Lines 215-230)

Table 2 has a "Reduction" column showing percentages (54.6%, 99.4%, etc.), but it is unclear whether these represent the fraction removed at each step or the fraction remaining. From context: 747M -> 406M is 54.6% pass-through (not reduction). But "Quality filters" 406M -> 2.37M showing 99.4% suggests that 99.4% were removed. The interpretation flips between steps.

On closer inspection:
- Row 1: 54.6% = fraction that are orphans (pass-through)
- Row 2: 99.4% = fraction removed by quality filters
- Row 3: 97.3% = fraction removed

The first row uses a different convention than the rest. Also, the column header "Reduction" suggests amount removed, but 54.6% of the original catalog being orphans is not a "reduction."

**Suggested fix:** Rename the column to "Reduction (\%)" and make row 1 consistent: the reduction from 747M to 406M is 45.4% (fraction removed), not 54.6%. Alternatively, add a tablecomment clarifying "Reduction shows the percentage of sources removed at each step."

### 3.6 Cover Letter Uses "Would Be" for Distance Claim (Line 45)

The cover letter states: "The nearest candidate, at 17.4+3.0/-2.6 pc, **would be** among the closest known free-floating room-temperature objects." The subjunctive "would be" appropriately hedges, but the manuscript abstract (line 41) uses the indicative "making it one of the nearest room-temperature objects known." These should be consistent.

**Suggested fix:** Make both use the same level of hedging. Given the provisional nature of the distance, the cover letter's more cautious phrasing is preferred. Update the abstract accordingly (see Issue 2.1).

### 3.7 Acknowledgments Missing Funding Statement (Lines 561-574)

The acknowledgments list data archives but do not include a statement about funding (or the lack thereof, for an independent researcher). ApJ typically expects either a funding acknowledgment or an explicit "no funding" statement.

**Suggested fix:** Add: "This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors."

### 3.8 Missing eROSITA Acknowledgment (Lines 561-574)

While eROSITA is listed in the acknowledgments (line 569), the eROSITA DR1 data use policy typically requires a specific acknowledgment text. Check the eROSITA data access policy for the required text.

**Suggested fix:** Add the standard eROSITA DR1 acknowledgment text as required by the data use policy.

---

## 4. LOW Issues

### 4.1 Minor Grammar and Style Issues

- **Line 64**: "The key insight is that computation requires energy" -- "The key insight" is slightly informal for a journal paper. Consider "The fundamental constraint is that..."
- **Line 70-71**: "No systematic search exists for:" -- This is a sentence fragment used to introduce a list. While acceptable in some styles, ApJ tends to prefer complete sentences. Consider: "No systematic search has been conducted for:"
- **Line 90**: "While natural explanations predominate" -- "predominate" is fine but slightly unusual in this context. Consider "While natural explanations are most likely" or "While natural astrophysical origins predominate."
- **Line 165**: "This approach is preferred over simple color--temperature relations because it utilizes all available photometric bands simultaneously." -- "utilizes" can be replaced with "uses" (simpler, no loss of meaning).
- **Line 397**: "far too slow to produce" -- somewhat informal; consider "orders of magnitude too slow to account for."
- **Line 438**: "JWST/MIRI or ground-based NIR spectroscopy is the critical next step" -- "critical" is strong; "essential" (used later on line 521) would be more appropriate here too for consistency.

### 4.2 Inconsistent Use of Em-Dash vs En-Dash

- Line 15 (title): Uses "---" (em-dash) for "Intelligence---Discovery" -- correct for AASTeX
- Line 34: Uses "--" (en-dash) for "55--306" -- correct for ranges
- These are consistent throughout. No issue found.

### 4.3 "\cite" vs "\citep" Inconsistency in Table 5 (Lines 442-463)

Table 5 (Y dwarf comparison) uses `\cite{...}` (lines 452-459) while the rest of the manuscript correctly uses `\citep{...}` for parenthetical citations and `\citet{...}` for textual. Inside a table column, `\cite` may render differently depending on the natbib configuration. This should be verified during compilation.

**Suggested fix:** Replace all `\cite{...}` in Table 5 with `\citet{...}` for consistency.

### 4.4 Abstract Keywords List (Lines 49-50)

The keywords include "technosignatures" which is appropriate given the SETI context, but the paper's actual results are entirely about natural sources (Y dwarf candidates). This keyword may attract attention from SETI researchers while the paper explicitly scores R < 1 on the Rio scale. This is a judgment call -- including it is defensible since the methodology is SETI-motivated.

### 4.5 Appendix Uses Synthetic Data (Lines 525-535)

The appendix (line 532) states the MCMC comparison uses "synthetic data consistent with J143046.35-025927.8" rather than the actual source data. This is a limitation that should be explicitly flagged. If the actual data were available, why not run the MCMC on the real data?

**Suggested fix:** Add a sentence explaining why synthetic data was used (e.g., "We use synthetic data to demonstrate the method's behavior in a controlled setting; application to the full NEOWISE epoch data for all three sources is planned for a subsequent analysis.").

### 4.6 Data Availability Section -- GitHub URL (Line 549)

The GitHub URL `https://github.com/denster32/TASNI` is given but should be verified to be live and public before submission. If the repository is not yet public, note this.

### 4.7 Software Citations Missing Version Numbers (Lines 580-585)

The `\software` block lists packages without version numbers. AAS journals recommend including version numbers for reproducibility (e.g., "Astropy v5.0").

**Suggested fix:** Add version numbers: "Astropy v5.0 \citep{...}, NumPy v1.24 \citep{...}, ..." etc.

### 4.8 Cover Letter -- Minor Phrasing

- Line 30 (cover letter): "the first systematic search for thermally anomalous objects among WISE catalog orphans" -- This is a strong claim. Verify that no prior work has done a similar search. The G-HAT survey (Wright et al. 2014, cited in the paper) searched for Dyson sphere signatures in WISE data, which is related though not identical.
- Line 51 (cover letter): "a potentially new class of nearby thermal sources" -- appropriate hedging.
- Overall tone is professional and appropriate for ApJ editors.

---

## 5. Logical Flow Assessment

### Section-by-Section Flow

1. **Introduction (Section 1)**: Good motivation from SETI -> thermodynamics -> WISE data. The "Detection Gap" subsection clearly identifies the niche. "Our Approach" previews the results. Flow is logical.

2. **Data and Methods (Section 2)**: Follows naturally. Data sources -> Pipeline -> Temperature -> Periodogram. However, the periodogram method is described in Section 2 but results are in Section 3.3, which is standard.

3. **Results (Section 3)**: Pipeline stats -> Fading orphans -> Periodograms -> Parallax -> X-ray. This is logical, building from the broadest sample to the most interesting sources.

4. **Discussion (Section 4)**: Interpretation -> Fading nature -> LMC source -> Y dwarf comparison -> SETI -> Limitations. Good flow. The limitations section (4.6) is appropriately placed at the end before conclusions.

5. **Conclusions (Section 5)**: Clean enumerated summary. Matches the abstract claims.

**Overall assessment**: The logical flow is solid. The main gap is the missing transition between Sections 3 and 4 (noted in Issue 2.4).

---

## 6. Abstract vs Body Consistency

| Claim in Abstract | Location in Body | Match? |
|---|---|---|
| 100 golden candidates | Table 2 (line 227) | Yes |
| T_eff = 251+/-35 to 293+/-47 K | Table 3 (lines 269-274) | Yes |
| mu = 55--306 mas/yr | Table 3 | Yes |
| Monotonic fading over 10 years | Section 3.2, Table 3 | Yes |
| J044024 LMC member at ~50 kpc | Section 4.3 (lines 412-427) | Yes |
| Periods 93--179 days are cadence aliases | Section 3.3 (lines 300-312) | Yes |
| Nearest at 17.4+3.0/-2.6 pc | Table 3 (line 269) | Yes |
| 59 sources in eROSITA footprint, no X-ray | Section 3.5 (lines 355-363) | Yes |
| Y dwarfs or edge-on disks most likely | Section 4.1 (lines 371-389) | Yes |
| Methodology reproducible, data public | Sections 5, Data Availability | Yes |

**Assessment**: Abstract accurately reflects body content. No orphaned claims.

---

## 7. TODO/FIXME/Placeholder Check

- No `TODO`, `FIXME`, `XXX`, `HACK`, or `\textcolor{red}` markers found in `manuscript.tex` or `cover_letter.tex`.
- `FIXME` comments exist only in `aastex701.cls` (the standard AAS class file), which is expected.
- `\received{\today}` is the only dynamic placeholder (see Issue 1.2).
- `\revised{}` and `\accepted{}` (lines 23-24) are appropriately empty for initial submission.

---

## 8. Cover Letter Assessment

The cover letter is well-structured and professional:
- Correctly identifies the journal (ApJ)
- Summarizes the four key results concisely
- Mentions reproducibility and data release
- States no dual submission and no conflicts of interest
- Provides author identification (ORCID, email)

**Issues found:**
- The "would be" hedging (Issue 3.6) is actually better than the manuscript's phrasing
- The claim of "first systematic search" (line 30) should be verified (Issue 4.8)
- Missing mention of the paper's key limitation (provisional distances) -- the cover letter could benefit from a single sentence acknowledging this to preempt reviewer concerns

---

## 9. Limitations Section (5.5) Assessment

The Limitations section (lines 480-499) covers five items:
1. Spatial resolution (WISE PSF)
2. Photometric precision
3. Spectroscopic confirmation needed
4. Astrometric precision (NEOWISE parallax limitations)
5. ML classification circularity

**Missing limitations that should be acknowledged:**
- **Proper-motion-induced fading artifact**: Discussed in Section 4.2 (lines 399-407) but not listed in the formal Limitations section. This is arguably the most important systematic concern and deserves explicit mention in Section 5.5.
- **Small number statistics**: Only 3 confirmed fading sources from 747M. While the filtering is rigorous, the statistical significance of conclusions drawn from N=3 deserves mention.
- **NEOWISE photometric stability assumption**: The paper cites the NEOWISE Explanatory Supplement (line 180-181) for stability, but does not discuss source-specific systematics (e.g., PSF fitting artifacts for very red sources at faint magnitudes).

---

## 10. Summary of All Issues

| # | Severity | Line(s) | Issue |
|---|----------|---------|-------|
| 1.1 | CRITICAL | 269-284 | Distance error bars inconsistent with stated propagation method |
| 1.2 | CRITICAL | 22 | `\received{\today}` should be fixed date or blank |
| 2.1 | HIGH | 41-42 | "Nearest room-temperature" claim lacks caveat in abstract |
| 2.2 | HIGH | 414-426 | LMC section logic is unclear/contradictory |
| 2.3 | HIGH | 94, 415 | Coincidental PM=165 for two different sources not addressed |
| 2.4 | HIGH | 202, 366 | Missing transition between Results and Discussion |
| 2.5 | HIGH | 150-154 | Injection recovery details insufficient |
| 3.1 | MEDIUM | 159, 326 | Passive voice inconsistency |
| 3.2 | MEDIUM | 468-473 | Hedging language in SETI section could be tightened |
| 3.3 | MEDIUM | 37-40 | Cadence alias explanation repeated 3 times |
| 3.4 | MEDIUM | 31 | "Fading thermal orphans" used before definition |
| 3.5 | MEDIUM | 215-230 | Table 2 "Reduction" column uses inconsistent convention |
| 3.6 | MEDIUM | 45 (CL) | Hedging inconsistency between cover letter and abstract |
| 3.7 | MEDIUM | 561-574 | Missing funding statement |
| 3.8 | MEDIUM | 561-574 | Missing standard eROSITA acknowledgment text |
| 4.1 | LOW | Various | Minor grammar/style (6 items) |
| 4.3 | LOW | 452-459 | `\cite` vs `\citep` in Table 5 |
| 4.4 | LOW | 49-50 | "technosignatures" keyword -- judgment call |
| 4.5 | LOW | 525-535 | Appendix uses synthetic data without clear justification |
| 4.6 | LOW | 549 | GitHub URL should be verified as live |
| 4.7 | LOW | 580-585 | Software citations missing version numbers |
| 4.8 | LOW | 30 (CL) | "First systematic search" claim should be verified |

**Total: 2 CRITICAL, 5 HIGH, 8 MEDIUM, 7 LOW**
