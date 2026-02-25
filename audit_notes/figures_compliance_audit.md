# TASNI Figures, Compliance & Bibliography Audit

**Date**: 2026-02-20
**Auditor**: figures-compliance agent (Claude Opus 4.6)
**Branch**: release/v1.0.0

---

## 1. Figure Audit

### 1.1 Figure Inventory

| File | Format | Size | Valid PDF | ApJ <10 MB |
|------|--------|------|-----------|------------|
| fig1_pipeline_flowchart.pdf | PDF 1.4 | 36 KB | YES | YES |
| fig2_allsky_galactic.pdf | PDF 1.4 | 33 KB | YES | YES |
| fig3_color_magnitude.pdf | PDF 1.4 | 23 KB | YES | YES |
| fig4_distributions.pdf | PDF 1.4 | 15 KB | YES | YES |
| fig5_variability.pdf | PDF 1.4 | 25 KB | YES | YES |
| fig6_periodograms.pdf | PDF 1.4 | 47 KB | YES | YES |
| fig_appendix_parallax_mcmc_ls.pdf | PDF 1.4 | 16 KB | YES | YES |

**Total figure payload**: ~195 KB (well under ApJ limits)

All 7 PDF figures verified as valid single-page PDF 1.4 documents.

**PNG duplicates present**: 6 PNG files also exist in the figures directory (269 KB -- 723 KB each). These are not referenced in the manuscript and should be removed before submission to avoid confusion, or retained only as supplementary screen-resolution copies.

**Ancillary file**: `parallax_mcmc_ls_results.json` (297 bytes) contains MCMC vs LS parallax comparison data for the appendix figure. This should be moved out of the figures directory or documented as a machine-readable supplement.

### 1.2 Figure Captions vs. Content

| Fig | Label | Caption Summary | Assessment |
|-----|-------|-----------------|------------|
| 1 | `fig:pipeline` | Pipeline flowchart: AllWISE input through cross-matching, ML, variability, golden sample | PASS -- describes figure content fully |
| 2 | `fig:allsky` | All-sky Galactic coordinates distribution of 100 candidates; notes clustering away from plane | PASS -- fully descriptive |
| 3 | `fig:colormag` | W1 vs W1-W2 CMD; marks 3 fading orphans as red stars; notes reddest colors | PASS -- fully descriptive |
| 4 | `fig:distributions` | Distributions of Teff, PM, W1-W2 color, anomaly score | PASS -- fully descriptive |
| 5 | `fig:variability` | Left: W1 light curves for 3 fading orphans; Right: variability classification distribution | PASS -- describes two-panel layout |
| 6 | `fig:periodograms` | Lomb-Scargle periodograms; peak periods noted; alias interpretation stated | PASS -- fully descriptive |
| A1 | `fig:appendix_mcmc` | LS vs MCMC parallax comparison; describes blue dashed line, gray histogram, 1-sigma interval | PASS -- fully descriptive |

### 1.3 Figure References Cross-Check

All 7 `\plotone{}` paths verified to exist on disk:
- `figures/fig1_pipeline_flowchart.pdf` -- OK
- `figures/fig2_allsky_galactic.pdf` -- OK
- `figures/fig3_color_magnitude.pdf` -- OK
- `figures/fig4_distributions.pdf` -- OK
- `figures/fig5_variability.pdf` -- OK
- `figures/fig6_periodograms.pdf` -- OK
- `figures/fig_appendix_parallax_mcmc_ls.pdf` -- OK

All figure `\label{}` tags are referenced by `\ref{}` in the text body:
- `\ref{fig:pipeline}` -- line 135
- `\ref{fig:allsky}` -- line 211
- `\ref{fig:colormag}` -- line 212
- `\ref{fig:distributions}` -- line 210
- `\ref{fig:variability}` -- line 257
- `\ref{fig:periodograms}` -- line 302
- `\ref{fig:appendix_mcmc}` -- line 531

### 1.4 Appendix Figure

`fig_appendix_parallax_mcmc_ls.pdf` EXISTS (16 KB, valid PDF 1.4).

**NOTE on MCMC results JSON**: The `parallax_mcmc_ls_results.json` file shows `ls_parallax_mas = 97.2` and `mcmc_median_mas = 97.2` (identical values), with MCMC 16th/84th percentiles at 82.0/111.5 mas. This is synthetic/illustrative data (consistent with the manuscript statement "synthetic data consistent with J143046.35-025927.8"). The actual manuscript Table 1 reports parallax = 57.6 +/- 9.9 mas for J143046. The discrepancy is expected since the appendix uses synthetic data, but this should be clearly noted.

---

## 2. ApJ Compliance Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Document class: `aastex701` | PASS | Line 6: `\documentclass[twocolumn]{aastex701}` |
| Two-column format | PASS | `twocolumn` option present |
| Author ORCID | PASS | Line 17: `\author[0009-0005-1026-5103]{Dennis Palucki}` |
| `\correspondingauthor` | PASS | Line 19 |
| `\email` | PASS | Line 18: `paluckide@yahoo.com` |
| Keywords (AAS vocabulary) | PASS with CAVEAT | Line 49-50: "brown dwarfs --- infrared: stars --- proper motions --- surveys --- techniques: photometric --- technosignatures" |
| `\facilities` | PASS | Line 578: WISE, NEOWISE, Gaia, eROSITA |
| `\software` | PASS | Lines 580-585: Astropy, NumPy, Pandas, scikit-learn, XGBoost, LightGBM |
| Data availability statement | PASS | Lines 547-553: Section present with GitHub URL and Zenodo plan |
| Code availability statement | PASS | Lines 555-559: Section present with GitHub URL and MIT license |
| `\received` | WARNING | Line 22: `\received{\today}` -- should be blank or set to actual submission date |
| `\revised` | PASS | Line 23: blank (correct for initial submission) |
| `\accepted` | PASS | Line 24: blank (correct for initial submission) |
| `\shorttitle` | PASS | Line 10 |
| `\shortauthors` | PASS | Line 11 |
| `\affiliation` | PASS | Line 20: "Independent Researcher" |
| `\bibliographystyle{aasjournalv7}` | PASS | Line 587 |

### 2.1 Keyword Compliance Details

AAS Unified Astronomy Thesaurus (UAT) check:
- "brown dwarfs" -- valid UAT term (concept 185)
- "infrared: stars" -- WARNING: should be "Infrared stars" (UAT concept 793) or separate terms; the colon separator format may not match UAT exactly
- "proper motions" -- valid UAT term (concept 1295)
- "surveys" -- valid UAT term (concept 1671)
- "techniques: photometric" -- WARNING: should be "Photometry" (UAT concept 1234) or "Photometric techniques"; colon-separated subheadings are legacy AAS style, not current UAT
- "technosignatures" -- valid UAT term (concept 2128, added 2018)

**RECOMMENDATION**: Verify keywords against the current UAT at https://astrothesaurus.org/. The colon-separated format ("infrared: stars", "techniques: photometric") is old-style AAS and may need updating to current UAT preferred terms.

### 2.2 `\received{\today}` Issue

Using `\received{\today}` will insert the compilation date. ApJ prefers either:
- Leave blank for initial submission (AAS editorial will fill in)
- Set to actual submission date

**RECOMMENDATION**: Change to `\received{}` for initial submission.

---

## 3. Bibliography Audit

### 3.1 Citation Cross-Check

**Manuscript cites 29 unique keys. Bibliography contains 29 entries.**

All 29 citation keys in the manuscript match a bib entry:

| Citation Key | In Manuscript | In .bib | Match |
|--------------|:---:|:---:|:---:|
| 2023A&A...674A...1G | YES | YES | OK |
| 2010AJ....140.1868W | YES | YES | OK |
| 2013wise.rept....1C | YES | YES | OK |
| 2014ApJ...792...13M | YES | YES | OK |
| 2020NEOWISE.DR | YES | YES | OK |
| 2012ApJ...753...56K | YES | YES | OK |
| 2011ApJ...743...50C | YES | YES | OK |
| 2014AJ....148...82W | YES | YES | OK |
| 2014ApJ...786...18L | YES | YES | OK |
| 2021ApJ...920...85M | YES | YES | OK |
| 2019BAAS...51c.389W | YES | YES | OK |
| 2015JBIS...68..142Y | YES | YES | OK |
| dyson1960search | YES | YES | OK |
| 1964SvA.....8..217K | YES | YES | OK |
| 1982ApJ...263..835S | YES | YES | OK |
| 1976Ap&SS..39..447L | YES | YES | OK |
| 2008MNRAS.385.1279B | YES | YES | OK |
| 2019IJAsB..18..336F | YES | YES | OK |
| 2022ApJ...935..167A | YES | YES | OK |
| harris2020array | YES | YES | OK |
| mckinney2010data | YES | YES | OK |
| sklearn2011 | YES | YES | OK |
| chen2016xgboost | YES | YES | OK |
| ke2017lightgbm | YES | YES | OK |
| 2021ApJS..253....8M | YES | YES | OK |
| 2021ApJS..253....7K | YES | YES | OK |
| 2020ApJ...899..123M | YES | YES | OK |
| 2014ApJ...792...27W | YES | YES | OK |
| 2023A&A...669A..91J | YES | YES | OK |

**No orphaned bib entries. No missing bib entries.**

### 3.2 Missing Key References Check

| Reference | Status | Severity |
|-----------|--------|----------|
| CatWISE2020 (Marocco+ 2021) | CITED | OK -- `\citep{2021ApJS..253....8M,2020ApJ...899..123M}` on line 144 |
| Backyard Worlds (Kuchner+ 2017) | NOT CITED | MEDIUM -- relevant citizen science Y dwarf discovery project; should be mentioned in Introduction or Discussion |
| JWST cold brown dwarf results (2023-2025) | NOT CITED | HIGH -- e.g., Beiler+ 2024 (JWST Y dwarf atmospheres), Luhman+ 2024 (JWST cold BD census); manuscript mentions JWST as follow-up but does not cite recent results |
| Zuckerman & Song (2009) or Y dwarf distance papers | NOT CITED | LOW -- not strictly required; Kirkpatrick 2021 covers the 20 pc census |
| NEOWISE mission decommissioning (2024) | NOT CITED | MEDIUM -- NEOWISE was decommissioned in Aug 2024; relevant context for the 10-year baseline discussion |

### 3.3 DOI Completeness

22 of 29 entries have DOIs. Missing DOIs:

| Key | Year | Reason / Action |
|-----|------|-----------------|
| `2013wise.rept....1C` | 2013 | Technical report -- no DOI available (acceptable) |
| `2019BAAS...51c.389W` | 2019 | BAAS white paper -- DOI likely available: `10.3847/2515-5172/ab0ec7` or similar; CHECK ADS |
| `2015JBIS...68..142Y` | 2015 | JBIS -- may not have DOI (acceptable) |
| `1964SvA.....8..217K` | 1964 | Soviet Astronomy translation -- no DOI (acceptable, pre-DOI era) |
| `2020NEOWISE.DR` | 2020 | Web resource/explanatory supplement -- no DOI (acceptable) |
| `sklearn2011` | 2011 | JMLR -- has a URL but DOI may not exist (acceptable) |
| `ke2017lightgbm` | 2017 | NeurIPS proceedings -- no formal DOI but has URL (acceptable) |

**RECOMMENDATION**: Check ADS for DOI on `2019BAAS...51c.389W` (Wright 2019 technosignatures white paper). All other missing DOIs are justifiable.

### 3.4 ADS Bibcode Formatting

7 of 29 entries have `adsurl` fields. The remaining entries use DOIs which ADS can resolve. Bibcode format check:

- `2023A%26A...674A...1G` -- contains URL-encoded `&` which is correct for URLs
- `2010AJ....140.1868W` -- standard 19-char bibcode format: PASS
- `2013wise.rept....1C` -- non-standard bibcode (technical report): acceptable
- `2015JBIS...68..142Y` -- standard format: PASS
- `2019BAAS...51c.389W` -- standard format: PASS
- `2014AJ....148...82W` -- standard format: PASS

All bibcodes appear properly formatted.

### 3.5 Bib Entry Type Consistency

| Key | Type | Notes |
|-----|------|-------|
| `2013wise.rept....1C` | `@misc` | Acceptable for tech report |
| `2020NEOWISE.DR` | `@misc` | Acceptable for web resource |
| `mckinney2010data` | `@inproceedings` | Correct |
| `chen2016xgboost` | `@inproceedings` | Correct |
| `ke2017lightgbm` | `@inproceedings` | Correct |
| All others | `@article` | Correct |

---

## 4. LaTeX Validation

### 4.1 Undefined References

**0 instances of `??` found** in manuscript.tex. No undefined references.

### 4.2 Label/Ref Matching

**Labels defined**: 31
**Refs used**: 14

All 14 `\ref{}` targets resolve to existing `\label{}` definitions:

| \ref target | Matching \label | Status |
|-------------|-----------------|--------|
| `fig:pipeline` | line 103 | OK |
| `fig:allsky` | line 191 | OK |
| `fig:colormag` | line 199 | OK |
| `fig:distributions` | line 237 | OK |
| `fig:variability` | line 294 | OK |
| `fig:periodograms` | line 320 | OK |
| `fig:appendix_mcmc` | line 544 | OK |
| `tab:datasources` | line 119 | OK |
| `tab:fading` | line 260 | OK |
| `tab:filtering` | line 216 | OK |
| `tab:y_comparison` | line 443 | OK |
| `subsec:parallax` | line 324 | OK |
| `subsec:periodogram_results` | line 298 | OK |
| `app:mcmc_parallax` | line 526 | OK |

**17 unreferenced labels** (all section/subsection labels -- normal and acceptable):
`sec:intro`, `sec:methods`, `sec:results`, `sec:discussion`, `sec:conclusions`,
`subsec:data`, `subsec:pipeline`, `subsec:teff`, `subsec:periodogram`,
`subsec:statistics`, `subsec:fading`, `subsec:interpretation`, `subsec:fading_nature`,
`subsec:lmc`, `subsec:comparison`, `subsec:seti`, `subsec:limitations`, `subsec:xray`

### 4.3 Environment Matching

| Environment | Opens | Closes | Balanced |
|-------------|-------|--------|----------|
| `figure` | 5 | 5 | YES |
| `figure*` | 2 | 2 | YES |
| `deluxetable` | 3 | 3 | YES |
| `deluxetable*` | 1 | 1 | YES |
| `equation` / `align` | 0 | 0 | N/A (no equations) |

All environments properly balanced.

### 4.4 Multiply-Defined Labels

No duplicate labels detected. All 31 labels are unique.

---

## 5. Summary of Issues

### CRITICAL (must fix before submission)
*None found.*

### HIGH (strongly recommended)

1. **Missing JWST citations**: Recent JWST cold brown dwarf results (2023-2025) should be cited in the Y dwarf comparison section. This is a significant omission given the paper discusses Y dwarf temperatures and distances that JWST has been actively characterizing.

2. **`\received{\today}` should be `\received{}`**: Using `\today` will embed the compilation date; ApJ editorial sets the received date upon submission.

### MEDIUM (recommended)

3. **Missing Backyard Worlds citation**: Kuchner+ 2017 (the Backyard Worlds: Planet 9 citizen science project) is a major Y dwarf discovery project and should be cited, at minimum in the Introduction or the Y dwarf comparison discussion.

4. **Missing NEOWISE decommissioning context**: NEOWISE was decommissioned in August 2024 after the final data release. This is relevant context for the 10-year baseline discussion and should be noted, possibly with a citation.

5. **Keyword format**: The colon-separated keywords ("infrared: stars", "techniques: photometric") use legacy AAS format. Current AAS journals use the Unified Astronomy Thesaurus (UAT). Verify terms at https://astrothesaurus.org/ and update to modern format (e.g., "Infrared stars", "Photometry").

6. **DOI for Wright 2019**: Check ADS for a DOI on `2019BAAS...51c.389W` (Searches for Technosignatures). It likely exists.

7. **PNG duplicates in figures/**: 6 PNG files (total ~1.9 MB) exist alongside the PDFs. Remove before submission to reduce package size and avoid confusion.

### LOW (optional improvements)

8. **Unreferenced section labels**: 17 section labels are defined but never referenced with `\ref{}`. This is normal but some could potentially support additional cross-references (e.g., "as discussed in Section~\ref{subsec:teff}").

9. **MCMC results JSON in figures/**: `parallax_mcmc_ls_results.json` should be in a `data/` or `supplementary/` directory, not in `figures/`.

10. **Appendix figure synthetic data note**: The JSON shows parallax values (97.2 mas) that differ significantly from Table 1 (57.6 mas). The manuscript correctly states "synthetic data," but a reviewer may question the discrepancy. Consider adding "Note: these illustrative values differ from the observed parallax" to the caption or a footnote.

---

## 6. File Checklist for Submission Package

| Item | Present | Notes |
|------|---------|-------|
| manuscript.tex | YES | 590 lines, aastex701 |
| references.bib | YES | 29 entries, 296 lines |
| fig1_pipeline_flowchart.pdf | YES | 36 KB |
| fig2_allsky_galactic.pdf | YES | 33 KB |
| fig3_color_magnitude.pdf | YES | 23 KB |
| fig4_distributions.pdf | YES | 15 KB |
| fig5_variability.pdf | YES | 25 KB |
| fig6_periodograms.pdf | YES | 47 KB |
| fig_appendix_parallax_mcmc_ls.pdf | YES | 16 KB |
| Machine-readable table (golden_sample_cds.txt) | YES | In tasni_paper_final/ |
| Cover letter | YES | cover_letter.tex |

**Total submission size estimate**: ~200 KB (figures) + ~50 KB (tex/bib) = ~250 KB

---

*Audit completed 2026-02-20. All critical items pass. 2 high-priority, 5 medium-priority, and 3 low-priority recommendations identified.*
