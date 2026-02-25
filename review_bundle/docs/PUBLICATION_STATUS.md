# TASNI Publication Readiness - Final Status Report

**Date**: 2026-02-20
**Version**: 1.0.0
**Status**: CONDITIONAL - HARDENING IN PROGRESS

---

## Hardening Reconciliation (Current)

This file previously declared unconditional readiness. During publication hardening,
that status is treated as provisional until all release gates pass.

### Required Gate Checks Before Final "Ready"

- Statistical integrity: multiple-testing correction and alias handling in operational periodogram path
- ML integrity: no circular proxy-label training in ranking pipeline
- Governance: complete release manifest + checksum verification + external-license coverage
- Reproducibility: clean-room rerun from documented instructions
- CI/reliability: unified workflows with release gates
- Metadata consistency: README/CITATION/pyproject/Zenodo alignment

### Source Of Truth

- Claim ledger: `docs/CLAIM_ARTIFACT_LEDGER.md`
- Baseline snapshot artifact: `output/release/baseline_snapshot.json`

---

## Completed Tasks Summary

### Phase 1: Manuscript Updates
| Task | Status | Details |
|------|--------|---------|
| ORCID ID | ✅ Complete | Updated to 0009-0005-1026-5103 |
| Uncertainty formatting | ✅ Complete | Fixed inconsistencies in abstract and conclusions |
| Rio Scale assessment | ✅ Complete | Added to SETI implications section |

### Phase 2: Publication Figures
| Figure | Status | File |
|--------|--------|------|
| Pipeline flowchart | ✅ Complete | `fig1_pipeline_flowchart.pdf` |
| All-sky Galactic map | ✅ Complete | `fig2_allsky_galactic.pdf` |
| Color-magnitude diagram | ✅ Complete | `fig3_color_magnitude.pdf` |
| Temperature/PM distributions | ✅ Complete | `fig4_distributions.pdf` |
| Variability analysis | ✅ Complete | `fig5_variability.pdf` |
| Periodograms | ✅ Complete | `fig6_periodograms.pdf` |
| Machine-readable table | ✅ Complete | `golden_sample_cds.txt` |

### Phase 3: GitHub Infrastructure
| File | Status | Purpose |
|------|--------|---------|
| `CODEOWNERS` | ✅ Complete | Repository governance |
| `SECURITY.md` | ✅ Complete | Security policy |
| `CODE_OF_CONDUCT.md` | ✅ Complete | Community standards |
| `AUTHORS` | ✅ Complete | Contributor credits |
| Feature request template | ✅ Complete | GitHub issue template |
| CI workflow updated | ✅ Complete | Python 3.11 + 3.12, coverage 10% |

### Phase 4: Code Quality
| Task | Status | Details |
|------|--------|---------|
| Python 3.12 support | ✅ Complete | Added to CI matrix |
| Coverage threshold | ✅ Complete | Raised to 10% |
| mypy configuration | ✅ Complete | Permissive mode for legacy code |

### Phase 5: Data Products
| File | Format | Status |
|------|--------|--------|
| Golden sample | CSV + Parquet | ✅ Complete |
| Parallax data | CSV + Parquet | ✅ Complete |
| Kinematics data | CSV + Parquet | ✅ Complete |
| eROSITA constraints | CSV + Parquet | ✅ Complete |
| Bayesian analysis | CSV + Parquet | ✅ Complete |
| Checksums | TXT | ✅ Complete |

### Phase 6: Documentation
| Document | Status | Location |
|----------|--------|----------|
| Data Availability | ✅ Complete | `docs/DATA_AVAILABILITY.md` |
| Release Preparation | ✅ Complete | `docs/RELEASE_PREPARATION.md` |
| Publication Checklist | ✅ Existing | `docs/publication_checklist.md` |
| Reproducibility Guide | ✅ Existing | `docs/REPRODUCIBILITY_QUICKSTART.md` |

---

## Remaining Tasks (Post-Zenodo)

The following tasks require action after Zenodo deposition:

1. **Update DOI placeholders** in:
   - `README.md` (line 6)
   - `CITATION.cff` (line 5)
   - `.zenodo.json` (line 6)
   - `docs/DATA_AVAILABILITY.md`

2. **Create GitHub release** following steps in `docs/RELEASE_PREPARATION.md`

3. **Submit to ApJ** after arXiv posting

---

## File Structure (Final)

```
tasni/
├── src/tasni/           # 116 Python modules
├── tests/               # 14 test files
├── docs/                # 57+ documentation files
├── data/processed/final/
│   ├── golden_improved.parquet
│   ├── golden_improved.csv
│   ├── golden_improved_parallax.parquet
│   ├── golden_improved_kinematics.parquet
│   ├── golden_improved_erosita.parquet
│   ├── golden_improved_bayesian.parquet
│   └── checksums.txt
├── tasni_paper_final/
│   ├── manuscript.tex
│   ├── references.bib
│   ├── aastex701.cls
│   ├── golden_sample_cds.txt
│   └── figures/
│       ├── fig1_pipeline_flowchart.pdf
│       ├── fig2_allsky_galactic.pdf
│       ├── fig3_color_magnitude.pdf
│       ├── fig4_distributions.pdf
│       ├── fig5_variability.pdf
│       └── fig6_periodograms.pdf
├── CODEOWNERS
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── AUTHORS
├── CITATION.cff
├── LICENSE
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── dockerfile
└── .github/
    ├── workflows/ci.yml
    ├── pull_request_template.md
    └── issue_template/
```

---

## Key Scientific Results

| Metric | Value |
|--------|-------|
| Total WISE sources processed | 747,634,026 |
| WISE orphans (no Gaia) | 406,387,755 |
| After quality filters | 2,371,667 |
| After NIR veto | 62,856 |
| Golden sample | 100 |
| **Fading thermal orphans** | **3** |

### The Three Confirmed Fading Thermal Orphans

| Designation | T_eff (K) | Distance (pc) | Period (days) |
|-------------|-----------|---------------|---------------|
| J143046.35-025927.8 | 293 ± 47 | 17.4 (+3.0/-2.6) | 116.3 |
| J231029.40-060547.3 | 258 ± 38 | 32.6 (+13.3/-8.0) | 178.6 |
| J193547.43+601201.5 | 251 ± 35 | --- | 92.6 |

> **Note:** J044024.40-731441.6 is located within the LMC and excluded from the confirmed sample.

---

## Pre-Publication Review Findings (2026-02-20)

A comprehensive review identified and addressed 25+ issues. Key fixes applied:

### Fixed Issues
| Issue | Severity | Fix Applied |
|-------|----------|-------------|
| LaTeX class: aastex710 → aastex701 | Showstopper | Fixed |
| `\drafttrue` still active | Showstopper | Commented out |
| No `\affiliation` | Showstopper | Added "Independent Researcher" |
| Temperature formula mismatch (3 different formulas) | Showstopper | Manuscript updated to describe actual SED fitting method |
| Periodogram results presented as real when earlier analysis called them aliases | Showstopper | Honestly acknowledged as NEOWISE cadence aliases |
| CDS table 85/100 rows zeros | Showstopper | Regenerated from valid data |
| Abstract claims μ > 100 but J1430 has μ = 55 | Critical | Fixed to "μ = 55--306" |
| LMC source contradiction (parallax vs membership) | Critical | Rewritten with MSX LMC 1152 identification |
| Distance errors inconsistent (two different values) | Critical | Harmonized throughout |
| ML training circularity not disclosed | Critical | Added to Limitations section |
| "Rotational modulation" claim for 90-180 day periods | Critical | Removed (Y dwarf rotation is 2-10 hours) |
| Missing J060501 documentation | High | Added footnote explaining removal |
| Empty Bayesian section | High | Removed (commented out with explanation) |
| Conflicting figure numbering (old + new scheme) | High | Old-scheme files deleted |
| Missing AllWISE/NEOWISE citations | High | Added Cutri 2013 and Mainzer 2014 |
| eROSITA coverage caveat missing | High | Added western hemisphere note |
| Two conflicting .bib files | Medium | Harmonized to single authoritative version |
| Wrong project name in `__init__.py` | Medium | Fixed |
| Only 3 cross-references in 490-line paper | Medium | Increased to 10 |
| Label prefix inconsistencies | Medium | Fixed all `subsubsec` → `subsec` |

### Data-Paper Mismatch - RESOLVED (2026-02-20)
**ROOT CAUSE**: `validation.py` selected top 150 from `ranked_tier5_improved.parquet` (ML output with mostly NaN coordinates) instead of using the correct `tasni_golden_targets.csv` (100 sources with full data).

**FIX**: Regenerated all data products from `tasni_golden_targets.csv`:
- `golden_improved.csv`: 100 rows, 54 columns, zero NaN, all 4 fading sources present
- `golden_improved_kinematics.csv`: 87 rows (PM > 100 mas/yr)
- `golden_improved_erosita.csv`: 100 rows (all X-ray quiet)
- `golden_improved_parallax.csv`: 29 rows (parallax > 5 mas)
- `golden_improved_bayesian.csv`: 10 rows (lowest false positive)
- `golden_sample_cds.txt`: 100 data rows with proper WISE designations
- `validation.py`: Fixed `nlargest(150)` → `nlargest(100)`
- `__main__.py`: Fixed default `top_n=150` → `top_n=100`

## Conclusion

The TASNI project is **ready for publication**. All critical issues from the comprehensive review have been resolved, data products match the manuscript, and all verification checks pass.

**Next Step**: Deposit on Zenodo → submit to ApJ.

---

*Report updated: 2026-02-20 (post-review)*

## Session History

| Session | Date | Commit | Key Changes |
|---------|------|--------|-------------|
| 4 | 2026-02-20 | db5b5fb | Pre-submission audit: 6 parallel agents, fixed \received, Fig3 caption, Table3 errors, UAT keywords, URLs, Zenodo DOI |
| 4 | 2026-02-20 | e0a9ed2 | Added comprehensive audit reports |
| 5 | 2026-02-20 | d0d3552 | Referee-proofing: W1-W2 color tension, Planck SED caveats, trend sign convention, Makefile fixes, 141 tests pass |
| 6 | 2026-02-20 | a7e60bf | ApJ major revision response: ATMO 2020 fitting, blend MC analysis, parallax injection-recovery, ML ablation study, Appendix B |
| 7 | 2026-02-20 | (pending) | Pre-submission bulletproofing: broken refs fixed, README counts corrected, pytest config added |
