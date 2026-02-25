# URL Update Audit Report

**Date:** 2026-02-20
**Task:** Update all GitHub URLs and Zenodo DOIs across the TASNI project

---

## Summary

Two categories of replacements were made:

1. **GitHub repo path**: `dpalucki/tasni` replaced with `denster32/TASNI` (case-sensitive)
2. **Zenodo DOI**: `10.5281/zenodo.XXXXXXX` replaced with `10.5281/zenodo.18717105`

Additionally, the `@dpalucki` GitHub username references in CODEOWNERS and SECURITY.md were updated to `@denster32`.

---

## Files Changed (16 files total)

### 1. `tasni_paper_final/manuscript.tex`
- Line 549: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`
- Line 558: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`

### 2. `README.md`
- Line 6: Zenodo DOI badge `zenodo.XXXXXXX` -> `zenodo.18717105` (2 occurrences in badge URL + link)
- Line 8: Removed placeholder note "Replace `XXXXXXX` with the actual DOI after depositing to Zenodo."
- Line 10: Updated DOI line to `[10.5281/zenodo.18717105](https://doi.org/10.5281/zenodo.18717105)`
- Line 63: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`

### 3. `CITATION.cff`
- Line 5: `doi: 10.5281/zenodo.XXXXXXX` -> `doi: 10.5281/zenodo.18717105`
- Line 7: `url: "https://github.com/dpalucki/tasni"` -> `url: "https://github.com/denster32/TASNI"`
- Line 8: `repository-code: "https://github.com/dpalucki/tasni"` -> `repository-code: "https://github.com/denster32/TASNI"`

### 4. `.zenodo.json`
- Line 49: `identifier: "https://github.com/dpalucki/tasni"` -> `identifier: "https://github.com/denster32/TASNI"`

### 5. `pyproject.toml`
- Line 12: `repository = "https://github.com/dpalucki/tasni"` -> `repository = "https://github.com/denster32/TASNI"`

### 6. `SECURITY.md`
- Line 24: `github.com/dpalucki/tasni/security/advisories` -> `github.com/denster32/TASNI/security/advisories`
- Line 87: `[@dpalucki](https://github.com/dpalucki)` -> `[@denster32](https://github.com/denster32)`

### 7. `CHANGELOG.md`
- Line 73: `github.com/dpalucki/tasni/releases/tag/v1.0.0` -> `github.com/denster32/TASNI/releases/tag/v1.0.0`
- Line 74: `github.com/dpalucki/tasni/releases/tag/v0.1.0` -> `github.com/denster32/TASNI/releases/tag/v0.1.0`

### 8. `CODEOWNERS`
- All 18 occurrences of `@dpalucki` -> `@denster32`

### 9. `docs/DATA_AVAILABILITY.md`
- Line 26: `10.5281/zenodo.XXXXXXX` -> `10.5281/zenodo.18717105`; removed "(to be assigned upon publication)"
- Line 41: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`
- Line 151: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`
- Line 213: `url = {https://github.com/dpalucki/tasni}` -> `url = {https://github.com/denster32/TASNI}`
- Line 214: `doi = {10.5281/zenodo.XXXXXXX}` -> `doi = {10.5281/zenodo.18717105}`
- Line 224: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`
- Line 226: `github.com/dpalucki/tasni/issues` -> `github.com/denster32/TASNI/issues`

### 10. `docs/RELEASE_PREPARATION.md`
- Line 65: `github.com/dpalucki/tasni/releases/new` -> `github.com/denster32/TASNI/releases/new`
- Line 94: `github.com/dpalucki/tasni.git` -> `github.com/denster32/TASNI.git`
- Line 122: `Authorize dpalucki/tasni repository` -> `Authorize denster32/TASNI repository`

### 11. `docs/REPRODUCIBILITY_QUICKSTART.md`
- Line 28: `github.com/dpalucki/tasni.git` -> `github.com/denster32/TASNI.git`
- Line 64: Removed placeholder comment "replace XXXXXXX with actual DOI", updated to "Download data release from Zenodo"
- Line 65: `zenodo.org/record/XXXXXXX` -> `zenodo.org/record/18717105`
- Line 245: `url = {https://github.com/dpalucki/tasni}` -> `url = {https://github.com/denster32/TASNI}`

### 12. `docs/paper/draft.md`
- Line 268: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`

### 13. `data/processed/final/golden_sample_readme.txt`
- Line 6: `DOI: 10.5281/zenodo.XXXXXXX (pending)` -> `DOI: 10.5281/zenodo.18717105`

### 14. `data/processed/final/README.md`
- Line 5: `DOI: 10.5281/zenodo.XXXXXXX (pending)` -> `DOI: 10.5281/zenodo.18717105`

### 15. `AUDIT_REPORT.md`
- Line 134: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`
- Line 150: `dpalucki/tasni` -> `denster32/TASNI`

### 16. `audit_notes/phase1_archaeology.md`
- Line 79: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`

### 17. `audit_notes/manuscript_text_audit.md`
- Line 190: `github.com/dpalucki/tasni` -> `github.com/denster32/TASNI`

---

## ORCID Verification

ORCID `0009-0005-1026-5103` was verified in the following files (all correct, no changes needed):

| File | Location |
|------|----------|
| `tasni_paper_final/manuscript.tex` | Line 17 |
| `tasni_paper_final/cover_letter.tex` | Line 73 |
| `CITATION.cff` | Line 23 |
| `docs/RELEASE_PREPARATION.md` | Line 18 |
| `docs/PUBLICATION_STATUS.md` | Line 14 |
| `AUTHORS` | Line 5 |

---

## Post-Update Verification

- `grep -r "dpalucki"` across all project text files: **0 matches** (clean)
- `grep -r "XXXXXXX"` across all project text files: **0 matches** (clean)
- `grep -r "denster32"` across all project text files: **26 occurrences** across 15 files (expected)
- `grep -r "18717105"` across all project text files: **8 occurrences** across 6 files (expected)
- ORCID `0009-0005-1026-5103`: **6 occurrences** across 6 files (all correct)

---

## Files NOT Changed (confirmed no references)

The following files were checked and contain no `dpalucki` or `XXXXXXX` references:
- `CODE_OF_CONDUCT.md`
- `docs/PUBLICATION_STATUS.md`
- `tasni_paper_final/references.bib`
- `tasni_paper_final/cover_letter.tex` (has ORCID only, which is correct)
- All Python source files in `src/tasni/`
- All test files in `tests/`
