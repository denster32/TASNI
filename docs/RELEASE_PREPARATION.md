# TASNI Release v1.0.0 Preparation

**Release Date**: 2026-02-15
**Status**: Ready for Release

---

## Pre-Release Checklist

### Code Quality
- [x] All tests passing (pytest)
- [x] Linting clean (ruff)
- [x] Type checking configured (mypy in permissive mode)
- [x] CI/CD passing on main branch
- [x] Documentation updated

### Manuscript
- [x] ORCID updated (0009-0005-1026-5103)
- [x] Abstract under 250 words
- [x] All figures generated
- [x] Tables formatted
- [x] References complete
- [x] Rio Scale assessment added
- [x] Data/Code availability statements

### Data Products
- [x] Golden sample (100 sources) - CSV and Parquet
- [x] Parallax measurements - CSV and Parquet
- [x] Kinematics data - CSV and Parquet
- [x] eROSITA constraints - CSV and Parquet
- [x] Bayesian analysis - CSV and Parquet
- [x] Checksums generated

### GitHub Repository
- [x] README.md complete
- [x] LICENSE (MIT)
- [x] CHANGELOG.md
- [x] CITATION.cff with ORCID
- [x] CODEOWNERS
- [x] SECURITY.md
- [x] CODE_OF_CONDUCT.md
- [x] CONTRIBUTING.md
- [x] PR template
- [x] Issue templates
- [x] CI workflow with Python 3.11/3.12

---

## GitHub Release Steps

### 1. Create Release Tag
```bash
git tag -a v1.0.0 -m "TASNI v1.0.0: Initial public release

- Discovery of 4 fading thermal orphans
- 100 golden candidates identified
- Complete cross-match pipeline
- ML ensemble classification
- Bayesian population inference"

git push origin v1.0.0
```

### 2. Create GitHub Release
Go to: https://github.com/dpalucki/tasni/releases/new

**Title**: TASNI v1.0.0 - Initial Public Release

**Description**:
```markdown
## TASNI: Thermal Anomaly Search for Non-communicating Intelligence

### Key Scientific Results

- **4 Fading Thermal Orphans** discovered with temperatures 251-466 K
- **100 golden candidates** from 747 million WISE sources
- Nearest object at 17.4 pc - one of the nearest room-temperature objects known
- 95% X-ray quiet, ruling out AGN

### Data Products

| File | Description |
|------|-------------|
| `golden_improved.parquet` | 100 top candidates |
| `golden_improved_parallax.parquet` | 75 distance measurements |
| `golden_improved_kinematics.parquet` | Proper motion data |
| `golden_improved_erosita.parquet` | X-ray constraints |

### Installation

```bash
pip install tasni
# or
git clone https://github.com/dpalucki/tasni.git
cd tasni && poetry install
```

### Citation

If you use TASNI, please cite:

```bibtex
@article{tasni2026,
    author = {{Palucki, Dennis}},
    title = "{TASNI: Thermal Anomaly Search for Non-communicating Intelligence}",
    journal = {The Astrophysical Journal},
    year = {2026}
}
```

### Full Changelog

See [CHANGELOG.md](CHANGELOG.md)
```

---

## Zenodo Deposit Steps

### 1. Automatic Deposit (Recommended)
- Link GitHub to Zenodo: https://zenodo.org/account/settings/github/
- Authorize dpalucki/tasni repository
- Create release on GitHub
- Zenodo will automatically create DOI

### 2. Manual Deposit (Alternative)
1. Go to https://zenodo.org/deposit
2. Create new upload
3. Upload data archive:
   - `data/processed/final/*.parquet`
   - `tasni_paper_final/figures/*.pdf`
   - `tasni_paper_final/manuscript.pdf` (compile first)
4. Fill metadata from `.zenodo.json`
5. Publish

### 3. Post-Deposit
- Update DOI in:
  - `README.md` (line 6)
  - `CITATION.cff` (line 5)
  - `.zenodo.json` (line 6)
  - `data/processed/final/README.md`
- Commit and push as v1.0.1

---

## File Manifest for Release

### Source Code
```
src/tasni/
├── __init__.py
├── __main__.py
├── core/           (3 files)
├── pipeline/       (7 files)
├── crossmatch/     (9 files)
├── analysis/       (20 files)
├── ml/             (7 files)
├── validation/     (5 files)
├── generation/     (8 files)
├── filtering/      (3 files)
├── download/       (6 files)
├── utils/          (9 files)
└── legacy/         (17 files, deprecated)
```

### Data Products
```
data/processed/final/
├── golden_improved.parquet (100 rows)
├── golden_improved.csv
├── golden_improved_parallax.parquet (75 rows)
├── golden_improved_kinematics.parquet
├── golden_improved_erosita.parquet
├── golden_improved_bayesian.parquet
├── checksums.txt
└── README.md
```

### Manuscript
```
tasni_paper_final/
├── manuscript.tex
├── references.bib
├── aastex701.cls
├── figures/
│   ├── fig1_pipeline_flowchart.pdf
│   ├── fig2_allsky_galactic.pdf
│   ├── fig3_color_magnitude.pdf
│   ├── fig4_distributions.pdf
│   ├── fig5_variability.pdf
│   └── fig6_periodograms.pdf
└── golden_sample_cds.txt
```

---

## Post-Release Tasks

1. [ ] Announce on social media (Twitter, etc.)
2. [ ] Post to arXiv (astro-ph.EP)
3. [ ] Submit to ApJ
4. [ ] Update project website
5. [ ] Create documentation site (Read the Docs)

---

**Release Prepared By**: Dennis Palucki
**Date**: 2026-02-15
