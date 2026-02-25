# Data Availability Statement

**TASNI: Thermal Anomaly Search for Non-communicating Intelligence**

This document provides the formal data availability statement for the TASNI project, compliant with AAS Journal and FAIR (Findable, Accessible, Interoperable, Reusable) data principles.

---

## Summary

All data products from this research are publicly available. The primary data products include:

1. **Golden Sample Catalog**: 100 high-priority thermal anomaly candidates
2. **Parallax Measurements**: NEOWISE distance estimates for 67 sources (significant parallax >5 mas)
3. **Kinematics Data**: Proper motion measurements for all candidates
4. **X-ray Constraints**: eROSITA cross-match results
5. **Bayesian Analysis**: False positive probability estimates
6. **Publication Figures**: All figures in PDF and PNG formats

---

## Primary Data Repository

### Zenodo Archive

**DOI**: `10.5281/zenodo.18717105`

The complete data release is archived on Zenodo under the MIT license. The archive includes:

| File | Size | Format | Description |
|------|------|--------|-------------|
| `golden_improved.parquet` | ~50 KB | Apache Parquet | Main golden sample (100 sources) |
| `golden_improved_parallax.parquet` | ~40 KB | Apache Parquet | Parallax measurements (67 sources) |
| `golden_improved_kinematics.parquet` | ~45 KB | Apache Parquet | Proper motion data |
| `golden_improved_erosita.parquet` | ~30 KB | Apache Parquet | X-ray constraints |
| `golden_improved_bayesian.parquet` | ~35 KB | Apache Parquet | Bayesian FP estimates |
| `figures.tar.gz` | ~15 MB | PDF/PNG | Publication figures |

### GitHub Repository

**URL**: https://github.com/denster32/TASNI

The source code and analysis pipeline are available under the MIT license. The repository includes:

- Complete Python pipeline (`src/tasni/`)
- Test suite (`tests/`)
- Documentation (`docs/`)
- Docker configuration (`dockerfile`)

---

## Source Data (External Archives)

This research uses data from publicly available surveys. Below are the access points and citations:

### WISE/NEOWISE

- **Archive**: NASA/IPAC Infrared Science Archive (IRSA)
- **URL**: https://irsa.ipac.caltech.edu/Missions/wise.html
- **Citation**: Wright et al. (2010), AJ, 140, 1868
- **Version**: AllWISE + NEOWISE Reactivation (through 2024)
- **Access**: Public, no registration required

### Gaia DR3

- **Archive**: ESA Gaia Archive
- **URL**: https://gea.esac.esa.int/archive/
- **Citation**: Gaia Collaboration et al. (2023), A&A, 674, A1
- **Version**: Data Release 3
- **Access**: Public, no registration required

### LAMOST DR12

- **Archive**: National Astronomical Data Center (China)
- **URL**: http://www.lamost.org/dr12/
- **Citation**: Cui et al. (2012), RAA, 12, 1197
- **Access**: Public, registration recommended

### Legacy Survey DR10

- **Archive**: NOIRLab Astro Data Lab
- **URL**: https://www.legacysurvey.org/dr10/
- **Citation**: Dey et al. (2019), AJ, 157, 168
- **Access**: Public, no registration required

### eROSITA DR1

- **Archive**: Max Planck Institute for Extraterrestrial Physics
- **URL**: https://erosita.mpe.mpg.de/dr1/
- **Citation**: Merloni et al. (2024), A&A, 682, A34
- **Access**: Public (EDR), registration required

### Sonora Cholla Models

- **Archive**: UCSB Brown Dwarf Research
- **URL**: https://zenodo.org/record/5063476
- **Citation**: Marley et al. (2021), ApJ, 920, 85
- **Access**: Public
- **Local attribution bundle**: `data/external/sonora_cholla/README.md` and `data/external/sonora_cholla/LICENSE`

### CatWISE2020

- **Archive**: NASA/IPAC Infrared Science Archive (IRSA)
- **URL**: https://irsa.ipac.caltech.edu/Missions/catwise.html
- **Citation**: Marocco et al. (2021), ApJS, 253, 8; Meisner et al. (2020), ApJ, 889, 74
- **Version**: CatWISE2020
- **Access**: Public, no registration required
- **Usage**: Proper motions and vetos for golden sample selection

---

## Derived Data Products

### Golden Sample Schema

The golden sample catalog (`golden_improved.parquet`) contains the following columns:

```
designation         : string   - WISE designation (JHHMMSS.ss+DDMMSS.s)
ra                  : float64  - Right Ascension (J2000, degrees)
dec                 : float64  - Declination (J2000, degrees)
w1mpro              : float32  - W1 magnitude (Vega, 3.4 micron)
w2mpro              : float32  - W2 magnitude (Vega, 4.6 micron)
w1sigmpro           : float32  - W1 uncertainty (mag)
w2sigmpro           : float32  - W2 uncertainty (mag)
T_eff_K             : float32  - Effective temperature (K)
T_eff_err           : float32  - Temperature uncertainty (K)
pmra                : float32  - Proper motion in RA (mas/yr)
pmdec               : float32  - Proper motion in Dec (mas/yr)
pm_total            : float32  - Total proper motion (mas/yr)
improved_composite_score : float32  - ML ensemble score (0-1)
variability_flag    : string   - FADING, STABLE, or VARIABLE
distance_pc         : float32  - Distance estimate (pc)
distance_err        : float32  - Distance uncertainty (pc)
has_parallax        : bool     - Has significant parallax detection
has_erosita         : bool     - Has eROSITA counterpart
```

### Data Integrity

Full checksums (including CSV exports and readme) are in `data/processed/final/checksums.txt`. Canonical parquet checksums:

```
9bd4b7010944a5bd0daeeb13701d9231db2ddd19b671ce276613a966e9ad3f1d  golden_improved.parquet
616de960ec37df20b50bbe6929b7ad381fa7ed9eac818b9c5d86c3ab403f9b01  golden_improved_bayesian.parquet
013c537b3d4ff5dd4b2e9998cd873e8cea92042083ebc9f154142d0af7bb8694  golden_improved_erosita.parquet
762eb7fe9d286db94fab39490434f86b646252aa5fc9313022a0926ff19e59c0  golden_improved_kinematics.parquet
d0aa1e4bda7e317dc0b81db3ef133198793feb03088d5d5075f516c6b93c5e69  golden_improved_parallax.parquet
```

Verify with: `sha256sum -c data/processed/final/checksums.txt`

---

## Reproducibility

### Pipeline Reproduction

The complete analysis pipeline can be reproduced from source data:

1. Clone the repository: `git clone https://github.com/denster32/TASNI.git`
2. Install dependencies: `poetry install`
3. Follow the [Reproducibility Quickstart](REPRODUCIBILITY_QUICKSTART.md)

**Estimated runtime**: 8-12 hours for full pipeline (requires ~500GB storage)

### Docker Reproduction

For guaranteed reproducibility:

```bash
docker build -t tasni:latest .
docker run -v $(pwd)/data:/data tasni:latest tasni pipeline all
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.11+ | 3.12 |
| RAM | 16 GB | 64 GB |
| Storage | 500 GB | 2 TB |
| CPU | 4 cores | 16+ cores |

---

## License

All data products are released under the **MIT License**:

> Copyright (c) 2026 Dennis Palucki
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software...

See [LICENSE](../LICENSE) for full text.

---

## Citation

If you use TASNI data or code, please cite:

```bibtex
@article{tasni2026,
    author = {{Palucki, Dennis}},
    title = "{TASNI: Thermal Anomaly Search for Non-communicating Intelligence}",
    journal = {The Astrophysical Journal},
    year = {2026},
    volume = {},
    pages = {},
    doi = {}
}

@software{tasni_code2026,
    author = {{Palucki, Dennis}},
    title = {TASNI: Thermal Anomaly Search for Non-communicating Intelligence},
    version = {1.0.0},
    year = {2026},
    url = {https://github.com/denster32/TASNI},
    doi = {10.5281/zenodo.18717105}
}
```

---

## Contact

**Corresponding Author**: Dennis Palucki (paluckide@yahoo.com)

**Project Repository**: https://github.com/denster32/TASNI

**Issue Tracker**: https://github.com/denster32/TASNI/issues

---

**Document Version**: 1.0
**Last Updated**: 2026-02-15
**Status**: Ready for Publication
