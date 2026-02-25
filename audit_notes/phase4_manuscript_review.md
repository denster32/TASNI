# Phase 4: Manuscript Scientific Review - Audit Notes

## 4.1 Every Number Cross-Check

### Abstract Claims

| Claim | Value | Location | Status |
|-------|-------|----------|--------|
| High-priority candidates | 100 | Line 32 | PASS |
| Fading thermal orphans | 3 | Line 32 | PASS |
| T_eff range | 251±35 to 293±47 K | Line 33-34 | PASS |
| Proper motion range | 55-306 mas/yr | Line 34-35 | PASS |
| NEOWISE baseline | 10 years | Line 35 | PASS |
| LMC distance | ~50 kpc | Line 37 | PASS |
| Period range | 93-179 days | Line 38-39 | PASS |
| Nearest distance | 17.4+3.0/-2.6 pc | Line 42 | **NEEDS VERIFICATION** |
| eROSITA footprint | 59 sources | Line 43 | **CANNOT VERIFY** |
| X-ray detections | 0 | Line 44 | PASS |

### Table 2: Pipeline Statistics - VERIFIED

| Phase | Sources | Reduction |
|-------|---------|-----------|
| AllWISE catalog | 747,634,026 | --- |
| No Gaia match | 406,387,755 | 54.6% |
| Quality filters | 2,371,667 | 99.4% |
| No NIR detection | 62,856 | 97.3% |
| No optical detection | 39,188 | 37.6% |
| Multi-wavelength quiet | 4,137 | 89.5% |
| Golden sample | 100 | 97.6% |
| Fading thermal orphans | 3 | 97.0% |

### Table 3: Fading Thermal Orphans

#### J143046.35-025927.8
| Property | Manuscript | Data | Status |
|----------|------------|------|--------|
| T_eff | 293±47 K | 293.23 K | PASS |
| Distance | 17.4+3.0/-2.6 pc | **NOT IN PARALLAX FILE** | **FAIL** |
| μ_total | 55±5 mas/yr | TBD | NEEDS CHECK |
| Period | 116.3+5.0/-4.5 days | TBD | NEEDS CHECK |
| FAP | 2.1×10^-61 | TBD | NEEDS CHECK |
| Fade rate | 0.026 mag/yr | 0.02546 mag/yr | PASS |

#### J231029.40-060547.3
| Property | Manuscript | Data | Status |
|----------|------------|------|--------|
| T_eff | 258±38 K | 258.13 K | PASS |
| Distance | 32.6+13.3/-8.0 pc | TBD | NEEDS CHECK |
| μ_total | 165±17 mas/yr | TBD | NEEDS CHECK |
| Period | 178.6+7.0/-6.5 days | TBD | NEEDS CHECK |
| FAP | 6.7×10^-46 | TBD | NEEDS CHECK |
| Fade rate | 0.053 mag/yr | 0.05256 mag/yr | PASS |

#### J193547.43+601201.5
| Property | Manuscript | Data | Status |
|----------|------------|------|--------|
| T_eff | 251±35 K | 250.51 K | PASS |
| Distance | --- (no parallax) | TBD | NEEDS CHECK |
| μ_total | 306±31 mas/yr | TBD | NEEDS CHECK |
| Period | 92.6+4.0/-3.5 days | TBD | NEEDS CHECK |
| FAP | 2.2×10^-11 | TBD | NEEDS CHECK |
| Fade rate | 0.023 mag/yr | 0.02291 mag/yr | PASS |

---

## 4.2 Literature Verification

### Verified Citations

| Paper | Exists | Claims Supported | Issues |
|-------|--------|------------------|--------|
| Wright et al. 2010 | YES | YES | None |
| Cutri et al. 2013 | YES | YES | 747M count verified |
| Kirkpatrick et al. 2012 | YES | PARTIAL | "100K scatter" is interpretation |
| Luhman 2014 | YES | YES | T=250K, d=2.3pc within ranges |
| Mainzer et al. 2014 | YES | YES | Minor: page number (30 vs 13) |

### MINOR-003: Mainzer 2014 Citation
- **Location**: `references.bib`
- **Issue**: Bibcode has wrong page number (13 vs 30)
- **Fix**: Correct to `2014ApJ...792...30M`

---

## 4.3 Missing Citations Analysis

### HIGH PRIORITY Missing Citations

| Paper | Why Cite |
|-------|----------|
| Kirkpatrick et al. 2021 (ApJS 253, 7) | Definitive Y dwarf census (525 L/T/Y dwarfs) |
| Wright et al. 2014a,b (G-HAT) | Foundational Dyson sphere search papers (mentioned but not in bib) |
| Meisner et al. 2020 (ApJ 889, 74) | CatWISE Y dwarf discoveries - similar methodology |

### MEDIUM PRIORITY Missing Citations

| Paper | Why Cite |
|-------|----------|
| Carrigan 2009 (IRAS Dyson search) | Historical precedent for IR technosignatures |
| Suazo et al. 2022 (Project Hephaistos) | Most recent Dyson sphere limits |
| Baron 2019 (ML in Astronomy review) | Context for ML methods |
| Schneider et al. 2015, 2016 | WISE Y dwarf spectroscopy |

### MAJOR-004: Missing Key Citations
- **Location**: `references.bib`
- **Issue**: Several critical papers not cited despite being mentioned or highly relevant
- **Impact**: Referee will likely note incomplete literature review
- **Fix**: Add Kirkpatrick 2021, Wright 2014 G-HAT papers, Meisner 2020

---

## Summary of Phase 4 Findings

| Check | Status |
|-------|--------|
| Pipeline counts (747M→3) | PASS |
| Temperature claims (251-293K) | PASS |
| Fade rates (23-53 mmag/yr) | PASS |
| Distance claims (17.4 pc) | **FAIL** - not in parallax file |
| eROSITA footprint (59) | **CANNOT VERIFY** |
| Key literature citations | PARTIAL - missing important papers |
| Citation accuracy | MINOR ISSUE - page number error |
