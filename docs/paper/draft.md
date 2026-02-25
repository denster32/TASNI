> **SUPERSEDED**: The authoritative manuscript is `tasni_paper_final/manuscript.tex`. This draft is retained for historical reference only.

# TASNI: Thermal Anomaly Search for Non-Communicating Intelligence

**Draft for arXiv submission - Updated 2026-02-15**

---

## Abstract

Current SETI methodology assumes intelligent civilizations broadcast intentionally (Wright 2019). This assumption may be fundamentally flawed. Civilizations—particularly post-biological AI-based ones—may go silent by choice or instinct while still obeying thermodynamic laws. They must dump waste heat. This project systematically searches for infrared thermal signatures in regions where no optical counterpart exists, using existing public datasets (WISE, Gaia) and novel cross-matching methodology. We identify 100 high-priority thermal anomaly candidates and discover four "fading thermal orphans" with effective temperatures of 293±47 K to 466±52 K, distances of 17.4+2.1/-1.8 pc to 30.5+4.3/-3.7 pc, and significant dimming trends (15-53 mmag/yr) over the 10-year NEOWISE baseline. The nearest source at 17.4 pc is one of the nearest room-temperature objects known.

**Keywords:** SETI, technosignatures, infrared astronomy, WISE, Gaia, thermal anomalies, brown dwarfs

---

## 1. Introduction

### 1.1 The Silent Civilization Problem

Traditional SETI assumes aliens want to be found. This anthropomorphizes alien intent.

Key observations:
- Computation requires energy → produces waste heat
- Thermodynamics is inescapable
- A silent civilization is still a warm civilization
- We should look for metabolism, not communication

### 1.2 The Detection Gap

No systematic search exists for:
- Infrared sources without optical counterparts
- Thermal signatures in "empty" space
- Heat anomalies that contradict natural models

### 1.3 Our Approach

Systematic cross-match of WISE (infrared) against Gaia (optical):
- Find sources bright in IR but invisible in optical
- Filter by quality and multi-wavelength vetoes
- Rank by thermal anomaly "weirdness"
- Identify candidates for follow-up

---

## 2. Methodology

### 2.1 Data Sources

| Survey | Band | Depth | Objects |
|--------|------|-------|---------|
| AllWISE | 3.4, 4.6, 12, 22 μm | ~16 mag | 747M | Cutri et al. (2013) |
| Gaia DR3 | optical | ~21 mag | 1.8B | Gaia Collaboration et al. (2023) |
| 2MASS | 1.2, 1.6, 2.2 μm | ~15 mag | 471M |
| Legacy Survey DR10 | optical/NIR | ~23 mag | 1.6B |
| eROSITA DR1 | 0.2-8 keV | ~10⁻¹³ erg/cm²/s | 1M |

### 2.2 Cross-Match Pipeline

```
WISE catalog (747M sources)
    ↓ Cross-match Gaia DR3 (3" radius)
WISE orphans (406M, no optical counterpart)
    ↓ Quality filters (SNR>5, clean flags)
High-quality IR sources (2.37M)
    ↓ Multi-wavelength vetoes (UKIDSS, VHS, CatWISE)
Vetoed survivors (62,856)
    ↓ Legacy Survey optical veto
No optical detection (39,188)
    ↓ eROSITA X-ray veto
X-ray quiet (~4,137)
    ↓ ML scoring & ranking
Golden targets (100)
    ↓ Variability analysis
Fading thermal orphans (4)
```

### 2.3 Temperature Estimation

Effective temperatures are estimated from W1-W2 colors using:

T_eff = 5614 / ((W1 - W2) + 0.98) K

Uncertainties are propagated from photometric errors using Monte Carlo sampling with 1000 iterations. Typical uncertainties are ±35-52 K depending on WISE photometric quality.

### 2.4 Periodogram Analysis

We use the Lomb-Scargle periodogram (Scargle 1982) as implemented in Astropy to analyze NEOWISE single-epoch photometry. The frequency grid spans 1/1000 to 2 day⁻¹ with 10,000 trial frequencies. False Alarm Probability (FAP) is computed using the Baluev (2008) approximation.

### 2.5 Validation and Quality Control

We validate the pipeline through multiple independent checks:

**1. Known Object Recovery**
- Tested on 100+ known Y, T, and late-L dwarfs from literature (Kirkpatrick et al. 2011, Best et al. 2021, Meisner et al. 2020)
- Recovery rate: >90% for sources meeting SNR thresholds

**2. Quality Flag Filtering**
All sources must pass strict WISE quality criteria:
- `cc_flags = '0000'`: No contamination from artifacts, diffraction spikes, or ghosts
- `ext_flg = 0`: Point sources only (rejects extended galaxies, nebulae)
- `ph_qual` in W1/W2: A or B grade photometry only
- SNR >= 5 in both W1 and W2 bands

**3. Multi-Epoch Persistence**
- Real astrophysical sources persist across 250+ NEOWISE epochs (2013-2024)
- Artifacts (cosmic rays, saturation bleeds, optical ghosts) do not produce consistent photometry over 10+ years
- Minimum 10 epochs required for variability classification

**4. Independent Catalog Verification**
- CatWISE2020 cross-match confirms independent pipeline detection
- Proper motion measurements (55-359 mas/yr) confirm real, nearby objects

**5. Visual Inspection**
- All 100 golden targets examined via AllWISE, unWISE, and Legacy Survey cutouts
- Human vetting system with artifact classifications receiving -1000 score penalty

---

## 3. Results

### 3.1 Filtering Statistics

| Phase | Sources | Reduction |
|-------|---------|-----------|
| WISE catalog | 747,000,000 | — |
| No Gaia match | 406,387,755 | 54.6% |
| Quality filter | 2,371,667 | 99.4% |
| No NIR detection | 62,856 | 97.3% |
| No optical detection | 39,188 | 37.6% |
| Multi-wavelength quiet | 4,137 | 89.5% |
| Golden targets | 100 | 97.6% |
| **Fading thermal orphans** | **4** | **96.0%** |

### 3.2 The Four Fading Thermal Orphans

| Designation | T_eff (K) | Distance (pc) | μ (mas/yr) | Fade (mmag/yr) |
|-------------|-----------|---------------|------------|----------------|
| J143046.35-025927.8 | 293 ± 47 | 17.4 +3.0/-2.6 | 55 ± 5 | 25.5 ± 2.1 |
| J044024.40-731441.6 | 466 ± 52 | 30.5 +1.3/-1.2 | 165 ± 17 | 16.1 ± 3.2 |
| J231029.40-060547.3 | 258 ± 38 | 32.6 +13.3/-8.0 | 165 ± 17 | 52.6 ± 4.3 |
| J193547.43+601201.5 | 251 ± 35 | --- | 306 ± 31 | 22.9 ± 1.8 |

### 3.3 Periodicity Analysis

We searched for periodic signals using Lomb-Scargle periodograms across the 10-year NEOWISE baseline:

- **No significant short-period (P < 30 days) variability detected** in any fading source
- Detected signals at 90-180 days correspond to **NEOWISE sampling aliases** (harmonics of the ~182-day orbital cadence)
- Known Y dwarf rotation periods are 2-10 hours, ruling out these long-period signals as rotational modulation

**Conclusion:** The fading trend is statistically significant, but the apparent "periodicity" is due to NEOWISE's observing cadence, not intrinsic variability.

### 3.4 X-ray Constraints

eROSITA DR1 cross-match shows that none of the 59 sources within the eROSITA footprint (western Galactic hemisphere) have X-ray detections, effectively ruling out AGN or stellar coronal activity.

### 3.5 Prior Catalog Cross-Check

Cross-matching all 100 golden targets against SIMBAD and VizieR with a 5" cone search radius returns **zero matches**. None are previously catalogued as known brown dwarfs, variable stars, AGN, or any other astronomical object class. This confirms all golden targets are novel discoveries.

### 3.6 Distance Measurements

Significant parallax detections (>5 mas) for 29/100 sources:
- Mean distance: 27.3 ± 3.1 pc (for detected sources)
- Nearest: J1430 at 17.4 +2.1/-1.8 pc

---

## 4. Discussion

### 4.1 What Are These Sources?

Natural explanations:

1. **Y dwarfs**: T_eff = 293-466 K is consistent with late-T/Y transition. The observed fading could be due to atmospheric variability, evolving cloud decks, or binary eclipses.

2. **Edge-on disk systems**: A debris disk viewed edge-on could produce gradual dimming while remaining bright in the mid-IR.

3. **Young stellar objects**: Unlikely due to high Galactic latitudes of most sources.

4. **Eclipsing binary brown dwarfs**: Two equal-mass brown dwarfs in an edge-on orbit would exhibit periodic dimming.

### 4.2 Causes of the Fading Trend

The observed fading (15-53 mmag/yr) could arise from several mechanisms:

- **Atmospheric variability**: Evolving cloud properties in cool brown dwarf atmospheres
- **Binary eclipses**: Eclipsing binary brown dwarf systems would show periodic dimming
- **Edge-on disk systems**: A debris disk viewed edge-on could produce gradual dimming

**Note:** While secular cooling (brown dwarfs radiating away formation heat) occurs on 10⁸-10⁹ year timescales and cannot explain the observed year-scale fading, we cannot distinguish between the above mechanisms without spectroscopic follow-up.

### 4.3 Implications for SETI

While artificial origins cannot be ruled out, natural astrophysical explanations (Y dwarfs, binary systems) are more parsimonious. The methodology remains valuable for identifying anomalous thermal sources for follow-up study.

### 4.4 Limitations

- Spatial resolution: 6" WISE PSF limits source separation
- Photometric precision: 0.05-0.1 mag NEOWISE uncertainties
- Spectroscopic confirmation: Required for definitive classification

### 4.5 Future Work

1. Spectroscopic follow-up with JWST/NIRSpec (proposed)
2. Enhanced proper motion from Gaia DR4
3. Expanded search with Roman Space Telescope
4. Real-time NEOWISE monitoring

---

## 5. Conclusion

We present TASNI, a systematic search for thermal anomalies:

1. **100 golden candidates** from 747M WISE sources
2. **4 fading thermal orphans** with T_eff = 293-466 K
3. **Nearest at 17.4 pc** - one of the nearest room-temperature objects
4. **Significant fading trends** (15-53 mmag/yr) over the 10-year NEOWISE baseline
5. **All 59 sources in eROSITA footprint are X-ray quiet**, ruling out AGN

The most likely interpretation is previously undiscovered Y dwarfs or eclipsing binary brown dwarf systems. Spectroscopic follow-up is essential.

---

## Acknowledgments

This research has made use of data from:
- NASA/IPAC Infrared Science Archive (IRSA)
- ESA Gaia Archive
- LAMOST DR12 (National Astronomical Data Center, China)
- Legacy Survey DR10 (NOIRLab)
- eROSITA DR1 (Max Planck Institute for Extraterrestrial Physics)

---

## References

### Survey Documentation

- **Cutri et al. (2013)**: Explanatory Supplement to the AllWISE Data Release Products. *Technical Report, IPAC*.
- **Gaia Collaboration et al. (2023)**: Gaia Data Release 3. Summary of the contents and survey properties. *A&A*, 674, A1.

### Brown Dwarf Background

- **Kirkpatrick et al. (2011)**: The First Hundred Brown Dwarfs Discovered by WISE. *ApJS*, 197, 19.
- **Cushing et al. (2011)**: The Discovery of Y Dwarfs using WISE. *ApJ*, 743, 50.
- **Luhman (2014)**: Discovery of a ~250 K Brown Dwarf at 2 pc. *ApJL*, 786, L18.

### Technosignature Framework

- **Wright (2019)**: Searches for Technosignatures in Astronomy and Astrophysics. *BAAS*, 51, 389.
- **Forgan et al. (2019)**: Rio 2.0: revising the Rio scale for SETI detections. *IJAsB*, 18, 336.

### Theoretical Framework

- **Dyson (1960)**: Search for Artificial Stellar Sources of Infrared Radiation. *Science*, 131, 1667.
- **Yu (2015)**: The Dark Forest Rule: One Solution to the Fermi Paradox. *JBIS*, 68, 142.

### Methods

- **Scargle (1982)**: Studies in astronomical time series analysis. II. *ApJ*, 263, 835.
- **Baluev (2008)**: Assessing the statistical significance of periodogram peaks. *MNRAS*, 385, 1279.

---

*Code and data available at: https://github.com/dpalucki/tasni*

*Author: Palucki, Dennis*
