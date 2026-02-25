# TASNI Paper - Methods Section

## 2. Methods

We developed the Thermal Anomaly Search for Non-communicating Intelligence (TASNI) pipeline to systematically identify infrared-bright sources that lack counterparts across the electromagnetic spectrum. Our approach leverages multi-wavelength archival data to isolate objects detectable only in the mid-infrared, then applies temporal analysis to characterize their long-term behavior. Below we describe our data sources, selection criteria, cross-matching strategy, and variability analysis methodology.

### 2.1 Data Sources

#### 2.1.1 Primary Infrared Catalog

Our parent sample is drawn from the **AllWISE Source Catalog** (Wright et al. 2010; Cutri et al. 2013), which contains 747,634,026 sources observed by the Wide-field Infrared Survey Explorer (WISE) during its cryogenic and post-cryogenic mission phases (2010-2011). AllWISE provides photometry in four mid-infrared bands: W1 (3.4 μm), W2 (4.6 μm), W3 (12 μm), and W4 (22 μm), with 5σ point source sensitivities of 0.068, 0.098, 0.86, and 5.4 mJy, respectively.

For temporal analysis, we utilize the **NEOWISE Reactivation Single-Exposure Source Table** (Mainzer et al. 2014), which provides multi-epoch photometry from 2013 to present. The combination of AllWISE and NEOWISE-R data enables variability analysis over a ~10-year baseline with typically 250-500 epochs per source at 6-month cadence.

#### 2.1.2 Optical and Near-Infrared Veto Catalogs

To identify sources lacking optical/NIR counterparts, we cross-match against:

- **Gaia DR3** (Gaia Collaboration 2023): 1.8 billion sources with G < 21 mag, providing optical photometry and astrometry. We use Gaia proper motions when available.

- **2MASS Point Source Catalog** (Skrutskie et al. 2006): 470 million sources with J, H, Ks photometry to Ks ≈ 14.3 mag.

- **Pan-STARRS DR1** (Chambers et al. 2016): 3 billion detections in grizy bands covering the sky north of δ = -30° to depths of g ≈ 23.3, r ≈ 23.2 mag.

- **DESI Legacy Imaging Surveys DR10** (Dey et al. 2019): Deep optical imaging (g ≈ 24.7, r ≈ 23.9, z ≈ 23.0 mag) covering 14,000 deg².

#### 2.1.3 Radio and X-ray Veto Catalogs

To exclude active galactic nuclei (AGN) and other high-energy sources:

- **NVSS** (Condon et al. 1998): 1.4 GHz radio survey covering δ > -40° with rms ≈ 0.45 mJy/beam.

- **ROSAT All-Sky Survey** (Voges et al. 1999): X-ray catalog (0.1-2.4 keV) for excluding coronal emitters and AGN.

#### 2.1.4 Spectroscopic Catalogs

We cross-match against spectroscopic surveys to identify previously classified objects:

- **LAMOST DR7** (Cui et al. 2012): Low-resolution (R ~ 1800) optical spectra for 10+ million sources, providing stellar classifications and radial velocities.

- **SIMBAD** (Wenger et al. 2000): Astronomical database queried with 5" cone search radius. We check against all object types, categorizing matches as known stars, brown dwarfs (BD*, L*, T*, Y* types), galaxies, or other classifications.

- **VizieR** (Ochsenbein et al. 2000): Supplemental catalog service for identification of previously observed sources across thousands of published catalogs.

### 2.2 Source Selection

Our selection pipeline applies successive filters to isolate thermally anomalous sources (Figure 6):

#### Tier 1: Optical Invisibility

We first select AllWISE sources lacking Gaia DR3 counterparts within 3":

```
N_Gaia(r < 3") = 0
```

This removes 341 million optically bright sources, yielding **406,387,755 candidates**.

#### Tier 2: Thermal Color Selection

We apply a color cut to select sources with thermal (blackbody-like) spectral energy distributions:

```
W1 - W2 > 0.5 mag
```

This criterion selects objects with effective temperatures T_eff ≲ 1000 K, consistent with cool brown dwarfs or room-temperature thermal emitters. Sources with W1-W2 > 2 mag correspond to T_eff ≲ 400 K.

#### Tier 3: Near-Infrared Invisibility

We remove sources with 2MASS counterparts within 3":

```
N_2MASS(r < 3") = 0
```

This eliminates late-type stars and most L/T dwarfs detectable in the near-infrared, reducing the sample to **62,856 candidates**.

#### Tier 4: Deep Optical Invisibility

We further require non-detection in Pan-STARRS DR1 (where available) and Legacy Survey DR10:

```
N_PS1(r < 3") = 0  AND  N_Legacy(r < 3") = 0
```

After this stage, **39,151 sources** remain—objects detectable *only* in mid-infrared wavelengths.

#### Tier 5: Radio Silence

To exclude AGN and radio-loud sources, we require:

```
N_NVSS(r < 30") = 0
```

This yields **4,137 radio-silent thermal anomaly candidates**.

#### Golden Sample Selection

From the Tier 5 sample, we select the **100 highest-scoring candidates** based on a composite anomaly score (Section 2.5). These "golden targets" are prioritized for detailed analysis and follow-up observations.

### 2.3 Cross-Matching Methodology

All catalog cross-matches are performed using positional coincidence with radius r_match:

```
Δθ = √[(α₁ - α₂)² cos²δ + (δ₁ - δ₂)²] < r_match
```

where α and δ are right ascension and declination. We adopt r_match = 3" for optical/NIR catalogs (accounting for WISE positional uncertainty of ~0.5" and proper motion over the ~10-year baseline) and r_match = 30" for radio catalogs (reflecting the larger NVSS beam).

For sources with Gaia proper motions, we propagate positions to the AllWISE epoch (2010.5) before cross-matching to account for source motion.

### 2.4 Physical Parameter Estimation

#### 2.4.1 Effective Temperature

We estimate effective temperatures by fitting a single-temperature blackbody to the W1 and W2 photometry. For our golden sample, we find:

- **Mean T_eff = 265 ± 36 K**
- Range: 200-500 K

#### 2.4.2 Distance Estimates

For sources with measured proper motions μ, we estimate distances assuming a typical tangential velocity v_⊥:

```
d = v_⊥ / (4.74 × μ) pc
```

where μ is in mas/yr and v_⊥ is in km/s. Adopting v_⊥ = 30 km/s (typical for disk objects), our golden sample spans estimated distances of **18-115 pc**, with ⟨d⟩ ≈ 50 pc.

### 2.5 Variability Analysis

#### 2.5.1 NEOWISE Light Curve Extraction

We query the NEOWISE Reactivation Single-Exposure Source Table via the IRSA TAP service for each golden target within a 3" cone. This yields multi-epoch W1 and W2 photometry spanning MJD 56639-60523 (2013.9-2024.5), with:

- **Mean: 387 epochs per source**
- **Baseline: 9.2 years**
- **Total: 38,700 epochs for 100 sources**

#### 2.5.2 Variability Metrics

For each source, we compute:

1. **Root-mean-square (RMS) variability**: σ_rms
2. **Reduced chi-squared**: χ²_ν
3. **Stetson J index**: For correlated variability between bands
4. **Linear trend slope**: dm/dt with p-value significance

#### 2.5.3 Variability Classification

Sources are classified as:

| Classification | Criteria | Count | Fraction |
|---------------|----------|-------|----------|
| **NORMAL** | Neither variable nor fading | 45 | 45% |
| **VARIABLE** | χ²_ν > 3 in either band | 50 | 50% |
| **FADING** | Significant negative trend (p < 0.01) with dm/dt > 15 mmag/yr | 5 | 5% |

#### 2.5.4 Periodogram Analysis

We search for periodic signals using the Lomb-Scargle periodogram implemented in `astropy.timeseries`. We sample 10,000 trial periods logarithmically spaced from 0.5 to 1000 days and compute the false alarm probability (FAP) for the highest peak. We consider periods significant if FAP < 0.01.

**Note**: The NEOWISE observing cadence (approximately 6-month intervals) introduces aliasing at periods near 180 days. Periods in this range should be interpreted with caution.

### 2.6 Anomaly Scoring

We assign each candidate a composite anomaly score based on multiple criteria:

```
S_total = S_stealth + S_thermal + S_kinematic + S_variability
```

#### Stealth Score (max 60 pts)
- +10 points for each veto catalog with non-detection
- Catalogs: Gaia, 2MASS, Pan-STARRS, Legacy, NVSS, ROSAT

#### Thermal Score (max 20 pts)
| Temperature | Points |
|-------------|--------|
| T_eff < 300 K | +20 |
| 300 ≤ T_eff < 400 K | +10 |
| 400 ≤ T_eff < 500 K | +5 |
| T_eff ≥ 500 K | 0 |

#### Kinematic Score
| Proper Motion | Points | Interpretation |
|---------------|--------|----------------|
| Parallax detected (3σ) | -30 | Nearby = natural |
| μ > 500 mas/yr | -15 | Very nearby |
| 100 < μ < 500 mas/yr | 0 | Moderate |
| μ < 100 mas/yr | +10 | Distant/interesting |

#### Variability Score
| Classification | Points | Rationale |
|---------------|--------|-----------|
| FADING | +25 | Highly unusual |
| VARIABLE | -10 | Common in astrophysical sources |
| NORMAL (stable) | +10 | Consistent with steady thermal emission |

### 2.7 Spectroscopic Follow-up Preparation

For the highest-priority candidates, we prepare spectroscopic target lists including:

1. Precise coordinates (J2000) with proper motion corrections
2. Finding charts from Legacy Survey DR10 imaging
3. Visibility calculations for major NIR facilities (Keck/NIRES, VLT/KMOS, Gemini/GNIRS)
4. Estimated exposure times based on W1 magnitude

**Target spectral features**:
- CH₄ absorption (1.6, 2.2 μm) — Y/T dwarf diagnostic
- H₂O bands (1.4, 1.9 μm)
- NH₃ features — coldest brown dwarfs

### 2.8 Software and Data Access

The TASNI pipeline is implemented in Python 3.10+ using:
- `astropy` — coordinate transformations, time series analysis
- `astroquery` — catalog access via TAP/Cone Search
- `pyvo` — Virtual Observatory services
- `pandas` — data manipulation
- `matplotlib` — visualization

**Catalog Access Points**:
- IRSA TAP: https://irsa.ipac.caltech.edu/TAP
- Gaia Archive: https://gea.esac.esa.int/archive
- VizieR: https://vizier.cds.unistra.fr
- Legacy Survey: https://www.legacysurvey.org

---

## Key Numbers Summary

| Stage | Sources | Reduction |
|-------|---------|-----------|
| AllWISE Catalog | 747,634,026 | — |
| No Gaia Optical | 406,387,755 | 54% remain |
| Thermal Selection | ~1,000,000 | — |
| No 2MASS NIR | 62,856 | 99.99% removed |
| No Pan-STARRS | 39,188 | 38% removed |
| No Legacy DR10 | 39,151 | <0.1% removed |
| Radio Silent | 4,137 | 89% removed |
| Golden Targets | 100 | Top-scored |
| **Fading Sources** | **4** | 4% of golden |

## Golden Sample Statistics

| Parameter | Value |
|-----------|-------|
| Mean W1 magnitude | 14.25 ± 0.89 mag |
| Mean W1-W2 color | 1.99 ± 0.36 mag |
| Mean T_eff | 265 ± 36 K |
| Mean proper motion | 216 ± 149 mas/yr |
| Mean baseline | 9.2 years |
| Mean N epochs | 387 |
