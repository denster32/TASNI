# TASNI Paper - Introduction

## 1. Introduction

### 1.1 Motivation: Searching for Thermal Anomalies

The identification of unusual astrophysical sources has historically driven major discoveries, from quasars to gamma-ray bursts. In the modern era of large-area sky surveys, systematic searches for anomalous objects—those that defy easy classification—offer a promising avenue for discovering new phenomena.

One class of potentially anomalous sources comprises objects that emit primarily in the thermal infrared while remaining undetected at other wavelengths. Such "thermal orphans" could arise from several physical mechanisms:

1. **Extremely cold brown dwarfs**: Objects with T_eff ≲ 300 K emit predominantly at wavelengths λ > 10 μm, with negligible optical flux. The coldest known brown dwarf, WISE J085510.83−071442.5, has T_eff ≈ 250 K (Luhman 2014) and is detectable only in the mid-infrared.

2. **Dust-obscured sources**: Objects embedded in optically thick dust shells re-radiate absorbed energy as thermal emission at temperatures set by the dust sublimation radius.

3. **Technosignatures**: Theoretical considerations suggest that advanced technological civilizations might be detectable through their waste heat (Dyson 1960, Kardashev 1964). A structure intercepting stellar luminosity would re-radiate at temperatures T ~ 300 K for Sun-like stars at 1 AU separation (Wright 2014a, 2014b).

While the first two explanations invoke known astrophysics, the third—though speculative—motivates careful characterization of any genuinely anomalous thermal sources. Even if all candidates prove to be natural objects, systematic searches constrain the prevalence of technological activity in the solar neighborhood.

### 1.2 The Y Dwarf Population

Brown dwarfs are substellar objects with masses below the hydrogen-burning limit (~0.075 M_☉). They cool continuously throughout their lifetimes, passing through spectral types M, L, T, and Y as their effective temperatures decline.

The Y dwarf spectral class, defined by T_eff ≲ 500 K, represents the coldest end of the brown dwarf sequence. These objects are characterized by:

| Property | Y Dwarf Characteristic |
|----------|------------------------|
| **Near-IR absorption** | Strong CH₄ and H₂O bands |
| **Mid-IR colors** | W1−W2 > 2.0 mag |
| **Coldest examples** | NH₃ features present |
| **Optical emission** | Negligible (M_V > 25 mag) |

Approximately 30 Y dwarfs are currently known, identified primarily through WISE color selection. The space density of Y dwarfs remains poorly constrained due to the difficulty of detecting these intrinsically faint objects. Population synthesis models predict a substantial population of cold (T < 300 K) brown dwarfs in the solar neighborhood awaiting discovery.

### 1.3 Wide-field Infrared Survey Explorer

The Wide-field Infrared Survey Explorer (WISE; Wright et al. 2010) mapped the entire sky in four mid-infrared bands:

| Band | Wavelength | 5σ Sensitivity |
|------|------------|----------------|
| W1 | 3.4 μm | 0.068 mJy |
| W2 | 4.6 μm | 0.098 mJy |
| W3 | 12 μm | 0.86 mJy |
| W4 | 22 μm | 5.4 mJy |

The AllWISE data release contains **747 million sources**, providing an unprecedented census of mid-infrared emitters.

For cold objects (T ≲ 500 K), the W1−W2 color provides a sensitive temperature diagnostic:

| Temperature | Expected W1−W2 |
|-------------|----------------|
| 400 K | ~1.5 mag |
| 300 K | ~2.0 mag |
| 250 K | ~3.0 mag |
| 200 K | ~4.0 mag |

These extreme colors distinguish cold thermal emitters from the stellar locus.

The NEOWISE reactivation mission has continued W1 and W2 observations since 2013, providing a **decade-long temporal baseline** for variability studies.

### 1.4 Multi-Wavelength Veto Strategy

Previous searches for ultracool dwarfs have relied primarily on color selection, identifying red objects in WISE photometry. While effective, this approach cannot distinguish genuinely "invisible" objects from those with faint but detectable counterparts at other wavelengths.

We adopt a complementary strategy based on **multi-wavelength non-detection**:

| Veto Catalog | Wavelength | Purpose |
|--------------|------------|---------|
| Gaia DR3 | Optical (G band) | Remove optically visible sources |
| 2MASS | Near-IR (J, H, Ks) | Remove NIR-detectable brown dwarfs |
| Pan-STARRS DR1 | Optical (grizy) | Deep optical veto |
| Legacy Survey DR10 | Optical (grz) | Deepest optical veto |
| NVSS | Radio (1.4 GHz) | Remove AGN and radio sources |

**Advantages of the veto strategy:**

1. **Purity**: Sources passing all vetoes are guaranteed to lack counterparts above survey detection limits
2. **Completeness**: Does not depend on assumed spectral energy distribution shapes
3. **Anomaly detection**: Naturally identifies unusual objects by selecting against known source classes

### 1.5 This Work

We present the **Thermal Anomaly Search for Non-communicating Intelligence (TASNI)**, a systematic pipeline to identify mid-infrared sources lacking counterparts across the electromagnetic spectrum.

**Goals:**

1. Quantify the population of genuinely "invisible" thermal emitters in AllWISE
2. Characterize their photometric, kinematic, and temporal properties
3. Identify candidates for spectroscopic follow-up
4. Constrain the prevalence of anomalous thermal signatures in the solar neighborhood

**Paper Organization:**

| Section | Content |
|---------|---------|
| §2 Methods | Data sources, selection criteria, analysis methodology |
| §3 Results | Pipeline results, discovery of fading thermal orphans |
| §4 Discussion | Physical interpretation, alternative hypotheses |
| §5 Conclusions | Summary and future observations |

---

## Key Context

| Concept | Relevance |
|---------|-----------|
| Y dwarfs | T_eff < 500 K, coldest brown dwarfs, ~30 known |
| WISE colors | W1−W2 > 2 mag indicates T < 400 K |
| Thermal orphans | Mid-IR only sources, no optical/NIR/radio counterparts |
| NEOWISE baseline | 10+ years enables variability analysis |
| Dyson spheres | Would have T ~ 280 K at 1 AU (similar to our sample!) |
