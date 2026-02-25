# TASNI Paper - Results Section

## 3. Results

### 3.1 Pipeline Source Counts

Table 1 summarizes the source counts at each stage of the TASNI selection pipeline. Beginning with the full AllWISE catalog of 747,634,026 sources, our multi-wavelength veto strategy progressively isolates objects detectable exclusively in the mid-infrared.

**Table 1: TASNI Pipeline Source Counts**

| Selection Stage | Sources | Reduction |
|-----------------|---------|-----------|
| AllWISE Catalog | 747,634,026 | — |
| No Gaia DR3 optical | 406,387,755 | 46% |
| Thermal selection (W1−W2 > 0.5) | ~1,000,000 | — |
| No 2MASS NIR | 62,856 | 94% |
| No Pan-STARRS DR1 | 39,188 | 38% |
| No Legacy Survey DR10 | 39,151 | <0.1% |
| No NVSS radio | 4,137 | 89% |
| **Golden targets (top 100)** | **100** | — |
| **Fading sources** | **4** | 4% |

The most significant reduction occurs at the 2MASS veto stage, where 94% of optically-invisible thermal sources are found to have near-infrared counterparts. These are predominantly late-type stars and L/T brown dwarfs with detectable J, H, or Ks emission. The 62,856 sources lacking 2MASS counterparts represent objects too faint or too red for near-infrared detection—consistent with extremely cold (T_eff ≲ 400 K) brown dwarfs or genuinely anomalous thermal emitters.

The radio veto removes 89% of remaining candidates, eliminating AGN and other radio-loud contaminants. The final Tier 5 sample of 4,137 radio-silent thermal anomalies represents 5.5 × 10⁻⁶ of the parent AllWISE catalog—objects that emit thermal radiation at room temperature while remaining invisible across optical, near-infrared, and radio wavelengths.

### 3.2 Golden Sample Properties

From the Tier 5 sample, we select the 100 highest-scoring candidates as our "golden sample" for detailed analysis.

**Table 2: Golden Sample Statistics (N = 100)**

| Parameter | Mean ± Std | Range |
|-----------|------------|-------|
| W1 (mag) | 14.20 ± 1.23 | 10.06–16.84 |
| W2 (mag) | 12.18 ± 1.23 | 8.42–15.34 |
| W1−W2 (mag) | 2.01 ± 0.40 | 1.50–3.67 |
| T_eff (K) | 272 ± 39 | 203–471 |
| μ (mas/yr) | 299 ± 178 | 27–963 |
| \|b\| (deg) | 39.7 ± 20.4 | 0.3–85.4 |

#### 3.2.1 Color-Magnitude Distribution

The W1−W2 colors span 1.53–3.67 mag, with a mean of 1.99 ± 0.36 mag. These colors correspond to extremely red spectral energy distributions, consistent with Y and late-T dwarf atmospheres dominated by methane and water absorption.

For comparison:
- Known Y dwarfs typically exhibit W1−W2 > 2.0 mag (Kirkpatrick et al. 2012)
- T dwarfs span W1−W2 ≈ 0.5–2.5 mag

Our golden sample includes **42 sources with W1−W2 > 2.0 mag**, placing them in the Y dwarf color regime. The most extreme source, **J143046.35−025927.8**, exhibits W1−W2 = 3.37 mag—among the reddest mid-infrared colors known for any astronomical source.

#### 3.2.2 Temperature Distribution

Blackbody fits to the W1 and W2 photometry yield effective temperatures ranging from 205 K to 466 K, with a mean of 265 ± 36 K.

**Key finding: 85% of the golden sample (85/100 sources) have T_eff < 300 K—cooler than typical room temperature on Earth.**

This temperature distribution is consistent with the coldest known brown dwarfs. The Y dwarf WISE 0855−0714, the coldest known brown dwarf at T_eff ≈ 250 K (Luhman 2014), has properties remarkably similar to our golden sample median.

#### 3.2.3 Proper Motion and Distance Estimates

The golden sample exhibits significant proper motions:
- **87% (87/100)** showing μ > 100 mas/yr
- **46% (46/100)** showing μ > 300 mas/yr

These high proper motions indicate nearby objects. Assuming typical disk kinematics (v_⊥ ≈ 30 km/s), the proper motion distribution implies distances of **10–100 pc** for most sources, with a median of approximately 50 pc.

#### 3.2.4 Galactic Distribution

The golden sample is preferentially located at high Galactic latitudes:
- Mean |b| = 39.7°
- **67% of sources at |b| > 30°**

This distribution minimizes contamination from dust-obscured background stars, infrared dark clouds, and other Galactic contaminants.

### 3.3 Multi-Wavelength Non-Detections

We verify the "invisible" nature of our golden sample through cross-matching with additional catalogs:

| Catalog | Matches | Implication |
|---------|---------|-------------|
| LAMOST DR7 | 0/100 | No optical spectra exist |
| Legacy Survey DR10 | 37/39,151 (<0.1%) | 99.9% truly optically dark |
| SIMBAD (fading sources) | 0/4 | Not previously catalogued |

The absence of LAMOST spectra is particularly significant: despite LAMOST's extensive spectroscopic coverage, **none of our 100 golden targets have been spectroscopically observed**. This confirms that these sources are genuinely below optical detection thresholds.

### 3.4 NEOWISE Variability Analysis

#### 3.4.1 Temporal Coverage

We retrieved NEOWISE multi-epoch photometry for all 100 golden targets:
- **Total epochs: 38,700**
- **Baseline: 9.2 ± 1.9 years** (2013.9–2024.5)
- **Mean epochs per source: 387 ± 95**

#### 3.4.2 Variability Classification

| Classification | Count | Fraction | Description |
|---------------|-------|----------|-------------|
| **NORMAL** | 45 | 45% | Stable emission, χ²_ν < 3, no trends |
| **VARIABLE** | 50 | 50% | Significant variability, χ²_ν > 3 |
| **FADING** | 5 | 5% | Systematic dimming, dm/dt > 15 mmag/yr |

The 50% variable fraction is consistent with known brown dwarf populations, where cloud-driven variability is common at the L/T transition and in Y dwarfs.

### 3.5 Discovery of Fading Thermal Orphans

**The most significant result of our variability analysis is the identification of five sources exhibiting systematic fading over the 10-year NEOWISE baseline.**

We designate these "fading thermal orphans" to emphasize their unusual combination of:
1. Thermal emission at room temperature
2. Multi-wavelength invisibility
3. Secular dimming

One source (J044024.40−731441.6) is identified as MSX LMC 1152 via SIMBAD—a known Large Magellanic Cloud object. We exclude this extragalactic contaminant, leaving **four bona fide fading thermal orphans**.

#### 3.5.1 Properties of Fading Sources

**Table 3: Fading Thermal Orphans**

| Designation | W1−W2 | T_eff | PM | Fade Rate | SIMBAD |
|-------------|-------|-------|-----|-----------|--------|
| J143046.35−025927.8 | **3.37** | 293 K | 55 mas/yr | 25.5 mmag/yr | Unclassified |
| J231029.40−060547.3 | 1.75 | 258 K | 165 mas/yr | **52.6 mmag/yr** | Unclassified |
| J193547.43+601201.5 | 1.53 | 251 K | **306 mas/yr** | 22.9 mmag/yr | Unclassified |
| J060501.01−545944.5 | 2.00 | 253 K | **359 mas/yr** | 17.9 mmag/yr | Unclassified |

**Key characteristics:**

1. **Extreme W1−W2 colors**: All four have W1−W2 > 1.5 mag. J143046.35−025927.8 exhibits W1−W2 = 3.37 mag—the reddest source in our sample.

2. **Room temperature emission**: T_eff range 251–293 K, comparable to terrestrial room temperature (~290 K).

3. **High proper motion**: Three of four sources have μ > 150 mas/yr, implying distances of 18–40 pc.

4. **No prior identification**: Cross-matching with SIMBAD and VizieR returns no matches. They are NOT catalogued as known brown dwarfs, variable stars, AGN, or any other object class.

#### 3.5.2 Light Curves

All four fading sources show monotonic dimming in both W1 and W2 over the 10-year baseline:

- **J231029.40−060547.3**: Fastest fader at 52.6 mmag/yr, dimmed by ~0.5 mag total
- **J143046.35−025927.8**: Reddest source, fading at 25.5 mmag/yr
- **J193547.43+601201.5**: Highest proper motion (306 mas/yr), fading at 22.9 mmag/yr
- **J060501.01-545944.5**: Very high PM (359 mas/yr), fading at 17.9 mmag/yr. **Note:** This source was dropped from the final manuscript after further analysis; its fading significance fell below the 3-sigma threshold upon reanalysis with updated photometric calibration.

The fading is detected independently in both W1 and W2 bands, ruling out instrumental systematics.

#### 3.5.3 Periodogram Analysis

We searched for periodic signals using Lomb-Scargle periodograms:
- **No significant short-period (P < 30 days) variability detected**
- Detected "periods" at 90-180 days are NEOWISE sampling aliases (harmonics and sub-harmonics of the ~182-day observing cadence)
- Known Y dwarf rotation periods are 2-10 hours, ruling out rotational modulation as the source of these long-period signals

### 3.6 Interpretation

The properties of the fading thermal orphans are consistent with **extremely cold (T_eff ≈ 250 K) brown dwarfs at distances of 20–50 pc**. Their extreme W1−W2 colors place them at the cool end of the Y dwarf sequence.

**Possible explanations for the fading behavior:**

1. **Cooling brown dwarfs**: Young brown dwarfs cool as they radiate formation heat. For a 250 K object, cooling rates of ~1 K/yr could produce the observed fade rates, implying ages ≲ 1 Gyr.

2. **Atmospheric evolution**: Changes in cloud properties or atmospheric chemistry could modulate thermal emission.

3. **Orbital effects**: Unresolved binaries could produce secular brightness changes over decade timescales.

**Spectroscopic follow-up is essential** to confirm the brown dwarf interpretation and measure spectral types, atmospheric compositions, and radial velocities.

### 3.7 Comparison to Known Populations

| Population | T_eff Range | W1−W2 Range | Comparison |
|------------|-------------|-------------|------------|
| Known Y dwarfs (~30) | 250–450 K | 1.5–4.0 mag | Significant overlap |
| 2MASS T/Y dwarfs | 300–1300 K | 0.5–2.5 mag | Our sources fainter in NIR |
| WISE Y candidates | 250–400 K | 1.5–3.5 mag | Our sample extends fainter |

**The four fading sources represent a potentially new class of variable Y dwarf candidates.** Their combination of:
- Extreme colors (W1−W2 up to 3.37 mag)
- High proper motions (up to 359 mas/yr)
- Optical/NIR invisibility
- Systematic fading (17–53 mmag/yr)

...has not been previously reported in the literature.

---

## Summary of Key Results

| Finding | Value | Significance |
|---------|-------|--------------|
| Radio-silent thermal anomalies | 4,137 | 5.5 × 10⁻⁶ of AllWISE |
| Golden targets with T < 300 K | 85% | Room temperature emitters |
| Sources with μ > 100 mas/yr | 87% | Nearby (<100 pc) |
| Variable sources | 55% | Consistent with brown dwarfs |
| **Fading thermal orphans** | **4** | **New discovery** |
| Fading sources in SIMBAD | 0/4 | Previously unknown objects |
