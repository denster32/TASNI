Title: TASNI Golden Sample - Machine-Readable Table
Description: 100 high-priority thermal anomaly candidates from the TASNI survey
Authors: Dennis Palucki
Version: 1.0.0
Date: 2026-02-15
DOI: 10.5281/zenodo.18717105

================================================================================
Column Definitions
================================================================================

Column Name              Units    Type      Description
--------------------------------------------------------------------------------
designation              ---      string    WISE designation (JHHMMSS.ss+DDMMSS.s)
ra                       deg      float     Right Ascension (J2000)
dec                      deg      float     Declination (J2000)
w1mpro                   mag      float     W1 Vega magnitude (3.4 micron)
w2mpro                   mag      float     W2 Vega magnitude (4.6 micron)
w3mpro                   mag      float     W3 Vega magnitude (12 micron)
w4mpro                   mag      float     W4 Vega magnitude (22 micron)
w1sigmpro                mag      float     W1 magnitude uncertainty
w2sigmpro                mag      float     W2 magnitude uncertainty
w3sigmpro                mag      float     W3 magnitude uncertainty
w4sigmpro                mag      float     W4 magnitude uncertainty
w1snr                    ---      float     W1 signal-to-noise ratio
w2snr                    ---      float     W2 signal-to-noise ratio
w3snr                    ---      float     W3 signal-to-noise ratio
w4snr                    ---      float     W4 signal-to-noise ratio
w1_w2_color              mag      float     W1 - W2 color index
T_eff_K                  K        float     Effective temperature (Planck SED fit)
pmra_value               mas/yr   float     Proper motion in RA
pmdec_value              mas/yr   float     Proper motion in Dec
pm_total                 mas/yr   float     Total proper motion
pm_angle                 deg      float     Proper motion position angle
pm_class                 ---      string    Proper motion classification
n_epochs                 ---      int       Number of NEOWISE epochs
baseline_years           yr       float     NEOWISE temporal baseline
rms_w1                   mag      float     W1 RMS variability
rms_w2                   mag      float     W2 RMS variability
chi2_w1                  ---      float     W1 chi-squared statistic
chi2_w2                  ---      float     W2 chi-squared statistic
trend_w1                 mag/yr   float     W1 linear brightness trend
trend_w2                 mag/yr   float     W2 linear brightness trend
is_variable              ---      bool      Significant variability flag
is_fading                ---      bool      Monotonic fading flag
variability_flag         ---      string    FADING, VARIABLE, or NORMAL
variability_score        ---      float     Composite variability metric
ps1_detected             ---      bool      Pan-STARRS1 detection flag
rosat_detected           ---      bool      ROSAT X-ray detection flag
detection_count          ---      int       Number of survey detections
detection_fraction       ---      float     Fraction of surveys with detection
mag_mean                 mag      float     Mean magnitude across epochs
mag_std                  mag      float     Standard deviation of magnitudes
mag_min                  mag      float     Minimum (brightest) magnitude
mag_max                  mag      float     Maximum (faintest) magnitude
mag_range                mag      float     Magnitude range (max - min)
color_mean               mag      float     Mean W1-W2 color across epochs
color_std                mag      float     Standard deviation of W1-W2 color
if_score                 ---      float     Isolation Forest anomaly score
xgb_score                ---      float     XGBoost classification score
lgb_score                ---      float     LightGBM classification score
ml_ensemble_score        ---      float     ML ensemble score (mean of IF, XGB, LGB)
improved_composite_score ---      float     Final composite ranking score (0-1)
rank                     ---      int       Overall ranking (1 = best candidate)
neowise_parallax_mas     mas      float     NEOWISE astrometric parallax (5-param fit)
neowise_parallax_err_mas mas      float     Formal parallax uncertainty
parallax_snr             ---      float     Parallax signal-to-noise ratio
distance_pc              pc       float     Distance from parallax (1000/pi)
distance_err_pc          pc       float     Symmetric distance uncertainty
p_false_positive_proxy   ---      float     False-positive probability proxy
golden_flag              ---      bool      Golden sample membership flag

================================================================================
Notes
================================================================================

1. Coordinates are from the AllWISE catalog.

2. Effective temperatures are estimated via Planck blackbody SED fitting to
   WISE W1 and W2 photometry.

3. Proper motions are from the CatWISE2020 catalog where available, otherwise
   computed from multi-epoch positional differences.

4. Variability flags:
   - FADING: Monotonic dimming with trend > 3 sigma from zero (4 sources)
   - VARIABLE: Significant RMS with no monotonic trend (51 sources)
   - NORMAL: No significant variability detected (45 sources)

5. Three confirmed FADING sources plus one LMC candidate (J0440-7314)
   are designated as "Fading Thermal Orphans" in the manuscript.

================================================================================
File Formats
================================================================================

1. golden_improved.csv - CSV format (human-readable)
2. golden_improved.parquet - Apache Parquet format (efficient for analysis)

================================================================================
Citation
================================================================================

If you use these data, please cite:

@article{tasni2026,
    author = {{Palucki, Dennis}},
    title = "{TASNI: Thermal Anomaly Search for Non-communicating Intelligence}",
    journal = {The Astrophysical Journal},
    year = {2026},
    volume = {},
    pages = {},
    doi = {}
}
