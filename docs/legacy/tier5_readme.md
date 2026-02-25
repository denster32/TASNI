# TIER 5: The Radio Silent Survivors

**Date:** December 30, 2025
**Count:** 4,137 Sources

## Definition
Tier 5 represents the "Ultra-Stealth" candidates of the TASNI project.
These objects satisfy the following stringent criteria:

1.  **Mid-IR Bright:** Detected by WISE (W1/W2 bands).
2.  **Optical Invisible:** No match in Gaia DR3 (limit ~21 mag).
3.  **NIR Invisible:** No match in 2MASS.
4.  **X-Ray Invisible:** No match in ROSAT (Rejects active stars/accretion).
5.  **Radio Invisible:** No match in NVSS (Rejects Quasars/AGN).
6.  **Ancillary Invisible:** No match in TESS or Spitzer archives.

## Key Files
- `tier5_radio_silent.parquet`: The full list of 4,137 survivors.
- `tier4_physics.parquet`: The calculated temperatures (T_eff) for the parent samples.
- `golden_targets.csv`: The Top 100 "Room Temperature" candidates selected from Tier 5.

## Next Steps
- **Visual Review:** Use `http://localhost:8000/blink.html` to manually inspect the Golden List.
- **Spectroscopy:** These targets are high-priority for spectral analysis to determine chemical composition (Methane? Dyson Shell?).
