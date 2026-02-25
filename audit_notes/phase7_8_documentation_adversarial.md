# Phase 7 & 8: Documentation and Adversarial Review - Notes

## Phase 7: Documentation Review

### MAJOR-005: Documentation Contradicts Reality

**PUBLICATION_STATUS.md claims:**
- "READY FOR PUBLICATION"
- "Figures: ✅ Complete" for all 6 figures
- File structure shows figures in `tasni_paper_final/figures/`

**Reality:**
- All 6 figures are MISSING
- Figures directory is empty

### MINOR-005: Data README Accuracy
- Claims `golden_improved_kinematics.csv` has "varies" rows
- Actual: 87 rows (should be documented)

---

## Phase 8: Adversarial Review

### Source Novelty Check

**Result: ALL 4 SOURCES ARE NOVEL**

| Source | Coordinates | Previously Reported? |
|--------|-------------|---------------------|
| J143046.35-025927.8 | (217.69, -2.99) | NO |
| J231029.40-060547.3 | (347.62, -6.10) | NO |
| J193547.43+601201.5 | (293.95, 60.20) | NO |
| J044024.40-731441.6 | (70.10, -73.24) | NO |

- No matches in SIMBAD
- No matches in VizieR catalogs
- Not in CatWISE motion catalog
- Not in Backyard Worlds discoveries
- No papers found mentioning these coordinates

**Novelty claim: SUPPORTED**

---

## Mock Referee Report Points

A hostile referee would likely raise:

### Critical Concerns
1. **ML Circular Validation**: The manuscript acknowledges circularity, but is it adequately addressed? The models are trained on proxy labels derived from the same scores being predicted.

2. **Distance Claims vs Data**: Manuscript claims 17.4 pc for J1430, but parallax file shows 50-199 pc range. This discrepancy is unexplained.

3. **Figure Missing**: Cannot properly review without figures.

4. **NEOWISE Parallax Reliability**: Deriving ~57 mas parallaxes from a 6 arcsec PSF is questionable. This limitation is not adequately addressed.

5. **Period vs Alias**: The 93-179 day "periods" are almost certainly NEOWISE cadence aliases. The manuscript acknowledges this but presents them as if they might be real.

### Major Concerns
6. **Three Sources**: Very small sample size for making population-level claims.

7. **Systematic Uncertainties**: Adding 100 K systematic to 35-47 K statistical gives huge total uncertainties (±105-115 K). Can we really call 251-293 K "room temperature" with ±100 K uncertainty?

8. **Technosignature Framing**: Does the SETI framing help or hurt? Might be better as a pure brown dwarf discovery paper.

9. **Missing Citations**: Key Y dwarf survey papers (Kirkpatrick 2021, Meisner 2020) not cited.

### Recommendation
The referee would likely recommend **major revisions** before publication, primarily:
1. Generate missing figures
2. Resolve distance/parallax discrepancy
3. Add missing citations
4. Better address NEOWISE parallax limitations
