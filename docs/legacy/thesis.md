# TASNI: Thermal Anomaly Search for Non-Communicating Intelligence

## Abstract

Current SETI methodology assumes intelligent civilizations broadcast intentionally. This assumption may be fundamentally flawed. Civilizations—particularly post-biological AI-based ones—may go silent by choice or instinct while still obeying thermodynamic laws. They must dump waste heat. This project systematically searches for infrared thermal signatures in regions where no optical counterpart exists, using existing public datasets (WISE, Gaia) and novel cross-matching methodology. We seek heat where there should be none.

---

## Core Premise

1. Traditional SETI anthropomorphizes alien intent
2. Early civilizations may recognize cosmic danger and go quiet
3. AI civilizations have no biological constraints—they can exist anywhere
4. Thermodynamics is inescapable: computation requires energy, energy produces waste heat
5. A silent civilization is still a warm civilization

---

## Current Status (Dec 30, 2025)

**Phase: VALIDATION & ANALYSIS**

We have successfully processed the entire AllWISE catalog (747 million sources) and cross-matched it against Gaia DR3, 2MASS, Pan-STARRS, and ROSAT.

**Results to Date:**
- **Total Sources Searched:** 747,000,000
- **Tier 1 (No Gaia Optical):** 406,387,755
- **Tier 2 (No 2MASS NIR):** 62,856
- **Tier 3 (No Pan-STARRS):** 39,188
- **Tier 4 (No X-Ray):** **39,188 "Ultra-Stealth" Candidates**

We have isolated ~39,000 objects that are bright in the Mid-IR but invisible in Optical, Near-IR, and X-Ray.

---

## Methodology

### Phase 1: Data Acquisition (COMPLETE)
- Downloaded AllWISE catalog (~300GB)
- Downloaded Gaia DR3 subset (~500GB)
- Verified server infrastructure (Ubuntu 24.04, 32GB RAM, RTX 3060 + Arc A770)

### Phase 2: Cross-Match Pipeline (COMPLETE)
- Implemented DuckDB + spatial indexing pipeline
- Filtered 747M -> 39k candidates based on "Invisible" criteria
- Criteria: W1/W2 detection > 0, No Gaia match < 3", No 2MASS match < 3"

### Phase 3: Multi-Wavelength Vetting (COMPLETE)
- **Pan-STARRS:** Deep optical check (removes faint stars) -> Passed.
- **ROSAT:** X-Ray check (removes active stars/accretion discs) -> Passed.

### Phase 4: Radio & Visual Validation (IN PROGRESS)
- **Radio Vetting (Tier 5):**
  - Cross-matching candidates with **NVSS (NRAO VLA Sky Survey)**.
  - Goal: Reject radio-loud sources (Quasars, AGN) to find "Radio Silent" anomalies.
  - Tool: `crossmatch_nvss.py` (running on server).

- **Visual Inspection:**
  - Downloading cutout images from Legacy Survey DR9.
  - Verification: Are they blank? Are they artifacts?
  - Tool: `fast_cutouts.py` (running 16x parallel on server).

### Phase 5: GPU-Accelerated Analysis (PLANNED)
**Hardware: NVIDIA RTX 3060 (12GB) & Intel Arc A770 (16GB)**

We will utilize the GPUs to analyze the 39,000 visual cutouts:
1.  **Artifact Rejection (CNN):** Train a simple classifier (ResNet-18) to identify "diffraction spikes," "halos," and "blank fields."
2.  **Morphology Classification:** Cluster the remaining "real" sources. Are they point sources? Extended blobs?
3.  **Compute Strategy:**
    - **RTX 3060:** Primary CUDA inference for PyTorch models.
    - **Arc A770:** Experimental XPU inference (batch processing).

---

## The Gap

No systematic search exists for:
- Infrared sources without optical counterparts
- Thermal signatures in "empty" space
- Heat anomalies that contradict natural astrophysical models

The data exists. The question hasn't been asked.

---

## Success Criteria

### Minimum viable outcome:
- Published methodology
- Open source pipeline
- Null results documented ("we looked, here's how, here's what we didn't find")

### Positive outcome:
- Catalog of unexplained thermal anomalies
- Targets for spectroscopic follow-up (Y-Dwarfs?)
- Contribution to technosignature search methodology

### Best case outcome:
- Anomalies that survive natural explanation
- Pattern or structure suggesting non-random origin
- Foundation for new SETI search paradigm

---

## Philosophical Framework

We reject the assumption that intelligence announces itself.
We propose that existence is detectable even when communication is not.
We look for the thermodynamic inevitability of life, not its intent to be found.
We publish honestly—successes, failures, and roadmaps for those who follow.

---

## Research Identity

Anonymous or pseudonymous publication.
Credit to methodology, not personality.
Open source. Open data. Open questions.

---

*TASNI Project*
*Cuchillo, New Mexico*
*2025*
