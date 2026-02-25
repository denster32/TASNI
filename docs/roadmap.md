# TASNI Roadmap & Future Work

This document outlines the planned enhancements and future development for the TASNI project.

## Table of Contents
1. [Current Status](#current-status)
2. [Short-term Goals (Q1 2025)](#short-term-goals-q1-2025)
3. [Medium-term Goals (Q2-Q3 2025)](#medium-term-goals-q2-q3-2025)
4. [Long-term Goals (Q4 2025+)](#long-term-goals-q4-2025)
5. [Research Priorities](#research-priorities)
6. [Technical Priorities](#technical-priorities)
7. [Collaboration Opportunities](#collaboration-opportunities)

---

## Current Status

**Project Phase:** Production ✅
**Publication Status:** Complete
**Discovery:** 4 fading thermal orphans
**Code Status:** Reorganized and modernized (February 2025)
**Data Status:** 747M WISE, 1.8B Gaia processed
**Pipeline Status:** Fully operational

### Achievements (2024-2025)
- ✅ Discovery of 4 fading thermal orphans
- ✅ Complete multi-wavelength pipeline
- ✅ Tier 5 radio-silent catalog (810K sources)
- ✅ Golden sample (100 top targets)
- ✅ Publication-ready analysis
- ✅ Code reorganization (97 files, 13 dirs)
- ✅ Git repository and CI/CD
- ✅ Comprehensive documentation
- ✅ Testing infrastructure
- ✅ Security and data management tools

---

## Short-term Goals (Q1 2025)

### 1. Spectroscopic Follow-up

**Priority:** 🔴 Critical
**Timeline:** 1-3 months
**Effort:** High

**Objectives:**
- Obtain spectroscopy for 4 fading orphans
- Confirm Y-dwarf nature
- Measure effective temperatures
- Search for atmospheric features

**Telescopes:**
- Keck/NIRES (1-5 μm, R~2700)
- VLT/KMOS (2-2.45 μm, R~4000)
- JWST/NIRSpec (0.6-5.3 μm, R~1000-2700)
- TMT/MICHI (future)

**Targets:**
1. J143046.35-025927.8 (T_eff=293 K)
2. J231029.40-060547.3 (T_eff=258 K)
3. J193547.43+601201.5 (T_eff=251 K)
4. J060501.01-545944.5 (T_eff=253 K)

**Expected Outcomes:**
- Spectral confirmation of Y-dwarfs
- Atmospheric composition analysis
- Temperature validation
- Publication of spectroscopic results

### 2. Parallax Measurements

**Priority:** 🟡 High
**Timeline:** 3-6 months
**Effort:** Medium

**Objectives:**
- Measure distances to fading orphans
- Calculate absolute luminosities
- Constrain mass estimates
- Validate brown dwarf models

**Methods:**
- Gaia DR4 (expected 2025)
- Hubble Space Telescope
- Spitzer (if operational)
- Ground-based astrometry (Keck, VLT)

**Targets:**
- High PM sources (>100 mas/yr)
- All 4 fading orphans

**Expected Outcomes:**
- Distance measurements (±10%)
- Luminosity constraints
- Mass-age relationship validation

### 3. Additional Variability Analysis

**Priority:** 🟢 Medium
**Timeline:** 1-2 months
**Effort:** Low

**Objectives:**
- Extended time series analysis
- Search for periodic variability
- Statistical analysis of 810K tier5 sources
- Identify additional fading candidates

**Data Sources:**
- NEOWISE extended mission (2014-2024+)
- Legacy Survey (observed 2013-2019)
- ZTF (optical, 2018-present)
- LSST (future)

**Analysis:**
- Multi-band variability trends
- Period detection (rotational, orbital)
- Correlation with proper motion
- Machine learning classification

**Expected Outcomes:**
- Additional fading candidates
- Periodic variable catalog
- Statistical characterization of variability

### 4. Publication of Discoveries

**Priority:** 🟡 High
**Timeline:** 2-4 months
**Effort:** High

**Objectives:**
- Submit main discovery paper to ApJ/MNRAS
- Submit spectroscopy paper
- Submit variability analysis paper
- Public data release

**Journals:**
- The Astrophysical Journal (ApJ)
- Monthly Notices of the RAS (MNRAS)
- Astronomical Journal (AJ)

**Papers:**
1. "Discovery of Four Fading Thermal Orphans..."
2. "Spectroscopic Confirmation of Y-Dwarfs..."
3. "NEOWISE Variability of Cold Brown Dwarfs..."

**Data Release:**
- Golden targets catalog
- Tier 5 radio-silent catalog
- NEOWISE light curves
- Variability metrics

---

## Medium-term Goals (Q2-Q3 2025)

### 5. Extended Search

**Priority:** 🟡 High
**Timeline:** 6-9 months
**Effort:** High

**Objectives:**
- Search entire AllWISE catalog (747M sources)
- Expand to other wavelengths
- Cross-check with new surveys
- Identify additional anomalies

**Surveys to Include:**
- **Near-IR:** UKIDSS, VISTA, VHS
- **Mid-IR:** Spitzer, WISE, NEOWISE
- **Far-IR:** Herschel, AKARI
- **Sub-mm:** SCUBA-2, JCMT
- **Radio:** VLASS, LOFAR, SKA precursors
- **X-ray:** eROSITA, Chandra, XMM

**Methods:**
- Full pipeline run on all sources
- Multi-wavelength cross-correlation
- Machine learning anomaly detection
- Statistical outlier analysis

**Expected Outcomes:**
- Complete anomaly catalog
- Additional fading orphans
- Rare object discoveries
- Statistical characterization

### 6. Machine Learning Classification

**Priority:** 🟢 Medium
**Timeline:** 3-6 months
**Effort:** High

**Objectives:**
- Train ML models on golden sample
- Classify all 810K tier5 sources
- Identify rare subpopulations
- Predict follow-up priority

**Models:**
- Random Forest (baseline)
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks (CNN for images)
- Unsupervised clustering (UMAP, DBSCAN)

**Features:**
- Photometry (all bands)
- Colors (W1-W2, J-H, etc.)
- Variability metrics
- Proper motion
- Galactic coordinates
- Multi-wavelength cross-match flags

**Training Data:**
- Known brown dwarfs (L, T, Y)
- Known variable stars
- Known galaxies, quasars
- Artifacts, spurious sources

**Expected Outcomes:**
- Probability scores for all tier5 sources
- Identification of rare subpopulations
- Automated candidate ranking
- Publication-ready ML catalog

### 7. JWST Observations

**Priority:** 🟡 High
**Timeline:** 12-18 months (proposal cycle)
**Effort:** High

**Objectives:**
- Obtain JWST MIRI spectra
- Study mid-IR features (5-28 μm)
- Atmospheric composition
- Temperature validation

**Programs:**
- MIRI IFU (4.9-27.9 μm, R~1500-3500)
- MIRI Imaging (F560W, F770W, F1000W, F1130W, F1500W, F1800W)
- NIRCam Imaging (0.6-5 μm)

**Targets:**
- Fading orphans (4 sources)
- Coldest Y-dwarfs (T<250 K)
- High-priority tier5 candidates (10-20 sources)

**Science:**
- CH₄, H₂O, NH₃ bands
- Metallicity indicators
- Cloud properties
- Comparison with models

**Expected Outcomes:**
- First mid-IR spectra of fading orphans
- Atmospheric characterization
- Publication of JWST results

### 8. Cloud Infrastructure

**Priority:** 🟢 Medium
**Timeline:** 3-6 months
**Effort:** Medium

**Objectives:**
- Migrate pipeline to cloud
- Enable scalable processing
- Provide public access to tools
- Real-time data analysis

**Providers:**
- AWS (Amazon Web Services)
- Google Cloud Platform
- Microsoft Azure
- Research computing centers

**Services:**
- **Compute:** EC2, Lambda, Batch
- **Storage:** S3, EBS
- **Database:** Redshift, Athena
- **Workflow:** Airflow, Prefect
- **Visualization:** Dash, Streamlit

**Architecture:**
```
Cloud Pipeline
├── Data Ingestion (API, S3 uploads)
├── Processing (Batch jobs, Lambda)
├── Database (PostgreSQL, Redshift)
├── API (REST, GraphQL)
├── Visualization (Dashboards)
└── Data Export (CSV, FITS)
```

**Expected Outcomes:**
- Scalable cloud pipeline
- Public API for queries
- Interactive dashboards
- Automated processing

---

## Long-term Goals (Q4 2025+)

### 9. LSST Integration

**Priority:** 🟡 High
**Timeline:** 2026+ (LSST operations)
**Effort:** High

**Objectives:**
- Integrate with LSST data streams
- Real-time anomaly detection
- Multi-epoch analysis
- Automated alert system

**LSST Data Products:**
- Deep stacked images (r~27.5 mag)
- Multi-epoch light curves (~1000 visits)
- Difference image alerts
- Photometric catalogs

**Analysis:**
- Real-time cross-match with WISE/Gaia
- Detect new fading sources
- Monitor known anomalies
- Statistical characterization

**Expected Outcomes:**
- Real-time anomaly detection
- LSST-WISE-Gaia cross-catalog
- Public alert system
- Long-term monitoring

### 10. Community Catalog

**Priority:** 🟡 High
**Timeline:** 2026+
**Effort:** High

**Objectives:**
- Publish comprehensive anomaly catalog
- Provide web interface for queries
- Enable community contributions
- Long-term maintenance

**Catalog Components:**
- AllWISE anomalies (full catalog)
- Tier 5 radio-silent sources (810K)
- Golden sample (100)
- Spectroscopic follow-up results
- Variability metrics
- ML classifications

**Web Interface:**
- Search by coordinates, magnitude, color
- Filter by anomaly type
- Download data (FITS, CSV)
- Visualization tools
- API access

**Expected Outcomes:**
- Public community resource
- 10,000+ user queries/year
- Scientific publications using catalog
- Educational use

### 11. Advanced Search Methods

**Priority:** 🟢 Medium
**Timeline:** 2026+
**Effort:** High

**Objectives:**
- Develop new search algorithms
- Explore non-thermal signatures
- Multi-messenger approaches
- Systematic anomaly detection

**New Search Directions:**
- **Multi-messenger:** Neutrino + IR, Gravitational waves + IR
- **Time-domain:** Transient searches, variability trends
- **Statistical:** Extreme outlier detection, unsupervised ML
- **Multi-wavelength:** Full SED analysis, template fitting
- **Polarimetry:** IR/optical polarimetry
- **Interferometry:** High-resolution imaging

**Expected Outcomes:**
- Novel detection methods
- Additional source types
- Publication of new techniques
- Broader search applicability

### 12. Theoretical Framework

**Priority:** 🟢 Medium
**Timeline:** 2026+
**Effort**: High

**Objectives:**
- Develop theoretical models
- Predict observational signatures
- Interpret discovered anomalies
- Guide future searches

**Theoretical Work:**
- **Dyson Spheres:** Waste heat signatures, partial coverage
- **Megastructures:** Thermal patterns, variability
- **Brown Dwarfs:** Cooling curves, atmospheric models
- **Exoplanets:** Transiting, free-floating
- **Artificial Sources:** Engineered emissions

**Expected Outcomes:**
- Comprehensive theoretical framework
- Signature predictions
- Guidance for observations
- Publications in theoretical journals

---

## Research Priorities

### High Priority (Critical)
1. **Spectroscopic Confirmation** - Validate 4 fading orphans
2. **Parallax Measurements** - Obtain distances
3. **Publication** - Submit discovery papers

### Medium Priority (Important)
4. **Extended Search** - Full catalog analysis
5. **JWST Observations** - Mid-IR spectroscopy
6. **ML Classification** - Automated ranking

### Low Priority (Exploratory)
7. **LSST Integration** - Real-time detection
8. **Community Catalog** - Public resource
9. **Theoretical Work** - New search methods

---

## Technical Priorities

### Code Development
- [ ] Increase test coverage to >80%
- [ ] Add API documentation (Sphinx)
- [ ] Docker registry deployment
- [ ] Cloud migration

### Pipeline Improvements
- [ ] Real-time processing
- [ ] Automated alerts
- [ ] Workflow manager (Airflow/Prefect)
- [ ] Parallel GPU acceleration

### Data Management
- [ ] Automated backups
- [ ] Data versioning (DVC)
- [ ] Public data release
- [ ] Long-term archiving

### Security & Performance
- [ ] Penetration testing
- [ ] Performance profiling
- [ ] Benchmark optimization
- [ ] Scalability testing

---

## Collaboration Opportunities

### Research Collaborations
- **Brown Dwarf Experts:** Spectroscopy, atmospheric modeling
- **Time-domain Experts:** Variability analysis, transient detection
- **Machine Learning Experts:** Anomaly detection, classification
- **Theoretical Physicists:** Megastructure modeling

### Telescope Access
- **Keck Observatory:** Near-IR spectroscopy
- **VLT:** Near-IR spectroscopy, imaging
- **JWST:** Mid-IR spectroscopy, imaging
- **LSST:** Time-domain observations

### Institutional Partnerships
- **NASA/JPL:** WISE/NEOWISE collaboration
- **ESA/Gaia:** DR4 access, parallax data
- **NOIRLab:** Legacy Survey, spectroscopy
- **CSDC:** Canadian astronomical data

### Open Source Community
- **GitHub:** Issues, PRs, discussions
- **Astronomy Software:** astropy, lightkurve
- **Machine Learning:** scikit-learn, pytorch
- **Citizen Science:** Zooniverse, Galaxy Zoo

---

## Funding Opportunities

### NASA
- **ADAP:** Astrophysics Data Analysis Program
- **Keck:** Telescope time proposals
- **JWST:** Guest observer programs
- **HST:** Telescope time proposals

### NSF
- **AST:** Astronomy Division
- **MRI:** Major Research Instrumentation
- **CAREER:** Early career awards
- **REU:** Research Experiences for Undergraduates

### International
- **ERC:** European Research Council
- **STFC:** UK Science and Technology Facilities Council
- **ARC:** Australian Research Council

### Private
- **Breakthrough Listen:** SETI funding
- **Simons Foundation:** Mathematics and physical sciences
- **Heising-Simons:** Astronomy and physics

---

## Timeline Summary

| Q1 2025 | Q2 2025 | Q3 2025 | Q4 2025+ |
|-----------|-----------|-----------|------------|
| Spectroscopy | Extended Search | JWST Proposals | LSST Integration |
| Parallax | ML Training | Cloud Migration | Community Catalog |
| Variability | Cloud Dev | Advanced Methods | Theoretical Work |
| Publication | | | |

---

## Success Metrics

### Scientific Impact
- [ ] 4 fading orphans confirmed as Y-dwarfs
- [ ] Parallax distances obtained
- [ ] JWST spectra acquired
- [ ] 5+ peer-reviewed publications
- [ ] 100+ citations

### Technical Impact
- [ ] Test coverage >80%
- [ ] Cloud deployment
- [ ] Public API
- [ ] 10,000+ catalog queries
- [ ] Community contributions

### Community Impact
- [ ] Public data release
- [ ] Web interface
- [ ] Educational use
- [ ] Citizen science integration
- [ ] Open-source adoption

---

## Conclusion

The TASNI project has achieved significant scientific success with the discovery of 4 fading thermal orphans. The future roadmap builds on this success through:

1. **Immediate:** Spectroscopic confirmation, parallax measurements, publication
2. **Short-term:** Extended search, ML classification, JWST observations
3. **Long-term:** LSST integration, community catalog, theoretical framework

The combination of cutting-edge data analysis, theoretical modeling, and observational follow-up positions TASNI for continued discoveries in brown dwarf science and the search for non-communicating intelligence.

---

**Roadmap Status:** Active
**Last Updated:** February 2, 2025
**Next Review:** Q2 2025

**For questions or collaboration opportunities, see:**
- `CONTRIBUTING.md` - How to contribute
- `docs/QUICKSTART.md` - Getting started
- GitHub Issues - Project discussions
