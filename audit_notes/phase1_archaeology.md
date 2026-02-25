# Phase 1: Project Archaeology - Audit Notes

## 1.1 File Inventory & Dependency Mapping

### Project Statistics
| Metric | Value |
|--------|-------|
| Total Size | 109 GB |
| Total Files | 86,185 |
| Python Files | 19,401 |
| CSV Files | 111 |
| Parquet Files | 34,193 |
| 0-Byte Files | 1,107 (mostly `__init__.py` and `.gitkeep`) |

### 0-Byte Files Assessment
Most are intentional:
- Empty `__init__.py` files for Python packages
- `.gitkeep` placeholder files
- Venv package markers (`py.typed`, `requested`)

### Suspicious Dates
- 19 files >1 year old: External reference data (Sonora Cholla spectra) - EXPECTED
- No future-dated files

## 1.2 Git History Forensics

### Repository Status
- **Location**: `/mnt/data/tasni` (symlinked to `/home/server/tasni`)
- **Total Commits**: 23
- **No Sensitive Info Found**: Searches for api_key, password, secret, token returned nothing

### Recent Commits (Selected)
1. `92eda1e` - docs: Add CLAUDE.md with project guidance
2. `15c19ed` - docs: Add comprehensive next steps summary
3. `7afaff1` - feat: Add spectroscopy planning and analysis workflow
4. `bceb592` - feat: Add complete ML pipeline infrastructure

### Uncommitted Changes
**WARNING**: Significant uncommitted changes detected:
- ~100+ deleted files (scripts/, docs/)
- 7 modified files
- 80+ untracked files/directories
- Major reorganization in progress

### Fix/Revert/Bug Commits
No actual bug fix or revert commits found (grep matched false positives from "verification", "security", "tests")

## 1.3 Configuration Audit

### Software Versions (for manuscript comparison)
| Software | Version in Config |
|----------|-------------------|
| Python | 3.11-3.13 |
| NumPy | 1.24.0+ |
| Pandas | 2.0.0+ |
| SciPy | 1.10.0+ |
| Astropy | 6.0.0+ |
| Astroquery | 0.4.6+ |
| HEALPy | 1.16.0+ |
| scikit-learn | 1.3.0+ |
| XGBoost | 3.1.3 |
| LightGBM | 4.6.0 |
| PyMC | 5.27.1 |
| UMAP-learn | 0.5.4+ |
| Matplotlib | 3.7.0+ |

### .zenodo.json Metadata
- Title: "TASNI: Thermal Anomaly Search for Non-communicating Intelligence - Data Release 1.0"
- Version: 1.0.0
- Publication Date: 2026-02-15
- DOI: (empty - pending deposition)
- License: MIT

---

## CRITICAL FINDINGS

### CRITICAL-001: GitHub Repository Does Not Exist
- **Location**: Manuscript references `https://github.com/denster32/TASNI`
- **Issue**: Repository returns 404 Not Found
- **Evidence**: Web fetch to GitHub URL returned 404 error
- **Impact**: Manuscript submission will have broken link
- **Fix**: Create the repository before submission or remove/modify the reference

### MAJOR-001: Uncommitted Changes in Repository
- **Location**: `/mnt/data/tasni`
- **Issue**: ~100+ files deleted, 7 modified, 80+ untracked
- **Evidence**: `git status` shows major reorganization in progress
- **Impact**: May indicate incomplete work or missing files
- **Fix**: Commit or stash changes, verify all necessary files are present

### MINOR-001: Git HEAD File Case Sensitivity
- **Location**: `.git/head` instead of `.git/HEAD`
- **Issue**: Minor case issue that may affect some git operations
- **Evidence**: Found and fixed during exploration
- **Impact**: Low
- **Fix**: Rename to uppercase `HEAD`
