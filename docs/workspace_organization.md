# Workspace Organization and Naming Convention Analysis

**TASNI Project**
Analysis Date: 2026-02-04
Version: 1.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Directory Structure](#current-directory-structure)
3. [Organizational Analysis](#organizational-analysis)
4. [Naming Convention Analysis](#naming-convention-analysis)
5. [Identified Issues](#identified-issues)
6. [Proposed Improvements](#proposed-improvements)
7. [Implementation Plan](#implementation-plan)
8. [Best Practices](#best-practices)
9. [Migration Guide](#migration-guide)
10. [Impact Analysis](#impact-analysis)

---

## Executive Summary

The TASNI project workspace has evolved organically through multiple development phases, resulting in a structure that is functional but exhibits several organizational inconsistencies. This document provides a comprehensive analysis of the current workspace organization, identifies areas for improvement, and proposes a systematic reorganization plan aligned with Python project best practices.

### Key Findings

**Strengths:**
- Well-organized `src/tasni/` directory with logical module separation
- Clear separation between source code, data, and output
- Comprehensive documentation in `docs/` directory
- Consistent use of Python conventions in source code

**Areas for Improvement:**
- Mixed naming conventions across documentation files
- Root-level documentation files that should be in `docs/`
- Duplicate documentation files
- Inconsistent data organization
- Output directory structure could be more hierarchical
- Missing standard Python project directories (`src/`, `tests/`)

---

## Current Directory Structure

```
tasni/
├── .github/                    # GitHub Actions workflows
├── benchmarks/                 # Performance benchmarks
├── data/interim/checkpoints/                # Pipeline checkpoints and intermediate data
├── data/                       # Input data files
│   ├── gaia_xmatch/           # Gaia cross-match data
│   ├── legacy_survey/         # Legacy survey data
│   ├── models/                # Atmospheric models
│   │   └── sonora_cholla/    # Sonora-Cholla model spectra
│   │       └── spectra_files/
│   └── secondary/            # Secondary catalog data
├── data_release/              # Public data release files
├── docs/                      # Documentation
│   ├── analysis/             # Analysis documentation
│   ├── api/                  # API documentation
│   ├── guides/               # User guides
│   ├── legacy/               # Legacy documentation
│   ├── output/               # Output documentation
│   └── paper/                # Paper-related documentation
├── logs/                      # Application logs
├── notebooks/                 # Jupyter notebooks
├── output/                    # Pipeline outputs
│   ├── figures/              # Generated figures and plots
│   ├── final/                # Final processed data
│   ├── features/             # ML features
│   ├── ml/                   # ML results and models
│   ├── periodogram/          # Periodogram analysis
│   └── spectroscopy/         # Spectroscopy outputs
├── tasni_paper_final/        # LaTeX paper source files (submission-ready)
│   ├── figures/              # Paper figures
│   └── manuscript.tex        # Main manuscript
├── src/tasni/                   # Main source code
│   ├── analysis/             # Analysis modules
│   ├── core/                 # Core utilities
│   ├── download/             # Data download scripts
│   ├── filtering/            # Filtering modules
│   ├── ml/                   # Machine learning modules
│   ├── optimized/            # Optimized implementations
│   └── utils/                # Utility functions
├── tests/                     # Test suite
├── validation/                # Validation scripts
├── validation_output/         # Validation results
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── .pre-commit-config.yaml   # Pre-commit hooks
├── CLAUDE.md                  # AI assistant instructions
├── CONTRIBUTING.md            # Contributing guidelines
├── Dockerfile                 # Docker configuration
├── FINAL_REPORT.md            # Final project report
├── Makefile                   # Build automation
├── README.md                  # Project README
├── README_UPDATE.md           # README updates
├── REORGANIZATION_SUMMARY.md  # Reorganization summary
├── requirements.txt           # Python dependencies
└── requirements-dev.txt       # Development dependencies
```

---

## Organizational Analysis

### Directory Organization Assessment

#### 1. Root Level

**Current State:**
- Contains 13 files and 10 directories
- Mix of configuration, documentation, and operational files
- Several documentation files that belong in `docs/`

**Assessment:**
- **Good:** Essential configuration files (`.env.example`, `.gitignore`, `Makefile`, `Dockerfile`)
- **Issue:** Documentation files scattered at root level (`CLAUDE.md`, `FINAL_REPORT.md`, `README_UPDATE.md`, `REORGANIZATION_SUMMARY.md`)
- **Issue:** `README.md` is appropriate at root, but `README_UPDATE.md` is redundant

#### 2. Source Code (`src/tasni/`)

**Current State:**
- Well-organized with 7 subdirectories
- Each subdirectory has an `__init__.py` file
- Clear functional separation

**Assessment:**
- **Excellent:** Logical module organization
- **Excellent:** Consistent naming conventions
- **Minor Issue:** Directory name `src/tasni/` is non-standard for Python projects (should be `src/` or `tasni/`)

#### 3. Documentation (`docs/`)

**Current State:**
- Contains 26+ markdown files
- Organized into subdirectories: `analysis/`, `api/`, `guides/`, `legacy/`, `output/`, `paper/`
- Mixed naming conventions (UPPER_CASE, lower_case, Title_Case)

**Assessment:**
- **Good:** Subdirectory organization is logical
- **Issue:** Inconsistent file naming (e.g., `ANALYSIS_WORKFLOW.md` vs `api/`, `guides/`)
- **Issue:** Duplicate files (`CLAUDE.md` and `claude.md`)
- **Issue:** Some files appear to be temporary or outdated

#### 4. Data Organization (`data/`, `data_release/`)

**Current State:**
- `data/` contains raw and processed input data
- `data_release/` contains public release files
- Subdirectories organized by data source

**Assessment:**
- **Good:** Clear separation between internal and public data
- **Good:** Logical organization by data source
- **Minor Issue:** Some data files at root of `data/` could be better organized

#### 5. Output Organization (`output/`)

**Current State:**
- 7 subdirectories organized by output type
- Contains figures, final data, ML results, spectroscopy outputs

**Assessment:**
- **Good:** Clear separation by output type
- **Minor Issue:** Could benefit from timestamped or versioned subdirectories
- **Minor Issue:** Some outputs might be better in `results/` directory

#### 6. Paper Directory (`paper/`)

**Current State:**
- Contains LaTeX source files
- Separate from `docs/paper/`
- Has its own `figures/` and `sections/` subdirectories

**Assessment:**
- **Issue:** Duplication with `docs/paper/`
- **Issue:** Figures duplicated between `paper/figures/` and `reports/figures/`
- **Recommendation:** Consolidate to single location

#### 7. Tests and Validation

**Current State:**
- `tests/` directory exists (empty or minimal)
- `validation/` and `validation_output/` separate

**Assessment:**
- **Issue:** Test organization unclear
- **Issue:** Validation should be part of testing infrastructure

---

## Naming Convention Analysis

### File Naming Patterns

#### 1. Documentation Files

| Pattern | Examples | Count | Assessment |
|---------|----------|-------|------------|
| UPPER_CASE | `ANALYSIS_WORKFLOW.md`, `API_REFERENCE.md` | 6 | Non-standard |
| lower_case | `claude.md` | 1 | Standard |
| Title_Case | `Architecture.md` | 0 | Not used |
| snake_case | `migration_guide.md` | 0 | Not used |

**Issues:**
- Mixed case conventions
- No consistent pattern
- Some files use underscores, others don't

#### 2. Source Code Files

**Python Files:**
- All use `snake_case` ✓
- Follows PEP 8 standards ✓
- Consistent module naming ✓

**Examples:**
- `classify_variability.py`
- `periodogram_significance.py`
- `optimized_pipeline.py`

#### 3. Data Files

| Extension | Pattern | Examples | Assessment |
|-----------|---------|----------|------------|
| .csv | snake_case | `tasni_golden_targets.csv` ✓ | Good |
| .parquet | snake_case | `tier5_cleaned.parquet` ✓ | Good |
| .json | snake_case | `tier5_variability_checkpoint.json` ✓ | Good |
| .dat | lowercase | `nvss.dat` ✓ | Good |
| .spec | snake_case | `500K_316g_logkzz7.spec` ✓ | Good |

#### 4. Output Files

**Figures:**
- Mixed patterns: `fig1_allsky_galactic.png`, `fading_cutouts.png`
- Inconsistent numbering scheme
- Some use descriptive names, others use `figX_` pattern

**Data Files:**
- Consistent `snake_case` ✓
- Clear naming ✓

### Directory Naming Patterns

| Directory | Pattern | Assessment |
|-----------|---------|------------|
| `src/tasni/` | lowercase | Non-standard for Python |
| `docs/` | lowercase | Standard ✓ |
| `data/` | lowercase | Standard ✓ |
| `output/` | lowercase | Standard ✓ |
| `benchmarks/` | lowercase | Standard ✓ |
| `notebooks/` | lowercase | Standard ✓ |
| `tests/` | lowercase | Standard ✓ |
| `data_release/` | snake_case | Good ✓ |
| `gaia_xmatch/` | snake_case | Good ✓ |
| `legacy_survey/` | snake_case | Good ✓ |

**Issues:**
- `src/tasni/` should be `src/` or `tasni/` for Python packages
- Some subdirectories use underscores, others don't

---

## Identified Issues

### Critical Issues

1. **Duplicate Documentation Files**
   - `CLAUDE.md` at root and `docs/CLAUDE.md` and `docs/claude.md`
   - `paper/` and `docs/paper/` duplication
   - Figures duplicated between `paper/figures/` and `reports/figures/`

2. **Root-Level Documentation Clutter**
   - `CLAUDE.md` - should be in `docs/`
   - `FINAL_REPORT.md` - should be in `docs/`
   - `README_UPDATE.md` - should be merged with `README.md` or moved to `docs/`
   - `REORGANIZATION_SUMMARY.md` - should be in `docs/`

3. **Non-Standard Source Directory**
   - `src/tasni/` is non-standard for Python projects
   - Should be `src/` or package name `tasni/`

### Medium Priority Issues

4. **Inconsistent Documentation Naming**
   - Mix of UPPER_CASE, lower_case, and Title_Case
   - No consistent pattern for documentation files

5. **Output Directory Structure**
   - Could benefit from better hierarchy
   - Some outputs might be better organized by date or version

6. **Test Organization**
   - `tests/` directory structure unclear
   - `validation/` separate from `tests/`

### Low Priority Issues

7. **Data File Organization**
   - Some data files at root of `data/` could be better organized
   - `data_release/` could be a subdirectory of `data/`

8. **Log Organization**
   - `logs/` directory structure unclear
   - No subdirectories for different log types

9. **Checkpoint Organization**
   - `data/interim/checkpoints/` could be better organized by pipeline stage or date

---

## Proposed Improvements

### 1. Root Level Cleanup

**Move to `docs/`:**
```
CLAUDE.md → docs/claude.md
FINAL_REPORT.md → docs/final_report.md
README_UPDATE.md → docs/readme_updates.md
REORGANIZATION_SUMMARY.md → docs/reorganization_summary.md
```

**Keep at root:**
- `README.md` (essential)
- `CONTRIBUTING.md` (essential)
- `.env.example`, `.gitignore`, `.pre-commit-config.yaml` (configuration)
- `Dockerfile`, `Makefile` (build)
- `requirements.txt`, `requirements-dev.txt` (dependencies)

### 2. Source Directory Reorganization

**Option A: Standard Python Package Structure**
```
src/
└── tasni/
    ├── __init__.py
    ├── analysis/
    ├── core/
    ├── download/
    ├── filtering/
    ├── ml/
    ├── optimized/
    └── utils/
```

**Option B: Keep Current but Rename**
```
tasni/  # Rename from src/tasni/
├── __init__.py
├── analysis/
├── core/
├── download/
├── filtering/
├── ml/
├── optimized/
└── utils/
```

**Recommendation:** Option A for better alignment with Python packaging standards

### 3. Documentation Standardization

**Adopt consistent naming:**
- Use `snake_case` for all documentation files
- Lowercase only
- Descriptive names

**Standardized naming examples:**
```
ANALYSIS_WORKFLOW.md → analysis_workflow.md
API_REFERENCE.md → api_reference.md
ARCHITECTURE.md → architecture.md
CLAUDE.md → claude.md
```

### 4. Paper Directory Consolidation

**Consolidate to single location:**
```
docs/paper/
├── figures/           # All paper figures
├── sections/          # Paper sections (both .md and .tex)
├── references.bib
├── tasni_paper.tex
├── tasni_paper.pdf
└── README.md
```

**Remove:**
- `paper/` directory at root level
- Duplicate figures in `reports/figures/`

### 5. Output Directory Enhancement

**Proposed structure:**
```
output/
├── figures/
│   ├── paper/         # Figures for paper
│   ├── analysis/      # Analysis figures
│   └── models/        # Model comparison figures
├── data/
│   ├── final/         # Final processed data
│   ├── features/      # ML features
│   └── ml/            # ML results
├── reports/
│   ├── health_check_report.json
│   └── VERIFICATION_REPORT.md
├── spectroscopy/
│   ├── finding_charts/
│   └── observation_plan/
└── periodogram/
```

### 6. Test Organization

**Proposed structure:**
```
tests/
├── unit/              # Unit tests
├── integration/       # Integration tests
├── validation/        # Validation tests (move from validation/)
└── fixtures/         # Test fixtures
```

### 7. Data Organization Enhancement

**Proposed structure:**
```
data/
├── raw/               # Raw input data
│   ├── nvss.dat
│   └── nvss.dat.gz
├── catalogs/          # Catalog data
│   ├── gaia/
│   ├── legacy_survey/
│   └── secondary/
├── models/            # Atmospheric models
│   └── sonora_cholla/
└── release/           # Public release data (move from data_release/)
```

### 8. Checkpoint Organization

**Proposed structure:**
```
data/interim/checkpoints/
├── downloads/         # Download checkpoints
├── processing/        # Processing checkpoints
└── ml/               # ML model checkpoints
```

---

## Implementation Plan

### Phase 1: Root Level Cleanup (Day 1)

**Tasks:**
1. Move documentation files to `docs/`
2. Update all internal references
3. Update `.gitignore` if needed
4. Test that all links still work

**Commands:**
```bash
# Move files
mv CLAUDE.md docs/claude.md
mv FINAL_REPORT.md docs/final_report.md
mv README_UPDATE.md docs/readme_updates.md
mv REORGANIZATION_SUMMARY.md docs/reorganization_summary.md

# Remove duplicate claude.md if it exists
rm docs/CLAUDE.md  # if duplicate
```

**Impact:** Low - mostly documentation moves

### Phase 2: Source Directory Reorganization (Day 2-3)

**Tasks:**
1. Create new `src/tasni/` structure
2. Move all Python modules
3. Update all imports
4. Update `requirements.txt` if needed
5. Update `Makefile` targets
6. Run tests to ensure nothing breaks

**Commands:**
```bash
# Create new structure
mkdir -p src/tasni

# Move modules
mv src/tasni/* src/tasni/

# Update imports (requires manual review or automated script)
# Find all files with imports and update
find . -name "*.py" -exec sed -i 's/from src/tasni/from tasni/g' {} \;
```

**Impact:** High - requires import updates across codebase

### Phase 3: Documentation Standardization (Day 4)

**Tasks:**
1. Rename all documentation files to `snake_case`
2. Update all internal references
3. Update table of contents
4. Verify all links work

**Commands:**
```bash
# Rename files in docs/
cd docs
for f in *.md; do
    newname=$(echo "$f" | tr '[:upper:]' '[:lower:]' | sed 's/_/ /g' | sed 's/ /_/g')
    mv "$f" "$newname"
done
```

**Impact:** Medium - requires reference updates

### Phase 4: Paper Directory Consolidation (Day 5)

**Tasks:**
1. Consolidate `paper/` into `docs/paper/`
2. Remove duplicate figures
3. Update all references
4. Update build scripts

**Commands:**
```bash
# Move paper files to docs/paper/
mv paper/* docs/paper/

# Remove duplicate figures (manual review required)
# Compare reports/figures/ and docs/paper/figures/
```

**Impact:** Medium - requires reference updates

### Phase 5: Output Directory Enhancement (Day 6)

**Tasks:**
1. Create new output subdirectories
2. Move files to appropriate locations
3. Update all references
4. Update pipeline scripts

**Commands:**
```bash
# Create new structure
mkdir -p reports/figures/paper reports/figures/analysis output/data output/reports

# Move files
mv data/processed/final/* output/data/
mv data/processed/features/* output/data/
mv data/processed/ml/* output/data/
mv output/health_check_report.json output/reports/
mv output/VERIFICATION_REPORT.md output/reports/
```

**Impact:** Medium - requires reference updates

### Phase 6: Test Organization (Day 7)

**Tasks:**
1. Create new test structure
2. Move validation tests to `tests/validation/`
3. Organize existing tests
4. Update test runner

**Commands:**
```bash
# Create new structure
mkdir -p tests/unit tests/integration tests/validation tests/fixtures

# Move validation tests
mv validation/* tests/validation/
```

**Impact:** Low - mostly test reorganization

### Phase 7: Data Organization Enhancement (Day 8)

**Tasks:**
1. Create new data structure
2. Move files to appropriate locations
3. Update all references
4. Update pipeline scripts

**Commands:**
```bash
# Create new structure
mkdir -p data/raw data/catalogs/gaia data/catalogs/legacy_survey data/catalogs/secondary data/release

# Move files
mv data/nvss* data/raw/
mv data/gaia_xmatch/* data/catalogs/gaia/
mv data/legacy_survey/* data/catalogs/legacy_survey/
mv data/secondary/* data/catalogs/secondary/
mv data_release/* data/release/
```

**Impact:** High - requires reference updates across codebase

### Phase 8: Checkpoint Organization (Day 9)

**Tasks:**
1. Create new checkpoint structure
2. Move files to appropriate locations
3. Update all references
4. Update pipeline scripts

**Commands:**
```bash
# Create new structure
mkdir -p data/interim/checkpoints/downloads data/interim/checkpoints/processing data/interim/checkpoints/ml

# Move files (requires manual review based on checkpoint type)
```

**Impact:** Medium - requires reference updates

### Phase 9: Verification and Testing (Day 10)

**Tasks:**
1. Run full test suite
2. Verify all imports work
3. Verify all documentation links work
4. Run pipeline end-to-end
5. Update documentation as needed

**Commands:**
```bash
# Run tests
pytest tests/

# Verify imports
python -c "import tasni; print('Import successful')"

# Check for broken links
# (use markdown link checker)
```

**Impact:** Verification only

---

## Best Practices

### Python Project Structure

#### Standard Layout

```
project/
├── src/                    # Source code
│   └── package_name/      # Main package
├── tests/                 # Tests
├── docs/                  # Documentation
├── data/                  # Data files
├── output/                # Generated outputs
├── src/tasni/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Dependencies
├── requirements-dev.txt   # Dev dependencies
├── setup.py              # Package setup
├── pyproject.toml        # Modern Python packaging
├── README.md             # Project README
├── LICENSE               # License
└── .gitignore            # Git ignore rules
```

#### Key Principles

1. **Separate source from tests:** `src/` and `tests/` should be separate
2. **Package structure:** Source code should be in a package directory
3. **Configuration at root:** Keep configuration files at root level
4. **Documentation in docs/:** All documentation should be in `docs/`
5. **Data separation:** Separate input data from generated outputs

### File Naming Conventions

#### Python Files
- Use `snake_case`: `my_module.py`, `my_function.py`
- Follow PEP 8 guidelines
- Descriptive names that indicate purpose

#### Documentation Files
- Use `snake_case`: `user_guide.md`, `api_reference.md`
- Lowercase only
- Descriptive names
- Avoid numbers unless part of a sequence

#### Data Files
- Use `snake_case`: `input_data.csv`, `results.parquet`
- Include version or date if applicable: `data_v2.csv`, `data_20240101.csv`
- Use appropriate extensions: `.csv`, `.parquet`, `.json`, `.fits`

#### Configuration Files
- Use standard names: `config.yaml`, `settings.json`
- Environment-specific: `config_dev.yaml`, `config_prod.yaml`

### Directory Naming Conventions

#### General Rules
- Use `lowercase` for directory names
- Use underscores for multi-word names: `data_files/`, `test_results/`
- Avoid spaces and special characters
- Be descriptive but concise

#### Specific Directories

| Directory | Purpose | Naming |
|-----------|---------|--------|
| Source code | Main package | `src/package_name/` |
| Tests | Test suite | `tests/` |
| Documentation | Docs | `docs/` |
| Data | Input data | `data/` |
| Output | Generated output | `output/` or `results/` |
| Scripts | Utility scripts | `src/tasni/` |
| Notebooks | Jupyter notebooks | `notebooks/` |
| Logs | Application logs | `logs/` |
| Config | Configuration | `config/` |

### Documentation Organization

#### Standard Structure

```
docs/
├── README.md              # Documentation index
├── getting_started/       # Getting started guides
├── user_guide/           # User guides
├── api_reference/        # API documentation
├── architecture/         # Architecture docs
├── development/          # Development guides
├── troubleshooting/      # Troubleshooting guides
└── changelog/            # Changelog
```

#### Key Principles

1. **Logical grouping:** Group related documentation together
2. **Clear hierarchy:** Use subdirectories to organize
3. **Consistent naming:** Use consistent naming conventions
4. **Index file:** Include a README.md or index.md in each directory
5. **Cross-references:** Use relative links for cross-references

### Data Management Best Practices

#### Data Organization

```
data/
├── raw/                  # Raw, unprocessed data
├── processed/            # Processed data
├── external/             # External data sources
├── models/               # Model files
└── config/               # Data configuration files
```

#### Key Principles

1. **Separate raw from processed:** Never modify raw data
2. **Document sources:** Include source information
3. **Version control:** Use versioning for data files
4. **Appropriate formats:** Use efficient formats (Parquet, HDF5)
5. **Metadata:** Include metadata files

### Output Management Best Practices

#### Output Organization

```
output/
├── figures/              # Generated figures
├── tables/               # Generated tables
├── reports/              # Analysis reports
├── models/               # Trained models
└── logs/                 # Output logs
```

#### Key Principles

1. **Clear separation:** Separate different types of outputs
2. **Timestamping:** Include timestamps in filenames
3. **Versioning:** Version outputs for reproducibility
4. **Cleanup:** Regular cleanup of old outputs
5. **Documentation:** Document output formats and schemas

---

## Migration Guide

### Pre-Migration Checklist

- [ ] Backup entire repository
- [ ] Create migration branch
- [ ] Document current state
- [ ] Identify all file references
- [ ] Plan rollback strategy

### Migration Steps

#### Step 1: Create Migration Branch

```bash
git checkout -b workspace-reorganization
git push -u origin workspace-reorganization
```

#### Step 2: Document Current State

```bash
# Create snapshot of current structure
tree -L 3 > docs/migration/current_structure.txt

# List all files
find . -type f > docs/migration/all_files_before.txt
```

#### Step 3: Execute Migration Phases

Follow the implementation plan in order, completing each phase before moving to the next.

#### Step 4: Update References

After each phase, update all references:

```bash
# Find all Python files with old imports
grep -r "from scripts" --include="*.py" . > docs/migration/import_references.txt

# Find all documentation references
grep -r "CLAUDE.md" --include="*.md" . > docs/migration/doc_references.txt
```

#### Step 5: Test After Each Phase

```bash
# Run tests
pytest tests/ -v

# Verify imports
python -c "import tasni; print('OK')"

# Check documentation links
# (use markdown link checker)
```

#### Step 6: Document Changes

After each phase, document what was changed:

```bash
# Create migration log
echo "Phase 1 completed: $(date)" >> docs/migration/migration_log.txt
echo "Files moved: ..." >> docs/migration/migration_log.txt
```

#### Step 7: Final Verification

After all phases:

```bash
# Run full test suite
pytest tests/ -v --cov

# Verify all imports work
python -m pytest tests/

# Check documentation
# (run documentation build if applicable)

# Verify pipeline
# (run end-to-end pipeline test)
```

#### Step 8: Create Pull Request

```bash
# Commit changes
git add .
git commit -m "Reorganize workspace structure"

# Push to remote
git push origin workspace-reorganization

# Create pull request
# Include migration notes in PR description
```

### Rollback Plan

If migration fails:

```bash
# Checkout original branch
git checkout main

# Delete migration branch
git branch -D workspace-reorganization

# Restore from backup if needed
# (use backup created in pre-migration)
```

---

## Impact Analysis

### Code Impact

#### High Impact Changes

1. **Source Directory Reorganization**
   - All imports need updating
   - All relative imports need updating
   - Test imports need updating
   - Documentation examples need updating

**Estimated Effort:** 2-3 days

**Affected Files:**
- All Python files in `src/tasni/`
- All test files
- All documentation with code examples
- Configuration files with import paths

#### Medium Impact Changes

2. **Data Directory Reorganization**
   - All data file references need updating
   - Pipeline scripts need updating
   - Configuration files need updating

**Estimated Effort:** 1-2 days

**Affected Files:**
- All pipeline scripts
- Configuration files
- Notebooks with data references

3. **Output Directory Reorganization**
   - All output file references need updating
   - Pipeline scripts need updating
   - Visualization scripts need updating

**Estimated Effort:** 1 day

**Affected Files:**
- All pipeline scripts
- Visualization scripts
- Notebooks with output references

#### Low Impact Changes

4. **Documentation Reorganization**
   - Documentation links need updating
   - Table of contents needs updating

**Estimated Effort:** 0.5 day

**Affected Files:**
- All documentation files
- README files

5. **Test Reorganization**
   - Test runner configuration may need updating
   - CI/CD configuration may need updating

**Estimated Effort:** 0.5 day

**Affected Files:**
- Test configuration files
- CI/CD configuration files

### Documentation Impact

#### User Documentation

- Getting started guides may need updating
- Installation instructions may need updating
- API documentation may need updating

**Estimated Effort:** 1 day

#### Developer Documentation

- Architecture documentation may need updating
- Development guides may need updating
- Contribution guidelines may need updating

**Estimated Effort:** 0.5 day

### CI/CD Impact

#### GitHub Actions

- Workflow files may need path updates
- Test commands may need updating
- Build commands may need updating

**Estimated Effort:** 0.5 day

#### Pre-commit Hooks

- Hook configurations may need updating
- File path patterns may need updating

**Estimated Effort:** 0.25 day

### Dependencies Impact

#### Python Packages

- No changes to dependencies required
- Import paths will change, but packages remain the same

#### External Tools

- Docker configuration may need path updates
- Makefile targets may need path updates

**Estimated Effort:** 0.5 day

### Total Estimated Effort

| Phase | Effort | Impact |
|-------|--------|--------|
| Root Level Cleanup | 0.5 day | Low |
| Source Directory Reorganization | 2-3 days | High |
| Documentation Standardization | 1 day | Medium |
| Paper Directory Consolidation | 1 day | Medium |
| Output Directory Enhancement | 1 day | Medium |
| Test Organization | 0.5 day | Low |
| Data Organization Enhancement | 1-2 days | High |
| Checkpoint Organization | 0.5 day | Medium |
| Verification and Testing | 1 day | Low |
| **Total** | **9-11 days** | **Mixed** |

---

## Recommendations Summary

### Immediate Actions (High Priority)

1. **Move root-level documentation to `docs/`**
   - Quick win with low risk
   - Improves root-level cleanliness

2. **Consolidate duplicate documentation**
   - Remove `CLAUDE.md` duplicates
   - Consolidate `paper/` directories

3. **Standardize documentation naming**
   - Adopt `snake_case` for all documentation files
   - Update all references

### Short-term Actions (Medium Priority)

4. **Reorganize source directory**
   - Move `src/tasni/` to `src/tasni/`
   - Update all imports
   - Follow Python packaging standards

5. **Enhance output directory structure**
   - Create better hierarchy
   - Organize by output type

6. **Organize test directory**
   - Create proper test structure
   - Move validation tests

### Long-term Actions (Low Priority)

7. **Enhance data directory structure**
   - Create better hierarchy
   - Separate raw from processed data

8. **Organize checkpoint directory**
   - Create better hierarchy
   - Organize by pipeline stage

9. **Regular cleanup**
   - Establish cleanup procedures
   - Archive old outputs

---

## Conclusion

The TASNI project workspace is generally well-organized but exhibits several inconsistencies that can be improved. The proposed reorganization plan addresses these issues systematically while minimizing disruption to the project.

The key improvements include:

1. **Standardized naming conventions** across all files and directories
2. **Better separation of concerns** with clear directory structure
3. **Alignment with Python best practices** for project organization
4. **Reduced duplication** of files and documentation
5. **Improved maintainability** through consistent organization

The implementation plan provides a phased approach that allows for incremental changes with verification at each step. This minimizes risk and ensures that the project remains functional throughout the reorganization process.

By following this plan, the TASNI project will achieve a more professional, maintainable, and scalable workspace structure that aligns with industry best practices.

---

## Appendices

### Appendix A: File Reference Inventory

A complete inventory of all file references that need updating during migration. This should be generated during the pre-migration phase.

### Appendix B: Migration Scripts

Example scripts for automated migration tasks.

#### Rename Documentation Files Script

```python
#!/usr/bin/env python3
"""Script to rename documentation files to snake_case."""

import os
import re
from pathlib import Path

def to_snake_case(name):
    """Convert filename to snake_case."""
    # Remove extension
    base = os.path.splitext(name)[0]
    ext = os.path.splitext(name)[1]

    # Convert to lowercase and replace spaces/hyphens with underscores
    snake = re.sub(r'[\s-]+', '_', base.lower())

    return snake + ext

def rename_docs(docs_dir):
    """Rename all documentation files in docs_dir."""
    docs_path = Path(docs_dir)

    for md_file in docs_path.glob('*.md'):
        new_name = to_snake_case(md_file.name)
        if new_name != md_file.name:
            new_path = md_file.parent / new_name
            print(f"Renaming: {md_file.name} -> {new_name}")
            md_file.rename(new_path)

if __name__ == '__main__':
    rename_docs('docs')
```

### Appendix C: Testing Checklist

Checklist for verifying migration success:

- [ ] All tests pass
- [ ] All imports work correctly
- [ ] All documentation links work
- [ ] Pipeline runs successfully
- [ ] All notebooks execute without errors
- [ ] CI/CD pipeline passes
- [ ] No broken file references
- [ ] All configuration files updated
- [ ] Documentation updated
- [ ] Team notified of changes

### Appendix D: Resources

#### Python Project Structure

- [Python Packaging User Guide](https://packaging.python.org/)
- [Cookiecutter Templates](https://github.com/cookiecutter/cookiecutter)
- [Scientific Python Project Structure](https://scikit-learn.org/stable/developers/contributing.html#project-structure)

#### Documentation Best Practices

- [Write the Docs](https://www.writethedocs.org/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)

#### Data Management

- [Data Management Best Practices](https://www.nature.com/articles/sdata201618)
- [FAIR Data Principles](https://www.go-fair.org/fair-principles/)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-04
**Author:** TASNI Project Team
**Status:** Draft - Ready for Review
