# TASNI Migration Guide

This guide helps you migrate from the old flat structure to the new organized structure.

## Overview

The TASNI project has been reorganized with a new directory structure. This guide helps you:

1. **Update import paths** in your code
2. **Locate moved scripts** in new directories
3. **Adapt to new configuration** (optional)
4. **Update shell scripts** and workflows

## Import Path Changes

### Old → New Mapping

```python
# OLD: Direct imports
from scripts.config import DATA_ROOT
from scripts.filter_anomalies_full import filter_anomalies
from scripts.generate_golden_list import generate_targets

# NEW: Organized imports
from scripts.core.config import DATA_ROOT
from scripts.filtering.filter_anomalies_full import filter_anomalies
from scripts.generation.generate_golden_list import generate_targets
```

### Complete Mapping Table

| Old Import | New Import |
|------------|-------------|
| `scripts.config` | `scripts.core.config` |
| `scripts.filter_anomalies_full` | `scripts.filtering.filter_anomalies_full` |
| `scripts.generate_golden_list` | `scripts.generation.generate_golden_list` |
| `scripts.analyze_kinematics` | `scripts.analysis.analyze_kinematics` |
| `scripts.compute_ir_variability` | `scripts.analysis.compute_ir_variability` |
| `scripts.periodogram_analysis` | `scripts.analysis.periodogram_analysis` |
| `scripts.crossmatch_full` | `scripts.crossmatch.crossmatch_full` |
| `scripts.download_wise_full` | `scripts.download.download_wise_full` |
| `scripts.download_gaia_full` | `scripts.download.download_gaia_full` |
| `scripts.multi_wavelength_scoring` | `scripts.filtering.multi_wavelength_scoring` |
| `scripts.validate_brown_dwarfs` | `scripts.filtering.validate_brown_dwarfs` |
| `scripts.gen_figures` | `scripts.generation.gen_figures` |
| `scripts.prepare_spectroscopy_targets` | `scripts.generation.prepare_spectroscopy_targets` |
| `scripts.quick_check` | `scripts.utils.quick_check` |
| `scripts.fast_cutouts` | `scripts.utils.fast_cutouts` |
| `scripts.check_spectra` | `scripts.checks.check_spectra` |
| `scripts.check_spitzer` | `scripts.checks.check_spitzer` |
| `scripts.check_tess` | `scripts.checks.check_tess` |
| `scripts.optimized_pipeline` | `scripts.optimized.optimized_pipeline` |
| `scripts.optimized_crossmatch` | `scripts.optimized.optimized_crossmatch` |
| `scripts.optimized_variability` | `scripts.optimized.optimized_variability` |

## Finding Moved Scripts

### Core Scripts

| Purpose | Old Location | New Location |
|----------|---------------|---------------|
| Configuration | `src/tasni/config.py` | `src/tasni/core/config.py` |
| Logging | `src/tasni/tasni_logging.py` | `src/tasni/core/tasni_logging.py` |

### Download Scripts

| Script | Old Location | New Location |
|---------|---------------|---------------|
| WISE downloader | `src/tasni/download_wise_full.py` | `src/tasni/download/download_wise_full.py` |
| Gaia downloader | `src/tasni/download_gaia_full.py` | `src/tasni/download/download_gaia_full.py` |
| Legacy Survey | `src/tasni/download_legacy_survey.py` | `src/tasni/download/download_legacy_survey.py` |
| LAMOST | `src/tasni/download_lamost.py` | `src/tasni/download/download_lamost.py` |
| Secondary catalogs | `src/tasni/download_secondary_catalogs.py` | `src/tasni/download/download_secondary_catalogs.py` |
| NEOWISE async | `src/tasni/async_neowise_query.py` | `src/tasni/download/async_neowise_query.py` |

### Crossmatch Scripts

| Script | Old Location | New Location |
|---------|---------------|---------------|
| Full crossmatch | `src/tasni/crossmatch_full.py` | `src/tasni/crossmatch/crossmatch_full.py` |
| Legacy crossmatch | `src/tasni/crossmatch_legacy.py` | `src/tasni/crossmatch/crossmatch_legacy.py` |
| LAMOST crossmatch | `src/tasni/crossmatch_lamost.py` | `src/tasni/crossmatch/crossmatch_lamost.py` |
| GPU crossmatch | `src/tasni/gpu_crossmatch.py` | `src/tasni/crossmatch/gpu_crossmatch.py` |
| Optimized | `src/tasni/optimized_crossmatch.py` | `src/tasni/optimized/optimized_crossmatch.py` |

### Analysis Scripts

| Script | Old Location | New Location |
|---------|---------------|---------------|
| Kinematics | `src/tasni/analyze_kinematics.py` | `src/tasni/analysis/analyze_kinematics.py` |
| IR variability | `src/tasni/compute_ir_variability.py` | `src/tasni/analysis/compute_ir_variability.py` |
| Periodogram | `src/tasni/periodogram_analysis.py` | `src/tasni/analysis/periodogram_analysis.py` |
| NEOWISE parallax | `src/tasni/extract_neowise_parallax.py` | `src/tasni/analysis/extract_neowise_parallax.py` |
| Population synthesis | `src/tasni/population_synthesis.py` | `src/tasni/analysis/population_synthesis.py` |

### Filtering Scripts

| Script | Old Location | New Location |
|---------|---------------|---------------|
| Filter anomalies | `src/tasni/filter_anomalies_full.py` | `src/tasni/filtering/filter_anomalies_full.py` |
| Multi-wave scoring | `src/tasni/multi_wavelength_scoring.py` | `src/tasni/filtering/multi_wavelength_scoring.py` |
| Validate BDs | `src/tasni/validate_brown_dwarfs.py` | `src/tasni/filtering/validate_brown_dwarfs.py` |

### Generation Scripts

| Script | Old Location | New Location |
|---------|---------------|---------------|
| Golden targets | `src/tasni/generate_golden_list.py` | `src/tasni/generation/generate_golden_list.py` |
| Publication figures | `src/tasni/generate_publication_figures.py` | `src/tasni/generation/generate_publication_figures.py` |
| GBT schedule | `src/tasni/generate_gbt_schedule.py` | `src/tasni/generation/generate_gbt_schedule.py` |
| Spectroscopy targets | `src/tasni/prepare_spectroscopy_targets.py` | `src/tasni/generation/prepare_spectroscopy_targets.py` |
| Gen figures | `src/tasni/gen_figures.py` | `src/tasni/generation/gen_figures.py` |

### Utility Scripts

| Script | Old Location | New Location |
|---------|---------------|---------------|
| Quick check | `src/tasni/quick_check.py` | `src/tasni/utils/quick_check.py` |
| Fast cutouts | `src/tasni/fast_cutouts.py` | `src/tasni/utils/fast_cutouts.py` |
| Combine chunks | `src/tasni/combine_chunks.py` | `src/tasni/utils/combine_chunks.py` |
| Server tagger | `src/tasni/server_tagger.py` | `src/tasni/utils/server_tagger.py` |
| Quality report | `src/tasni/generate_quality_report.py` | `src/tasni/generation/generate_quality_report.py` |
| Teff calculator | `src/tasni/calculate_teff.py` | `src/tasni/utils/calculate_teff.py` |
| Data manager | `src/tasni/data_manager.py` | `src/tasni/utils/data_manager.py` (NEW) |
| Security audit | `src/tasni/security_audit.py` | `src/tasni/utils/security_audit.py` (NEW) |

## Configuration Migration (Optional)

### Old: Hardcoded in config.py

```python
# src/tasni/core/config.py (old)
DATA_ROOT = Path("/mnt/data/tasni")
WISE_DIR = DATA_ROOT / "data" / "wise"
GAIA_DIR = DATA_ROOT / "data" / "gaia"
```

### New: Environment variables (optional)

```bash
# .env file
TASNI_DATA_ROOT=/mnt/data/tasni
TASNI_WISE_DIR=${TASNI_DATA_ROOT}/data/wise
TASNI_GAIA_DIR=${TASNI_DATA_ROOT}/data/gaia
```

**Note:** The old config.py still works! The new environment-based configuration is optional.

## Shell Script Updates

### Old: Direct script paths

```bash
#!/bin/bash
# OLD
python /mnt/data/tasni/src/tasni/download_wise_full.py
python /mnt/data/tasni/src/tasni/crossmatch_full.py
python /mnt/data/tasni/src/tasni/filter_anomalies_full.py
```

### New: Use Makefile or relative paths

```bash
#!/bin/bash
# NEW: Use Makefile
cd /mnt/data/tasni
make download-wise
make crossmatch-cpu
make filter-anomalies

# OR: Use relative paths
python src/tasni/download/download_wise_full.py
python src/tasni/crossmatch/crossmatch_full.py
python src/tasni/filtering/filter_anomalies_full.py
```

## Jupyter Notebook Updates

### Old: Direct imports

```python
# OLD
import sys
sys.path.insert(0, '/mnt/data/tasni/scripts')
from config import DATA_ROOT
from filter_anomalies_full import filter_anomalies
```

### New: Organized imports

```python
# NEW
import sys
sys.path.insert(0, '/mnt/data/tasni/scripts')
from core.config import DATA_ROOT
from filtering.filter_anomalies_full import filter_anomalies
```

## Automated Migration Script

If you have many files to update, use this script:

```python
#!/usr/bin/env python3
"""
Migration script to update import paths
"""
import re
from pathlib import Path

# Import path mappings
mappings = {
    'from scripts.config import': 'from scripts.core.config import',
    'from scripts.filter_anomalies_full import': 'from scripts.filtering.filter_anomalies_full import',
    'from scripts.generate_golden_list import': 'from scripts.generation.generate_golden_list import',
    'from scripts.analyze_kinematics import': 'from scripts.analysis.analyze_kinematics import',
    'from scripts.compute_ir_variability import': 'from scripts.analysis.compute_ir_variability import',
    'from scripts.periodogram_analysis import': 'from scripts.analysis.periodogram_analysis import',
    'from scripts.crossmatch_full import': 'from scripts.crossmatch.crossmatch_full import',
    'from scripts.download_wise_full import': 'from scripts.download.download_wise_full import',
    'from scripts.download_gaia_full import': 'from scripts.download.download_gaia_full import',
    'from scripts.multi_wavelength_scoring import': 'from scripts.filtering.multi_wavelength_scoring import',
    'from scripts.gen_figures import': 'from scripts.generation.gen_figures import',
    'from scripts.quick_check import': 'from scripts.utils.quick_check import',
    'from scripts.check_spectra import': 'from scripts.checks.check_spectra import',
    'from scripts.optimized_pipeline import': 'from scripts.optimized.optimized_pipeline import',
}

# Scan and update files
tasni_root = Path('/mnt/data/tasni')

# Update Python files in notebooks/
for notebook in tasni_root.rglob('*.ipynb'):
    content = notebook.read_text()
    changed = False

    for old, new in mappings.items():
        if old in content:
            content = content.replace(old, new)
            changed = True

    if changed:
        notebook.write_text(content)
        print(f"Updated: {notebook}")

# Update Python files in custom scripts
for py_file in tasni_root.rglob('*.py'):
    # Skip reorganized scripts
    if 'src/tasni/' in str(py_file):
        continue

    content = py_file.read_text()
    changed = False

    for old, new in mappings.items():
        if old in content:
            content = content.replace(old, new)
            changed = True

    if changed:
        py_file.write_text(content)
        print(f"Updated: {py_file}")
```

## Testing Migration

### Verify imports work

```python
# Test script
import sys
sys.path.insert(0, '/mnt/data/tasni/scripts')

# Test core
from core.config import DATA_ROOT
print(f"✓ Core config: {DATA_ROOT}")

# Test download
from download.download_wise_full import download_wise
print("✓ Download module")

# Test crossmatch
from crossmatch.crossmatch_full import Crossmatcher
print("✓ Crossmatch module")

# Test analysis
from analysis.analyze_kinematics import calculate_pm
print("✓ Analysis module")

# Test filtering
from filtering.filter_anomalies_full import filter_anomalies
print("✓ Filtering module")

# Test generation
from generation.generate_golden_list import generate_targets
print("✓ Generation module")

print("\n✓ All imports successful!")
```

### Test Makefile commands

```bash
cd /mnt/data/tasni

# Test basic commands
make help
make pipeline-status

# Test configuration
python src/tasni/core/config.py

# Test new tools
python src/tasni/utils/data_manager.py --status
python src/tasni/utils/security_audit.py --status
```

## Common Migration Issues

### Issue 1: ImportError: No module named 'config'

**Old:**
```python
from scripts.config import DATA_ROOT
```

**New:**
```python
from scripts.core.config import DATA_ROOT
```

### Issue 2: Script not found

**Old:**
```bash
python src/tasni/download_wise_full.py
```

**New:**
```bash
python src/tasni/download/download_wise_full.py
# or
make download-wise
```

### Issue 3: Missing configuration

**Solution:** The old config.py is still available at `src/tasni/core/config.py`. You don't need to change anything unless you want to use environment variables.

## Rollback Plan

If you need to rollback, the old flat structure scripts are preserved in `src/tasni/legacy/` directory.

To restore:
```bash
cd /mnt/data/tasni/scripts
cp legacy/*.py .
```

However, this is not recommended as you'll lose all the benefits of the new organization.

## Need Help?

- **Documentation:** See `docs/QUICKSTART.md` and `docs/ARCHITECTURE.md`
- **Configuration:** See `docs/pipeline.md`
- **Contributing:** See `CONTRIBUTING.md`
- **Issues:** Open a GitHub issue with the "question" label

## Summary

- **97 scripts** moved to 13 organized subdirectories
- **Import paths** updated for all modules
- **Makefile** provides 50+ commands for common tasks
- **Configuration** supports environment variables (optional)
- **Backward compatibility** maintained via legacy/ directory

The migration is straightforward and the old configuration still works. The new structure provides better organization, maintainability, and scalability.

---

**Migration Status:** ✅ Complete
**Date:** February 2, 2025
