# TASNI API Reference

Complete API documentation for TASNI pipeline components.

## Table of Contents
1. [Core Modules](#core-modules)
2. [Download Modules](#download-modules)
3. [Crossmatch Modules](#crossmatch-modules)
4. [Analysis Modules](#analysis-modules)
5. [Filtering Modules](#filtering-modules)
6. [Generation Modules](#generation-modules)
7. [Utility Modules](#utility-modules)
8. [Configuration](#configuration)

---

## Core Modules

### config

TASNI configuration management.

**Location:** `src/tasni/core/config.py`

**Main Variables:**

```python
# Paths
DATA_ROOT: Path              # Base data directory
WISE_DIR: Path               # WISE catalog directory
GAIA_DIR: Path               # Gaia catalog directory
OUTPUT_DIR: Path              # Output directory
LOG_DIR: Path                # Log directory
CHECKPOINT_DIR: Path          # Checkpoint directory

# HEALPix
HEALPIX_NSIDE: int           # HEALPix NSIDE parameter (32)
HEALPIX_NPIXELS: int         # Number of HEALPix pixels (12,288)

# Matching
MATCH_RADIUS_ARCSEC: float   # Crossmatch radius in arcsec (3.0)

# Processing
N_WORKERS: int               # Number of parallel workers (16)

# Catalogs
WISE_CATALOG: str            # AllWISE catalog name
GAIA_CATALOG: str            # Gaia DR3 catalog name
WISE_COLUMNS: List[str]      # WISE columns to download
GAIA_COLUMNS: List[str]      # Gaia columns to download

# TAP URLs
WISE_TAP_URL: str            # WISE TAP service URL
GAIA_TAP_URL: str            # Gaia TAP service URL
```

**Functions:**

```python
def print_config() -> None
    """Print current TASNI configuration"""
```

**Example:**

```python
from scripts.core.config import DATA_ROOT, HEALPIX_NSIDE

print(f"Data root: {DATA_ROOT}")
print(f"HEALPix NSIDE: {HEALPIX_NSIDE}")
```

### tasni_logging

Logging configuration for TASNI.

**Location:** `src/tasni/core/tasni_logging.py`

**Functions:**

```python
def setup_logging(
    name: str = 'tasni',
    level: str = 'INFO',
    log_file: bool = True,
    console: bool = True
) -> logging.Logger
    """
    Setup logging for TASNI modules.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Enable file logging
        console: Enable console logging

    Returns:
        Configured logger
    """
```

**Example:**

```python
from scripts.core.tasni_logging import setup_logging

logger = setup_logging('my_module', level='DEBUG')
logger.info('Module initialized')
```

---

## Download Modules

### download_wise_full

Download AllWISE catalog using HEALPix tiling.

**Location:** `src/tasni/download/download_wise_full.py`

**Main Function:**

```python
def download_wise(
    hpix: int = None,
    nside: int = 32,
    output_dir: Path = None,
    batch_size: int = 1000,
    overwrite: bool = False
) -> Path
    """
    Download WISE catalog for specific HEALPix tile.

    Args:
        hpix: HEALPix tile number (None = download all)
        nside: HEALPix NSIDE parameter
        output_dir: Output directory
        batch_size: TAP query batch size
        overwrite: Overwrite existing files

    Returns:
        Path to downloaded parquet file
    """
```

**Example:**

```python
from scripts.download.download_wise_full import download_wise

# Download specific tile
download_wise(hpix=1234)

# Download all tiles
download_wise()
```

### download_gaia_full

Download Gaia DR3 catalog using HEALPix tiling.

**Location:** `src/tasni/download/download_gaia_full.py`

**Main Function:**

```python
def download_gaia(
    hpix: int = None,
    nside: int = 32,
    output_dir: Path = None,
    parallel: bool = True,
    workers: int = 8,
    overwrite: bool = False
) -> Path
    """
    Download Gaia catalog for specific HEALPix tile.

    Args:
        hpix: HEALPix tile number (None = download all)
        nside: HEALPix NSIDE parameter
        output_dir: Output directory
        parallel: Enable parallel downloads
        workers: Number of parallel workers
        overwrite: Overwrite existing files

    Returns:
        Path to downloaded parquet file
    """
```

**Example:**

```python
from scripts.download.download_gaia_full import download_gaia

# Download all tiles in parallel
download_gaia(parallel=True, workers=16)
```

### async_neowise_query

Asynchronous NEOWISE epoch queries.

**Location:** `src/tasni/download/async_neowise_query.py`

**Main Function:**

```python
async def query_neowise_epochs(
    sources: pd.DataFrame,
    max_concurrent: int = 10,
    timeout: int = 600
) -> pd.DataFrame
    """
    Query NEOWISE epochs asynchronously.

    Args:
        sources: DataFrame with designation column
        max_concurrent: Maximum concurrent requests
        timeout: Query timeout in seconds

    Returns:
        DataFrame with NEOWISE epoch data
    """
```

**Example:**

```python
import asyncio
from scripts.download.async_neowise_query import query_neowise_epochs
import pandas as pd

sources = pd.read_csv('data/processed/final/golden_targets.csv')

async def main():
    epochs = await query_neowise_epochs(sources)
    epochs.to_parquet('data/processed/final/neowise_epochs.parquet')

asyncio.run(main())
```

---

## Crossmatch Modules

### crossmatch_full

Main crossmatching module for WISE × Gaia.

**Location:** `src/tasni/crossmatch/crossmatch_full.py`

**Classes:**

```python
class Crossmatcher:
    """WISE × Gaia crossmatcher"""

    def __init__(
        self,
        match_radius_arcsec: float = 3.0,
        use_gpu: bool = False
    ):
        """
        Initialize crossmatcher.

        Args:
            match_radius_arcsec: Matching radius in arcsec
            use_gpu: Enable GPU acceleration
        """

    def match(
        self,
        wise_df: pd.DataFrame,
        gaia_df: pd.DataFrame,
        by_healpix: bool = True,
        nside: int = 32
    ) -> pd.DataFrame:
        """
        Crossmatch WISE and Gaia catalogs.

        Args:
            wise_df: WISE catalog DataFrame
            gaia_df: Gaia catalog DataFrame
            by_healpix: Use HEALPix-based matching
            nside: HEALPix NSIDE parameter

        Returns:
            Matched DataFrame with columns:
            - wise_designation
            - wise_ra, wise_dec
            - gaia_source_id
            - gaia_ra, gaia_dec
            - separation_arcsec
            - matched (bool)
        """

    def match_by_healpix(
        self,
        wise_df: pd.DataFrame,
        gaia_df: pd.DataFrame,
        nside: int = 32
    ) -> pd.DataFrame:
        """
        Crossmatch using HEALPix tiling.

        Args:
            wise_df: WISE catalog DataFrame
            gaia_df: Gaia catalog DataFrame
            nside: HEALPix NSIDE parameter

        Returns:
            Matched DataFrame
        """
```

**Example:**

```python
import pandas as pd
from scripts.crossmatch.crossmatch_full import Crossmatcher

# Load catalogs
wise_df = pd.read_parquet('data/wise/wise_hp00000.parquet')
gaia_df = pd.read_parquet('data/gaia/gaia_hp00000.parquet')

# Crossmatch
crossmatcher = Crossmatcher(match_radius_arcsec=3.0)
matched_df = crossmatcher.match(wise_df, gaia_df)

# Save results
matched_df.to_parquet('data/crossmatch/orphans_hp00000.parquet')
```

---

## Analysis Modules

### analyze_kinematics

Proper motion and kinematics analysis.

**Location:** `src/tasni/analysis/analyze_kinematics.py`

**Functions:**

```python
def calculate_pm(
    pmra: pd.Series,
    pmdec: pd.Series
) -> pd.Series
    """
    Calculate total proper motion.

    Args:
        pmra: Proper motion in RA (mas/yr)
        pmdec: Proper motion in Dec (mas/yr)

    Returns:
        Total proper motion (mas/yr)
    """

def calculate_pm_angle(
    pmra: pd.Series,
    pmdec: pd.Series
) -> pd.Series
    """
    Calculate proper motion direction angle.

    Args:
        pmra: Proper motion in RA (mas/yr)
        pmdec: Proper motion in Dec (mas/yr)

    Returns:
        Direction angle (degrees, 0 = North, 90 = East)
    """

def classify_motion(
    pm: pd.Series,
    pm_threshold: float = 20.0
) -> pd.Series
    """
    Classify proper motion.

    Args:
        pm: Total proper motion (mas/yr)
        pm_threshold: Threshold for "high" PM (mas/yr)

    Returns:
        Classification string (HIGH, MEDIUM, LOW, ZERO)
    """
```

**Example:**

```python
import pandas as pd
from scripts.analysis.analyze_kinematics import calculate_pm, classify_motion

df = pd.read_parquet('data/processed/final/golden_kinematics.parquet')

# Calculate total PM
df['pm_total'] = calculate_pm(df['pmra'], df['pmdec'])

# Classify motion
df['pm_class'] = classify_motion(df['pm_total'])
```

### compute_ir_variability

NEOWISE IR variability analysis.

**Location:** `src/tasni/analysis/compute_ir_variability.py`

**Functions:**

```python
def compute_variability_metrics(
    epochs_df: pd.DataFrame,
    min_epochs: int = 20
) -> pd.DataFrame
    """
    Compute variability metrics for a source.

    Args:
        epochs_df: DataFrame with NEOWISE epochs
        min_epochs: Minimum number of epochs

    Returns:
        DataFrame with variability metrics:
        - n_epochs: Number of epochs
        - w1_rms: RMS scatter in W1
        - w2_rms: RMS scatter in W2
        - w1_chi2: Chi-squared statistic
        - w2_chi2: Chi-squared statistic
        - stetson_j: Stetson J index
        - fade_rate: Linear trend (mag/yr)
        - is_variable: Boolean (True if variable)
    """

def calculate_stetson_j(
    w1: pd.Series,
    w2: pd.Series,
    w1_err: pd.Series,
    w2_err: pd.Series
) -> float
    """
    Calculate Stetson J index.

    Args:
        w1: W1 magnitudes
        w2: W2 magnitudes
        w1_err: W1 uncertainties
        w2_err: W2 uncertainties

    Returns:
        Stetson J index
    """

def detect_trend(
    mjd: pd.Series,
    mag: pd.Series,
    mag_err: pd.Series
) -> Tuple[float, float]
    """
    Detect linear trend in light curve.

    Args:
        mjd: Modified Julian Date
        mag: Magnitudes
        mag_err: Magnitude uncertainties

    Returns:
        (slope, slope_err) where slope is mag/yr
    """
```

**Example:**

```python
import pandas as pd
from scripts.analysis.compute_ir_variability import compute_variability_metrics

# Load NEOWISE epochs
epochs_df = pd.read_parquet('data/processed/final/neowise_epochs.parquet')

# Compute variability for each source
variability_df = compute_variability_metrics(epochs_df)

# Save results
variability_df.to_csv('data/processed/final/golden_variability.csv', index=False)
```

---

## Filtering Modules

### filter_anomalies_full

Multi-wavelength anomaly filtering.

**Location:** `src/tasni/filtering/filter_anomalies_full.py`

**Main Function:**

```python
def filter_anomalies(
    orphans_df: pd.DataFrame,
    gaia_df: pd.DataFrame = None,
    twomass_df: pd.DataFrame = None,
    ps1_df: pd.DataFrame = None,
    legacy_df: pd.DataFrame = None,
    nvss_df: pd.DataFrame = None,
    w1_w2_threshold: float = 0.5,
    match_radius_arcsec: float = 3.0
) -> pd.DataFrame
    """
    Filter orphans using multi-wavelength criteria.

    Args:
        orphans_df: WISE sources (no Gaia match)
        gaia_df: Gaia catalog (optional, pre-crossmatched)
        twomass_df: 2MASS catalog (optional)
        ps1_df: Pan-STARRS catalog (optional)
        legacy_df: Legacy Survey catalog (optional)
        nvss_df: NVSS catalog (optional)
        w1_w2_threshold: W1-W2 color threshold (mag)
        match_radius_arcsec: Crossmatch radius (arcsec)

    Returns:
        Filtered DataFrame with columns:
        - All original WISE columns
        - passed_thermal (bool)
        - passed_twomass (bool)
        - passed_ps1 (bool)
        - passed_legacy (bool)
        - passed_nvss (bool)
        - passed_all (bool)
    """
```

**Example:**

```python
import pandas as pd
from scripts.filtering.filter_anomalies_full import filter_anomalies

# Load orphan sources
orphans_df = pd.read_parquet('data/crossmatch/orphans.parquet')

# Apply filters
filtered_df = filter_anomalies(
    orphans_df,
    w1_w2_threshold=0.5,
    match_radius_arcsec=3.0
)

# Save results
filtered_df.to_parquet('output/anomalies_filtered.parquet')
```

---

## Generation Modules

### generate_golden_list

Generate golden target list from filtered candidates.

**Location:** `src/tasni/generation/generate_golden_list.py`

**Main Function:**

```python
def generate_golden_list(
    anomalies_df: pd.DataFrame,
    variability_df: pd.DataFrame = None,
    kinematics_df: pd.DataFrame = None,
    top_n: int = 100,
    scoring_weights: Dict[str, float] = None
) -> pd.DataFrame
    """
    Generate golden target list from anomalies.

    Args:
        anomalies_df: Filtered anomalies DataFrame
        variability_df: Variability metrics (optional)
        kinematics_df: Kinematics data (optional)
        top_n: Number of top candidates
        scoring_weights: Scoring weight dictionary

    Returns:
        Golden target list with columns:
        - designation, ra, dec
        - w1mag, w2mag, w1_w2_color
        - composite_score
        - rank (1 = highest score)
        - variability_metrics (if provided)
        - kinematics (if provided)
    """
```

**Example:**

```python
import pandas as pd
from scripts.generation.generate_golden_list import generate_golden_list

# Load filtered anomalies
anomalies_df = pd.read_parquet('output/anomalies_ranked.parquet')

# Generate golden targets
golden_df = generate_golden_list(anomalies_df, top_n=100)

# Save results
golden_df.to_csv('data/processed/final/golden_targets.csv', index=False)
```

---

## Utility Modules

### data_manager

Data lifecycle management.

**Location:** `src/tasni/utils/data_manager.py`

**Classes:**

```python
class DataManager:
    """Manage TASNI data lifecycle"""

    def __init__(self, tasni_root: Path = None):
        """
        Initialize data manager.

        Args:
            tasni_root: TASNI root directory (auto-detect if None)
        """

    def cleanup_old_logs(
        self,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Compress or remove old log files.

        Args:
            dry_run: Preview changes without executing

        Returns:
            Dictionary with cleanup statistics
        """

    def archive_intermediate_files(
        self,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Move intermediate files to archive.

        Args:
            dry_run: Preview changes without executing

        Returns:
            Dictionary with archive statistics
        """

    def cleanup_archive(
        self,
        dry_run: bool = False
    ) -> Dict[str, int]:
        """
        Remove old/duplicate files from archive.

        Args:
            dry_run: Preview changes without executing

        Returns:
            Dictionary with cleanup statistics
        """

    def check_archive_size(self) -> Dict[str, float]:
        """
        Check current archive size.

        Returns:
            Dictionary with size statistics
        """

    def generate_manifest(self) -> str:
        """
        Generate manifest of all data files.

        Returns:
            Path to manifest JSON file
        """

    def run_cleanup_cycle(
        self,
        dry_run: bool = False
    ) -> Dict:
        """
        Run complete cleanup cycle.

        Args:
            dry_run: Preview changes without executing

        Returns:
            Dictionary with complete cleanup results
        """
```

**Example:**

```python
from scripts.utils.data_manager import DataManager

# Initialize
manager = DataManager()

# Run cleanup
results = manager.run_cleanup_cycle()

# Generate manifest
manifest_path = manager.generate_manifest()
print(f"Manifest: {manifest_path}")
```

### security_audit

Security scanning tool.

**Location:** `src/tasni/utils/security_audit.py`

**Functions:**

```python
def scan_for_secrets(file_path: Path) -> List[Tuple[str, int, str]]:
    """
    Scan a file for potential secrets.

    Args:
        file_path: File to scan

    Returns:
        List of (file, line_num, description) tuples
    """

def scan_for_hardcoded_paths(file_path: Path) -> List[Tuple[str, int, str]]:
    """
    Scan for hardcoded absolute paths.

    Args:
        file_path: File to scan

    Returns:
        List of (file, line_num, description) tuples
    """

def check_file_permissions(tasni_root: Path) -> List[Tuple[str, str]]:
    """
    Check for files with overly permissive permissions.

    Args:
        tasni_root: TASNI root directory

    Returns:
        List of (file, issue) tuples
    """
```

**Example:**

```python
import subprocess

# Run security audit
subprocess.run(['python', 'src/tasni/utils/security_audit.py'])
```

---

## Configuration

### Environment Variables

TASNI supports configuration via environment variables (`.env` file).

**Example .env:**

```bash
# Paths
TASNI_DATA_ROOT=/mnt/data/tasni
TASNI_WISE_DIR=${TASNI_DATA_ROOT}/data/wise
TASNI_GAIA_DIR=${TASNI_DATA_ROOT}/data/gaia

# HEALPix
HEALPIX_NSIDE=32

# Processing
N_WORKERS=16
MATCH_RADIUS_ARCSEC=3.0

# Filtering
W1_W2_THRESHOLD=0.5
TEMP_THRESHOLD=500

# GPU
USE_CUDA=auto
USE_XPU=auto

# Logging
LOG_LEVEL=INFO
```

**Loading Environment Variables:**

```python
# Option 1: Use python-dotenv
from dotenv import load_dotenv
load_dotenv()

# Option 2: Use config_env module
from scripts.core.config_env import *

# Option 3: Use os.environ
import os
workers = int(os.getenv('N_WORKERS', '16'))
```

---

## Quick API Reference

| Module | Purpose | Key Function/Class |
|---------|----------|-------------------|
| `core.config` | Configuration | `DATA_ROOT`, `HEALPIX_NSIDE` |
| `core.tasni_logging` | Logging | `setup_logging()` |
| `download.download_wise_full` | Download WISE | `download_wise()` |
| `download.download_gaia_full` | Download Gaia | `download_gaia()` |
| `download.async_neowise_query` | NEOWISE epochs | `query_neowise_epochs()` |
| `crossmatch.crossmatch_full` | Crossmatching | `Crossmatcher` class |
| `analysis.analyze_kinematics` | Kinematics | `calculate_pm()`, `classify_motion()` |
| `analysis.compute_ir_variability` | Variability | `compute_variability_metrics()` |
| `filtering.filter_anomalies_full` | Filtering | `filter_anomalies()` |
| `generation.generate_golden_list` | Golden targets | `generate_golden_list()` |
| `utils.data_manager` | Data management | `DataManager` class |
| `utils.security_audit` | Security scanning | `scan_for_secrets()` |

---

**For more detailed information, see:**
- `docs/ARCHITECTURE.md` - System design
- `docs/pipeline.md` - Pipeline execution guide
- Source code docstrings
