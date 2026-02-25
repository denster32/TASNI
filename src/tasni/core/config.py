"""
TASNI Configuration
====================

Single source of truth for TASNI paths and runtime settings.
Environment variables override defaults.
"""

import os
from pathlib import Path

# Load environment variables from .env when python-dotenv is available.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

# =============================================================================
# PATHS
# =============================================================================

# Base data directory
DATA_ROOT = Path(os.environ.get("TASNI_DATA_ROOT", str(Path(__file__).resolve().parents[3])))

# Subdirectories
WISE_DIR = DATA_ROOT / "data" / "wise"
GAIA_DIR = DATA_ROOT / "data" / "gaia"
CROSSMATCH_DIR = DATA_ROOT / "data" / "crossmatch"
SECONDARY_DIR = DATA_ROOT / "data" / "secondary"  # 2MASS, Spitzer, etc.
OUTPUT_DIR = DATA_ROOT / "output"
LOG_DIR = DATA_ROOT / "logs"
SCRIPTS_DIR = DATA_ROOT / "scripts"
CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
VALIDATION_DIR = DATA_ROOT / "validation"
BENCHMARK_DIR = DATA_ROOT / "benchmarks"

# =============================================================================
# CATALOG SETTINGS
# =============================================================================

# AllWISE Source Catalog
WISE_CATALOG = "allwise_p3as_psd"
WISE_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# Columns we need from WISE
WISE_COLUMNS = [
    "designation",  # WISE designation
    "ra",
    "dec",  # Position (degrees)
    "w1mpro",
    "w2mpro",
    "w3mpro",
    "w4mpro",  # Profile-fit magnitudes
    "w1sigmpro",
    "w2sigmpro",
    "w3sigmpro",
    "w4sigmpro",  # Uncertainties
    "w1snr",
    "w2snr",
    "w3snr",
    "w4snr",  # Signal-to-noise
    "pmra",
    "pmdec",  # Proper motion (mas/yr)
    "cc_flags",  # Contamination/confusion flags
    "ext_flg",  # Extended source flag
    "ph_qual",  # Photometric quality flag
    "nb",  # Number of blend components
    "na",  # Active deblend flag
]

# Gaia DR3
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
GAIA_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "parallax",
    "pmra",
    "pmdec",
]

# =============================================================================
# SECONDARY CATALOGS (Multi-Wavelength)
# =============================================================================

# 2MASS (Near-IR: J, H, Ks bands)
TWO_MASS_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
TWO_MASS_TABLE = "fp_psc"
TWO_MASS_COLUMNS = ["ra", "dec", "j_m", "h_m", "ks_m", "ph_qual", "rd_flg"]

# Spitzer (Mid-IR: IRAC 3.6, 4.5, 5.8, 8.0 μm)
SPITZER_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
SPITZER_TABLE = "irac_spet"
SPITZER_COLUMNS = ["ra", "dec", "i1_mag", "i2_mag", "i3_mag", "i4_mag"]

# AKARI (Far-IR)
AKARI_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# Chandra (X-ray)
CHANDRA_URL = "https://cxc.harvard.edu/csc/"

# Fermi (Gamma-ray)
FERMI_URL = "https://fermi.gsfc.nasa.gov/ssc/data/access/"

# VLASS (Radio 3GHz)
VLASS_URL = "https://archive.nrao.edu/vlass/"

# NVSS (Radio 1.4GHz)
NVSS_FILE = DATA_ROOT / "data" / "nvss.dat"

# =============================================================================
# CHINESE VO CATALOGS (NADC - nadc.china-vo.org)
# =============================================================================

# BASS/Legacy Survey DR10 (Deep Optical g,r,z - includes BASS + MzLS)
# Depth: g=24.2, r=23.6, z=23.0 (5σ) - deeper than Pan-STARRS!
# Coverage: Northern Galactic Cap (Dec > +32°), 5400 sq deg
LEGACY_SWEEP_URL = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/north/sweep/"
LEGACY_DIR = DATA_ROOT / "data" / "legacy_survey"
LEGACY_COLUMNS = [
    "ra",
    "dec",
    "flux_g",
    "flux_r",
    "flux_z",  # Fluxes in nanomaggies
    "flux_ivar_g",
    "flux_ivar_r",
    "flux_ivar_z",  # Inverse variance
    "type",  # Morphological type (PSF, REX, EXP, DEV, SER)
    "ref_cat",
    "ref_id",  # Cross-ref to Gaia/Tycho
    "maskbits",  # Quality mask
    "fitbits",
]

# LAMOST DR12 (Spectroscopy - stellar parameters)
# 28 million spectra, 8.3M stellar parameters
# Parameters: Teff, logg, [Fe/H], RV, elemental abundances
LAMOST_TAP_URL = "https://www.lamost.org/dr12/v1.1/voservice/tap"
LAMOST_DIR = DATA_ROOT / "data" / "lamost"
LAMOST_COLUMNS = [
    "obsid",  # Observation ID
    "ra",
    "dec",  # Coordinates
    "snr_g",
    "snr_r",
    "snr_i",  # Signal-to-noise per band
    "teff",
    "teff_err",  # Effective temperature
    "logg",
    "logg_err",  # Surface gravity
    "feh",
    "feh_err",  # Metallicity [Fe/H]
    "rv",
    "rv_err",  # Radial velocity
    "class",  # Object classification (STAR, GALAXY, QSO, Unknown)
    "subclass",  # Spectral subtype (A0, K2, M5, etc.)
]

# LAMOST stellar types that explain IR emission (use as veto)
LAMOST_KNOWN_IR_TYPES = [
    "M",
    "L",
    "T",
    "Y",  # Cool dwarfs
    "C",  # Carbon stars
    "S",  # S-type stars
    "WC",
    "WN",  # Wolf-Rayet
    "Be",  # Be stars (circumstellar disk)
]

# =============================================================================
# CROSSMATCH SETTINGS
# =============================================================================

# Match radius in arcseconds
MATCH_RADIUS_ARCSEC = 3.0

# HEALPix settings for chunked processing
HEALPIX_NSIDE = 32  # 12,288 tiles, ~3.4 sq deg each
HEALPIX_ORDER = "nested"

# =============================================================================
# QUALITY FILTERS
# =============================================================================

# Minimum SNR in W1 and W2
MIN_SNR_W1 = 5.0
MIN_SNR_W2 = 5.0

# Photometric quality - require A or B in W1 and W2
GOOD_PH_QUAL = ["A", "B"]

# Clean contamination flags
CLEAN_CC_FLAGS = "0000"

# =============================================================================
# YSO CONTAMINATION FILTERS (Added for data quality improvement)
# =============================================================================

# Galactic plane filter - YSOs concentrate in disk
# Cold brown dwarfs should be distributed isotropically
MIN_GALACTIC_LATITUDE = 5.0  # |b| > 5° required (removes ~20% of sky but ~50% of YSOs)

# W3/W4 vetting - cold brown dwarfs (T_eff < 500K) should be W3/W4 faint
# YSOs, AGB stars, and dust-embedded objects are W3/W4 bright
W3_FAINT_THRESHOLD = 14.0  # W3 > 14 mag expected for cold BDs (or undetected)
W4_FAINT_THRESHOLD = 12.0  # W4 > 12 mag expected for cold BDs (or undetected)

# Submillimeter contamination - protostars have submm emission
SUBMM_MATCH_RADIUS = 30.0  # arcsec - larger radius for submm beam size

# =============================================================================
# WEIRDNESS SCORING
# =============================================================================

# Color thresholds for anomaly detection
W1_W2_ANOMALY_THRESHOLD = -0.5  # Anomalously blue (shouldn't happen)
W3_BRIGHT_THRESHOLD = 12.0  # Bright in mid-IR
W1_FAINT_THRESHOLD = 15.0  # Faint in near-IR

# Isolation bonus (log10 of separation in arcsec)
ISOLATION_WEIGHT = 0.5

# Anomaly thresholds (to be refined by baseline analysis)
WEIRDNESS_THRESHOLD_LOW = 2.0  # Mildly interesting
WEIRDNESS_THRESHOLD_MED = 5.0  # Definitely weird
WEIRDNESS_THRESHOLD_HIGH = 10.0  # Extreme anomalies

# Baseline statistics (TODO: run analyze_baseline.py to update from data)
BASELINE_W1_W2_MEAN = 0.5  # Typical stellar W1-W2 color
BASELINE_W1_W2_STD = 0.3  # Standard deviation
ISOLATION_THRESHOLD_95 = 300.0  # 95th percentile isolation (arcsec)

# =============================================================================
# CHECKPOINT SETTINGS
# =============================================================================

# Checkpoint files
CHECKPOINT_WISE_DOWNLOAD = CHECKPOINT_DIR / "wise_download.json"
CHECKPOINT_GAIA_DOWNLOAD = CHECKPOINT_DIR / "gaia_download.json"
CHECKPOINT_CROSSMATCH = CHECKPOINT_DIR / "crossmatch.json"
CHECKPOINT_FILTER = CHECKPOINT_DIR / "filter.json"

# Checkpoint interval (save every N tiles processed)
CHECKPOINT_INTERVAL = 100

# Resume behavior
RESUME_FROM_CHECKPOINT = True  # Automatically resume if checkpoint exists

# =============================================================================
# NETWORK & RETRY SETTINGS
# =============================================================================

# Retry configuration for network operations
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2  # Exponential backoff base
RETRY_BACKOFF_MAX = 60  # Max seconds between retries
REQUEST_TIMEOUT = 300  # Seconds to wait for TAP response

# Connection pool settings
MAX_CONNECTIONS = 10
CONNECTION_POOL_SIZE = 20

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Known object catalogs for validation
BROWN_DWARF_CATALOG = "simbad_wise_ultracool"  # SIMBAD query for L/T/Y dwarfs
STAR_FORMING_REGIONS = [
    "Orion",
    "Taurus",
    "Ophiuchus",
    "Perseus",
]

# Validation test regions (small patches for quick testing)
VALIDATION_PATCH_SIZE = 10  # square degrees
VALIDATION_PATCHES = [
    {"ra": 83.63, "dec": 22.01, "radius": 5},  # Orion
    {"ra": 56.0, "dec": 24.0, "radius": 5},  # Random sparse region
]

# Expected detection rates (for validation)
EXPECTED_BROWN_DWARF_RECOVERY = 0.95  # Should recover 95% of known dwarfs
FALSE_POSITIVE_RATE_TARGET = 0.01  # <1% false positives

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "pdf"  # Default output format
COLOR_MAP = "RdYlBu_r"  # Blue to red colormap (thermal)

# Plot styles
STYLE_SCATTER_ALPHA = 0.6
STYLE_LINE_WIDTH = 2
STYLE_FONT_SIZE = 12

# Interactive plots
INTERACTIVE_PLOTS = True  # Generate HTML versions with plotly
PLOTLY_HEIGHT = 600
PLOTLY_WIDTH = 800

# =============================================================================
# HARDWARE
# =============================================================================

# GPU Configuration
# -----------------
# NVIDIA RTX 3060 12GB - CUDA (card1, renderD128)
#   - Use for: RAPIDS/cuDF, CUDA-specific workloads
#   - Env: set COMPUTE_ENV to activate
#
# Intel Arc A770 16GB - XPU (card2, renderD129)
#   - Use for: PyTorch XPU, larger tensor ops
#   - Env: set XPU_ENV to activate
#   - 16GB VRAM > 12GB VRAM

USE_GPU = True
USE_CUDA = True  # NVIDIA RTX 3060
USE_XPU = True  # Intel Arc A770

# Memory limits (leave headroom)
CUDA_MEMORY_LIMIT = 10 * 1024  # MB (10GB of 12GB)
XPU_MEMORY_LIMIT = 14 * 1024  # MB (14GB of 16GB)

# Parallel processing
N_WORKERS = 16  # Leave 4 cores for system on i9-10850K (20 threads)

# Chunk size for processing
CHUNK_SIZE = 1_000_000  # rows per chunk

# Environment paths
COMPUTE_ENV = os.environ.get("COMPUTE_ENV", "")
XPU_ENV = os.environ.get("XPU_ENV", "")

# =============================================================================
# MULTI-WAVELENGTH SCORING
# =============================================================================

# Cross-match radii for different catalogs
RADIUS_WISE_GAIA = 3.0  # arcsec
RADIUS_WISE_2MASS = 2.0
RADIUS_WISE_SPITZER = 2.0
RADIUS_WISE_RADIO = 5.0  # Larger for radio positions
RADIUS_WISE_XRAY = 5.0
RADIUS_WISE_LEGACY = 2.0  # BASS/Legacy Survey deep optical
RADIUS_WISE_LAMOST = 3.0  # LAMOST spectroscopy

# Multi-wavelength anomaly scoring weights
IR_BRIGHTNESS_WEIGHT = 1.0
COLOR_BONUS_WEIGHT = 2.0
OPTICAL_PENALTY_WEIGHT = -10.0
RADIO_PENALTY_WEIGHT = -5.0
XRAY_PENALTY_WEIGHT = -3.0
ISOLATION_BONUS_WEIGHT = 0.5

# BASS/Legacy Survey scoring (deep optical veto)
LEGACY_DEEP_OPTICAL_PENALTY = -15.0  # Stronger than Gaia - if visible in deep optical, not stealthy
LEGACY_FAINT_THRESHOLD_G = 23.5  # Fainter than this = borderline detection
LEGACY_FAINT_THRESHOLD_R = 23.0

# LAMOST spectral scoring
LAMOST_KNOWN_TYPE_PENALTY = -20.0  # Known stellar type explains IR emission
LAMOST_UNKNOWN_BONUS = 5.0  # Has spectrum but unknown type = interesting
LAMOST_TEMP_MISMATCH_BONUS = 10.0  # Spectral Teff disagrees with IR Teff = very interesting

# Tier thresholds (percentile)
TIER5_THRESHOLD = 99.9  # Top 0.1% - "gold" sample
TIER4_THRESHOLD = 99.0  # Top 1%
TIER3_THRESHOLD = 95.0  # Top 5%
TIER2_THRESHOLD = 90.0  # Top 10%
TIER1_THRESHOLD = 75.0  # Top 25%

# =============================================================================
# VARIABILITY THRESHOLDS
# =============================================================================

# Fade rate threshold (mmag/yr) - minimum for FADING classification
# Based on: 15 mmag/yr = 0.015 mag/yr
# Justification: >3x expected from measurement noise, significantly above
# typical stellar variability, consistent with detected fade rates (17-53 mmag/yr)
FADE_RATE_THRESHOLD_MMAG_YR = 15.0

# Trend threshold (mag/yr) - same as above but in magnitudes
TREND_THRESHOLD_MAG_YR = 0.015

# Chi-squared threshold for variability detection
# Sources with chi2_nu > this are considered variable
CHI2_VARIABILITY_THRESHOLD = 3.0

# P-value threshold for fade significance
# P < this indicates statistically significant fading
FADE_P_VALUE_THRESHOLD = 0.01

# Minimum baseline (years) for reliable fade detection
MIN_BASELINE_YEARS = 2.0

# Minimum epochs for variability analysis
MIN_EPOCHS_VARIABILITY = 10

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_dirs():
    """Create all directories if they don't exist"""
    for d in [
        WISE_DIR,
        GAIA_DIR,
        CROSSMATCH_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        CHECKPOINT_DIR,
        VALIDATION_DIR,
        BENCHMARK_DIR,
        LEGACY_DIR,
        LAMOST_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def get_checkpoint_file(phase: str) -> Path:
    """Get checkpoint file path for a given pipeline phase"""
    checkpoint_files = {
        "wise_download": CHECKPOINT_WISE_DOWNLOAD,
        "gaia_download": CHECKPOINT_GAIA_DOWNLOAD,
        "crossmatch": CHECKPOINT_CROSSMATCH,
        "filter": CHECKPOINT_FILTER,
    }
    return checkpoint_files.get(phase, CHECKPOINT_DIR / f"{phase}.json")


def get_output_filename(name: str) -> Path:
    """Get path for output file"""
    output_files = {
        "anomalies": "anomalies_ranked.parquet",
        "anomalies_csv": "top_anomalies.csv",
        "extreme_csv": "extreme_anomalies.csv",
        "orphans": "wise_no_gaia_match.parquet",
        "summary": "summary.json",
        "report": "validation_report.html",
    }
    return OUTPUT_DIR / output_files.get(name, name)


def get_wise_file(healpix_idx=None):
    """Get path to WISE data file"""
    if healpix_idx is not None:
        return WISE_DIR / f"wise_hp{healpix_idx:05d}.parquet"
    return WISE_DIR / "allwise_full.parquet"


def get_gaia_file(healpix_idx=None):
    """Get path to Gaia data file"""
    if healpix_idx is not None:
        return GAIA_DIR / f"gaia_hp{healpix_idx:05d}.parquet"
    return GAIA_DIR / "gaia_full.parquet"


def get_orphan_file(healpix_idx=None):
    """Get path to orphan (no-match) file"""
    if healpix_idx is not None:
        return CROSSMATCH_DIR / f"orphans_hp{healpix_idx:05d}.parquet"
    return OUTPUT_DIR / "wise_no_gaia_match.parquet"


def get_anomaly_file():
    """Get path to final anomaly catalog"""
    return OUTPUT_DIR / "anomalies_ranked.parquet"


if __name__ == "__main__":
    ensure_dirs()
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"WISE_DIR: {WISE_DIR}")
    print(f"GAIA_DIR: {GAIA_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"Available space: check with df -h {DATA_ROOT}")
