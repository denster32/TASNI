"""
TASNI - Thermal Anomaly Search for Non-communicating Intelligence

A systematic search for mid-infrared sources without optical counterparts
that exhibit systematic dimming, using WISE/NEOWISE and Gaia DR3 data.
"""

__version__ = "1.0.0"
__author__ = "Dennis Palucki"

# Import main modules for easy access
from . import analysis, core, download, filtering, ml, utils

__all__ = [
    "analysis",
    "core",
    "download",
    "filtering",
    "ml",
    "utils",
]
