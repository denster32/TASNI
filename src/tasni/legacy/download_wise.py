"""
TASNI: Download AllWISE Catalog
================================

Downloads the AllWISE Source Catalog from IRSA.

Full catalog is ~747 million sources, ~300GB.
This script downloads in chunks by sky region.

Usage:
    python download_wise.py [--test]

    --test: Download small test region only (1 sq degree)
"""

import argparse
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.irsa import Irsa

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "wise"
CATALOG = "allwise_p3as_psd"  # AllWISE Source Catalog

# Columns we need
COLUMNS = [
    "designation",  # WISE designation
    "ra",
    "dec",  # Position
    "w1mpro",
    "w2mpro",
    "w3mpro",
    "w4mpro",  # Magnitudes in 4 bands
    "w1sigmpro",
    "w2sigmpro",
    "w3sigmpro",
    "w4sigmpro",  # Uncertainties
    "w1snr",
    "w2snr",
    "w3snr",
    "w4snr",  # Signal-to-noise
    "pmra",
    "pmdec",  # Proper motion
    "cc_flags",  # Contamination/confusion flags
    "ext_flg",  # Extended source flag
    "ph_qual",  # Photometric quality
]


def download_test_region():
    """Download a small test region (1 sq degree centered on Bootes Void)"""

    print("Downloading test region (Bootes Void area)...")

    # Bootes Void approximate center
    coord = SkyCoord(ra=218.0, dec=26.0, unit="deg")

    result = Irsa.query_region(
        coord,
        catalog=CATALOG,
        spatial="Box",
        width=1 * u.degree,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "wise_test_region.parquet"

    df = result.to_pandas()
    df.to_parquet(output_file)

    print(f"Downloaded {len(df)} sources to {output_file}")
    return df


def download_full_catalog():
    """
    Download full AllWISE catalog.

    Strategy: Query by HEALPix tiles to manage memory and allow resume.
    """

    print("Full catalog download not yet implemented.")
    print("For full download, use IRSA's bulk download service:")
    print("https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd")
    print()
    print("Or use the TAP service for programmatic access:")
    print("https://irsa.ipac.caltech.edu/TAP")

    # TODO: Implement chunked download by HEALPix region
    # TODO: Add resume capability
    # TODO: Add integrity verification


def main():
    parser = argparse.ArgumentParser(description="Download AllWISE catalog")
    parser.add_argument("--test", action="store_true", help="Download test region only")
    args = parser.parse_args()

    if args.test:
        download_test_region()
    else:
        download_full_catalog()


if __name__ == "__main__":
    main()
