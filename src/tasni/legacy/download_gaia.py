"""
TASNI: Download Gaia DR3 Catalog
=================================

Downloads Gaia DR3 source positions and photometry.

Full catalog is ~1.8 billion sources.
We only need: position, G magnitude, parallax for cross-matching.

Usage:
    python download_gaia.py [--test]

    --test: Download small test region only (1 sq degree)
"""

import argparse
from pathlib import Path

from astroquery.gaia import Gaia

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "gaia"

# Columns we need (minimal for cross-matching)
GAIA_COLUMNS = """
    source_id,
    ra, dec,
    phot_g_mean_mag,
    parallax,
    pmra, pmdec
"""


def download_test_region():
    """Download a small test region matching WISE test region"""

    print("Downloading Gaia test region (Bootes Void area)...")

    # Same region as WISE test
    query = f"""
    SELECT {GAIA_COLUMNS}
    FROM gaiadr3.gaia_source
    WHERE ra BETWEEN 217.5 AND 218.5
    AND dec BETWEEN 25.5 AND 26.5
    """

    job = Gaia.launch_job_async(query)
    result = job.get_results()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "gaia_test_region.parquet"

    df = result.to_pandas()
    df.to_parquet(output_file)

    print(f"Downloaded {len(df)} sources to {output_file}")
    return df


def download_full_catalog():
    """
    Download full Gaia DR3 catalog.

    Strategy: Download pre-computed crossmatch tables or bulk files.
    """

    print("Full Gaia download not yet implemented.")
    print()
    print("Options for full catalog:")
    print("1. Gaia Archive bulk download: https://gea.esac.esa.int/archive/")
    print("2. Use pre-computed WISE x Gaia crossmatch from CDS")
    print("3. Query in HEALPix chunks via TAP")
    print()
    print("Recommended: Download the pre-computed crossmatch table first,")
    print("then we only need to find WISE sources NOT in that table.")

    # TODO: Implement chunked download
    # TODO: Consider using CDS crossmatch tables


def main():
    parser = argparse.ArgumentParser(description="Download Gaia DR3 catalog")
    parser.add_argument("--test", action="store_true", help="Download test region only")
    args = parser.parse_args()

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1  # No limit

    if args.test:
        download_test_region()
    else:
        download_full_catalog()


if __name__ == "__main__":
    main()
