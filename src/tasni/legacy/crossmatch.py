"""
TASNI: Cross-Match WISE and Gaia
=================================

Cross-matches WISE infrared sources against Gaia optical sources.
Outputs WISE sources with NO Gaia counterpart within search radius.

These are our candidates: heat without light.

Usage:
    python crossmatch.py [--test] [--radius 3.0]

    --test: Use test region data only
    --radius: Match radius in arcseconds (default: 3.0)
"""

import argparse
from pathlib import Path

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def load_wise(test=False):
    """Load WISE catalog"""
    if test:
        path = DATA_DIR / "wise" / "wise_test_region.parquet"
    else:
        path = DATA_DIR / "wise" / "allwise_full.parquet"

    print(f"Loading WISE from {path}...")
    return pd.read_parquet(path)


def load_gaia(test=False):
    """Load Gaia catalog"""
    if test:
        path = DATA_DIR / "gaia" / "gaia_test_region.parquet"
    else:
        path = DATA_DIR / "gaia" / "gaia_full.parquet"

    print(f"Loading Gaia from {path}...")
    return pd.read_parquet(path)


def crossmatch(wise_df, gaia_df, radius_arcsec=3.0):
    """
    Cross-match WISE against Gaia.

    Returns WISE sources with NO match within radius.
    """

    print("Building coordinate arrays...")

    wise_coords = SkyCoord(ra=wise_df["ra"].values * u.degree, dec=wise_df["dec"].values * u.degree)

    gaia_coords = SkyCoord(ra=gaia_df["ra"].values * u.degree, dec=gaia_df["dec"].values * u.degree)

    print(
        f"Cross-matching {len(wise_coords)} WISE sources against {len(gaia_coords)} Gaia sources..."
    )
    print(f"Match radius: {radius_arcsec} arcsec")

    # Find nearest Gaia source for each WISE source
    idx, sep2d, _ = match_coordinates_sky(wise_coords, gaia_coords)

    # Find WISE sources with NO match within radius
    no_match_mask = sep2d.arcsec > radius_arcsec

    orphans = wise_df[no_match_mask].copy()
    orphans["nearest_gaia_sep_arcsec"] = sep2d.arcsec[no_match_mask]

    print(f"Found {len(orphans)} WISE sources with no Gaia counterpart")
    print(f"({len(orphans)/len(wise_df)*100:.2f}% of WISE sources)")

    return orphans


def main():
    parser = argparse.ArgumentParser(description="Cross-match WISE and Gaia")
    parser.add_argument("--test", action="store_true", help="Use test region only")
    parser.add_argument("--radius", type=float, default=3.0, help="Match radius (arcsec)")
    args = parser.parse_args()

    # Load data
    wise_df = load_wise(test=args.test)
    gaia_df = load_gaia(test=args.test)

    # Cross-match
    orphans = crossmatch(wise_df, gaia_df, radius_arcsec=args.radius)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_test" if args.test else "_full"
    output_file = OUTPUT_DIR / f"wise_no_gaia_match{suffix}.parquet"

    orphans.to_parquet(output_file)
    print(f"Saved {len(orphans)} orphan sources to {output_file}")


if __name__ == "__main__":
    main()
