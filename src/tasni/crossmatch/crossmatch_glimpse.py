#!/usr/bin/env python3
"""
TASNI: GLIMPSE/Spitzer Cross-Match
==================================

Cross-match sources against GLIMPSE (Galactic Legacy Infrared Mid-Plane Survey
Extraordinaire) from Spitzer for independent mid-IR confirmation.

GLIMPSE provides IRAC 3.6, 4.5, 5.8, 8.0 μm photometry that can be compared
to WISE W1, W2 for consistency checking.

Catalogs:
- GLIMPSE I (II/293/glimpse1)
- GLIMPSE II (II/293/glimpse2)
- GLIMPSE 3D (J/ApJ/778/15)

Usage:
    python crossmatch_glimpse.py [--input FILE] [--output FILE]
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tasni.core.config import LOG_DIR, OUTPUT_DIR, RADIUS_WISE_SPITZER, ensure_dirs

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "crossmatch_glimpse.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def query_irsa_glimpse(ra, dec, radius_arcsec=3.0):
    """
    Query IRSA TAP for GLIMPSE sources near a position.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Search radius in arcseconds

    Returns:
        DataFrame of GLIMPSE matches or None
    """
    from pyvo.dal import TAPService

    service = TAPService("https://irsa.ipac.caltech.edu/TAP")

    # GLIMPSE tables at IRSA
    tables = ["glimpse_s07", "glimpse2_s07"]

    for table in tables:
        try:
            query = f"""
            SELECT *
            FROM {table}
            WHERE CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600})
            ) = 1
            """
            result = service.run_sync(query, timeout=30)
            if result and len(result.to_table()) > 0:
                return result.to_table().to_pandas()
        except Exception as e:
            logger.debug(f"Error querying {table}: {e}")
            continue

    return None


def crossmatch_glimpse_batch(sources, radius_arcsec=3.0, batch_pause=1.0):
    """
    Cross-match sources against GLIMPSE catalog.

    Args:
        sources: DataFrame with 'ra' and 'dec' columns
        radius_arcsec: Search radius in arcseconds
        batch_pause: Seconds to pause between batches (rate limiting)

    Returns:
        DataFrame with GLIMPSE match information added
    """
    logger.info(f"Cross-matching {len(sources)} sources against GLIMPSE...")
    logger.info(f"Search radius: {radius_arcsec} arcsec")

    sources = sources.copy()
    sources["glimpse_match"] = False
    sources["glimpse_sep_arcsec"] = np.nan
    sources["glimpse_i1"] = np.nan  # IRAC 3.6 μm
    sources["glimpse_i2"] = np.nan  # IRAC 4.5 μm
    sources["glimpse_i3"] = np.nan  # IRAC 5.8 μm
    sources["glimpse_i4"] = np.nan  # IRAC 8.0 μm

    n_matches = 0
    start_time = time.time()

    for idx, (row_idx, row) in enumerate(sources.iterrows()):
        if idx > 0 and idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            logger.info(f"  Processed {idx}/{len(sources)} ({rate:.1f}/s), {n_matches} matches")
            time.sleep(batch_pause)

        try:
            result = query_irsa_glimpse(row["ra"], row["dec"], radius_arcsec)

            if result is not None and len(result) > 0:
                sources.at[row_idx, "glimpse_match"] = True
                n_matches += 1

                # Get photometry from first match
                row_data = result.iloc[0]

                # IRAC magnitudes (column names vary by table)
                for col, dest in [
                    ("mag_3_6", "glimpse_i1"),
                    ("mag_4_5", "glimpse_i2"),
                    ("mag_5_8", "glimpse_i3"),
                    ("mag_8_0", "glimpse_i4"),
                ]:
                    if col in row_data:
                        sources.at[row_idx, dest] = row_data[col]

        except Exception as e:
            logger.debug(f"Error for source {row_idx}: {e}")
            continue

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"Total GLIMPSE matches: {n_matches} ({100*n_matches/len(sources):.2f}%)")

    return sources


def main():
    parser = argparse.ArgumentParser(description="Cross-match sources against GLIMPSE catalog")
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_radio_silent.parquet"),
        help="Input source file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_glimpse.parquet"),
        help="Output file with GLIMPSE matches",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS_WISE_SPITZER,
        help=f"Match radius in arcseconds (default: {RADIUS_WISE_SPITZER})",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: only process first 10 sources"
    )
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: GLIMPSE Cross-Match")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Load sources
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        sources = pd.read_parquet(input_path)
    else:
        sources = pd.read_csv(input_path)

    logger.info(f"Loaded {len(sources)} sources")

    if args.test:
        logger.info("TEST MODE: Processing first 10 sources only")
        sources = sources.head(10)

    # Run cross-match
    sources = crossmatch_glimpse_batch(sources, radius_arcsec=args.radius)

    # Summary
    n_matches = sources["glimpse_match"].sum()
    logger.info("")
    logger.info("=== SUMMARY ===")
    logger.info(f"Total sources: {len(sources)}")
    logger.info(f"GLIMPSE matches: {n_matches} ({100*n_matches/len(sources):.2f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        sources.to_parquet(output_path, index=False)
    else:
        sources.to_csv(output_path, index=False)

    logger.info(f"Results saved to: {output_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
