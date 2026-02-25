#!/usr/bin/env python3
"""
TASNI: UKIDSS Deep Near-IR Cross-Match
======================================

Cross-match sources against UKIDSS (UKIRT Infrared Deep Sky Survey) for
deep near-IR veto. UKIDSS is deeper than 2MASS and can detect faint
optical counterparts that 2MASS misses.

Surveys included:
- Large Area Survey (LAS) - J=19.6, K=18.2
- Galactic Plane Survey (GPS) - J=19.8, K=18.1
- Galactic Clusters Survey (GCS)
- Deep Extragalactic Survey (DXS)

Usage:
    python crossmatch_ukidss.py [--input FILE] [--output FILE]
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from tasni.core.config import LOG_DIR, OUTPUT_DIR, RADIUS_WISE_2MASS, ensure_dirs

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "crossmatch_ukidss.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def query_vizier_ukidss(ra, dec, radius_arcsec=3.0):
    """
    Query VizieR for UKIDSS sources near a position.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Search radius in arcseconds

    Returns:
        DataFrame of UKIDSS matches or None
    """
    from astroquery.vizier import Vizier

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    radius = radius_arcsec * u.arcsec

    # UKIDSS catalogs at VizieR
    # II/319 = UKIDSS DR10
    catalogs = [
        "II/319/las10",  # Large Area Survey
        "II/319/gps10",  # Galactic Plane Survey
        "II/319/gcs10",  # Galactic Clusters Survey
        "II/319/dxs10",  # Deep Extragalactic Survey
    ]

    v = Vizier(columns=["*"], row_limit=10)

    for cat in catalogs:
        try:
            result = v.query_region(coord, radius=radius, catalog=cat)
            if result and len(result) > 0:
                return result[0].to_pandas()
        except Exception as e:
            logger.debug(f"Error querying {cat}: {e}")
            continue

    return None


def crossmatch_ukidss_batch(sources, radius_arcsec=3.0, batch_pause=1.0):
    """
    Cross-match sources against UKIDSS catalog.

    Args:
        sources: DataFrame with 'ra' and 'dec' columns
        radius_arcsec: Search radius in arcseconds
        batch_pause: Seconds to pause between batches (rate limiting)

    Returns:
        DataFrame with UKIDSS match information added
    """
    logger.info(f"Cross-matching {len(sources)} sources against UKIDSS...")
    logger.info(f"Search radius: {radius_arcsec} arcsec")

    sources = sources.copy()
    sources["ukidss_match"] = False
    sources["ukidss_sep_arcsec"] = np.nan
    sources["ukidss_j"] = np.nan
    sources["ukidss_h"] = np.nan
    sources["ukidss_k"] = np.nan
    sources["ukidss_survey"] = ""

    n_matches = 0
    start_time = time.time()

    for idx, (row_idx, row) in enumerate(sources.iterrows()):
        if idx > 0 and idx % 50 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            logger.info(f"  Processed {idx}/{len(sources)} ({rate:.1f}/s), {n_matches} matches")
            time.sleep(batch_pause)

        try:
            result = query_vizier_ukidss(row["ra"], row["dec"], radius_arcsec)

            if result is not None and len(result) > 0:
                sources.at[row_idx, "ukidss_match"] = True
                n_matches += 1

                # Get photometry from first match
                row_data = result.iloc[0]

                # J, H, K magnitudes
                for col, dest in [
                    ("Jmag", "ukidss_j"),
                    ("Hmag", "ukidss_h"),
                    ("Kmag", "ukidss_k"),
                    ("Japermag3", "ukidss_j"),
                    ("Hapermag3", "ukidss_h"),
                    ("Kapermag3", "ukidss_k"),
                ]:
                    if col in row_data and pd.notna(row_data[col]):
                        sources.at[row_idx, dest] = row_data[col]

                # Separation if available
                if "_r" in row_data:
                    sources.at[row_idx, "ukidss_sep_arcsec"] = row_data["_r"]

        except Exception as e:
            logger.debug(f"Error for source {row_idx}: {e}")
            continue

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"Total UKIDSS matches: {n_matches} ({100*n_matches/len(sources):.2f}%)")

    return sources


def check_deep_detections(sources):
    """
    Analyze UKIDSS detections that were missed by 2MASS.

    UKIDSS is ~3 mag deeper than 2MASS, so sources detected in UKIDSS
    but not 2MASS may indicate faint optical/near-IR counterparts.
    """
    ukidss_only = sources[
        sources["ukidss_match"] & (~sources.get("twomass_match", True))  # No 2MASS match
    ]

    if len(ukidss_only) > 0:
        logger.info("")
        logger.info("=== DEEP DETECTIONS (UKIDSS only, no 2MASS) ===")
        logger.info(f"Sources with UKIDSS but no 2MASS: {len(ukidss_only)}")

        # These are interesting - faint near-IR counterparts
        for col in ["ukidss_j", "ukidss_k"]:
            if col in ukidss_only.columns:
                median = ukidss_only[col].median()
                logger.info(f"  Median {col}: {median:.2f} mag")

    return sources


def main():
    parser = argparse.ArgumentParser(description="Cross-match sources against UKIDSS deep near-IR")
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_radio_silent.parquet"),
        help="Input source file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_ukidss.parquet"),
        help="Output file with UKIDSS matches",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=RADIUS_WISE_2MASS,
        help=f"Match radius in arcseconds (default: {RADIUS_WISE_2MASS})",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: only process first 10 sources"
    )
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: UKIDSS Deep Near-IR Cross-Match")
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
    sources = crossmatch_ukidss_batch(sources, radius_arcsec=args.radius)

    # Analyze deep detections
    sources = check_deep_detections(sources)

    # Summary
    n_matches = sources["ukidss_match"].sum()
    logger.info("")
    logger.info("=== SUMMARY ===")
    logger.info(f"Total sources: {len(sources)}")
    logger.info(f"UKIDSS matches: {n_matches} ({100*n_matches/len(sources):.2f}%)")
    logger.info(f"No UKIDSS: {len(sources) - n_matches} (truly invisible in near-IR)")

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
