#!/usr/bin/env python3
"""
TASNI: Submillimeter Cross-Match for YSO Contamination Detection
================================================================

Cross-matches sources against submillimeter catalogs to identify protostars
and YSOs that could contaminate the brown dwarf sample.

Catalogs used:
- Bolocam Galactic Plane Survey (II/268) - 1.1mm continuum
- SCOPE: SCUBA-2 Continuum Observations of Pre-protostellar Evolution (J/ApJS/254/33)
- ATLASGAL: APEX Telescope Large Area Survey of the Galaxy (J/A+A/568/A41)

Protostars have strong submillimeter emission from cold dust envelopes,
which cold brown dwarfs lack.

Usage:
    python crossmatch_submm.py [--input FILE] [--output FILE] [--test]
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

from tasni.core.config import LOG_DIR, OUTPUT_DIR, SUBMM_MATCH_RADIUS, ensure_dirs

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "crossmatch_submm.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# VizieR catalog identifiers
SUBMM_CATALOGS = {
    "bolocam_gps": "II/268/bolocat",  # Bolocam Galactic Plane Survey
    "scope": "J/ApJS/254/33/table1",  # SCOPE protostellar catalog
    "atlasgal": "J/A+A/568/A41/table1",  # ATLASGAL compact sources
}


def query_vizier_cone(ra, dec, radius_arcsec, catalogs):
    """
    Query VizieR for submillimeter sources near a position.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Search radius in arcseconds
        catalogs: List of VizieR catalog identifiers

    Returns:
        dict with catalog name -> DataFrame of matches
    """
    from astroquery.vizier import Vizier

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    radius = radius_arcsec * u.arcsec

    # Configure VizieR query
    v = Vizier(columns=["*"], row_limit=100)

    results = {}
    for cat_name, cat_id in catalogs.items():
        try:
            result = v.query_region(coord, radius=radius, catalog=cat_id)
            if result and len(result) > 0:
                results[cat_name] = result[0].to_pandas()
        except Exception as e:
            logger.debug(f"Error querying {cat_name}: {e}")
            continue

    return results


def batch_query_vizier(sources, radius_arcsec=30, batch_size=50):
    """
    Batch query VizieR for multiple sources efficiently.

    Args:
        sources: DataFrame with 'ra' and 'dec' columns
        radius_arcsec: Search radius in arcseconds
        batch_size: Number of sources to query before pausing

    Returns:
        DataFrame with submm match information added
    """

    logger.info(f"Querying VizieR for {len(sources)} sources...")
    logger.info(f"Search radius: {radius_arcsec} arcsec")
    logger.info(f"Catalogs: {list(SUBMM_CATALOGS.keys())}")

    # Initialize result columns
    sources = sources.copy()
    sources["submm_match"] = False
    sources["submm_catalog"] = ""
    sources["submm_sep_arcsec"] = np.nan
    sources["submm_flux"] = np.nan

    n_matches = 0
    start_time = time.time()

    for idx, (row_idx, row) in enumerate(sources.iterrows()):
        if idx > 0 and idx % 100 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            logger.info(
                f"  Processed {idx}/{len(sources)} sources ({rate:.1f}/s), {n_matches} matches"
            )

        # Rate limiting - be nice to VizieR
        if idx > 0 and idx % batch_size == 0:
            time.sleep(1.0)

        try:
            matches = query_vizier_cone(row["ra"], row["dec"], radius_arcsec, SUBMM_CATALOGS)

            if matches:
                # Take first match from any catalog
                for cat_name, match_df in matches.items():
                    if len(match_df) > 0:
                        sources.at[row_idx, "submm_match"] = True
                        sources.at[row_idx, "submm_catalog"] = cat_name
                        n_matches += 1

                        # Try to get separation and flux
                        if "_r" in match_df.columns:
                            sources.at[row_idx, "submm_sep_arcsec"] = match_df["_r"].iloc[0]
                        break

        except Exception as e:
            logger.debug(f"Error querying source {row_idx}: {e}")
            continue

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s")
    logger.info(f"Total submm matches: {n_matches} ({100*n_matches/len(sources):.2f}%)")

    return sources


def crossmatch_submm_bulk(sources, radius_arcsec=30):
    """
    Alternative: Download catalogs and do local cross-match.

    This is faster for large source lists but requires more setup.
    For Tier5 (~4000 sources), individual queries are acceptable.
    """

    logger.info("Using bulk cross-match method...")

    # For now, use the per-source query method
    # NOTE: For > 10000 sources, implement bulk download + local match for performance

    return batch_query_vizier(sources, radius_arcsec)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-match sources against submillimeter catalogs"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_radio_silent.parquet"),
        help="Input source file (parquet or csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_submm.parquet"),
        help="Output file with submm matches",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: only process first 10 sources"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=SUBMM_MATCH_RADIUS,
        help=f"Match radius in arcseconds (default: {SUBMM_MATCH_RADIUS})",
    )
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: Submillimeter Cross-Match")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Match radius: {args.radius} arcsec")

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
    logger.info("")
    sources = batch_query_vizier(sources, radius_arcsec=args.radius)

    # Summary
    n_matches = sources["submm_match"].sum()
    logger.info("")
    logger.info("=== SUMMARY ===")
    logger.info(f"Total sources: {len(sources)}")
    logger.info(f"Submm matches: {n_matches} ({100*n_matches/len(sources):.2f}%)")
    logger.info(f"Clean (no submm): {len(sources) - n_matches}")

    if n_matches > 0:
        logger.info("")
        logger.info("Matches by catalog:")
        for cat in SUBMM_CATALOGS.keys():
            count = (sources["submm_catalog"] == cat).sum()
            if count > 0:
                logger.info(f"  {cat}: {count}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        sources.to_parquet(output_path, index=False)
    else:
        sources.to_csv(output_path, index=False)

    logger.info(f"Results saved to: {output_path}")

    # Also save list of contaminated sources
    if n_matches > 0:
        contam_path = output_path.parent / "submm_contaminated.csv"
        contam = sources[sources["submm_match"]]
        contam.to_csv(contam_path, index=False)
        logger.info(f"Contaminated sources: {contam_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
