#!/usr/bin/env python3
"""
TASNI: Query NEOWISE Multi-Epoch Photometry

Queries IRSA TAP service for NEOWISE single-exposure source photometry
to enable variability analysis across the 2010-2024 baseline.

NEOWISE observations:
- AllWISE: 2010-2011 (cryogenic + post-cryo)
- NEOWISE-R: 2013-2024 (reactivation, ongoing)
- ~6-month cadence with multiple epochs per visit

Data Access:
- IRSA TAP: https://irsa.ipac.caltech.edu/TAP/
- Tables:
  - allwise_p3as_psd: AllWISE single-exposure photometry
  - neowiser_p1bs_psd: NEOWISE-R single-exposure photometry

Usage:
    python query_neowise_variability.py [--input FILE] [--output FILE]
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.irsa import Irsa

    HAS_ASTROQUERY = True
except ImportError:
    HAS_ASTROQUERY = False

try:
    import pyvo

    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [NEOWISE-VAR] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# IRSA TAP Service
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# Search parameters
SEARCH_RADIUS_ARCSEC = 3.0  # Match radius in arcsec
BATCH_SIZE = 50  # Number of sources per TAP query

# Output columns
OUTPUT_COLUMNS = [
    "designation",
    "ra",
    "dec",
    "mjd",
    "w1mpro_ep",
    "w1sigmpro_ep",
    "w2mpro_ep",
    "w2sigmpro_ep",
    "qual_frame",
    "source_table",
]


def query_neowise_tap(
    ra: float,
    dec: float,
    radius_arcsec: float = 3.0,
    table: str = "neowiser_p1bs_psd",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Query NEOWISE single-exposure source table via TAP.

    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Search radius in arcseconds
        table: IRSA table name
        max_retries: Number of retry attempts

    Returns:
        DataFrame with multi-epoch photometry
    """
    if not HAS_PYVO:
        logger.error("pyvo not installed. Run: pip install pyvo")
        return pd.DataFrame()

    # TAP query - limit to 500 epochs per source
    query = f"""
    SELECT TOP 500
        ra, dec, mjd,
        w1mpro as w1mpro_ep, w1sigmpro as w1sigmpro_ep,
        w2mpro as w2mpro_ep, w2sigmpro as w2sigmpro_ep,
        qual_frame
    FROM {table}
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0})) = 1
    ORDER BY mjd
    """

    for attempt in range(max_retries):
        try:
            service = pyvo.dal.TAPService(IRSA_TAP_URL)
            result = service.run_sync(query, timeout=30)
            df = result.to_table().to_pandas()
            df["source_table"] = table
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.debug(f"Retry {attempt + 1} for ({ra:.4f}, {dec:.4f}) after {wait_time}s")
                time.sleep(wait_time)
            else:
                logger.debug(
                    f"TAP query failed for ({ra}, {dec}) after {max_retries} attempts: {e}"
                )

    return pd.DataFrame()


def query_allwise_mep(ra: float, dec: float, radius_arcsec: float = 3.0) -> pd.DataFrame:
    """
    Query AllWISE Multi-Epoch Photometry table.

    Uses allwise_p3as_psd (single-exposure source photometry).
    """
    return query_neowise_tap(ra, dec, radius_arcsec, table="allwise_p3as_psd")


def query_neowise_reactivation(ra: float, dec: float, radius_arcsec: float = 3.0) -> pd.DataFrame:
    """
    Query NEOWISE Reactivation single-epoch source table.

    Uses neowiser_p1bs_psd (2013-present).
    """
    return query_neowise_tap(ra, dec, radius_arcsec, table="neowiser_p1bs_psd")


def query_cone_batch(sources: pd.DataFrame, radius_arcsec: float = 3.0) -> pd.DataFrame:
    """
    Query multiple sources using TAP with cone search.

    Args:
        sources: DataFrame with 'ra', 'dec', 'designation' columns
        radius_arcsec: Search radius

    Returns:
        DataFrame with all matched epochs
    """
    all_epochs = []

    for idx, row in sources.iterrows():
        ra, dec = row["ra"], row["dec"]
        designation = row["designation"]

        if pd.isna(ra) or pd.isna(dec):
            continue

        # Query NEOWISE-R only (AllWISE MEP queries are slow/unreliable)
        # epochs_aw = query_allwise_mep(ra, dec, radius_arcsec)
        epochs_nw = query_neowise_reactivation(ra, dec, radius_arcsec)

        if len(epochs_nw) > 0:
            epochs_nw["designation"] = designation
            epochs_nw["target_ra"] = ra
            epochs_nw["target_dec"] = dec
            all_epochs.append(epochs_nw)
            logger.info(f"[{idx + 1}/{len(sources)}] {designation}: {len(epochs_nw)} epochs")
        else:
            logger.debug(f"[{idx + 1}/{len(sources)}] {designation}: no epochs found")

        # Rate limiting between queries
        time.sleep(0.5)

        if (idx + 1) % 10 == 0:
            total_epochs = sum(len(e) for e in all_epochs)
            sources_with_data = len(all_epochs)
            logger.info(
                f"Progress: {idx + 1}/{len(sources)} sources, {sources_with_data} with data, {total_epochs} total epochs"
            )

    if all_epochs:
        return pd.concat(all_epochs, ignore_index=True)
    return pd.DataFrame()


def query_bulk_tap(
    sources: pd.DataFrame, radius_arcsec: float = 3.0, batch_size: int = 50
) -> pd.DataFrame:
    """
    Query multiple sources efficiently using bulk TAP queries.

    Uses ADQL with multiple cones in a single query for efficiency.
    """
    if not HAS_PYVO:
        logger.error("pyvo not installed. Run: pip install pyvo")
        return pd.DataFrame()

    all_epochs = []
    n_sources = len(sources)

    for start_idx in range(0, n_sources, batch_size):
        batch = sources.iloc[start_idx : start_idx + batch_size]

        # Build multi-cone ADQL query
        cone_clauses = []
        for _, row in batch.iterrows():
            if pd.notna(row["ra"]) and pd.notna(row["dec"]):
                cone_clauses.append(
                    f"CONTAINS(POINT('ICRS', ra, dec), "
                    f"CIRCLE('ICRS', {row['ra']}, {row['dec']}, {radius_arcsec/3600.0})) = 1"
                )

        if not cone_clauses:
            continue

        where_clause = " OR ".join(cone_clauses)

        # Query NEOWISE-R (most epochs)
        query = f"""
        SELECT
            ra, dec, mjd,
            w1mpro as w1mpro_ep, w1sigmpro as w1sigmpro_ep,
            w2mpro as w2mpro_ep, w2sigmpro as w2sigmpro_ep,
            qual_frame, qi_fact, saa_sep
        FROM neowiser_p1bs_psd
        WHERE ({where_clause})
        """

        try:
            service = pyvo.dal.TAPService(IRSA_TAP_URL)
            result = service.run_sync(query, timeout=300)
            df = result.to_table().to_pandas()

            if len(df) > 0:
                df["source_table"] = "neowiser"
                all_epochs.append(df)
                logger.info(f"Batch {start_idx//batch_size + 1}: {len(df)} epochs")

        except Exception as e:
            logger.warning(f"Bulk query failed for batch {start_idx//batch_size + 1}: {e}")
            # Fall back to individual queries for this batch
            for _, row in batch.iterrows():
                if pd.notna(row["ra"]) and pd.notna(row["dec"]):
                    epochs = query_neowise_reactivation(row["ra"], row["dec"], radius_arcsec)
                    if len(epochs) > 0:
                        epochs["designation"] = row["designation"]
                        all_epochs.append(epochs)
                    time.sleep(0.2)

        time.sleep(0.5)  # Rate limiting between batches

    if all_epochs:
        return pd.concat(all_epochs, ignore_index=True)
    return pd.DataFrame()


def match_epochs_to_sources(
    epochs: pd.DataFrame, sources: pd.DataFrame, radius_arcsec: float = 3.0
) -> pd.DataFrame:
    """
    Match retrieved epochs back to source designations.

    Args:
        epochs: DataFrame with epoch photometry (ra, dec columns)
        sources: DataFrame with source catalog (ra, dec, designation)
        radius_arcsec: Match radius

    Returns:
        Epochs DataFrame with designation column added
    """
    if "designation" in epochs.columns:
        return epochs

    if len(epochs) == 0:
        return epochs

    # Simple nearest-neighbor matching
    matched = []
    for _, epoch in epochs.iterrows():
        dists = (
            np.sqrt((sources["ra"] - epoch["ra"]) ** 2 + (sources["dec"] - epoch["dec"]) ** 2)
            * 3600
        )  # to arcsec

        min_idx = dists.idxmin()
        min_dist = dists[min_idx]

        if min_dist <= radius_arcsec:
            epoch_dict = epoch.to_dict()
            epoch_dict["designation"] = sources.loc[min_idx, "designation"]
            epoch_dict["match_dist_arcsec"] = min_dist
            matched.append(epoch_dict)

    if matched:
        return pd.DataFrame(matched)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="TASNI NEOWISE Variability Query")
    parser.add_argument(
        "--input",
        "-i",
        default="./data/processed/golden_targets.csv",
        help="Input CSV file with target positions",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./data/processed/neowise_epochs.parquet",
        help="Output parquet file with multi-epoch photometry",
    )
    parser.add_argument("--radius", type=float, default=3.0, help="Search radius in arcseconds")
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for bulk TAP queries"
    )
    parser.add_argument(
        "--method",
        choices=["individual", "bulk"],
        default="individual",
        help="Query method (individual or bulk)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: NEOWISE Multi-Epoch Photometry Query")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Search radius: {args.radius} arcsec")

    # Check dependencies
    if not HAS_PYVO:
        logger.error("pyvo not installed. Run: pip install pyvo")
        return

    # Load sources
    if args.input.endswith(".parquet"):
        sources = pd.read_parquet(args.input)
    else:
        sources = pd.read_csv(args.input)

    logger.info(f"Loaded {len(sources)} sources")

    # Query epochs
    logger.info(f"Querying NEOWISE epochs (method: {args.method})...")

    if args.method == "bulk":
        epochs = query_bulk_tap(sources, args.radius, args.batch_size)
        # Match back to sources
        epochs = match_epochs_to_sources(epochs, sources, args.radius)
    else:
        epochs = query_cone_batch(sources, args.radius)

    if len(epochs) == 0:
        logger.warning("No epochs retrieved!")
        return

    logger.info(f"Retrieved {len(epochs)} total epochs")

    # Summary statistics
    n_sources_with_epochs = epochs["designation"].nunique()
    epochs_per_source = epochs.groupby("designation").size()

    logger.info("=" * 60)
    logger.info("NEOWISE Multi-Epoch Summary:")
    logger.info(f"  Sources with epochs: {n_sources_with_epochs}/{len(sources)}")
    logger.info(f"  Total epochs: {len(epochs)}")
    logger.info(f"  Mean epochs per source: {epochs_per_source.mean():.1f}")
    logger.info(f"  Min epochs: {epochs_per_source.min()}")
    logger.info(f"  Max epochs: {epochs_per_source.max()}")

    # Time baseline
    if "mjd" in epochs.columns:
        mjd_range = epochs["mjd"].max() - epochs["mjd"].min()
        years_baseline = mjd_range / 365.25
        logger.info(f"  Time baseline: {years_baseline:.1f} years")
        logger.info(f"  MJD range: {epochs['mjd'].min():.1f} - {epochs['mjd'].max():.1f}")

    logger.info("=" * 60)

    # Save results
    output_path = Path(args.output)
    epochs.to_parquet(output_path, index=False)
    logger.info(f"Saved epochs to {output_path}")

    # Also save summary per source
    summary_path = output_path.with_suffix(".summary.csv")
    summary = epochs.groupby("designation").agg(
        {
            "mjd": ["min", "max", "count"],
            "w1mpro_ep": ["mean", "std", "min", "max"],
            "w2mpro_ep": ["mean", "std", "min", "max"],
        }
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved epoch summary to {summary_path}")

    return epochs


if __name__ == "__main__":
    main()
