#!/usr/bin/env python3
"""
TASNI: Re-Filter Tier5 Sample with Enhanced Quality Cuts
========================================================

Applies all new data quality filters to the existing Tier5 sample:
1. Galactic plane filter (|b| > 5°) - remove YSO contamination
2. W3/W4 vetting - flag/remove dusty objects
3. Submillimeter cross-match - identify protostars (optional)

This creates a "cleaned" Tier5 sample with higher confidence.

Usage:
    python refilter_tier5.py [--apply-submm] [--strict]

Options:
    --apply-submm: Run submillimeter cross-match (slower, needs internet)
    --strict: Remove W3/W4 flagged sources (default: just flag them)
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from tasni.core.config import (
    LOG_DIR,
    MIN_GALACTIC_LATITUDE,
    OUTPUT_DIR,
    W3_FAINT_THRESHOLD,
    W4_FAINT_THRESHOLD,
    ensure_dirs,
)

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "refilter_tier5.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_tier5():
    """Load current Tier5 sample."""
    path = OUTPUT_DIR / "final" / "tier5_radio_silent.parquet"
    logger.info(f"Loading Tier5 from {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} sources")
    return df


def apply_galactic_filter(df):
    """
    Remove sources in Galactic plane (|b| < MIN_GALACTIC_LATITUDE).

    YSOs concentrate in the disk, cold brown dwarfs are distributed isotropically.
    """
    original = len(df)

    logger.info(f"Applying Galactic plane filter (|b| > {MIN_GALACTIC_LATITUDE}°)...")

    # Convert to Galactic coordinates
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
    galactic = coords.galactic

    # Add Galactic coordinates
    df = df.copy()
    df["gal_l"] = galactic.l.deg
    df["gal_b"] = galactic.b.deg

    # Filter
    mask = np.abs(df["gal_b"]) > MIN_GALACTIC_LATITUDE
    df_filtered = df[mask].copy()

    removed = original - len(df_filtered)
    logger.info(
        f"  Removed {removed:,} sources at |b| < {MIN_GALACTIC_LATITUDE}° ({100*removed/original:.1f}%)"
    )
    logger.info(f"  Remaining: {len(df_filtered):,}")

    return df_filtered


def flag_w34_bright(df):
    """
    Flag sources with bright W3/W4 (potential YSO/dusty contamination).

    Cold brown dwarfs should be W3/W4 faint or undetected.
    """
    df = df.copy()

    logger.info("Flagging W3/W4 bright sources...")

    df["w34_yso_flag"] = False

    if "w3mpro" in df.columns:
        w3_bright = (df["w3mpro"] < W3_FAINT_THRESHOLD) & df["w3mpro"].notna()
        df.loc[w3_bright, "w34_yso_flag"] = True
        logger.info(f"  W3 < {W3_FAINT_THRESHOLD}: {w3_bright.sum():,} flagged")

    if "w4mpro" in df.columns:
        w4_bright = (df["w4mpro"] < W4_FAINT_THRESHOLD) & df["w4mpro"].notna()
        df.loc[w4_bright, "w34_yso_flag"] = True
        logger.info(f"  W4 < {W4_FAINT_THRESHOLD}: {w4_bright.sum():,} flagged")

    total_flagged = df["w34_yso_flag"].sum()
    logger.info(f"  Total flagged: {total_flagged:,} ({100*total_flagged/len(df):.1f}%)")

    return df


def apply_submm_check(df):
    """
    Cross-match against submillimeter catalogs to identify protostars.

    This is slower as it requires network queries.
    """
    from crossmatch_submm import batch_query_vizier

    logger.info("Running submillimeter cross-match...")
    df = batch_query_vizier(df, radius_arcsec=30)

    n_matches = df["submm_match"].sum()
    logger.info(f"  Submm matches: {n_matches:,}")

    return df


def generate_summary(df_original, df_cleaned):
    """Generate summary statistics."""

    logger.info("")
    logger.info("=" * 60)
    logger.info("RE-FILTER SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Original Tier5: {len(df_original):,} sources")
    logger.info(f"After filters: {len(df_cleaned):,} sources")
    logger.info(
        f"Removed: {len(df_original) - len(df_cleaned):,} ({100*(len(df_original) - len(df_cleaned))/len(df_original):.1f}%)"
    )

    logger.info("")
    logger.info("Galactic distribution:")
    if "gal_b" in df_cleaned.columns:
        logger.info(
            f"  |b| range: {df_cleaned['gal_b'].abs().min():.1f}° to {df_cleaned['gal_b'].abs().max():.1f}°"
        )
        logger.info(f"  Mean |b|: {df_cleaned['gal_b'].abs().mean():.1f}°")

    if "w34_yso_flag" in df_cleaned.columns:
        n_flagged = df_cleaned["w34_yso_flag"].sum()
        n_clean = len(df_cleaned) - n_flagged
        logger.info("")
        logger.info("W3/W4 status:")
        logger.info(f"  Clean (W3/W4 faint): {n_clean:,}")
        logger.info(f"  Flagged (W3/W4 bright): {n_flagged:,}")

    if "submm_match" in df_cleaned.columns:
        n_submm = df_cleaned["submm_match"].sum()
        logger.info("")
        logger.info("Submillimeter contamination:")
        logger.info(f"  Submm matches: {n_submm:,}")
        logger.info(f"  Clean: {len(df_cleaned) - n_submm:,}")


def main():
    parser = argparse.ArgumentParser(description="Re-filter Tier5 with enhanced quality cuts")
    parser.add_argument(
        "--apply-submm", action="store_true", help="Run submillimeter cross-match (slower)"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Remove (not just flag) W3/W4 bright sources"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_cleaned.parquet"),
        help="Output file for cleaned Tier5",
    )
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: Re-Filter Tier5 Sample")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Galactic cut: |b| > {MIN_GALACTIC_LATITUDE}°")
    logger.info(f"W3 threshold: > {W3_FAINT_THRESHOLD} mag")
    logger.info(f"W4 threshold: > {W4_FAINT_THRESHOLD} mag")
    logger.info(f"Strict mode: {args.strict}")
    logger.info(f"Submm check: {args.apply_submm}")

    # Load original Tier5
    df_original = load_tier5()

    # Apply filters
    logger.info("")
    df = apply_galactic_filter(df_original)

    logger.info("")
    df = flag_w34_bright(df)

    # Optional submm check
    if args.apply_submm:
        logger.info("")
        df = apply_submm_check(df)

    # Strict mode: remove flagged sources
    if args.strict:
        before = len(df)
        df = df[~df["w34_yso_flag"]].copy()
        logger.info(f"Strict mode: removed {before - len(df):,} W3/W4 bright sources")

        if "submm_match" in df.columns:
            before = len(df)
            df = df[~df["submm_match"]].copy()
            logger.info(f"Strict mode: removed {before - len(df):,} submm matches")

    # Generate summary
    generate_summary(df_original, df)

    # Save cleaned catalog
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("")
    logger.info(f"Cleaned Tier5 saved to: {output_path}")
    logger.info(f"Total sources: {len(df):,}")

    # Also save summary CSV of flagged sources for review
    flagged = df[df.get("w34_yso_flag", False) | df.get("submm_match", False)]
    if len(flagged) > 0:
        flagged_path = output_path.parent / "tier5_flagged_for_review.csv"
        flagged.to_csv(flagged_path, index=False)
        logger.info(f"Flagged sources for review: {flagged_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
