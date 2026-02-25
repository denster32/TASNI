#!/usr/bin/env python3
"""
TASNI: Kinematic Analysis for Golden Targets

Analyzes proper motion data to identify nearby sources (high PM = likely natural).
Sources without optical counterparts but with high proper motion are almost certainly
nearby brown dwarfs or other natural objects, not distant megastructures.

Scoring Logic:
- HIGH_PM_PENALTY: High proper motion indicates nearby object (natural)
- LOW_PM_BONUS: Very low PM is consistent with distant object (interesting)
- GALACTIC_HALO_BONUS: Halo kinematics are more unusual

Usage:
    python analyze_kinematics.py [--input FILE] [--output FILE]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [KINEMATICS] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Kinematic Scoring Constants
HIGH_PM_THRESHOLD = 500.0  # mas/yr - very high PM
MEDIUM_PM_THRESHOLD = 200.0  # mas/yr - moderate PM
LOW_PM_THRESHOLD = 50.0  # mas/yr - low PM

HIGH_PM_PENALTY = -30.0  # Very high PM = definitely nearby
MEDIUM_PM_PENALTY = -15.0  # Moderate PM = likely nearby
LOW_PM_BONUS = 10.0  # Low PM = consistent with distant

# Galactic latitude thresholds
HIGH_LATITUDE_THRESHOLD = 30.0  # |b| > 30 deg = halo-like
LOW_LATITUDE_THRESHOLD = 10.0  # |b| < 10 deg = disk-like

HALO_LOCATION_BONUS = 5.0  # High galactic latitude is less confused
DISK_LOCATION_PENALTY = -5.0  # Low galactic latitude has more confusion


def compute_galactic_coordinates(ra: np.ndarray, dec: np.ndarray) -> tuple:
    """
    Convert equatorial (RA, Dec) to Galactic (l, b) coordinates.

    Uses standard transformation with J2000 pole at:
    - RA_NGP = 192.85948 deg
    - Dec_NGP = 27.12825 deg
    - l_NCP = 122.93192 deg
    """
    # Convert to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)

    # North Galactic Pole (J2000)
    ra_ngp = np.radians(192.85948)
    dec_ngp = np.radians(27.12825)
    l_ncp = np.radians(122.93192)

    # Compute Galactic latitude b
    sin_b = np.sin(dec_ngp) * np.sin(dec_rad) + np.cos(dec_ngp) * np.cos(dec_rad) * np.cos(
        ra_rad - ra_ngp
    )
    b = np.arcsin(np.clip(sin_b, -1, 1))

    # Compute Galactic longitude l
    cos_b = np.cos(b)
    # Avoid division by zero for sources at poles
    cos_b = np.where(np.abs(cos_b) < 1e-10, 1e-10, cos_b)

    sin_l_minus_lncp = np.cos(dec_rad) * np.sin(ra_rad - ra_ngp) / cos_b
    cos_l_minus_lncp = (
        np.cos(dec_ngp) * np.sin(dec_rad)
        - np.sin(dec_ngp) * np.cos(dec_rad) * np.cos(ra_rad - ra_ngp)
    ) / cos_b

    l = l_ncp - np.arctan2(sin_l_minus_lncp, cos_l_minus_lncp)

    # Normalize l to [0, 2*pi)
    l = np.mod(l, 2 * np.pi)

    return np.degrees(l), np.degrees(b)


def classify_pm_category(pm_total: float) -> str:
    """Classify proper motion into categories."""
    if pm_total >= HIGH_PM_THRESHOLD:
        return "very_high"
    elif pm_total >= MEDIUM_PM_THRESHOLD:
        return "high"
    elif pm_total >= LOW_PM_THRESHOLD:
        return "moderate"
    else:
        return "low"


def classify_galactic_region(b: float) -> str:
    """Classify galactic region based on latitude."""
    abs_b = abs(b)
    if abs_b >= HIGH_LATITUDE_THRESHOLD:
        return "halo"
    elif abs_b >= LOW_LATITUDE_THRESHOLD:
        return "thick_disk"
    else:
        return "thin_disk"


def estimate_distance_from_pm(pm_total: float, assumed_v_tan: float = 50.0) -> float:
    """
    Estimate rough distance from proper motion.

    Uses: d (pc) = v_tan (km/s) / (4.74 * pm (arcsec/yr))

    Args:
        pm_total: Total proper motion in mas/yr
        assumed_v_tan: Assumed tangential velocity in km/s (default 50 km/s typical for stars)

    Returns:
        Estimated distance in parsecs
    """
    if pm_total <= 0:
        return np.inf

    pm_arcsec = pm_total / 1000.0  # Convert mas to arcsec
    d_pc = assumed_v_tan / (4.74 * pm_arcsec)
    return d_pc


def analyze_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform kinematic analysis on candidates.

    Adds columns:
    - gal_l, gal_b: Galactic coordinates
    - pm_category: Proper motion classification
    - galactic_region: Disk/halo classification
    - est_distance_pc: Rough distance estimate from PM
    - kinematic_score: Score adjustment from kinematics
    - kinematic_flag: Summary flag
    """
    logger.info(f"Analyzing kinematics for {len(df)} candidates")

    # Ensure we have the required columns
    required_cols = ["ra", "dec", "pmra", "pmdec"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute pm_total if not present
    if "pm_total" not in df.columns:
        df["pm_total"] = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)

    # Compute Galactic coordinates
    logger.info("Computing Galactic coordinates...")
    df["gal_l"], df["gal_b"] = compute_galactic_coordinates(df["ra"].values, df["dec"].values)

    # Classify proper motion
    logger.info("Classifying proper motion...")
    df["pm_category"] = df["pm_total"].apply(classify_pm_category)

    # Classify galactic region
    df["galactic_region"] = df["gal_b"].apply(classify_galactic_region)

    # Estimate distance from PM (assuming typical stellar velocity)
    df["est_distance_pc"] = df["pm_total"].apply(estimate_distance_from_pm)

    # Compute kinematic score
    logger.info("Computing kinematic scores...")
    df["kinematic_score"] = 0.0

    # Proper motion penalties/bonuses
    df.loc[df["pm_category"] == "very_high", "kinematic_score"] += HIGH_PM_PENALTY
    df.loc[df["pm_category"] == "high", "kinematic_score"] += MEDIUM_PM_PENALTY
    df.loc[df["pm_category"] == "low", "kinematic_score"] += LOW_PM_BONUS

    # Galactic location modifiers
    df.loc[df["galactic_region"] == "halo", "kinematic_score"] += HALO_LOCATION_BONUS
    df.loc[df["galactic_region"] == "thin_disk", "kinematic_score"] += DISK_LOCATION_PENALTY

    # Create summary flag
    def get_kinematic_flag(row):
        flags = []
        if row["pm_category"] in ["very_high", "high"]:
            flags.append("HIGH_PM")
        if row["galactic_region"] == "halo":
            flags.append("HALO")
        if row["pm_category"] == "low":
            flags.append("LOW_PM")
        return "|".join(flags) if flags else "NORMAL"

    df["kinematic_flag"] = df.apply(get_kinematic_flag, axis=1)

    return df


def print_summary(df: pd.DataFrame):
    """Print kinematic analysis summary."""
    logger.info("=" * 60)
    logger.info("Kinematic Analysis Summary")
    logger.info("=" * 60)

    # PM distribution
    logger.info("Proper Motion Distribution:")
    pm_counts = df["pm_category"].value_counts()
    for cat in ["low", "moderate", "high", "very_high"]:
        if cat in pm_counts:
            pct = 100 * pm_counts[cat] / len(df)
            logger.info(f"  {cat}: {pm_counts[cat]} ({pct:.1f}%)")

    # Galactic region distribution
    logger.info("\nGalactic Region Distribution:")
    region_counts = df["galactic_region"].value_counts()
    for region in ["thin_disk", "thick_disk", "halo"]:
        if region in region_counts:
            pct = 100 * region_counts[region] / len(df)
            logger.info(f"  {region}: {region_counts[region]} ({pct:.1f}%)")

    # Score impact
    logger.info("\nKinematic Score Statistics:")
    logger.info(f"  Mean: {df['kinematic_score'].mean():.2f}")
    logger.info(f"  Min: {df['kinematic_score'].min():.2f}")
    logger.info(f"  Max: {df['kinematic_score'].max():.2f}")

    # High-PM candidates (potential natural sources)
    n_high_pm = len(df[df["pm_category"].isin(["high", "very_high"])])
    logger.info(f"\nHigh-PM candidates (likely nearby/natural): {n_high_pm}")

    # Low-PM candidates (interesting)
    n_low_pm = len(df[df["pm_category"] == "low"])
    logger.info(f"Low-PM candidates (consistent with distant): {n_low_pm}")

    # Estimated distance distribution
    finite_dist = df["est_distance_pc"][df["est_distance_pc"] < np.inf]
    if len(finite_dist) > 0:
        logger.info("\nEstimated Distance (assuming v_tan=50 km/s):")
        logger.info(f"  Median: {finite_dist.median():.1f} pc")
        logger.info(f"  Min: {finite_dist.min():.1f} pc")
        logger.info(f"  Max: {finite_dist.max():.1f} pc")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TASNI Kinematic Analysis")
    parser.add_argument(
        "--input",
        "-i",
        default="./data/processed/golden_targets.csv",
        help="Input CSV file with candidates",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./data/processed/golden_kinematics.csv",
        help="Output CSV file with kinematic data",
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Update golden_targets.csv with kinematic scores",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Kinematic Analysis")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Load data
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} candidates")

    # Run analysis
    df = analyze_kinematics(df)

    # Print summary
    print_summary(df)

    # Save results
    output_path = Path(args.output)
    if str(output_path).endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    logger.info(f"Saved kinematic analysis to {output_path}")

    # Optionally update golden targets with new total score
    if args.update_golden:
        golden_path = Path("./data/processed/golden_targets.csv")
        if golden_path.exists():
            # Merge kinematic scores back
            golden = pd.read_csv(golden_path)
            kinematic_cols = [
                "designation",
                "gal_l",
                "gal_b",
                "pm_category",
                "galactic_region",
                "est_distance_pc",
                "kinematic_score",
                "kinematic_flag",
            ]
            kinematic_data = df[kinematic_cols]

            # Merge and update score
            golden = golden.merge(kinematic_data, on="designation", how="left")
            golden["score"] = golden["score"] + golden["kinematic_score"].fillna(0)

            # Re-sort by score
            golden = golden.sort_values("score", ascending=False)
            golden.to_csv(golden_path, index=False)
            logger.info(f"Updated {golden_path} with kinematic scores")

    # Show top candidates after kinematic analysis
    logger.info("\nTop 10 candidates after kinematic analysis:")
    display_cols = [
        "designation",
        "pm_total",
        "pm_category",
        "galactic_region",
        "kinematic_score",
        "T_eff_K",
        "score",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    # Sort by combined score if available
    if "score" in df.columns:
        top = df.nlargest(10, "score")
    else:
        top = df.nsmallest(10, "pm_total")  # Lowest PM is most interesting

    print(top[display_cols].to_string())

    return df


if __name__ == "__main__":
    main()
