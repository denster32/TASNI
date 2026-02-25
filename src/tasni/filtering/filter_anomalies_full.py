"""
TASNI: Filter and Rank Anomalies (Production)
==============================================

Takes the merged orphan catalog and applies filters + ranking.

Filters:
1. Quality flags (artifacts, extended sources)
2. SNR thresholds
3. Photometric quality

Ranking:
1. Thermal profile anomalies
2. Isolation score
3. Combined weirdness metric

Usage:
    python filter_anomalies_full.py [--gpu]
"""

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from tasni.core.config import (
    CLEAN_CC_FLAGS,
    GOOD_PH_QUAL,
    ISOLATION_WEIGHT,
    LOG_DIR,
    MIN_GALACTIC_LATITUDE,
    MIN_SNR_W1,
    MIN_SNR_W2,
    OUTPUT_DIR,
    W1_FAINT_THRESHOLD,
    W1_W2_ANOMALY_THRESHOLD,
    W3_BRIGHT_THRESHOLD,
    W3_FAINT_THRESHOLD,
    W4_FAINT_THRESHOLD,
    ensure_dirs,
)

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "filter_anomalies.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_orphans():
    """Load merged orphan catalog"""
    path = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    logger.info(f"Loading orphans from {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} orphan sources")
    return df


def filter_quality(df):
    """Remove sources with quality issues"""

    original = len(df)

    # Contamination flags - must be clean
    if "cc_flags" in df.columns:
        df = df[df["cc_flags"] == CLEAN_CC_FLAGS]
        logger.info(f"  After cc_flags filter: {len(df):,}")

    # Extended source flag - want point sources only
    if "ext_flg" in df.columns:
        df = df[df["ext_flg"] == 0]
        logger.info(f"  After ext_flg filter: {len(df):,}")

    # SNR thresholds
    if "w1snr" in df.columns:
        df = df[df["w1snr"] >= MIN_SNR_W1]
        logger.info(f"  After W1 SNR filter: {len(df):,}")

    if "w2snr" in df.columns:
        df = df[df["w2snr"] >= MIN_SNR_W2]
        logger.info(f"  After W2 SNR filter: {len(df):,}")

    # Photometric quality
    if "ph_qual" in df.columns:
        # First two chars must be A or B
        mask = df["ph_qual"].str[0].isin(GOOD_PH_QUAL) & df["ph_qual"].str[1].isin(GOOD_PH_QUAL)
        df = df[mask]
        logger.info(f"  After ph_qual filter: {len(df):,}")

    logger.info(f"Quality filter: {original:,} -> {len(df):,} ({100*len(df)/original:.1f}%)")
    return df.copy()


def filter_galactic_plane(df):
    """
    Remove sources in Galactic plane to reduce YSO contamination.

    YSOs (Young Stellar Objects) concentrate in the Galactic disk and can
    mimic cold brown dwarf signatures (fading, red colors). Cold brown dwarfs
    should be distributed more isotropically.

    Args:
        df: DataFrame with 'ra' and 'dec' columns

    Returns:
        DataFrame filtered to |b| > MIN_GALACTIC_LATITUDE
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    original = len(df)

    if "ra" not in df.columns or "dec" not in df.columns:
        logger.warning("No ra/dec columns - skipping Galactic filter")
        return df

    logger.info(f"Applying Galactic plane filter (|b| > {MIN_GALACTIC_LATITUDE}°)...")

    # Convert to Galactic coordinates
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
    galactic = coords.galactic

    # Add Galactic coordinates to dataframe
    df = df.copy()
    df["gal_l"] = galactic.l.deg
    df["gal_b"] = galactic.b.deg

    # Filter by Galactic latitude
    mask = np.abs(df["gal_b"]) > MIN_GALACTIC_LATITUDE
    df_filtered = df[mask]

    removed = original - len(df_filtered)
    logger.info(f"  Galactic plane filter: {original:,} -> {len(df_filtered):,}")
    logger.info(
        f"  Removed {removed:,} sources at |b| < {MIN_GALACTIC_LATITUDE}° ({100*removed/original:.1f}%)"
    )

    return df_filtered


def flag_w34_bright(df):
    """
    Flag sources with bright W3/W4 that are likely YSOs or dusty objects.

    Cold brown dwarfs (T_eff < 500K) should be faint or undetected in W3/W4
    because they lack dust emission. YSOs, AGB stars, and embedded objects
    have circumstellar dust that emits strongly at 12-22 μm.

    Args:
        df: DataFrame with w3mpro and w4mpro columns

    Returns:
        DataFrame with 'w34_yso_flag' column added
    """
    df = df.copy()

    logger.info("Flagging W3/W4 bright sources (potential YSO contamination)...")

    # Initialize flag
    df["w34_yso_flag"] = False

    # Check W3 brightness (sources brighter than threshold are suspicious)
    if "w3mpro" in df.columns:
        w3_bright = df["w3mpro"] < W3_FAINT_THRESHOLD
        # Only flag if W3 is detected (not null)
        w3_bright = w3_bright & df["w3mpro"].notna()
        df.loc[w3_bright, "w34_yso_flag"] = True
        logger.info(f"  W3 < {W3_FAINT_THRESHOLD} mag: {w3_bright.sum():,} sources flagged")

    # Check W4 brightness
    if "w4mpro" in df.columns:
        w4_bright = df["w4mpro"] < W4_FAINT_THRESHOLD
        w4_bright = w4_bright & df["w4mpro"].notna()
        df.loc[w4_bright, "w34_yso_flag"] = True
        logger.info(f"  W4 < {W4_FAINT_THRESHOLD} mag: {w4_bright.sum():,} sources flagged")

    total_flagged = df["w34_yso_flag"].sum()
    logger.info(f"  Total W3/W4 flagged: {total_flagged:,} ({100*total_flagged/len(df):.1f}%)")

    return df


def compute_thermal_profile(df):
    """Compute color indices from 4-band photometry"""

    logger.info("Computing thermal profiles...")

    df = df.copy()

    # Color indices (magnitude differences)
    # Negative = bluer (hotter at shorter wavelength)
    # Positive = redder (hotter at longer wavelength)

    cols = ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]
    if all(c in df.columns for c in cols):
        df["w1_w2"] = df["w1mpro"] - df["w2mpro"]  # Near-IR color
        df["w2_w3"] = df["w2mpro"] - df["w3mpro"]  # Near to mid-IR
        df["w3_w4"] = df["w3mpro"] - df["w4mpro"]  # Mid-IR color
        df["w1_w4"] = df["w1mpro"] - df["w4mpro"]  # Overall IR slope

        # Flux ratios (for physical interpretation)
        # Magnitude to flux: F = F0 * 10^(-m/2.5)
        # Ratio: F1/F2 = 10^((m2-m1)/2.5)
        df["flux_ratio_w2_w1"] = 10 ** ((df["w1mpro"] - df["w2mpro"]) / 2.5)
        df["flux_ratio_w4_w1"] = 10 ** ((df["w1mpro"] - df["w4mpro"]) / 2.5)

    return df


def compute_weirdness_score(df):
    """
    Compute anomaly score for each source.

    We're looking for sources that:
    1. Have unusual thermal profiles (not matching known object classes)
    2. Are highly isolated (far from any known optical source)
    3. Show signatures of warm emission without hot stellar component

    Known populations and their typical colors:
    - Main sequence stars: W1-W2 ≈ 0
    - Brown dwarfs: W1-W2 = 0.5 to 2.0 (redder = cooler)
    - Distant galaxies: W1-W2 = 0.5 to 1.0
    - AGN/Quasars: W1-W2 > 0.8, very red overall
    - Planetary nebulae: Variable, often W3/W4 excess
    - Asteroids: Solar colors, but they move

    ANOMALIES we're looking for:
    - W1-W2 < 0: "Bluer than blue" - physically weird
    - High W3/W4 flux with faint W1: Warm but not hot
    - Extreme isolation: Nothing nearby in optical
    """

    logger.info("Computing weirdness scores...")

    df = df.copy()
    df["weirdness"] = 0.0
    df["anomaly_flags"] = ""

    # === TIER 1: Physically weird colors ===

    # Anomalously blue W1-W2 (shouldn't happen in nature)
    if "w1_w2" in df.columns:
        blue_mask = df["w1_w2"] < W1_W2_ANOMALY_THRESHOLD
        df.loc[blue_mask, "weirdness"] += 3.0
        df.loc[blue_mask, "anomaly_flags"] += "BLUE_W1W2,"
        logger.info(f"  Blue W1-W2 anomalies: {blue_mask.sum():,}")

    # === TIER 2: Warm but not hot ===

    # Detected in W3/W4 (warm) but faint in W1 (not hot)
    # This is the "waste heat" signature we're looking for
    if all(c in df.columns for c in ["w1mpro", "w3mpro"]):
        warm_not_hot = (df["w3mpro"] < W3_BRIGHT_THRESHOLD) & (df["w1mpro"] > W1_FAINT_THRESHOLD)
        df.loc[warm_not_hot, "weirdness"] += 2.0
        df.loc[warm_not_hot, "anomaly_flags"] += "WARM_NOT_HOT,"
        logger.info(f"  Warm-not-hot anomalies: {warm_not_hot.sum():,}")

    # === TIER 3: Isolation ===

    # Sources far from any Gaia counterpart get bonus
    if "nearest_gaia_sep_arcsec" in df.columns:
        # Log scale: 3 arcsec = 0.5 pts, 30 arcsec = 1.5 pts, 300 arcsec = 2.5 pts
        iso_score = np.log10(df["nearest_gaia_sep_arcsec"].clip(lower=1)) * ISOLATION_WEIGHT
        df["weirdness"] += iso_score
        df["isolation_score"] = iso_score

    # === TIER 4: Extreme colors ===

    # Very red W1-W4 (could be cool dust, could be interesting)
    if "w1_w4" in df.columns:
        very_red = df["w1_w4"] > 5.0
        df.loc[very_red, "weirdness"] += 1.0
        df.loc[very_red, "anomaly_flags"] += "VERY_RED,"

    # Very blue W1-W4 (unusual)
    if "w1_w4" in df.columns:
        very_blue = df["w1_w4"] < -1.0
        df.loc[very_blue, "weirdness"] += 1.5
        df.loc[very_blue, "anomaly_flags"] += "VERY_BLUE,"

    # Clean up flags
    df["anomaly_flags"] = df["anomaly_flags"].str.rstrip(",")

    logger.info(
        f"Weirdness score range: {df['weirdness'].min():.2f} to {df['weirdness'].max():.2f}"
    )
    logger.info(f"Mean weirdness: {df['weirdness'].mean():.2f}")

    return df


def rank_and_export(df):
    """Sort by weirdness and export results"""

    logger.info("Ranking and exporting...")

    # Sort by weirdness (highest first)
    df = df.sort_values("weirdness", ascending=False)

    # Add rank
    df["rank"] = range(1, len(df) + 1)

    # Save full catalog
    full_path = OUTPUT_DIR / "anomalies_ranked.parquet"
    df.to_parquet(full_path, index=False)
    logger.info(f"Full catalog: {full_path} ({len(df):,} sources)")

    # Save top 10000 as CSV for easy viewing
    top_path = OUTPUT_DIR / "top_anomalies.csv"
    top_cols = [
        "rank",
        "designation",
        "ra",
        "dec",
        "w1mpro",
        "w2mpro",
        "w3mpro",
        "w4mpro",
        "w1_w2",
        "w1_w4",
        "nearest_gaia_sep_arcsec",
        "isolation_score",
        "weirdness",
        "anomaly_flags",
    ]
    top_cols = [c for c in top_cols if c in df.columns]
    df.head(10000)[top_cols].to_csv(top_path, index=False)
    logger.info(f"Top 10000: {top_path}")

    # Save extreme anomalies (weirdness > 5)
    extreme = df[df["weirdness"] > 5.0]
    if len(extreme) > 0:
        extreme_path = OUTPUT_DIR / "extreme_anomalies.csv"
        extreme[top_cols].to_csv(extreme_path, index=False)
        logger.info(f"Extreme anomalies (>5.0): {extreme_path} ({len(extreme):,} sources)")

    # Summary stats
    logger.info("")
    logger.info("=== SUMMARY ===")
    logger.info(f"Total anomalies: {len(df):,}")
    logger.info(f"Weirdness > 3: {(df['weirdness'] > 3).sum():,}")
    logger.info(f"Weirdness > 5: {(df['weirdness'] > 5).sum():,}")
    logger.info(f"Weirdness > 7: {(df['weirdness'] > 7).sum():,}")

    # Flag breakdown
    if "anomaly_flags" in df.columns:
        logger.info("")
        logger.info("Anomaly flag counts:")
        for flag in ["BLUE_W1W2", "WARM_NOT_HOT", "VERY_RED", "VERY_BLUE"]:
            count = df["anomaly_flags"].str.contains(flag).sum()
            logger.info(f"  {flag}: {count:,}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Filter and rank anomalies")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: Filter and Rank Anomalies")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    # Load data
    df = load_orphans()

    # Filter
    logger.info("")
    logger.info("Applying quality filters...")
    df = filter_quality(df)

    # YSO contamination filters (new)
    logger.info("")
    df = filter_galactic_plane(df)

    logger.info("")
    df = flag_w34_bright(df)

    # Compute profiles
    logger.info("")
    df = compute_thermal_profile(df)

    # Score
    logger.info("")
    df = compute_weirdness_score(df)

    # Export
    logger.info("")
    df = rank_and_export(df)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
