"""
TASNI: Cross-match with Chinese VO Catalogs
=============================================

Cross-matches WISE orphans with:
1. Legacy Survey DR10 (BASS + MzLS) - Deep optical veto
2. LAMOST DR12 - Spectral classification veto

This adds two powerful filters to the TASNI pipeline:
- Sources visible in deep optical (g<24.2) are NOT thermally stealthy
- Sources with known spectral types (M dwarfs, carbon stars, etc.) are explained

Usage:
    python crossmatch_chinese_vo.py [--input orphans.parquet] [--catalog legacy|lamost|all]
"""

import argparse
import logging
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tasni.core.config import (
    HEALPIX_NSIDE,
    LAMOST_DIR,
    LAMOST_KNOWN_IR_TYPES,
    LAMOST_KNOWN_TYPE_PENALTY,
    LAMOST_TEMP_MISMATCH_BONUS,
    LAMOST_UNKNOWN_BONUS,
    LEGACY_DEEP_OPTICAL_PENALTY,
    LEGACY_DIR,
    LEGACY_FAINT_THRESHOLD_G,
    LOG_DIR,
    OUTPUT_DIR,
    RADIUS_WISE_LAMOST,
    RADIUS_WISE_LEGACY,
    ensure_dirs,
)

ensure_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [XMATCH-CVO] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "crossmatch_chinese_vo.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def crossmatch_kdtree(ra1, dec1, ra2, dec2, radius_arcsec):
    """
    Fast cross-match using KD-tree on 3D Cartesian coordinates.
    Returns indices of matches and separations.
    """
    # Convert to radians
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)

    # Convert to 3D Cartesian (unit sphere)
    def to_cartesian(ra, dec):
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.column_stack([x, y, z])

    xyz1 = to_cartesian(ra1_rad, dec1_rad)
    xyz2 = to_cartesian(ra2_rad, dec2_rad)

    # Build tree on catalog 2
    tree = cKDTree(xyz2)

    # Convert radius to 3D chord length
    radius_rad = np.radians(radius_arcsec / 3600)
    chord = 2 * np.sin(radius_rad / 2)

    # Query
    distances, indices = tree.query(xyz1, k=1, distance_upper_bound=chord)

    # Convert chord distance back to angular separation (arcsec)
    valid = np.isfinite(distances)
    sep_arcsec = np.full(len(ra1), np.inf)
    sep_arcsec[valid] = 2 * np.degrees(np.arcsin(distances[valid] / 2)) * 3600

    # -1 for no match
    match_idx = np.where(valid, indices, -1)

    return match_idx, sep_arcsec


def crossmatch_legacy_tile(orphan_tile_file, legacy_tile_file):
    """Cross-match a single HEALPix tile with Legacy Survey"""

    if not legacy_tile_file.exists():
        return None

    orphans = pd.read_parquet(orphan_tile_file)
    legacy = pd.read_parquet(legacy_tile_file)

    if len(orphans) == 0 or len(legacy) == 0:
        return orphans  # No matches possible

    # Cross-match
    match_idx, sep_arcsec = crossmatch_kdtree(
        orphans["ra"].values,
        orphans["dec"].values,
        legacy["ra"].values,
        legacy["dec"].values,
        RADIUS_WISE_LEGACY,
    )

    # Add Legacy Survey columns to orphans
    has_match = match_idx >= 0

    orphans["legacy_match"] = has_match
    orphans["legacy_sep_arcsec"] = sep_arcsec

    # Get matched magnitudes
    orphans["legacy_mag_g"] = np.where(
        has_match, legacy["mag_g"].values[np.maximum(match_idx, 0)], 99.0
    )
    orphans["legacy_mag_r"] = np.where(
        has_match, legacy["mag_r"].values[np.maximum(match_idx, 0)], 99.0
    )
    orphans["legacy_mag_z"] = np.where(
        has_match, legacy["mag_z"].values[np.maximum(match_idx, 0)], 99.0
    )

    # Morphological type (PSF = star, others = galaxy)
    if "type" in legacy.columns:
        orphans["legacy_type"] = np.where(
            has_match, legacy["type"].values[np.maximum(match_idx, 0)], ""
        )

    return orphans


def crossmatch_lamost_tile(orphan_tile_file, lamost_tile_file):
    """Cross-match a single HEALPix tile with LAMOST"""

    if not lamost_tile_file.exists():
        return None

    orphans = pd.read_parquet(orphan_tile_file)
    lamost = pd.read_parquet(lamost_tile_file)

    if len(orphans) == 0 or len(lamost) == 0:
        return orphans

    # Cross-match
    match_idx, sep_arcsec = crossmatch_kdtree(
        orphans["ra"].values,
        orphans["dec"].values,
        lamost["ra"].values,
        lamost["dec"].values,
        RADIUS_WISE_LAMOST,
    )

    has_match = match_idx >= 0

    orphans["lamost_match"] = has_match
    orphans["lamost_sep_arcsec"] = sep_arcsec

    # Get spectral parameters
    for col in ["teff", "logg", "feh", "rv", "class", "subclass"]:
        if col in lamost.columns:
            if lamost[col].dtype == "object":
                orphans[f"lamost_{col}"] = np.where(
                    has_match, lamost[col].values[np.maximum(match_idx, 0)], ""
                )
            else:
                orphans[f"lamost_{col}"] = np.where(
                    has_match, lamost[col].values[np.maximum(match_idx, 0)], np.nan
                )

    return orphans


def crossmatch_all_tiles(catalog="all"):
    """Cross-match all HEALPix tiles with Chinese VO catalogs"""

    npix = hp.nside2npix(HEALPIX_NSIDE)

    # Find orphan files
    orphan_dir = OUTPUT_DIR
    orphan_files = list(orphan_dir.glob("orphans_hp*.parquet"))

    if not orphan_files:
        # Try alternative location
        orphan_files = list((OUTPUT_DIR.parent / "data" / "crossmatch").glob("orphans_hp*.parquet"))

    if not orphan_files:
        logger.error("No orphan tile files found!")
        logger.info("Run the main crossmatch pipeline first to generate orphans_hp*.parquet files")
        return

    logger.info(f"Found {len(orphan_files)} orphan tile files")

    # Determine catalog directories
    legacy_healpix = LEGACY_DIR / "north_healpix"
    lamost_healpix = LAMOST_DIR / "healpix"

    # Check what's available
    has_legacy = legacy_healpix.exists() and any(legacy_healpix.glob("*.parquet"))
    has_lamost = lamost_healpix.exists() and any(lamost_healpix.glob("*.parquet"))

    logger.info(f"Legacy Survey available: {has_legacy}")
    logger.info(f"LAMOST available: {has_lamost}")

    if not has_legacy and not has_lamost:
        logger.error("No Chinese VO catalogs found!")
        logger.info("Run download_legacy_survey.py and/or download_lamost.py first")
        return

    # Output directory
    output_dir = OUTPUT_DIR / "chinese_vo_crossmatch"
    output_dir.mkdir(exist_ok=True)

    # Process tiles
    total_orphans = 0
    legacy_matches = 0
    lamost_matches = 0

    start_time = time.time()

    for i, orphan_file in enumerate(orphan_files):
        # Extract tile index from filename
        tile_idx = int(orphan_file.stem.split("hp")[1])

        orphans = pd.read_parquet(orphan_file)
        n_orphans = len(orphans)
        total_orphans += n_orphans

        if n_orphans == 0:
            continue

        # Legacy Survey cross-match
        if has_legacy and catalog in ["legacy", "all"]:
            legacy_file = legacy_healpix / f"legacy_hp{tile_idx:05d}.parquet"
            if legacy_file.exists():
                legacy_df = pd.read_parquet(legacy_file)
                if len(legacy_df) > 0:
                    match_idx, sep = crossmatch_kdtree(
                        orphans["ra"].values,
                        orphans["dec"].values,
                        legacy_df["ra"].values,
                        legacy_df["dec"].values,
                        RADIUS_WISE_LEGACY,
                    )
                    has_match = match_idx >= 0
                    orphans["legacy_match"] = has_match
                    orphans["legacy_sep_arcsec"] = sep

                    for col in ["mag_g", "mag_r", "mag_z", "type"]:
                        if col in legacy_df.columns:
                            orphans[f"legacy_{col}"] = np.where(
                                has_match,
                                legacy_df[col].values[np.maximum(match_idx, 0)],
                                99.0 if "mag" in col else "",
                            )

                    legacy_matches += has_match.sum()

        # LAMOST cross-match
        if has_lamost and catalog in ["lamost", "all"]:
            lamost_file = lamost_healpix / f"lamost_hp{tile_idx:05d}.parquet"
            if lamost_file.exists():
                lamost_df = pd.read_parquet(lamost_file)
                if len(lamost_df) > 0:
                    match_idx, sep = crossmatch_kdtree(
                        orphans["ra"].values,
                        orphans["dec"].values,
                        lamost_df["ra"].values,
                        lamost_df["dec"].values,
                        RADIUS_WISE_LAMOST,
                    )
                    has_match = match_idx >= 0
                    orphans["lamost_match"] = has_match
                    orphans["lamost_sep_arcsec"] = sep

                    for col in ["teff", "logg", "feh", "rv", "class", "subclass"]:
                        if col in lamost_df.columns:
                            dtype = lamost_df[col].dtype
                            if dtype == "object":
                                default = ""
                            else:
                                default = np.nan
                            orphans[f"lamost_{col}"] = np.where(
                                has_match,
                                lamost_df[col].values[np.maximum(match_idx, 0)],
                                default,
                            )

                    lamost_matches += has_match.sum()

        # Save updated orphans
        output_file = output_dir / f"orphans_cvo_hp{tile_idx:05d}.parquet"
        orphans.to_parquet(output_file, index=False)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            logger.info(
                f"[{i + 1}/{len(orphan_files)}] {rate:.1f} tiles/s, "
                f"Legacy: {legacy_matches:,}, LAMOST: {lamost_matches:,}"
            )

    elapsed = time.time() - start_time

    # Summary
    logger.info("=" * 60)
    logger.info("Chinese VO Cross-match Complete")
    logger.info("=" * 60)
    logger.info(f"Total orphans processed: {total_orphans:,}")
    logger.info(
        f"Legacy Survey matches: {legacy_matches:,} ({100 * legacy_matches / max(1, total_orphans):.2f}%)"
    )
    logger.info(
        f"LAMOST matches: {lamost_matches:,} ({100 * lamost_matches / max(1, total_orphans):.2f}%)"
    )
    logger.info(f"Time: {elapsed:.0f}s")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)


def apply_chinese_vo_scoring(input_file=None, output_file=None):
    """
    Apply scoring adjustments based on Chinese VO cross-matches.

    Scoring logic:
    - Legacy match (g < 24): Strong penalty (visible in deep optical = not stealthy)
    - LAMOST known type: Strong penalty (explained astrophysics)
    - LAMOST unknown type: Bonus (has spectrum but unexplained)
    - Temperature mismatch: Bonus (spectral Teff != IR Teff)
    """

    if input_file is None:
        input_file = OUTPUT_DIR / "chinese_vo_crossmatch"
    if output_file is None:
        output_file = OUTPUT_DIR / "orphans_cvo_scored.parquet"

    input_path = Path(input_file)

    if input_path.is_dir():
        # Merge all tile files
        files = list(input_path.glob("orphans_cvo_hp*.parquet"))
        if not files:
            logger.error(f"No cross-matched files found in {input_path}")
            return

        logger.info(f"Merging {len(files)} tile files...")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_parquet(input_path)

    logger.info(f"Scoring {len(df):,} sources")

    # Initialize score adjustment column
    df["cvo_score_adj"] = 0.0
    df["cvo_flags"] = ""

    # Legacy Survey scoring
    if "legacy_match" in df.columns:
        # Sources with deep optical detection
        has_legacy = df["legacy_match"].fillna(False)

        # Bright in g-band = definitely has optical counterpart
        bright_g = df.get("legacy_mag_g", 99) < LEGACY_FAINT_THRESHOLD_G

        # Apply penalty
        legacy_penalty = has_legacy & bright_g
        df.loc[legacy_penalty, "cvo_score_adj"] += LEGACY_DEEP_OPTICAL_PENALTY
        df.loc[legacy_penalty, "cvo_flags"] += "LEGACY_OPTICAL;"

        logger.info(f"Legacy optical penalty applied: {legacy_penalty.sum():,} sources")

    # LAMOST scoring
    if "lamost_match" in df.columns:
        has_lamost = df["lamost_match"].fillna(False)

        # Check spectral type
        subclass = df.get("lamost_subclass", "").fillna("").astype(str)

        # Known IR-emitting types (M dwarfs, carbon stars, etc.)
        known_ir_type = subclass.str.match("|".join(LAMOST_KNOWN_IR_TYPES), case=False, na=False)

        # Apply penalties and bonuses
        lamost_known = has_lamost & known_ir_type
        df.loc[lamost_known, "cvo_score_adj"] += LAMOST_KNOWN_TYPE_PENALTY
        df.loc[lamost_known, "cvo_flags"] += "LAMOST_KNOWN_TYPE;"

        logger.info(f"LAMOST known-type penalty applied: {lamost_known.sum():,} sources")

        # Unknown type bonus (has spectrum but classification failed)
        lamost_unknown = has_lamost & ~known_ir_type & (subclass == "")
        df.loc[lamost_unknown, "cvo_score_adj"] += LAMOST_UNKNOWN_BONUS
        df.loc[lamost_unknown, "cvo_flags"] += "LAMOST_UNKNOWN;"

        logger.info(f"LAMOST unknown-type bonus applied: {lamost_unknown.sum():,} sources")

        # Temperature mismatch check
        if "lamost_teff" in df.columns and "w1mpro" in df.columns:
            lamost_teff = pd.to_numeric(df["lamost_teff"], errors="coerce")

            # Estimate IR temperature from W1-W2 color (rough approximation)
            # Hotter = bluer (W1-W2 < 0), Cooler = redder (W1-W2 > 0)
            w1 = df.get("w1mpro", np.nan)
            w2 = df.get("w2mpro", np.nan)

            # Very rough: T ~ 5000K * 10^(-0.4 * (W1-W2))
            # This is approximate but catches large mismatches
            with np.errstate(invalid="ignore"):
                ir_teff_approx = 5000 * np.power(10, -0.4 * (w1 - w2))

            # Significant mismatch (>1000K difference)
            teff_mismatch = has_lamost & (np.abs(lamost_teff - ir_teff_approx) > 1000)
            teff_mismatch = teff_mismatch.fillna(False)

            df.loc[teff_mismatch, "cvo_score_adj"] += LAMOST_TEMP_MISMATCH_BONUS
            df.loc[teff_mismatch, "cvo_flags"] += "TEFF_MISMATCH;"

            logger.info(f"Temperature mismatch bonus applied: {teff_mismatch.sum():,} sources")

    # Summary statistics
    logger.info("=" * 60)
    logger.info("Chinese VO Scoring Summary")
    logger.info("=" * 60)
    logger.info(f"Total sources: {len(df):,}")
    logger.info(f"Mean score adjustment: {df['cvo_score_adj'].mean():.2f}")
    logger.info(f"Sources with negative adj (penalized): {(df['cvo_score_adj'] < 0).sum():,}")
    logger.info(f"Sources with positive adj (boosted): {(df['cvo_score_adj'] > 0).sum():,}")
    logger.info(f"Sources with no adjustment: {(df['cvo_score_adj'] == 0).sum():,}")

    # Save
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved to {output_file}")

    # Also save CSV of top candidates (after adjustment)
    if "multiwave_score" in df.columns:
        df["total_score"] = df["multiwave_score"] + df["cvo_score_adj"]
    else:
        df["total_score"] = df["cvo_score_adj"]

    top_candidates = df.nlargest(1000, "total_score")
    top_file = OUTPUT_DIR / "top_candidates_with_cvo.csv"
    top_candidates.to_csv(top_file, index=False)
    logger.info(f"Saved top 1000 candidates to {top_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Cross-match with Chinese VO catalogs")
    parser.add_argument(
        "--catalog",
        choices=["legacy", "lamost", "all"],
        default="all",
        help="Which catalog to cross-match (default: all)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only apply scoring (skip cross-matching)",
    )
    parser.add_argument("--input", type=str, help="Input file or directory for scoring")


def main():
    parser = argparse.ArgumentParser(description="Cross-match with Chinese VO catalogs")
    parser.add_argument(
        "--catalog",
        choices=["legacy", "lamost", "all"],
        default="all",
        help="Which catalog to cross-match (default: all)",
    )
    parser.add_argument(
        "--score-only", action="store_true", help="Only apply scoring (skip cross-matching)"
    )
    parser.add_argument("--input", type=str, help="Input file or directory for scoring")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Chinese VO Cross-Matching")
    logger.info("=" * 60)
    logger.info("Catalogs: Legacy Survey DR10 (BASS), LAMOST DR12")
    logger.info("=" * 60)

    if args.score_only:
        apply_chinese_vo_scoring(input_file=args.input)
    else:
        crossmatch_all_tiles(catalog=args.catalog)
        apply_chinese_vo_scoring()


if __name__ == "__main__":
    main()
