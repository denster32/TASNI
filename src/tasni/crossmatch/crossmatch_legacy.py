"""
TASNI: Cross-match Candidates with Legacy Survey DR10 (BASS + MzLS)
===================================================================

Cross-matches TASNI anomaly candidates with Legacy Survey DR10 deep optical
data to apply a stronger optical veto than Pan-STARRS.

Legacy Survey Depth vs Pan-STARRS:
- Legacy g=24.2 vs PS g=23.2 (1 mag deeper)
- Legacy r=23.6 vs PS r=23.2 (0.4 mag deeper)
- Legacy z=23.0 vs PS z=22.3 (0.7 mag deeper)

If a candidate is visible in Legacy Survey, it's NOT truly optically dark.

Scoring Logic (from config.py):
- LEGACY_DEEP_OPTICAL_PENALTY = -15.0 (visible in deep optical = not stealthy)
- LEGACY_FAINT_THRESHOLD_G = 23.5 (fainter than this = borderline detection)

Usage:
    python crossmatch_legacy.py --input output/tier4_final.parquet
    python crossmatch_legacy.py --use-healpix  # Use pre-organized HEALPix tiles
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, match_coordinates_sky

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    import healpy as hp

    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False

try:
    import pyvo as vo

    PYVO_AVAILABLE = True
except ImportError:
    PYVO_AVAILABLE = False

# NOIRLab TAP service for Legacy Survey DR10
NOIRLAB_TAP_URL = "https://datalab.noirlab.edu/tap"
from tasni.core.config import (
    HEALPIX_NSIDE,
    LEGACY_DEEP_OPTICAL_PENALTY,
    LEGACY_DIR,
    LEGACY_FAINT_THRESHOLD_G,
    LEGACY_FAINT_THRESHOLD_R,
    LOG_DIR,
    OUTPUT_DIR,
    RADIUS_WISE_LEGACY,
    ensure_dirs,
)

ensure_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [LEGACY-XMATCH] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "crossmatch_legacy.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_legacy_healpix_tile(healpix_idx, region="north"):
    """
    Load Legacy Survey data for a specific HEALPix tile.

    Args:
        healpix_idx: HEALPix pixel index
        region: 'north' or 'south'

    Returns:
        DataFrame with Legacy Survey data, or None if not available
    """
    healpix_dir = LEGACY_DIR / f"{region}_healpix"
    tile_file = healpix_dir / f"legacy_hp{healpix_idx:05d}.parquet"

    if tile_file.exists():
        return pd.read_parquet(tile_file)

    return None


def load_legacy_sweep_files(region="north"):
    """
    Load all Legacy Survey sweep files for a region.

    Args:
        region: 'north' or 'south'

    Returns:
        DataFrame with all Legacy Survey data
    """
    sweep_dir = LEGACY_DIR / region

    if not sweep_dir.exists():
        logger.warning(f"Legacy Survey directory not found: {sweep_dir}")
        logger.info("Run: python download_legacy_survey.py --region north")
        return None

    sweep_files = list(sweep_dir.glob("sweep-*.parquet"))

    if not sweep_files:
        logger.warning(f"No sweep files found in {sweep_dir}")
        return None

    logger.info(f"Loading {len(sweep_files)} Legacy Survey sweep files...")

    dfs = []
    for f in sweep_files:
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(result):,} Legacy Survey sources")

    return result


def crossmatch_candidates_with_legacy(candidates_df, legacy_df, radius_arcsec=2.0):
    """
    Cross-match candidate positions with Legacy Survey catalog.

    Args:
        candidates_df: DataFrame with 'ra', 'dec' columns
        legacy_df: DataFrame with Legacy Survey data
        radius_arcsec: Match radius in arcseconds

    Returns:
        DataFrame with Legacy match columns added
    """
    if not ASTROPY_AVAILABLE:
        logger.error("astropy not available for coordinate matching")
        return None

    logger.info(
        f"Cross-matching {len(candidates_df):,} candidates with "
        f'{len(legacy_df):,} Legacy sources (radius={radius_arcsec}")'
    )

    # Create SkyCoord objects
    cand_coords = SkyCoord(
        ra=candidates_df["ra"].values * u.deg, dec=candidates_df["dec"].values * u.deg, frame="icrs"
    )

    # Find RA/Dec columns in Legacy data
    legacy_ra = legacy_df["ra"].values if "ra" in legacy_df.columns else legacy_df["RA"].values
    legacy_dec = legacy_df["dec"].values if "dec" in legacy_df.columns else legacy_df["DEC"].values

    legacy_coords = SkyCoord(ra=legacy_ra * u.deg, dec=legacy_dec * u.deg, frame="icrs")

    # Cross-match
    idx, sep2d, _ = match_coordinates_sky(cand_coords, legacy_coords)

    # Create result DataFrame
    result = candidates_df.copy()

    # Initialize Legacy columns
    result["legacy_match"] = False
    result["legacy_sep_arcsec"] = np.inf
    result["legacy_mag_g"] = 99.0
    result["legacy_mag_r"] = 99.0
    result["legacy_mag_z"] = 99.0
    result["legacy_type"] = None

    # Apply matches within radius
    match_mask = sep2d.arcsec < radius_arcsec

    for i, (is_match, match_idx, sep) in enumerate(
        zip(match_mask, idx, sep2d.arcsec, strict=False)
    ):
        if is_match:
            legacy_row = legacy_df.iloc[match_idx]
            result.loc[result.index[i], "legacy_match"] = True
            result.loc[result.index[i], "legacy_sep_arcsec"] = sep

            # Get magnitudes (handle different column name conventions)
            result.loc[result.index[i], "legacy_mag_g"] = legacy_row.get(
                "mag_g", legacy_row.get("FLUX_G", 99)
            )
            result.loc[result.index[i], "legacy_mag_r"] = legacy_row.get(
                "mag_r", legacy_row.get("FLUX_R", 99)
            )
            result.loc[result.index[i], "legacy_mag_z"] = legacy_row.get(
                "mag_z", legacy_row.get("FLUX_Z", 99)
            )
            result.loc[result.index[i], "legacy_type"] = legacy_row.get(
                "type", legacy_row.get("TYPE")
            )

    n_matches = match_mask.sum()
    logger.info(f"Found {n_matches:,} Legacy matches ({100*n_matches/len(result):.1f}%)")

    return result


def compute_legacy_scores(df):
    """
    Compute Legacy Survey-based scoring.

    If visible in deep optical, apply penalty (not truly dark).
    If very faint detection, apply smaller penalty.

    Args:
        df: DataFrame with Legacy match columns

    Returns:
        DataFrame with legacy_score column added
    """
    result = df.copy()
    result["legacy_score"] = 0.0
    result["legacy_is_bright"] = False
    result["legacy_is_faint"] = False

    for idx, row in result.iterrows():
        if not row.get("legacy_match", False):
            # No Legacy match - source is truly dark in deep optical
            result.loc[idx, "legacy_score"] = 0.0  # No penalty (good)
            continue

        # Has Legacy match - visible in deep optical
        g_mag = row.get("legacy_mag_g", 99)
        r_mag = row.get("legacy_mag_r", 99)

        if g_mag < LEGACY_FAINT_THRESHOLD_G or r_mag < LEGACY_FAINT_THRESHOLD_R:
            # Bright detection - strong penalty
            result.loc[idx, "legacy_score"] = LEGACY_DEEP_OPTICAL_PENALTY  # -15
            result.loc[idx, "legacy_is_bright"] = True
        else:
            # Faint detection (near detection limit) - mild penalty
            result.loc[idx, "legacy_score"] = LEGACY_DEEP_OPTICAL_PENALTY / 3  # -5
            result.loc[idx, "legacy_is_faint"] = True

    return result


def crossmatch_with_tap(candidates_df, batch_size=1000):
    """
    Cross-match candidates against Legacy Survey DR10 using NOIRLab TAP.

    Args:
        candidates_df: DataFrame with 'ra', 'dec' columns
        batch_size: Number of candidates per query batch

    Returns:
        DataFrame with Legacy match columns added
    """
    if not PYVO_AVAILABLE:
        logger.error("pyvo required for TAP crossmatch")
        return None

    logger.info(
        f"Cross-matching {len(candidates_df):,} candidates with Legacy Survey via NOIRLab TAP"
    )
    logger.info(f"TAP URL: {NOIRLAB_TAP_URL}")

    # Initialize TAP service
    try:
        tap = vo.dal.TAPService(NOIRLAB_TAP_URL)
    except Exception as e:
        logger.error(f"Failed to connect to NOIRLab TAP: {e}")
        return None

    # Initialize result DataFrame
    result = candidates_df.copy()
    result["legacy_match"] = False
    result["legacy_sep_arcsec"] = np.inf
    result["legacy_mag_g"] = 99.0
    result["legacy_mag_r"] = 99.0
    result["legacy_mag_z"] = 99.0
    result["legacy_type"] = None

    # Process in batches
    n_batches = (len(candidates_df) + batch_size - 1) // batch_size
    total_matches = 0

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(candidates_df))
        batch = candidates_df.iloc[start_idx:end_idx]

        if len(batch) == 0:
            continue

        # Get RA/Dec bounds for this batch
        ra_min, ra_max = batch["ra"].min() - 0.01, batch["ra"].max() + 0.01
        dec_min, dec_max = batch["dec"].min() - 0.01, batch["dec"].max() + 0.01

        # Query Legacy Survey DR10 via TAP
        # Using tractor table which has all sources
        query = f"""
        SELECT ra, dec, type, flux_g, flux_r, flux_z
        FROM ls_dr10.tractor
        WHERE ra BETWEEN {ra_min} AND {ra_max}
          AND dec BETWEEN {dec_min} AND {dec_max}
          AND flux_g > 0
        """

        try:
            tap_result = tap.search(query, maxrec=100000)

            if len(tap_result) > 0:
                legacy_df = tap_result.to_table().to_pandas()

                # Convert flux to magnitude (nanomaggies to AB mag)
                # mag = 22.5 - 2.5 * log10(flux)
                for band in ["g", "r", "z"]:
                    flux_col = f"flux_{band}"
                    if flux_col in legacy_df.columns:
                        flux = legacy_df[flux_col].values
                        valid = flux > 0
                        legacy_df[f"mag_{band}"] = 99.0
                        legacy_df.loc[valid, f"mag_{band}"] = 22.5 - 2.5 * np.log10(flux[valid])

                # Cross-match this batch
                if not legacy_df.empty and "ra" in legacy_df.columns:
                    # Filter out NaN coordinates
                    valid_mask = ~(legacy_df["ra"].isna() | legacy_df["dec"].isna())
                    legacy_df = legacy_df[valid_mask]

                    if len(legacy_df) > 0:
                        batch_coords = SkyCoord(
                            ra=batch["ra"].values * u.deg,
                            dec=batch["dec"].values * u.deg,
                            frame="icrs",
                        )

                        legacy_coords = SkyCoord(
                            ra=legacy_df["ra"].values * u.deg,
                            dec=legacy_df["dec"].values * u.deg,
                            frame="icrs",
                        )

                        idx, sep2d, _ = match_coordinates_sky(batch_coords, legacy_coords)
                        match_mask = sep2d.arcsec < RADIUS_WISE_LEGACY

                        for i, (orig_idx, is_match, match_idx, sep) in enumerate(
                            zip(batch.index, match_mask, idx, sep2d.arcsec, strict=False)
                        ):
                            if is_match:
                                legacy_row = legacy_df.iloc[match_idx]
                                result.loc[orig_idx, "legacy_match"] = True
                                result.loc[orig_idx, "legacy_sep_arcsec"] = sep
                                result.loc[orig_idx, "legacy_mag_g"] = legacy_row.get("mag_g", 99.0)
                                result.loc[orig_idx, "legacy_mag_r"] = legacy_row.get("mag_r", 99.0)
                                result.loc[orig_idx, "legacy_mag_z"] = legacy_row.get("mag_z", 99.0)
                                result.loc[orig_idx, "legacy_type"] = legacy_row.get("type")
                                total_matches += 1

        except Exception as e:
            logger.warning(f"Batch {batch_idx+1}: TAP query failed: {e}")

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"[{batch_idx+1}/{n_batches}] {total_matches} matches so far")

        time.sleep(0.3)  # Rate limit

    logger.info(
        f"TAP crossmatch complete: {total_matches:,} matches "
        f"({100*total_matches/len(result):.1f}%)"
    )

    return result


def crossmatch_tier4_with_legacy(
    input_file=None, output_file=None, use_healpix=True, region="north"
):
    """
    Main function to cross-match Tier 4 candidates with Legacy Survey.

    Args:
        input_file: Path to Tier 4 candidate parquet
        output_file: Path to save results
        use_healpix: Use HEALPix-organized tiles (faster if available)
        region: 'north' or 'south'

    Returns:
        DataFrame with Legacy match data and scores
    """
    if input_file is None:
        input_file = OUTPUT_DIR / "tier4_final.parquet"
    if output_file is None:
        output_file = OUTPUT_DIR / "tier4_with_legacy.parquet"

    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return None

    # Load candidates
    logger.info(f"Loading candidates from {input_file}")
    candidates = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(candidates):,} candidates")

    # Check if HEALPix tiles are available
    healpix_dir = LEGACY_DIR / f"{region}_healpix"
    use_healpix = use_healpix and healpix_dir.exists() and HEALPY_AVAILABLE

    if use_healpix:
        logger.info("Using HEALPix-organized Legacy Survey data")

        # Compute HEALPix indices for candidates
        cand_pixels = hp.ang2pix(
            HEALPIX_NSIDE, candidates["ra"].values, candidates["dec"].values, nest=True, lonlat=True
        )

        # Get unique pixels
        unique_pixels = np.unique(cand_pixels)
        logger.info(f"Candidates span {len(unique_pixels)} HEALPix tiles")

        # Process tile by tile
        results = []
        for pix in unique_pixels:
            # Get candidates in this tile
            mask = cand_pixels == pix
            tile_candidates = candidates[mask].copy()

            # Load Legacy data for this tile (and neighbors for edge effects)
            neighbor_pixels = hp.get_all_neighbours(HEALPIX_NSIDE, pix, nest=True)
            pixels_to_load = [pix] + [p for p in neighbor_pixels if p >= 0]

            tile_legacy_dfs = []
            for p in pixels_to_load:
                leg_df = load_legacy_healpix_tile(p, region)
                if leg_df is not None and len(leg_df) > 0:
                    tile_legacy_dfs.append(leg_df)

            if not tile_legacy_dfs:
                # No Legacy data for this region
                tile_candidates["legacy_match"] = False
                tile_candidates["legacy_sep_arcsec"] = np.inf
                tile_candidates["legacy_mag_g"] = 99.0
                tile_candidates["legacy_mag_r"] = 99.0
                tile_candidates["legacy_mag_z"] = 99.0
                tile_candidates["legacy_type"] = None
            else:
                tile_legacy = pd.concat(tile_legacy_dfs, ignore_index=True)
                tile_candidates = crossmatch_candidates_with_legacy(
                    tile_candidates, tile_legacy, RADIUS_WISE_LEGACY
                )

            results.append(tile_candidates)

        result = pd.concat(results, ignore_index=True)

    else:
        # Load all sweep files
        logger.info("Loading full Legacy Survey sweep files...")
        legacy_df = load_legacy_sweep_files(region)

        if legacy_df is None or len(legacy_df) == 0:
            logger.info("No local Legacy Survey data available")
            logger.info("Falling back to NOIRLab TAP query...")

            if PYVO_AVAILABLE:
                result = crossmatch_with_tap(candidates)
                if result is None:
                    logger.error("TAP crossmatch failed")
                    return None
            else:
                logger.error("pyvo not available for TAP crossmatch")
                logger.info("Run: python download_legacy_survey.py --region north")
                return None
        else:
            result = crossmatch_candidates_with_legacy(candidates, legacy_df, RADIUS_WISE_LEGACY)

    # Compute scores
    logger.info("Computing Legacy Survey-based scores...")
    result = compute_legacy_scores(result)

    # Summary statistics
    n_total = len(result)
    n_matches = result["legacy_match"].sum()
    n_bright = result["legacy_is_bright"].sum()
    n_faint = result["legacy_is_faint"].sum()
    n_dark = n_total - n_matches

    logger.info("=" * 60)
    logger.info("Legacy Survey Cross-match Summary:")
    logger.info(f"  Total candidates: {n_total:,}")
    logger.info(
        f"  Legacy matches (visible in deep optical): {n_matches:,} ({100*n_matches/n_total:.1f}%)"
    )
    logger.info(f"    Bright (strong veto): {n_bright:,}")
    logger.info(f"    Faint (mild veto): {n_faint:,}")
    logger.info(f"  Truly dark (no Legacy match): {n_dark:,} ({100*n_dark/n_total:.1f}%)")
    logger.info("  Score distribution:")
    logger.info(f"    Mean: {result['legacy_score'].mean():.2f}")
    logger.info("=" * 60)

    # Save results
    result.to_parquet(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    # Save "truly dark" candidates
    dark_candidates = result[~result["legacy_match"]].copy()
    if len(dark_candidates) > 0:
        dark_file = output_file.parent / "tier4_truly_dark.parquet"
        dark_candidates.to_parquet(dark_file, index=False)
        logger.info(f"Saved {len(dark_candidates)} truly dark candidates to {dark_file}")

    # Save summary JSON
    summary = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "n_candidates": n_total,
        "n_legacy_matches": int(n_matches),
        "n_bright_detections": int(n_bright),
        "n_faint_detections": int(n_faint),
        "n_truly_dark": int(n_dark),
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = output_file.parent / "legacy_crossmatch_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-match TASNI candidates with Legacy Survey")
    parser.add_argument("--input", type=str, help="Input candidate parquet file")
    parser.add_argument("--output", type=str, help="Output file with Legacy matches")
    parser.add_argument(
        "--region",
        choices=["north", "south"],
        default="north",
        help="Legacy Survey region (default: north for BASS)",
    )
    parser.add_argument(
        "--use-healpix", action="store_true", help="Use HEALPix-organized tiles (faster)"
    )
    parser.add_argument(
        "--use-sweeps",
        action="store_true",
        help="Use raw sweep files (works without HEALPix reorganization)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Legacy Survey DR10 Cross-match")
    logger.info("=" * 60)
    logger.info("Deep Optical Veto: g=24.2, r=23.6, z=23.0")
    logger.info("=" * 60)

    use_healpix = args.use_healpix or not args.use_sweeps

    crossmatch_tier4_with_legacy(
        input_file=args.input, output_file=args.output, use_healpix=use_healpix, region=args.region
    )


if __name__ == "__main__":
    main()
