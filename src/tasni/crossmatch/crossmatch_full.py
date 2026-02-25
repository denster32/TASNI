"""
TASNI: Cross-Match WISE and Gaia (Full Sky)
============================================

Processes HEALPix tiles in parallel:
- For each tile, load WISE and Gaia
- Find WISE sources with NO Gaia counterpart
- Save orphans to crossmatch directory

Usage:
    python crossmatch_full.py [--workers N] [--start-tile N] [--end-tile N]
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import healpy as hp
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

from tasni.core.config import (
    CROSSMATCH_DIR,
    GAIA_DIR,
    HEALPIX_NSIDE,
    LOG_DIR,
    MATCH_RADIUS_ARCSEC,
    N_WORKERS,
    OUTPUT_DIR,
    WISE_DIR,
    ensure_dirs,
)

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "crossmatch.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

N_TILES = hp.nside2npix(HEALPIX_NSIDE)


def process_tile(tile_idx):
    """Process a single HEALPix tile - find WISE sources with no Gaia match"""

    wise_file = WISE_DIR / f"wise_hp{tile_idx:05d}.parquet"
    gaia_file = GAIA_DIR / f"gaia_hp{tile_idx:05d}.parquet"
    output_file = CROSSMATCH_DIR / f"orphans_hp{tile_idx:05d}.parquet"

    # Skip if already done
    if output_file.exists():
        return tile_idx, 0, "skipped"

    # Check inputs exist
    if not wise_file.exists():
        return tile_idx, 0, "no_wise"
    if not gaia_file.exists():
        return tile_idx, 0, "no_gaia"

    try:
        # Load data
        wise_df = pd.read_parquet(wise_file)
        gaia_df = pd.read_parquet(gaia_file)

        if len(wise_df) == 0:
            # No WISE sources, save empty
            wise_df.to_parquet(output_file)
            return tile_idx, 0, "empty_wise"

        if len(gaia_df) == 0:
            # No Gaia sources - ALL WISE are orphans!
            wise_df["nearest_gaia_sep_arcsec"] = np.inf
            wise_df.to_parquet(output_file)
            return tile_idx, len(wise_df), "all_orphans"

        # Build coordinate arrays
        wise_coords = SkyCoord(
            ra=wise_df["ra"].values * u.degree, dec=wise_df["dec"].values * u.degree
        )
        gaia_coords = SkyCoord(
            ra=gaia_df["ra"].values * u.degree, dec=gaia_df["dec"].values * u.degree
        )

        # Cross-match
        idx, sep2d, _ = match_coordinates_sky(wise_coords, gaia_coords)

        # Find orphans (no match within radius)
        no_match_mask = sep2d.arcsec > MATCH_RADIUS_ARCSEC

        orphans = wise_df[no_match_mask].copy()
        orphans["nearest_gaia_sep_arcsec"] = sep2d.arcsec[no_match_mask]

        # Save
        orphans.to_parquet(output_file, index=False)

        return tile_idx, len(orphans), "ok"

    except Exception as e:
        return tile_idx, 0, f"error: {e}"


def get_ready_tiles():
    """Get tiles that have both WISE and Gaia data"""
    wise_tiles = {int(f.stem.split("hp")[1]) for f in WISE_DIR.glob("wise_hp*.parquet")}
    gaia_tiles = {int(f.stem.split("hp")[1]) for f in GAIA_DIR.glob("gaia_hp*.parquet")}
    return wise_tiles & gaia_tiles


def get_completed_tiles():
    """Get tiles already processed"""
    return {int(f.stem.split("hp")[1]) for f in CROSSMATCH_DIR.glob("orphans_hp*.parquet")}


def merge_orphans():
    """Merge all orphan files into single catalog"""

    logger.info("Merging orphan files...")

    orphan_files = sorted(CROSSMATCH_DIR.glob("orphans_hp*.parquet"))
    if not orphan_files:
        logger.warning("No orphan files found!")
        return

    dfs = []
    for f in orphan_files:
        df = pd.read_parquet(f)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        logger.warning("All orphan files were empty!")
        return

    merged = pd.concat(dfs, ignore_index=True)
    output_file = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    merged.to_parquet(output_file, index=False)

    logger.info(f"Merged {len(merged):,} total orphans -> {output_file}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Cross-match WISE and Gaia")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="Parallel workers")
    parser.add_argument("--start-tile", type=int, default=0)
    parser.add_argument("--end-tile", type=int, default=N_TILES)
    parser.add_argument(
        "--merge-only", action="store_true", help="Just merge existing orphan files"
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.merge_only:
        merge_orphans()
        return

    logger.info("=" * 60)
    logger.info("TASNI: Full-Sky Cross-Match")
    logger.info("=" * 60)
    logger.info(f"Match radius: {MATCH_RADIUS_ARCSEC} arcsec")
    logger.info(f"Workers: {args.workers}")

    # Find tiles to process
    ready = get_ready_tiles()
    completed = get_completed_tiles()
    to_process = sorted(ready - completed)
    to_process = [t for t in to_process if args.start_tile <= t < args.end_tile]

    logger.info(f"Tiles ready: {len(ready)}")
    logger.info(f"Tiles completed: {len(completed)}")
    logger.info(f"Tiles to process: {len(to_process)}")

    if not to_process:
        logger.info("Nothing to process. Running merge...")
        merge_orphans()
        return

    # Process in parallel
    total_orphans = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_tile, t): t for t in to_process}

        for i, future in enumerate(as_completed(futures)):
            tile_idx, n_orphans, status = future.result()
            total_orphans += n_orphans

            if "error" in status:
                failed.append(tile_idx)
                logger.error(f"Tile {tile_idx}: {status}")
            else:
                logger.info(
                    f"[{i+1}/{len(to_process)}] Tile {tile_idx}: {n_orphans:,} orphans ({status})"
                )

    # Merge
    merge_orphans()

    # Summary
    logger.info("=" * 60)
    logger.info("Cross-match complete")
    logger.info(f"Total orphans: {total_orphans:,}")
    if failed:
        logger.warning(f"Failed tiles: {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
