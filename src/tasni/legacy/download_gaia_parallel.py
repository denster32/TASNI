"""
TASNI: Download Gaia DR3 Catalog (PARALLEL VERSION)
====================================================

Downloads multiple tiles simultaneously to maximize bandwidth.
Uses ThreadPoolExecutor for parallel HTTP requests.

Usage:
    python download_gaia_parallel.py [--workers 10]
"""

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import healpy as hp
import numpy as np
from astroquery.gaia import Gaia

from tasni.core.config import GAIA_COLUMNS, GAIA_DIR, HEALPIX_NSIDE, LOG_DIR, ensure_dirs

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "download_gaia_parallel.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configure Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1

N_TILES = hp.nside2npix(HEALPIX_NSIDE)  # 12288

# Thread-safe counters
lock = threading.Lock()
stats = {"completed": 0, "failed": 0, "sources": 0}


def get_healpix_center(nside, ipix, nest=True):
    theta, phi = hp.pix2ang(nside, ipix, nest=nest)
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)
    return ra, dec


def get_healpix_cone_radius(nside):
    area = hp.nside2pixarea(nside)
    radius = np.degrees(np.sqrt(area / np.pi)) * 1.1
    return radius


def download_tile(tile_idx):
    """Download a single tile - thread-safe"""

    output_file = GAIA_DIR / f"gaia_hp{tile_idx:05d}.parquet"

    if output_file.exists():
        return tile_idx, 0, "exists"

    ra_center, dec_center = get_healpix_center(HEALPIX_NSIDE, tile_idx, nest=True)
    radius = get_healpix_cone_radius(HEALPIX_NSIDE)

    columns = ", ".join(GAIA_COLUMNS)

    query = f"""
    SELECT {columns}
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center}, {dec_center}, {radius})
    ) = 1
    """

    try:
        start_time = time.time()

        job = Gaia.launch_job_async(query)
        result = job.get_results()
        df = result.to_pandas()

        elapsed = time.time() - start_time

        if len(df) > 0:
            pixels = hp.ang2pix(
                HEALPIX_NSIDE, df["ra"].values, df["dec"].values, nest=True, lonlat=True
            )
            df = df[pixels == tile_idx]

        df.to_parquet(output_file, index=False)

        with lock:
            stats["completed"] += 1
            stats["sources"] += len(df)

        return tile_idx, len(df), f"{elapsed:.1f}s"

    except Exception as e:
        with lock:
            stats["failed"] += 1
        return tile_idx, 0, f"error: {e}"


def get_completed_tiles():
    completed = set()
    for f in GAIA_DIR.glob("gaia_hp*.parquet"):
        try:
            idx = int(f.stem.split("hp")[1])
            completed.add(idx)
        except:
            pass
    return completed


def main():
    parser = argparse.ArgumentParser(description="Download Gaia DR3 (parallel)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--start-tile", type=int, default=0)
    parser.add_argument("--end-tile", type=int, default=N_TILES)
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: Gaia DR3 PARALLEL Download")
    logger.info("=" * 60)
    logger.info(f"Workers: {args.workers}")
    logger.info(f"HEALPix tiles: {N_TILES}")
    logger.info("")

    completed = get_completed_tiles()
    logger.info(f"Already completed: {len(completed)} tiles")

    # Get tiles to download
    to_download = [t for t in range(args.start_tile, args.end_tile) if t not in completed]
    logger.info(f"Tiles to download: {len(to_download)}")

    if not to_download:
        logger.info("Nothing to download!")
        return

    start_time = time.time()
    last_report = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_tile, t): t for t in to_download}

        for i, future in enumerate(as_completed(futures)):
            tile_idx, n_sources, status = future.result()

            # Log every tile
            if "error" in status:
                logger.error(f"[{i+1}/{len(to_download)}] Tile {tile_idx}: {status}")
            else:
                logger.info(
                    f"[{i+1}/{len(to_download)}] Tile {tile_idx}: {n_sources:,} sources ({status})"
                )

            # Rate report every 30 sec
            if time.time() - last_report > 30:
                elapsed = time.time() - start_time
                rate = stats["completed"] / elapsed * 60
                remaining = len(to_download) - i - 1
                eta_min = remaining / rate if rate > 0 else 0
                logger.info(
                    f">>> RATE: {rate:.1f} tiles/min | ETA: {eta_min:.0f} min | Sources: {stats['sources']:,}"
                )
                last_report = time.time()

    # Final stats
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Download complete in {elapsed/60:.1f} minutes")
    logger.info(f"Completed: {stats['completed']} | Failed: {stats['failed']}")
    logger.info(f"Total sources: {stats['sources']:,}")
    logger.info(f"Rate: {stats['completed'] / elapsed * 60:.1f} tiles/min")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
