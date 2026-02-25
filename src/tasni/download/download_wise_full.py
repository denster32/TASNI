"""
TASNI: Download AllWISE Catalog (Full Sky)
===========================================

Downloads the complete AllWISE Source Catalog via IRSA TAP service.
~747 million sources, downloads in HEALPix chunks for manageability.

Strategy:
- Split sky into HEALPix tiles (NSIDE=32 = 12,288 tiles)
- Download each tile as a separate parquet file
- Track progress, support resume
- Estimated total: ~300GB

Usage:
    python download_wise_full.py [--start-tile N] [--end-tile N] [--dry-run]
"""

import argparse
import logging
import time

import healpy as hp
import numpy as np
import pyvo as vo
from astropy import units as u
from astropy.coordinates import SkyCoord

from tasni.core.config import (
    HEALPIX_NSIDE,
    LOG_DIR,
    WISE_COLUMNS,
    WISE_DIR,
    WISE_TAP_URL,
    ensure_dirs,
)

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "download_wise.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# TAP service
TAP_SERVICE = vo.dal.TAPService(WISE_TAP_URL)

# Total tiles at NSIDE=32
N_TILES = hp.nside2npix(HEALPIX_NSIDE)  # 12288


def get_healpix_corners(nside, ipix, nest=True):
    """Get the corner coordinates of a HEALPix pixel"""
    corners = hp.boundaries(nside, ipix, step=1, nest=nest)
    # corners shape is (3, 4) - x,y,z for 4 corners
    theta, phi = hp.vec2ang(corners.T)
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)
    return ra, dec


def get_healpix_center(nside, ipix, nest=True):
    """Get the center coordinates of a HEALPix pixel"""
    theta, phi = hp.pix2ang(nside, ipix, nest=nest)
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)
    return ra, dec


def get_healpix_cone_radius(nside):
    """Get radius that fully contains a HEALPix pixel (in degrees)"""
    # Area of pixel in steradians
    area = hp.nside2pixarea(nside)
    # Approximate as circle, get radius, add 10% margin
    radius = np.degrees(np.sqrt(area / np.pi)) * 1.1
    return radius


def download_tile(tile_idx, dry_run=False):
    """Download all WISE sources in a single HEALPix tile"""

    output_file = WISE_DIR / f"wise_hp{tile_idx:05d}.parquet"

    # Skip if already exists
    if output_file.exists():
        logger.info(f"Tile {tile_idx} already exists, skipping")
        return True

    # Get tile center and search radius
    ra_center, dec_center = get_healpix_center(HEALPIX_NSIDE, tile_idx, nest=True)
    radius = get_healpix_cone_radius(HEALPIX_NSIDE)

    # Build query
    columns = ", ".join(WISE_COLUMNS)
    query = f"""
    SELECT {columns}
    FROM allwise_p3as_psd
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center}, {dec_center}, {radius})
    ) = 1
    """

    if dry_run:
        logger.info(
            f"Tile {tile_idx}: center=({ra_center:.2f}, {dec_center:.2f}), radius={radius:.2f}°"
        )
        logger.info(f"Query:\n{query[:200]}...")
        return True

    logger.info(
        f"Downloading tile {tile_idx}/{N_TILES} - center=({ra_center:.2f}, {dec_center:.2f})"
    )

    try:
        start_time = time.time()

        # Execute query
        job = TAP_SERVICE.submit_job(query)
        job.run()
        job.wait()
        job.raise_if_error()

        # Get results
        result = job.fetch_result()
        table = result.to_table()
        df = table.to_pandas()

        elapsed = time.time() - start_time

        if len(df) == 0:
            logger.warning(f"Tile {tile_idx}: No sources found")
            # Save empty file to mark as processed
            df.to_parquet(output_file)
            return True

        # Filter to exact HEALPix pixel (cone query returns slightly more)
        coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
        pixels = hp.ang2pix(HEALPIX_NSIDE, coords.ra.deg, coords.dec.deg, nest=True, lonlat=True)
        df = df[pixels == tile_idx]

        # Save
        df.to_parquet(output_file, index=False)

        logger.info(f"Tile {tile_idx}: {len(df):,} sources in {elapsed:.1f}s -> {output_file.name}")
        return True

    except Exception as e:
        logger.error(f"Tile {tile_idx} failed: {e}")
        return False


def get_completed_tiles():
    """Get set of already-downloaded tile indices"""
    completed = set()
    for f in WISE_DIR.glob("wise_hp*.parquet"):
        try:
            idx = int(f.stem.split("hp")[1])
            completed.add(idx)
        except (ValueError, IndexError):
            continue  # Skip malformed filenames
    return completed


def main():
    parser = argparse.ArgumentParser(description="Download AllWISE catalog")
    parser.add_argument("--start-tile", type=int, default=0, help="Starting tile index")
    parser.add_argument("--end-tile", type=int, default=N_TILES, help="Ending tile index")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: AllWISE Full-Sky Download")
    logger.info("=" * 60)
    logger.info(f"HEALPix NSIDE: {HEALPIX_NSIDE} ({N_TILES} tiles)")
    logger.info(f"Tile range: {args.start_tile} to {args.end_tile}")
    logger.info(f"Output: {WISE_DIR}")

    # Check progress
    completed = get_completed_tiles()
    logger.info(f"Already completed: {len(completed)} tiles")

    # Download tiles
    failed = []
    for tile_idx in range(args.start_tile, args.end_tile):
        if tile_idx in completed:
            continue

        success = download_tile(tile_idx, dry_run=args.dry_run)
        if not success:
            failed.append(tile_idx)

        # Brief pause to be nice to the server
        if not args.dry_run:
            time.sleep(0.5)

    # Summary
    logger.info("=" * 60)
    logger.info("Download complete")
    logger.info(f"Completed: {len(get_completed_tiles())} / {N_TILES} tiles")
    if failed:
        logger.warning(f"Failed tiles: {failed}")
        logger.info("Re-run script to retry failed tiles")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
