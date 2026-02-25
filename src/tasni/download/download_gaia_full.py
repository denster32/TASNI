"""
TASNI: Download Gaia DR3 Catalog (Full Sky)
============================================

Downloads Gaia DR3 positions via ESA TAP service.
We only need: position, magnitude, parallax for crossmatching.

~1.8 billion sources total, but we download in HEALPix chunks.

ALTERNATIVE (FASTER):
---------------------
Instead of downloading all of Gaia, consider using CDS X-Match:
http://cdsxmatch.u-strasbg.fr/

Upload WISE orphans -> get Gaia matches -> what's left are true orphans.
This is MUCH faster than downloading 1.8B sources.

See: crossmatch_cds.py for this approach.

Usage:
    python download_gaia_full.py [--start-tile N] [--end-tile N] [--dry-run]
"""

import argparse
import logging
import time

import healpy as hp
import numpy as np
from astroquery.gaia import Gaia

from tasni.core.config import GAIA_COLUMNS, GAIA_DIR, HEALPIX_NSIDE, LOG_DIR, ensure_dirs

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "download_gaia.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configure Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1  # No limit

# Total tiles at NSIDE=32
N_TILES = hp.nside2npix(HEALPIX_NSIDE)  # 12288


def get_healpix_center(nside, ipix, nest=True):
    """Get the center coordinates of a HEALPix pixel"""
    theta, phi = hp.pix2ang(nside, ipix, nest=nest)
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)
    return ra, dec


def get_healpix_cone_radius(nside):
    """Get radius that fully contains a HEALPix pixel (in degrees)"""
    area = hp.nside2pixarea(nside)
    radius = np.degrees(np.sqrt(area / np.pi)) * 1.1
    return radius


def download_tile(tile_idx, dry_run=False):
    """Download all Gaia sources in a single HEALPix tile"""

    output_file = GAIA_DIR / f"gaia_hp{tile_idx:05d}.parquet"

    # Skip if already exists
    if output_file.exists():
        logger.info(f"Tile {tile_idx} already exists, skipping")
        return True

    # Get tile center and search radius
    ra_center, dec_center = get_healpix_center(HEALPIX_NSIDE, tile_idx, nest=True)
    radius = get_healpix_cone_radius(HEALPIX_NSIDE)

    columns = ", ".join(GAIA_COLUMNS)

    # ADQL query with cone search
    query = f"""
    SELECT {columns}
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_center}, {dec_center}, {radius})
    ) = 1
    """

    if dry_run:
        logger.info(
            f"Tile {tile_idx}: center=({ra_center:.2f}, {dec_center:.2f}), radius={radius:.2f}°"
        )
        return True

    logger.info(
        f"Downloading tile {tile_idx}/{N_TILES} - center=({ra_center:.2f}, {dec_center:.2f})"
    )

    try:
        start_time = time.time()

        # Execute query
        job = Gaia.launch_job_async(query)
        result = job.get_results()
        df = result.to_pandas()

        elapsed = time.time() - start_time

        if len(df) == 0:
            logger.warning(f"Tile {tile_idx}: No sources found")
            df.to_parquet(output_file)
            return True

        # Filter to exact HEALPix pixel
        pixels = hp.ang2pix(
            HEALPIX_NSIDE, df["ra"].values, df["dec"].values, nest=True, lonlat=True
        )
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
    for f in GAIA_DIR.glob("gaia_hp*.parquet"):
        try:
            idx = int(f.stem.split("hp")[1])
            completed.add(idx)
        except (ValueError, IndexError):
            continue  # Skip malformed filenames
    return completed


def main():
    parser = argparse.ArgumentParser(description="Download Gaia DR3 catalog")
    parser.add_argument("--start-tile", type=int, default=0, help="Starting tile index")
    parser.add_argument("--end-tile", type=int, default=N_TILES, help="Ending tile index")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: Gaia DR3 Full-Sky Download")
    logger.info("=" * 60)
    logger.info(f"HEALPix NSIDE: {HEALPIX_NSIDE} ({N_TILES} tiles)")
    logger.info(f"Tile range: {args.start_tile} to {args.end_tile}")
    logger.info(f"Output: {GAIA_DIR}")
    logger.info("")
    logger.info("NOTE: Consider using crossmatch_cds.py instead - it's faster!")
    logger.info("")

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

        # Pause to be nice to ESA
        if not args.dry_run:
            time.sleep(1.0)

    # Summary
    logger.info("=" * 60)
    logger.info("Download complete")
    logger.info(f"Completed: {len(get_completed_tiles())} / {N_TILES} tiles")
    if failed:
        logger.warning(f"Failed tiles: {failed}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
