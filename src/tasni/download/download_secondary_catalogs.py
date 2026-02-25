"""
TASNI: Multi-Wavelength Catalog Downloader
==========================================

Downloads secondary astronomical catalogs for enhanced anomaly detection:
- 2MASS (Near-IR): J, H, Ks bands
- Spitzer (Mid-IR): IRAC bands
- AKARI (Far-IR)
- GALEX (UV)
- Chandra (X-ray)
- Fermi (Gamma-ray)
- VLASS/NVSS (Radio)

Usage:
    python download_secondary_catalogs.py [--catalog 2mass,spitzer] [--workers N]
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import healpy as hp
import numpy as np
import pandas as pd

from tasni.core.config import DATA_ROOT, HEALPIX_NSIDE, LOG_DIR, ensure_dirs

# Setup
ensure_dirs()

# Create secondary catalog directory
SECONDARY_DIR = DATA_ROOT / "data" / "secondary"
SECONDARY_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [DOWNLOAD] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "secondary_download.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Catalog definitions
CATALOGS = {
    "2mass": {
        "name": "2MASS Point Source Catalog",
        "tap_url": "https://irsa.ipac.caltech.edu/TAP",
        "table": "fp_psc",
        "columns": [
            "ra",
            "dec",
            "j_m",
            "h_m",
            "ks_m",  # Magnitudes
            "j_msig",
            "h_msig",
            "ks_msig",  # Uncertainties
            "ph_qual",  # Photometric quality
            "rd_flg",  # Read flag
            "glon",
            "glat",  # Galactic coords
        ],
        "priority": 1,
        "n_sources": 470_000_000,
        "description": "Near-IR (1.2, 1.6, 2.2 μm)",
    },
    "spitzer": {
        "name": "Spitzer IRAC",
        "tap_url": "https://irsa.ipac.caltech.edu/TAP",
        "table": "irac_spet",
        "columns": [
            "ra",
            "dec",
            "i1_mag",
            "i2_mag",
            "i3_mag",
            "i4_mag",  # IRAC channels
            "i1_sig",
            "i2_sig",
        ],
        "priority": 2,
        "n_sources": 100_000_000,
        "description": "Mid-IR (3.6, 4.5, 5.8, 8.0 μm)",
    },
    "akari": {
        "name": "AKARI/IRC",
        "tap_url": "https://irsa.ipac.caltech.edu/TAP",
        "table": "akari_irc_psc",
        "columns": ["ra", "dec", "f_color_1", "f_color_2"],
        "priority": 3,
        "description": "Far-IR (9, 18 μm)",
    },
    "chandra": {
        "name": "Chandra Source Catalog",
        "url": "https://cxc.harvard.edu/csc/",
        "priority": 4,
        "description": "X-ray (0.1-10 keV)",
        "download_type": "http",
    },
    "fermi": {
        "name": "Fermi-LAT",
        "url": "https://fermi.gsfc.nasa.gov/ssc/data/access/",
        "priority": 5,
        "description": "Gamma-ray (>100 MeV)",
    },
    "vlass": {
        "name": "VLASS (Radio)",
        "url": "https://archive.nrao.edu/vlass/",
        "priority": 6,
        "description": "Radio (3 GHz)",
    },
    "nvss": {
        "name": "NVSS (Radio)",
        "table": "nvss",
        "priority": 7,
        "description": "Radio (1.4 GHz)",
    },
}


def download_2mass_healpix_tile(tile_idx):
    """Download 2MASS data for a single HEALPix tile via IRSA TAP"""

    output_file = SECONDARY_DIR / f"2mass_hp{tile_idx:05d}.parquet"

    if output_file.exists():
        return tile_idx, 0, "skipped"

    try:
        # Get HEALPix boundary
        npix = hp.nside2npix(HEALPIX_NSIDE)

        # Calculate tile center and radius
        theta, phi = hp.pix2ang(HEALPIX_NSIDE, tile_idx, nest=False)
        ra = np.degrees(phi)
        dec = 90 - np.degrees(theta)

        # Approximate tile radius (HEALPix NSIDE=32 ~ 3.4 deg)
        radius_deg = 2.0

        # Query 2MASS
        query = f"""
        SELECT {','.join(CATALOGS['2mass']['columns'])}
        FROM fp_psc
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                      CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
        """

        # Use astroquery
        from astroquery.irsa import Irsa

        result = Irsa.query_tap(query, maxrec=10_000_000)

        if len(result) == 0:
            # Empty tile
            pd.DataFrame().to_parquet(output_file)
            return tile_idx, 0, "empty"

        # Convert to DataFrame and save
        df = result.to_pandas()
        df.to_parquet(output_file, index=False)

        return tile_idx, len(df), "ok"

    except Exception as e:
        return tile_idx, 0, f"error: {e}"


def download_spitzer_healpix_tile(tile_idx):
    """Download Spitzer data for a single HEALPix tile"""

    output_file = SECONDARY_DIR / f"spitzer_hp{tile_idx:05d}.parquet"

    if output_file.exists():
        return tile_idx, 0, "skipped"

    try:
        # Get tile center
        theta, phi = hp.pix2ang(HEALPIX_NSIDE, tile_idx, nest=False)
        ra = np.degrees(phi)
        dec = 90 - np.degrees(theta)
        radius_deg = 2.0

        # Query Spitzer
        query = f"""
        SELECT ra, dec, i1_mag, i2_mag, i3_mag, i4_mag
        FROM irac_spet
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                      CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
        """

        from astroquery.irsa import Irsa

        result = Irsa.query_tap(query, maxrec=1_000_000)

        if len(result) == 0:
            pd.DataFrame().to_parquet(output_file)
            return tile_idx, 0, "empty"

        df = result.to_pandas()
        df.to_parquet(output_file, index=False)

        return tile_idx, len(df), "ok"

    except Exception as e:
        return tile_idx, 0, f"error: {e}"


def download_catalog(catalog_name, workers=4):
    """Download all tiles for a catalog"""

    if catalog_name not in CATALOGS:
        logger.error(f"Unknown catalog: {catalog_name}")
        return

    cat = CATALOGS[catalog_name]
    logger.info(f"Downloading {cat['name']}")
    logger.info(f"Description: {cat['description']}")

    npix = hp.nside2npix(HEALPIX_NSIDE)

    if catalog_name == "2mass":
        func = download_2mass_healpix_tile
    elif catalog_name == "spitzer":
        func = download_spitzer_healpix_tile
    else:
        logger.error(f"Download not implemented for {catalog_name}")
        return

    total_sources = 0
    failed = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(func, tile): tile for tile in range(npix)}

        for i, future in enumerate(as_completed(futures)):
            tile_idx, n_sources, status = future.result()
            total_sources += n_sources

            if "error" in status:
                failed.append(tile_idx)
                logger.error(f"Tile {tile_idx}: {status}")
            else:
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(
                        f"[{i+1}/{npix}] Tile {tile_idx}: {n_sources} sources ({rate:.1f} tiles/s)"
                    )

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"{cat['name']} download complete")
    logger.info(f"Total sources: {total_sources:,}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if failed:
        logger.warning(f"Failed tiles: {len(failed)}")
    logger.info("=" * 60)

    # Save summary
    summary = {
        "catalog": catalog_name,
        "n_tiles": npix,
        "total_sources": total_sources,
        "failed_tiles": failed,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = SECONDARY_DIR / f"{catalog_name}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download secondary catalogs")
    parser.add_argument(
        "--catalog", type=str, default="2mass", help="Catalog to download (2mass, spitzer, akari)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--all", action="store_true", help="Download all high-priority catalogs")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Multi-Wavelength Catalog Downloader")
    logger.info("=" * 60)

    if args.all:
        # Download high-priority catalogs
        for cat in ["2mass", "spitzer"]:
            logger.info(f"Starting {cat} download...")
            download_catalog(cat, args.workers)
    else:
        download_catalog(args.catalog, args.workers)


if __name__ == "__main__":
    main()
