"""
TASNI: Download LAMOST DR12 Stellar Parameters
===============================================

Downloads LAMOST DR12 stellar parameter catalog via TAP service.
LAMOST = Large Sky Area Multi-Object Fiber Spectroscopic Telescope

Key data:
- 28 million spectra total
- 8.3 million stellar parameter measurements (Teff, logg, [Fe/H])
- Spectral classifications (STAR, GALAXY, QSO)
- Radial velocities

Why LAMOST for TASNI?
- If a WISE orphan has LAMOST classification -> we know what it is -> veto
- Known M/L/T dwarfs, carbon stars explain IR emission
- Temperature mismatch (spectral Teff vs IR Teff) -> anomalous

Usage:
    python download_lamost.py [--mode full|sample] [--workers N]
    python download_lamost.py --test  # Download small test region
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

try:
    import pyvo as vo

    PYVO_AVAILABLE = True
except ImportError:
    PYVO_AVAILABLE = False

try:
    from astroquery.vizier import Vizier

    VIZIER_AVAILABLE = True
except ImportError:
    VIZIER_AVAILABLE = False

from tasni.core.config import (
    CHECKPOINT_DIR,
    HEALPIX_NSIDE,
    LAMOST_DIR,
    LAMOST_TAP_URL,
    LOG_DIR,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    ensure_dirs,
)

ensure_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [LAMOST] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download_lamost.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Alternative: VizieR mirror of LAMOST (may be more accessible)
VIZIER_LAMOST_CATALOG = "V/164/dr7"  # LAMOST DR7 on VizieR (older but accessible)
VIZIER_LAMOST_DR9 = "J/ApJS/266/15"  # LAMOST DR9 stellar params

# Checkpoint
CHECKPOINT_FILE = CHECKPOINT_DIR / "lamost_download.json"


def get_healpix_cone(nside, ipix, margin=1.1):
    """Get cone search parameters for a HEALPix pixel"""
    theta, phi = hp.pix2ang(nside, ipix, nest=True)
    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)

    # Pixel radius with margin
    area = hp.nside2pixarea(nside)
    radius = np.degrees(np.sqrt(area / np.pi)) * margin

    return ra, dec, radius


def download_lamost_tap_tile(tile_idx, tap_service=None):
    """Download LAMOST data for a HEALPix tile via TAP"""

    output_file = LAMOST_DIR / f"lamost_hp{tile_idx:05d}.parquet"

    if output_file.exists():
        return tile_idx, 0, "skipped"

    ra, dec, radius = get_healpix_cone(HEALPIX_NSIDE, tile_idx)

    # Build ADQL query
    columns = ", ".join(
        [
            "obsid",
            "ra",
            "dec",
            "snr_g",
            "snr_r",
            "snr_i",
            "teff",
            "teff_err",
            "logg",
            "logg_err",
            "feh",
            "feh_err",
            "rv",
            "rv_err",
            "class",
            "subclass",
        ]
    )

    query = f"""
    SELECT {columns}
    FROM dr12_v1_1_lr_stellar
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius})
    )
    """

    for attempt in range(MAX_RETRIES):
        try:
            if tap_service is None:
                tap_service = vo.dal.TAPService(LAMOST_TAP_URL)

            job = tap_service.submit_job(query)
            job.run()
            job.wait(timeout=REQUEST_TIMEOUT)
            job.raise_if_error()

            result = job.fetch_result()
            table = result.to_table()
            df = table.to_pandas()

            if len(df) == 0:
                # Empty tile - save marker
                pd.DataFrame().to_parquet(output_file)
                return tile_idx, 0, "empty"

            # Filter to exact HEALPix pixel
            pixels = hp.ang2pix(
                HEALPIX_NSIDE, df["ra"].values, df["dec"].values, nest=True, lonlat=True
            )
            df = df[pixels == tile_idx]

            df.to_parquet(output_file, index=False)
            return tile_idx, len(df), "ok"

        except Exception as e:
            logger.warning(f"Tile {tile_idx} attempt {attempt + 1} failed: {e}")
            time.sleep(2**attempt)

    return tile_idx, 0, f"failed after {MAX_RETRIES} attempts"


def download_lamost_vizier_tile(tile_idx):
    """Download LAMOST data via VizieR (alternative if TAP fails)"""

    output_file = LAMOST_DIR / f"lamost_hp{tile_idx:05d}.parquet"

    if output_file.exists():
        return tile_idx, 0, "skipped"

    ra, dec, radius = get_healpix_cone(HEALPIX_NSIDE, tile_idx)

    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier

        # Configure Vizier
        v = Vizier(columns=["*"], row_limit=-1)

        coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))

        # Query LAMOST catalog
        result = v.query_region(coord, radius=radius * u.deg, catalog=VIZIER_LAMOST_DR9)

        if not result or len(result) == 0:
            pd.DataFrame().to_parquet(output_file)
            return tile_idx, 0, "empty"

        df = result[0].to_pandas()

        # Rename columns to match our schema
        col_map = {
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Teff": "teff",
            "e_Teff": "teff_err",
            "logg": "logg",
            "e_logg": "logg_err",
            "[Fe/H]": "feh",
            "e_[Fe/H]": "feh_err",
            "RV": "rv",
            "e_RV": "rv_err",
            "Class": "class",
            "SubClass": "subclass",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Filter to HEALPix pixel
        if "ra" in df.columns and "dec" in df.columns:
            pixels = hp.ang2pix(
                HEALPIX_NSIDE, df["ra"].values, df["dec"].values, nest=True, lonlat=True
            )
            df = df[pixels == tile_idx]

        df.to_parquet(output_file, index=False)
        return tile_idx, len(df), "ok"

    except Exception as e:
        return tile_idx, 0, f"error: {e}"


def download_lamost_bulk_via_http():
    """
    Download pre-built LAMOST catalogs from official FTP/HTTP.
    This is more reliable than TAP for bulk downloads.
    """
    import urllib.request

    # LAMOST provides bulk CSV/FITS downloads
    base_url = "https://www.lamost.org/dr12/v1.1/catalogue/"

    catalogs = [
        ("lr_stellar.csv.gz", "Low-resolution stellar parameters"),
        ("lr_classification.csv.gz", "Classification catalog"),
    ]

    for filename, desc in catalogs:
        output_file = LAMOST_DIR / filename

        if output_file.exists():
            logger.info(f"Already have {filename}")
            continue

        url = base_url + filename
        logger.info(f"Downloading {desc}: {url}")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TASNI/1.0"})
            with urllib.request.urlopen(req, timeout=3600) as response:
                with open(output_file, "wb") as f:
                    # Stream download for large files
                    while True:
                        chunk = response.read(8192 * 1024)  # 8MB chunks
                        if not chunk:
                            break
                        f.write(chunk)

            logger.info(f"Downloaded {filename}")

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")


def convert_bulk_to_healpix(input_file, nside=32):
    """Convert bulk LAMOST catalog to HEALPix tiles"""

    logger.info(f"Converting {input_file} to HEALPix tiles (NSIDE={nside})")

    # Read in chunks for memory efficiency
    healpix_dir = LAMOST_DIR / "healpix"
    healpix_dir.mkdir(exist_ok=True)

    npix = hp.nside2npix(nside)
    tile_data = {i: [] for i in range(npix)}

    chunksize = 500_000

    if str(input_file).endswith(".gz"):
        import gzip

        opener = gzip.open
    else:
        opener = open

    total_rows = 0

    try:
        for chunk in pd.read_csv(
            input_file, chunksize=chunksize, low_memory=False, on_bad_lines="skip"
        ):
            # Find RA/Dec columns
            ra_col = next((c for c in chunk.columns if c.lower() in ["ra", "raj2000"]), None)
            dec_col = next((c for c in chunk.columns if c.lower() in ["dec", "dej2000"]), None)

            if ra_col is None or dec_col is None:
                logger.error(f"Cannot find RA/Dec columns in {chunk.columns.tolist()}")
                return

            # Compute HEALPix
            ra = chunk[ra_col].values
            dec = chunk[dec_col].values

            valid = ~(np.isnan(ra) | np.isnan(dec))
            chunk = chunk[valid]
            ra = ra[valid]
            dec = dec[valid]

            pixels = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
            chunk["healpix"] = pixels

            for pix in np.unique(pixels):
                tile_data[pix].append(chunk[chunk["healpix"] == pix].drop(columns=["healpix"]))

            total_rows += len(chunk)
            logger.info(f"Processed {total_rows:,} rows")

    except Exception as e:
        logger.error(f"Error reading {input_file}: {e}")
        return

    # Save tiles
    saved = 0
    for pix in range(npix):
        if tile_data[pix]:
            tile_df = pd.concat(tile_data[pix], ignore_index=True)
            tile_file = healpix_dir / f"lamost_hp{pix:05d}.parquet"
            tile_df.to_parquet(tile_file, index=False)
            saved += 1

    logger.info(f"Saved {saved} HEALPix tiles with {total_rows:,} total sources")


def download_test_region():
    """Download a small test region to verify connectivity"""

    logger.info("Downloading test region (Orion area)")

    # Orion: RA~83.6, Dec~-5.4
    test_query = """
    SELECT obsid, ra, dec, teff, logg, feh, rv, class, subclass, snr_g
    FROM dr12_v1_1_lr_stellar
    WHERE ra BETWEEN 82 AND 85
      AND dec BETWEEN -7 AND -4
    """

    output_file = LAMOST_DIR / "test_orion.parquet"

    try:
        if PYVO_AVAILABLE:
            logger.info(f"Querying LAMOST TAP: {LAMOST_TAP_URL}")
            tap = vo.dal.TAPService(LAMOST_TAP_URL)

            job = tap.submit_job(test_query)
            job.run()
            job.wait(timeout=120)
            job.raise_if_error()

            result = job.fetch_result()
            df = result.to_table().to_pandas()

            logger.info(f"Retrieved {len(df):,} sources from LAMOST")
            df.to_parquet(output_file, index=False)

            # Show sample
            logger.info("Sample data:")
            logger.info(df.head(10).to_string())

            return True
        else:
            logger.error("pyvo not available - install with: pip install pyvo")
            return False

    except Exception as e:
        logger.error(f"TAP query failed: {e}")
        logger.info("Trying VizieR fallback...")

        if VIZIER_AVAILABLE:
            try:
                import astropy.units as u
                from astropy.coordinates import SkyCoord

                v = Vizier(columns=["*"], row_limit=10000)
                coord = SkyCoord(ra=83.6, dec=-5.4, unit=(u.deg, u.deg))
                result = v.query_region(coord, radius=2 * u.deg, catalog="V/164")

                if result:
                    df = result[0].to_pandas()
                    logger.info(f"VizieR returned {len(df)} sources")
                    df.to_parquet(output_file, index=False)
                    return True

            except Exception as ve:
                logger.error(f"VizieR also failed: {ve}")

        return False


def download_full(workers=4, method="tap"):
    """Download full LAMOST catalog"""

    npix = hp.nside2npix(HEALPIX_NSIDE)

    logger.info("Downloading LAMOST DR12 stellar parameters")
    logger.info(f"Method: {method}")
    logger.info(f"HEALPix tiles: {npix}")

    if method == "bulk":
        # Try bulk HTTP download first (fastest)
        download_lamost_bulk_via_http()

        # Convert to HEALPix
        bulk_file = LAMOST_DIR / "lr_stellar.csv.gz"
        if bulk_file.exists():
            convert_bulk_to_healpix(bulk_file)
        return

    # TAP or VizieR method - tile by tile
    download_func = download_lamost_tap_tile if method == "tap" else download_lamost_vizier_tile

    # Load checkpoint
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            state = json.load(f)
    else:
        state = {"completed": [], "failed": [], "total_sources": 0}

    completed = set(state.get("completed", []))
    remaining = [i for i in range(npix) if i not in completed]

    logger.info(f"Already completed: {len(completed)} tiles")
    logger.info(f"Remaining: {len(remaining)} tiles")

    if not remaining:
        logger.info("All tiles already downloaded!")
        return

    total_sources = state.get("total_sources", 0)
    failed = list(state.get("failed", []))

    start_time = time.time()

    # Create TAP service once if using TAP
    tap_service = None
    if method == "tap" and PYVO_AVAILABLE:
        try:
            tap_service = vo.dal.TAPService(LAMOST_TAP_URL)
        except:
            pass

    with ThreadPoolExecutor(max_workers=workers) as executor:
        if method == "tap":
            futures = {
                executor.submit(download_lamost_tap_tile, t, tap_service): t for t in remaining
            }
        else:
            futures = {executor.submit(download_lamost_vizier_tile, t): t for t in remaining}

        for i, future in enumerate(as_completed(futures)):
            tile_idx, n_sources, status = future.result()

            if status in ["ok", "empty", "skipped"]:
                completed.add(tile_idx)
                total_sources += n_sources
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 3600
                    logger.info(
                        f"[{i + 1}/{len(remaining)}] {rate:.0f} tiles/hour, {total_sources:,} sources"
                    )
            else:
                failed.append(tile_idx)
                logger.warning(f"Tile {tile_idx}: {status}")

            # Checkpoint
            if (i + 1) % 50 == 0:
                state = {
                    "completed": list(completed),
                    "failed": failed,
                    "total_sources": total_sources,
                    "last_update": datetime.now().isoformat(),
                }
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(state, f)

    # Final save
    elapsed = time.time() - start_time
    state = {
        "completed": list(completed),
        "failed": failed,
        "total_sources": total_sources,
        "elapsed_seconds": elapsed,
        "last_update": datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)

    logger.info("=" * 60)
    logger.info("LAMOST download complete")
    logger.info(f"Total sources: {total_sources:,}")
    logger.info(f"Completed tiles: {len(completed)}/{npix}")
    logger.info(f"Failed tiles: {len(failed)}")
    logger.info(f"Time: {elapsed / 3600:.1f} hours")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Download LAMOST DR12 catalog")
    parser.add_argument(
        "--method",
        choices=["tap", "vizier", "bulk"],
        default="tap",
        help="Download method (default: tap)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for TAP/VizieR")
    parser.add_argument("--test", action="store_true", help="Download small test region only")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: LAMOST DR12 Downloader")
    logger.info("=" * 60)
    logger.info("Large Sky Area Multi-Object Fiber Spectroscopic Telescope")
    logger.info("28 million spectra, 8.3 million stellar parameters")
    logger.info("=" * 60)

    if args.test:
        success = download_test_region()
        if success:
            logger.info("Test successful! Ready for full download.")
        else:
            logger.error("Test failed. Check network/authentication.")
        return

    download_full(workers=args.workers, method=args.method)


if __name__ == "__main__":
    main()
