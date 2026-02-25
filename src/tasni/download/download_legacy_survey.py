"""
TASNI: Download Legacy Survey DR10 (BASS + MzLS)
=================================================

Downloads the DESI Legacy Imaging Surveys DR10 sweep catalogs.
This includes BASS (g,r bands) and MzLS (z band) for the northern sky.

Why Legacy Survey instead of raw BASS?
- Pre-combined, calibrated photometry
- Includes cross-matches to Gaia
- Sweep files are manageable chunks (~1GB each)
- Deeper than Pan-STARRS: g=24.2, r=23.6, z=23.0 (5σ)

Coverage: Dec > ~+32° (northern galactic cap)
Total: ~2 billion sources

Usage:
    python download_legacy_survey.py [--region north|south] [--dry-run]
    python download_legacy_survey.py --list  # List available sweep files
"""

import argparse
import json
import logging
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from html.parser import HTMLParser

import numpy as np
import pandas as pd
from astropy.io import fits

from tasni.core.config import (
    CHECKPOINT_DIR,
    LEGACY_COLUMNS,
    LEGACY_DIR,
    LOG_DIR,
    MAX_RETRIES,
    ensure_dirs,
)

ensure_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [LEGACY] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "download_legacy.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Base URLs for sweep catalogs
SWEEP_URLS = {
    "north": "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/north/sweep/10.0/",
    "south": "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/sweep/10.0/",
}

# Checkpoint file
CHECKPOINT_FILE = CHECKPOINT_DIR / "legacy_download.json"


class SweepFileParser(HTMLParser):
    """Parse directory listing to extract sweep file names"""

    def __init__(self):
        super().__init__()
        self.files = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value.startswith("sweep-") and value.endswith(".fits"):
                    self.files.append(value)


def get_sweep_file_list(region="north"):
    """Fetch list of available sweep files from NERSC"""
    url = SWEEP_URLS.get(region, SWEEP_URLS["north"])

    logger.info(f"Fetching sweep file list from {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TASNI/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            html = response.read().decode("utf-8")

        parser = SweepFileParser()
        parser.feed(html)

        logger.info(f"Found {len(parser.files)} sweep files")
        return sorted(parser.files)

    except Exception as e:
        logger.error(f"Failed to fetch file list: {e}")
        return []


def parse_sweep_filename(filename):
    """
    Parse sweep filename to get RA/Dec coverage.
    Format: sweep-{ra_min}{dec_sign}{dec_min}-{ra_max}{dec_sign}{dec_max}.fits
    Example: sweep-000p005-010p010.fits covers RA 0-10, Dec +5 to +10
    """
    match = re.match(r"sweep-(\d{3})([mp])(\d{3})-(\d{3})([mp])(\d{3})\.fits", filename)
    if match:
        ra_min = int(match.group(1))
        dec_min = int(match.group(3)) * (1 if match.group(2) == "p" else -1)
        ra_max = int(match.group(4))
        dec_max = int(match.group(6)) * (1 if match.group(5) == "p" else -1)
        return {
            "ra_min": ra_min,
            "ra_max": ra_max,
            "dec_min": dec_min,
            "dec_max": dec_max,
            "filename": filename,
        }
    return None


def download_sweep_file(filename, region="north", output_dir=None):
    """Download a single sweep file and convert to parquet"""

    if output_dir is None:
        output_dir = LEGACY_DIR / region
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output as parquet (smaller, faster)
    parquet_file = output_dir / filename.replace(".fits", ".parquet")

    # Skip if already downloaded
    if parquet_file.exists():
        return filename, 0, "skipped"

    url = SWEEP_URLS[region] + filename
    fits_file = output_dir / filename

    for attempt in range(MAX_RETRIES):
        try:
            # Download FITS file
            logger.info(f"Downloading {filename} (attempt {attempt + 1})")

            req = urllib.request.Request(url, headers={"User-Agent": "TASNI/1.0"})
            with urllib.request.urlopen(req, timeout=300) as response:
                with open(fits_file, "wb") as f:
                    f.write(response.read())

            # Convert to parquet with only columns we need
            with fits.open(fits_file) as hdul:
                data = hdul[1].data

                # Extract columns (handle missing gracefully)
                df_data = {}
                for col in LEGACY_COLUMNS:
                    if col.upper() in data.names:
                        df_data[col] = data[col.upper()]
                    elif col in data.names:
                        df_data[col] = data[col]

                # Always need RA/Dec
                if "ra" not in df_data and "RA" in data.names:
                    df_data["ra"] = data["RA"]
                if "dec" not in df_data and "DEC" in data.names:
                    df_data["dec"] = data["DEC"]

                # Convert fluxes to magnitudes for easier use
                for band in ["g", "r", "z"]:
                    flux_col = f"flux_{band}"
                    if flux_col.upper() in data.names:
                        flux = data[flux_col.upper()]
                        # nanomaggies to AB magnitude: m = 22.5 - 2.5*log10(flux)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            mag = 22.5 - 2.5 * np.log10(np.maximum(flux, 1e-10))
                            mag[flux <= 0] = 99.0  # Non-detection
                        df_data[f"mag_{band}"] = mag.astype(np.float32)
                        df_data[flux_col] = flux

                df = pd.DataFrame(df_data)

            # Save as parquet
            df.to_parquet(parquet_file, index=False)

            # Remove FITS file to save space
            fits_file.unlink()

            return filename, len(df), "ok"

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if fits_file.exists():
                fits_file.unlink()
            time.sleep(2**attempt)  # Exponential backoff

    return filename, 0, f"failed after {MAX_RETRIES} attempts"


def load_checkpoint():
    """Load download progress checkpoint"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "total_sources": 0}


def save_checkpoint(state):
    """Save download progress checkpoint"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2)


def download_all(region="north", workers=4, dry_run=False):
    """Download all sweep files for a region"""

    sweep_files = get_sweep_file_list(region)
    if not sweep_files:
        logger.error("No sweep files found")
        return

    # Load checkpoint
    state = load_checkpoint()
    completed = set(state.get("completed", []))

    # Filter to remaining files
    remaining = [f for f in sweep_files if f not in completed]

    logger.info(f"Total sweep files: {len(sweep_files)}")
    logger.info(f"Already completed: {len(completed)}")
    logger.info(f"Remaining: {len(remaining)}")

    if dry_run:
        logger.info("Dry run - would download:")
        for f in remaining[:10]:
            info = parse_sweep_filename(f)
            if info:
                logger.info(
                    f"  {f}: RA {info['ra_min']}-{info['ra_max']}, Dec {info['dec_min']}-{info['dec_max']}"
                )
        if len(remaining) > 10:
            logger.info(f"  ... and {len(remaining) - 10} more")
        return

    # Download with thread pool
    total_sources = state.get("total_sources", 0)
    failed = list(state.get("failed", []))

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_sweep_file, f, region): f for f in remaining}

        for i, future in enumerate(as_completed(futures)):
            filename, n_sources, status = future.result()

            if status == "ok":
                completed.add(filename)
                total_sources += n_sources
                logger.info(f"[{i + 1}/{len(remaining)}] {filename}: {n_sources:,} sources")
            elif status == "skipped":
                completed.add(filename)
            else:
                failed.append(filename)
                logger.error(f"[{i + 1}/{len(remaining)}] {filename}: {status}")

            # Checkpoint every 10 files
            if (i + 1) % 10 == 0:
                state = {
                    "completed": list(completed),
                    "failed": failed,
                    "total_sources": total_sources,
                    "last_update": datetime.now().isoformat(),
                }
                save_checkpoint(state)

    # Final checkpoint
    elapsed = time.time() - start_time
    state = {
        "completed": list(completed),
        "failed": failed,
        "total_sources": total_sources,
        "elapsed_seconds": elapsed,
        "last_update": datetime.now().isoformat(),
    }
    save_checkpoint(state)

    logger.info("=" * 60)
    logger.info("Download complete")
    logger.info(f"Total sources: {total_sources:,}")
    logger.info(f"Completed files: {len(completed)}")
    logger.info(f"Failed files: {len(failed)}")
    logger.info(f"Time: {elapsed:.0f}s ({elapsed / 3600:.1f} hours)")
    logger.info("=" * 60)


def merge_to_healpix(region="north", nside=32):
    """
    Reorganize downloaded sweep files into HEALPix tiles.
    This makes cross-matching faster by matching spatial organization with WISE data.
    """
    import healpy as hp

    output_dir = LEGACY_DIR / region
    healpix_dir = LEGACY_DIR / f"{region}_healpix"
    healpix_dir.mkdir(parents=True, exist_ok=True)

    npix = hp.nside2npix(nside)

    logger.info(f"Reorganizing {region} sweep files into {npix} HEALPix tiles (NSIDE={nside})")

    # Process each sweep file
    sweep_files = list(output_dir.glob("sweep-*.parquet"))

    # Accumulate sources per HEALPix tile
    tile_data = {i: [] for i in range(npix)}

    for sweep_file in sweep_files:
        logger.info(f"Processing {sweep_file.name}")

        df = pd.read_parquet(sweep_file)

        if len(df) == 0:
            continue

        # Compute HEALPix indices
        pixels = hp.ang2pix(nside, df["ra"].values, df["dec"].values, nest=True, lonlat=True)

        df["healpix"] = pixels

        # Group by pixel
        for pix, group in df.groupby("healpix"):
            tile_data[pix].append(group.drop(columns=["healpix"]))

    # Save each tile
    total_sources = 0
    for pix in range(npix):
        if tile_data[pix]:
            tile_df = pd.concat(tile_data[pix], ignore_index=True)
            tile_file = healpix_dir / f"legacy_hp{pix:05d}.parquet"
            tile_df.to_parquet(tile_file, index=False)
            total_sources += len(tile_df)

            if (pix + 1) % 500 == 0:
                logger.info(f"Saved {pix + 1}/{npix} tiles, {total_sources:,} sources so far")

    logger.info(f"Reorganization complete: {total_sources:,} sources in {npix} tiles")


def main():
    parser = argparse.ArgumentParser(description="Download Legacy Survey DR10 sweep catalogs")
    parser.add_argument(
        "--region",
        choices=["north", "south", "both"],
        default="north",
        help="Sky region to download (default: north for BASS coverage)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available sweep files without downloading",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument(
        "--merge-healpix",
        action="store_true",
        help="Reorganize downloaded files into HEALPix tiles",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Legacy Survey DR10 Downloader")
    logger.info("=" * 60)
    logger.info("Includes BASS (g,r) + MzLS (z) - Deep Optical for Northern Sky")
    logger.info("Depth: g=24.2, r=23.6, z=23.0 mag (5σ)")
    logger.info("=" * 60)

    if args.list:
        for region in ["north", "south"]:
            files = get_sweep_file_list(region)
            logger.info(f"{region}: {len(files)} sweep files")
            for f in files[:5]:
                info = parse_sweep_filename(f)
                if info:
                    logger.info(
                        f"  {f}: RA {info['ra_min']}-{info['ra_max']}, Dec {info['dec_min']}-{info['dec_max']}"
                    )
            if len(files) > 5:
                logger.info(f"  ... and {len(files) - 5} more")
        return

    if args.merge_healpix:
        merge_to_healpix(args.region)
        return

    if args.region == "both":
        for region in ["north", "south"]:
            logger.info(f"Downloading {region} region...")
            download_all(region, args.workers, args.dry_run)
    else:
        download_all(args.region, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
