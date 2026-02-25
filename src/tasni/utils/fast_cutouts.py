import concurrent.futures
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from tasni.core.config_env import DATA_ROOT, N_WORKERS

    _processed = DATA_ROOT / "data" / "processed"
    INPUT_FILE = _processed / "tier4_prime.csv"
    CUTOUT_DIR = _processed / "cutouts"
    MAX_WORKERS = N_WORKERS
except ImportError:
    _root = Path(__file__).resolve().parents[3]
    _processed = _root / "data" / "processed"
    INPUT_FILE = _processed / "tier4_prime.csv"
    CUTOUT_DIR = _processed / "cutouts"
    MAX_WORKERS = 16

TIMEOUT = 10
logger = logging.getLogger(__name__)


def download_image(row):
    # Unpack row
    if isinstance(row, pd.Series):
        ra, dec, designation = row["ra"], row["dec"], row["designation"]
    else:
        # If passed as tuple/dict
        ra, dec, designation = row

    fname_base = designation.replace(" ", "_").replace("+", "p").replace("-", "m")
    fname = CUTOUT_DIR / f"dr9_{fname_base}.jpg"

    # Skip if exists and valid size (>1KB)
    if os.path.exists(fname) and os.path.getsize(fname) > 1000:
        return "skipped"

    url = "https://www.legacysurvey.org/viewer/jpeg-cutout"
    params = {
        "ra": ra,
        "dec": dec,
        "width": 150,  # Increased slightly
        "height": 150,
        "layer": "ls-dr9",
        "pixscale": 0.262,
    }

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            # Check for "blank" images (often small filesize ~800-900 bytes or specific signature)
            # We save everything for now, filter later.
            with open(fname, "wb") as f:
                f.write(resp.content)
            return "success"
        else:
            return "failed_status"
    except (requests.RequestException, OSError) as e:
        logger.debug("Download failed for %s: %s", designation, e)
        return "failed_error"


def main():
    input_path = str(INPUT_FILE)
    print(f"Loading {input_path}...")
    # Supports both parquet and csv
    if str(INPUT_FILE).endswith(".parquet"):
        df = pd.read_parquet(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE)

    cutout_dir = str(CUTOUT_DIR)
    os.makedirs(cutout_dir, exist_ok=True)

    print(f"Starting parallel download for {len(df)} sources with {MAX_WORKERS} workers...")

    # Convert to list of dicts for easier iteration
    rows = [row for _, row in df.iterrows()]

    count_success = 0
    count_skipped = 0
    count_failed = 0

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map returns in order
        results = executor.map(download_image, rows)

        for i, res in enumerate(results):
            if res == "success":
                count_success += 1
            elif res == "skipped":
                count_skipped += 1
            else:
                count_failed += 1

            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(
                    f"Progress: {i}/{len(df)} | Rate: {rate:.1f} img/s | Success: {count_success} | Skipped: {count_skipped}",
                    end="\r",
                )

    print(f"\nDone. Success: {count_success}, Skipped: {count_skipped}, Failed: {count_failed}")


if __name__ == "__main__":
    main()
