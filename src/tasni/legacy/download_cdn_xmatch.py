"""
Download WISE×Gaia crossmatch from Gaia CDN
Much faster than TAP queries - 8GB vs 5+ days of querying
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "http://cdn.gea.esac.esa.int/Gaia/gedr3/cross_match/allwise_best_neighbour/"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = _PROJECT_ROOT / "data" / "cdn_xmatch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(_LOG_DIR / "cdn_xmatch_download.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def download_file(file_num):
    """Download one crossmatch file"""
    filename = f"allwiseBestNeighbour{file_num:04d}.csv.gz"
    url = f"{BASE_URL}{filename}"
    output_path = OUTPUT_DIR / filename

    if output_path.exists():
        size = output_path.stat().st_size
        logger.info(f"File {file_num:04d}: already exists ({size:,} bytes)")
        return file_num, "exists", size

    try:
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        size = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                size += len(chunk)

        logger.info(f"File {file_num:04d}: downloaded ({size:,} bytes)")
        return file_num, "success", size
    except Exception as e:
        logger.error(f"File {file_num:04d}: error - {e}")
        return file_num, "error", str(e)


def main():
    logger.info("=" * 60)
    logger.info("TASNI: CDN Cross-Match Download")
    logger.info("=" * 60)
    logger.info(f"URL: {BASE_URL}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("Files: 33")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_file, i): i for i in range(1, 34)}

        completed = 0
        total_size = 0
        errors = []

        for future in as_completed(futures):
            file_num, status, detail = future.result()
            completed += 1

            if status == "success":
                total_size += detail
            elif status == "error":
                errors.append((file_num, detail))

            logger.info(f"[{completed}/33] File {file_num:04d}: {status}")

    logger.info("=" * 60)
    logger.info(f"Download complete: {completed}/33 files")
    logger.info(f"Total size: {total_size / (1024**3):.2f} GB")
    if errors:
        logger.error(f"Errors: {len(errors)} files")
        for num, err in errors:
            logger.error(f"  File {num:04d}: {err}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
