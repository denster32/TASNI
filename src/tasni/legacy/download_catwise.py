import os
from pathlib import Path

import requests

# Configuration
# CatWISE2020 Main Catalog (Point Sources)
# URL: https://catwise.github.io/
# Data hosted at IPAC.
# Structure: 72 files or similar split by RA/DEC?
# Actually typically it's hosted at NERSC or IPAC in splits.
# Let's use the explicit raw FITS tables or Parquet if available (unlikely).
# We will download the "CatWISE2020_main.parquet" if it exists, otherwise the FITS splits.
# The official archive is IRSA.
# URL Pattern: https://irsa.ipac.caltech.edu/data/WISE/CatWISE/2020/catwise2020.html

OUTPUT_DIR = str(Path(__file__).resolve().parents[3] / "data" / "catwise2020")
BASE_URL = "https://irsa.ipac.caltech.edu/data/WISE/CatWISE/2020/CatWISE2020_main.TBL"
# Wait, the main table is split.
# Let's use a simpler approach: The "reject" list is best checked against the UnWISE catalog.
# Actually, for "Motion" check, we just need the motion columns from CatWISE.

# Let's try downloading the index or using wget for the folder if possible.
# Since we can't easily crawl, we'll write a script that targets the specific files.
# For now, let's look for "UnWISE" co-adds which are easier? No, CatWISE is better for proper motions.

# Reverting to explicit file list strategy.
# Common IRSA structure: /catwise_2020_main_N.fits
# Let's start by downloading the 500MB test region to verify speed.


def download_file(url, dest):
    if os.path.exists(dest):
        return "skipped"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return "success"
    except Exception as e:
        return f"failed: {e}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Prepping CatWISE2020 Download...")

    # URL list is massive.
    # Alternative: Use "wget -r" via subprocess?
    # This is "pushing it harder" - utilizing the connection.
    cmd = f"wget -nc -r -l1 -nd -P {OUTPUT_DIR} -A 'CatWISE2020_main_*.fits' https://irsa.ipac.caltech.edu/data/WISE/CatWISE/2020/"

    print(f"Executing: {cmd}")
    # We will let the user approve running this as a command object rather than python internal
    # because it's more robust.


if __name__ == "__main__":
    main()
