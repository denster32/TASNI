import gzip
import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii

# CDS URL for NVSS (The definitive source)
# format: table
URL = "http://cdsarc.u-strasbg.fr/ftp/VIII/65/nvss.dat.gz"
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
RAW_FILE = _PROJECT_ROOT + "/data/nvss.dat.gz"
PARQUET_FILE = _PROJECT_ROOT + "/data/nvss.parquet"


def download_nvss():
    if os.path.exists(PARQUET_FILE):
        print("NVSS Parquet exists. Skipping download.")
        return pd.read_parquet(PARQUET_FILE)

    print(f"Downloading NVSS bulk from {URL}...")
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)

    # Download
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        with open(RAW_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Parsing fixed-width text file...")
    # NVSS Format from CDS ReadMe:
    # Bytes 1-12: RA (h m s)
    # Bytes 14-25: DEC (d m s)
    # ... flux ...
    # Actually astropy.io.ascii.read can handle 'cds' format or we can just parse the dat.
    # We will try astropy first.

    try:
        # Unzip first
        dat_file = RAW_FILE.replace(".gz", "")
        with gzip.open(RAW_FILE, "rb") as f_in:
            with open(dat_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Read with astropy
        # Use simpler whitespace reader if CDS format fails
        # NVSS is usually fixed width.
        # Columns: RA (0-12), DEC (13-25), Flux(26-32)...
        # Let's rely on astropy guessing or use generic fixed width.

        tab = ascii.read(
            dat_file,
            format="fixed_width_no_header",
            col_starts=(0, 14, 27),
            col_ends=(12, 26, 33),
            names=("ra_s", "dec_s", "flux_mjy"),
        )

        # We need to convert hms/dms to deg.
        coords = SkyCoord(tab["ra_s"], tab["dec_s"], unit=(u.hourangle, u.deg))

        df = pd.DataFrame({"ra": coords.ra.deg, "dec": coords.dec.deg, "flux_mjy": tab["flux_mjy"]})

        print(f"Parsed {len(df)} sources.")
        df.to_parquet(PARQUET_FILE)
        return df

    except Exception as e:
        print(f"Parsing failed: {e}")
        # Fallback to pure pandas fixed width if needed?
        return None


def main():
    df = download_nvss()
    if df is not None:
        print("NVSS Database Ready.")


if __name__ == "__main__":
    main()
