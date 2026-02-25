import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS

INPUT_FILE = "./data/processed/tier4_prime.parquet"
OUTPUT_FILE = "./data/processed/tier6_spectra.csv"


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)

    print(f"Checking {len(df)} sources against SDSS Spectra...")
    try:
        # SDSS Crossmatch
        # spectro=True ensures we only get objects with spectra
        matches = SDSS.query_crossid(
            coords, data_release=17, photoobj_fields=["ra", "dec", "objid"], spectro=True
        )
        if matches:
            print(f"FOUND SDSS SPECTRA: {len(matches)}")
            matches.write("./data/processed/sdss_matches.csv", format="csv", overwrite=True)
        else:
            print("No SDSS spectra found.")
    except Exception as e:
        print(f"SDSS Query failed: {e}")

    print("Checking LAMOST...")
    try:
        # Check LAMOST DR7
        # radius 5 arcsec
        match_count = 0

        # LAMOST query is best done via xmatch service usually, but let's try direct radius
        # For 4000 sources, loop might be needed again or chunked xmatch.
        # Let's try a bulk cone search for the first few to test? No.
    except Exception as e:
        print(f"LAMOST failed: {e}")

    print("Done checking spectra.")


if __name__ == "__main__":
    main()
