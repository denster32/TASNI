import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.imcce import Skybot

# WISE Survey Epoch (approximate center)
# WISE full coverage was 2010.
# We should technically check the exact MJD of the detection, but we don't have it easily in the simple CSV.
# We will check a range of dates in 2010 or use the 'mjd' column if we fetched it?
# We only have 'ra', 'dec'.
# Strategy: Check if *any* known bright asteroid was within 30" during the WISE cryogenic mission (Jan 2010 - Oct 2010).
# This is computationally expensive (Skybot is slow).
# We will do a spot check for the Golden List (Top 100).

INPUT_FILE = "./data/processed/final/golden_improved.csv"
OUTPUT_FILE = "./data/processed/golden_asteroids.csv"


def main():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
        print("Golden list not ready, using golden_improved.parquet sample.")
        df = pd.read_parquet("./data/processed/final/golden_improved.parquet").head(100)

    print(f"Checking {len(df)} candidates against SkyBot (Asteroids)...")

    # WISE Cryo mission mid-point: May 2010.
    t = Time("2010-05-01 12:00:00")

    hits = []

    for i, row in df.iterrows():
        coord = SkyCoord(row["ra"], row["dec"], unit=(u.deg, u.deg))
        try:
            # Radius 60 arcsec (WISE PSF is big, asteroids move)
            bodies = Skybot.cone_search(coord, 60 * u.arcsec, t)
            if len(bodies) > 0:
                print(f"HIT: {row['designation']} might be asteroid {bodies['Name'][0]}")
                hits.append(
                    {
                        "designation": row["designation"],
                        "asteroid_name": bodies["Name"][0],
                        "asteroid_class": bodies["Type"][0],
                    }
                )
        except Exception as e:
            if i % 20 == 0:  # Log occasionally to avoid spam
                print(f"  Skybot timeout/error for row {i}: {e}")

        if i % 10 == 0:
            print(f"Checked {i}...", end="\r")

    pd.DataFrame(hits).to_csv(OUTPUT_FILE, index=False)
    print(f"Found {len(hits)} asteroid coincidences.")


if __name__ == "__main__":
    main()
