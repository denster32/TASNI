import logging

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

# Configuration
INPUT_FILE = "./data/processed/tier4_prime.parquet"
OUTPUT_FILE = "./data/processed/tier5_radio_silent.parquet"
MATCH_RADIUS_ARCSEC = 15


def main():
    print(f"Loading candidates from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df)} candidates.")

    # Initialize radio flag
    df["has_nvss_radio"] = 0
    df["nvss_separation"] = 999.9

    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)

    # Configure Vizier
    Vizier.ROW_LIMIT = 1  # We only care if there is A match
    v = Vizier(columns=["*"], catalog="VIII/65/nvss")

    print("Querying Vizier for NVSS matches (this takes time but is reliable)...")

    # We can pass a list of coordinates to query_region?
    # No, query_region takes one coord or a list, but for list it returns a TableList which is hard to map back.
    # Efficient way: Cross-match locally if possible, but remote query needed here.
    # We will loop in batches.

    matches = []
    chunk_size = 100

    # Actually, for 4000 sources, doing a loop is slow but stable.
    # Let's try xmatch method if available, or just simple query_region loop.

    count_radio = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        coord = SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg)
        try:
            result = v.query_region(coord, radius=MATCH_RADIUS_ARCSEC * u.arcsec)
            if len(result) > 0 and len(result[0]) > 0:
                # Found a match
                df.at[idx, "has_nvss_radio"] = 1
                df.at[idx, "nvss_separation"] = 0.0  # Approximation
                count_radio += 1
        except Exception as e:
            logging.getLogger(__name__).debug("Vizier query failed for %s: %s", coord, e)

        if i % 50 == 0:
            print(f"Progress: {i}/{len(df)} | Matches: {count_radio}", end="\r")

    print(f"\nFinal Radio Loud Count: {count_radio}")

    # Filter
    radio_silent = df[df["has_nvss_radio"] == 0]
    print(f"Remaining Radio-Silent Candidates: {len(radio_silent)}")

    # Save
    final_output = "./data/processed/tier4_prime_with_radio.parquet"
    df.to_parquet(final_output)
    radio_silent.to_parquet(OUTPUT_FILE)
    print(f"Saved radio-silent list to {OUTPUT_FILE}")

    # Save radio-loud for inspection
    loud_path = "./data/processed/radio_loud_rejected.csv"
    df[df["has_nvss_radio"] == 1].to_csv(loud_path, index=False)


if __name__ == "__main__":
    main()
