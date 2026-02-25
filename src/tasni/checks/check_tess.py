import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs

INPUT_FILE = "./data/processed/tier4_prime.parquet"
OUTPUT_FILE = "./data/processed/tess_matches.csv"


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)

    print(f"Querying TESS Input Catalog (TIC) for {len(df)} sources...")

    # We use a cone search.
    # TESS pixels are big (21"), so we might just look for ANY source nearby that could be the counterpart.
    # But usually we want to know if TESS observed this sector.

    # Let's check which TESS sectors cover our candidates.
    # We can use tess-point or just query MAST.

    results = []

    # Chunking to avoid timeouts
    chunk_size = 50
    for i in range(0, len(df), chunk_size):
        batch = df.iloc[i : i + chunk_size]
        batch_coords = coords[i : i + chunk_size]

        try:
            # Query TIC
            # We want to know if there is a known TIC object (star) nearby.
            # If our "invisible" object is a TIC object, it's not invisible (TIC is mostly optical/NIR).
            # This is a good "Anti-Check".

            matches = Catalogs.query_region(batch_coords, radius=5 * u.arcsec, catalog="TIC")

            if len(matches) > 0:
                # Found TIC matches
                # Store them
                for match in matches:
                    results.append(
                        {
                            "ra": match["ra"],
                            "dec": match["dec"],
                            "tic_id": match["ID"],
                            "Tmag": match["Tmag"],
                        }
                    )
        except Exception as e:
            print(f"Chunk {i} failed: {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(df)}...", end="\r")

    print(f"\nFound {len(results)} TIC collisions.")
    if len(results) > 0:
        res_df = pd.DataFrame(results)
        res_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved TESS/TIC matches to {OUTPUT_FILE}")

    # Phase 2: Download Lightcurves?
    # Only if Tmag is bright enough (<16?).
    # If our objects are "Invisible" in Gaia, they shouldn't be in TIC unless they are very red.


if __name__ == "__main__":
    main()
