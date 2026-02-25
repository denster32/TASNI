import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

INPUT_FILE = "./data/processed/tier4_prime.parquet"
OUTPUT_FILE = "./data/processed/spitzer_matches.csv"


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)

    print(f"Querying Spitzer Enhanced Imaging Products (SEIP) for {len(df)} sources...")

    # Irsa.query_region
    # Catalog: 'seip_source_cat' (Spitzer Source List)

    matches = []

    # IRSA allows table upload for xmatch usually.
    # But astroquery wrap usually prefers cone search loop or massive list?
    # Let's try region query loop for robust handling (we have time)

    count = 0
    Irsa.ROW_LIMIT = 1

    for i, (idx, row) in enumerate(df.iterrows()):
        coord = SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg)
        try:
            # Search 2 arcsec (Spitzer is sharp)
            # Catalog list: https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd
            # "spitzer_seip_source_cat"
            res = Irsa.query_region(coord, catalog="spitzer_seip_source_cat", radius=2 * u.arcsec)

            if len(res) > 0:
                count += 1
                row_match = res[0]
                matches.append(
                    {
                        "designation": row["designation"],
                        "spitzer_id": row_match.get("designation", "unknown"),
                        "i1_flux": row_match.get("i1_f_ap1", -99),  # 3.6um
                        "i2_flux": row_match.get("i2_f_ap1", -99),  # 4.5um
                    }
                )
        except Exception:
            # Often objects aren't in Spitzer footprint
            pass

        if i % 50 == 0:
            print(f"Checked {i}/{len(df)} | Matches: {count}", end="\r")

    print(f"\nFinal Spitzer Matches: {count}")

    if len(matches) > 0:
        res_df = pd.DataFrame(matches)
        res_df.to_csv(OUTPUT_FILE, index=False)
        print("Saved Spitzer matches.")


if __name__ == "__main__":
    main()
