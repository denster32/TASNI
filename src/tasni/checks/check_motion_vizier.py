import logging
import time

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

INPUT_FILE = "./data/processed/tier5_radio_silent.parquet"
OUTPUT_FILE = "./data/processed/tier5_motions.csv"


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    # Check Top 500 sorted by something? Or just all? 4000 is doable.
    # Vizier CatWISE catalog: II/365/catwise

    print("Querying CatWISE2020 via Vizier for Proper Motions...")

    Vizier.ROW_LIMIT = 1
    v = Vizier(columns=["*"], catalog="II/365/catwise")

    results = []

    # We loop or batch.
    start_time = time.time()

    for i, (idx, row) in enumerate(df.iterrows()):
        coord = SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg)
        try:
            # 6 arcsec radius (WISE matching)
            res = v.query_region(coord, radius=6 * u.arcsec)

            if len(res) > 0 and len(res[0]) > 0:
                match = res[0][0]  # First match
                # PMRA, PMDEC in mas/yr
                pmra = match["PMRA"]
                pmdec = match["PMDE"]

                # If motion > 100 mas/yr -> High Motion
                total_pm = (pmra**2 + pmdec**2) ** 0.5

                results.append(
                    {
                        "designation": row["designation"],
                        "catwise_id": match["Name"],
                        "pm_total": total_pm,
                        "pm_ra": pmra,
                        "pm_dec": pmdec,
                    }
                )
        except Exception as e:
            logging.getLogger(__name__).debug("Vizier query failed for row %s: %s", i, e)

        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            print(
                f"Checked {i}/{len(df)} | Rate: {rate:.1f} obj/s | Matches: {len(results)}",
                end="\r",
            )

    print("\nDone.")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)

    # Identify Fast Movers
    fast = res_df[res_df["pm_total"] > 100]
    print(f"Found {len(fast)} High Motion Objects (Likely Y-Dwarfs).")
    print(f"Found {len(res_df) - len(fast)} Stationary Objects (Ghosts).")


if __name__ == "__main__":
    main()
