"""
Download pre-computed WISE x Gaia crossmatch from CDS.

Much faster than downloading Gaia and crossmatching ourselves.
CDS already did the work - just download their results.

Table: wise/allwise_best_neighbour_gaiaedr3
~400M rows, WISE sources matched to Gaia DR3
"""

from pathlib import Path

from astroquery.utils.tap import TapPlus

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "cds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_cds_crossmatch():
    """Download WISE x Gaia crossmatch from CDS"""

    print("Connecting to CDS TAP...")
    cds = TapPlus(url="http://tap.cds.unistra.fr/TAP")

    # First, get total count
    count_query = """
    SELECT COUNT(*)
    FROM wise.allwise_best_neighbour_gaiaedr3
    """

    job = cds.launch_job(count_query)
    total = job.get_results()
    print(f"Total WISE-Gaia matches: {total[0][0]}")

    # Download in chunks (CDS has row limits)
    # We'll download by HEALPix to match our structure

    query = """
    SELECT
        wise_allwise_skymfid, wise_ra, wise_dec,
        wise_designation, wise_w1mpro, wise_w2mpro, wise_w3mpro, wise_w4mpro,
        wise_ph_qual, wise_cc_flags, wise_ext_flg,
        gaiaedr3_source_id, gaiaedr3_ra, gaiaedr3_dec,
        gaiaedr3_phot_g_mean_mag, gaiaedr3_parallax,
        angDist
    FROM wise.allwise_best_neighbour_gaiaedr3
    WHERE angDist > 3.0  -- ONLY non-matches (our orphans!)
    """

    print("This query gets WISE sources with NO Gaia match within 3 arcsec")
    print("These are our orphan candidates - heat without light.")

    # For now, let's get a sample to test
    sample_query = query + " LIMIT 1000000"

    print("Downloading 1M row sample...")
    job = cds.launch_job(sample_query)
    results = job.get_results()

    df = results.to_pandas()
    print(f"Downloaded {len(df)} rows")

    output_file = OUTPUT_DIR / "wise_no_gaia_orphans.parquet"
    df.to_parquet(output_file)
    print(f"Saved to {output_file}")

    return df


if __name__ == "__main__":
    download_cds_crossmatch()
