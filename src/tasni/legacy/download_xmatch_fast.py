import logging
from pathlib import Path

import pandas as pd
from astroquery.gaia import Gaia

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "xmatch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_full_xmatch():
    """Download the complete crossmatch table"""

    # Download in chunks to avoid timeout
    chunk_size = 100_000_000  # 100M rows per chunk
    offset = 0
    all_dfs = []

    while True:
        query = """
        SELECT source_id, original_ext_source_id, angular_distance
        FROM gaiaedr3.allwise_best_neighbour
        ORDER BY source_id
        """

        logger.info("Downloading crossmatch table...")
        job = Gaia.launch_job(query, dump_to_file=False)
        results = job.get_results()

        df = results.to_pandas()

        if len(df) == 0:
            break

        all_dfs.append(df)
        logger.info(f"Downloaded {len(df):,} rows")

        # If we got fewer rows than expected, we're done
        if len(df) < chunk_size:
            break

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        # Save
        output_file = OUTPUT_DIR / "gaia_wise_xmatch_full.parquet"
        combined.to_parquet(output_file)
        logger.info(f"Saved {len(combined):,} rows to {output_file}")

        # Get stats
        unique_gaia = combined["source_id"].nunique()
        unique_wise = combined["original_ext_source_id"].nunique()
        logger.info(f"Unique Gaia sources: {unique_gaia:,}")
        logger.info(f"Unique WISE source IDs: {unique_wise:,}")

        return combined

    return None


if __name__ == "__main__":
    download_full_xmatch()
