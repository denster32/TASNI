import logging
import time
from pathlib import Path

import pandas as pd
from astroquery.gaia import Gaia

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "xmatch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_xmatch_by_ra_chunks():
    """Download crossmatch in RA chunks to get more data"""

    all_dfs = []

    # Divide sky into RA strips
    for ra_min in range(0, 360, 30):
        ra_max = ra_min + 30

        query = f"""
        SELECT
            source_id,
            original_ext_source_id,
            angular_distance,
            ra as gaia_ra,
            dec as gaia_dec
        FROM gaiaedr3.allwise_best_neighbour
        WHERE ra >= {ra_min} AND ra < {ra_max}
        """

        logger.info(f"Downloading RA {ra_min}-{ra_max}...")

        try:
            job = Gaia.launch_job(query, dump_to_file=False)
            results = job.get_results()
            df = results.to_pandas()

            if len(df) > 0:
                all_dfs.append(df)
                logger.info(f"  Got {len(df):,} rows")

            time.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"  Error: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)

        output_file = OUTPUT_DIR / "gaia_wise_xmatch_sample.parquet"
        combined.to_parquet(output_file)
        logger.info(f"Saved {len(combined):,} rows to {output_file}")

        unique_wise = combined["original_ext_source_id"].nunique()
        logger.info(f"Unique WISE source IDs: {unique_wise:,}")

        return combined

    return None


if __name__ == "__main__":
    download_xmatch_by_ra_chunks()
