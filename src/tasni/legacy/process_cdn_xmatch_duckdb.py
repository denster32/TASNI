"""DuckDB crossmatch - FIXED CSV parsing"""

import logging
from pathlib import Path

import duckdb
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CDN_DIR = _PROJECT_ROOT / "data" / "cdn_xmatch"
WISE_DIR = _PROJECT_ROOT / "data" / "wise"
OUTPUT_DIR = _PROJECT_ROOT / "output"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_orphans_duckdb():
    logger.info("=" * 60)
    logger.info("TASNI: DuckDB Crossmatch (Fixed)")
    logger.info("=" * 60)

    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA temp_directory='{_PROJECT_ROOT / 'tmp'}'")

    con.execute("PRAGMA memory_limit='20GB'")

    # Create table from CDN files - proper CSV parsing with quote handling
    logger.info("Creating matched WISE IDs table from CDN files...")

    con.execute(f"""
        CREATE TABLE matched_ids AS
        SELECT DISTINCT trim(original_ext_source_id, '"') as designation
        FROM read_csv('{CDN_DIR}/*.csv.gz', header=true, quote='"', delim=',')
    """)

    n_matched = con.sql("SELECT COUNT(*) as n FROM matched_ids").fetchone()[0]
    logger.info(f"Loaded {n_matched:,} unique matched WISE IDs")

    wise_files = sorted(WISE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(wise_files)} WISE tiles")

    BATCH_SIZE = 500
    orphan_chunks = []
    total_wise = 0
    total_orphans = 0

    for batch_start in range(0, len(wise_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(wise_files))
        batch_files = wise_files[batch_start:batch_end]

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(wise_files) - 1) // BATCH_SIZE + 1

        logger.info(f"Batch {batch_num}/{total_batches} (tiles {batch_start+1}-{batch_end})...")

        file_list = ",".join(f"'{str(wf)}'" for wf in batch_files)

        # Anti-join
        result = con.execute(f"""
            SELECT w.*, CAST(NULL AS DOUBLE) as nearest_gaia_sep_arcsec
            FROM read_parquet([{file_list}]) w
            WHERE w.designation NOT IN (SELECT designation FROM matched_ids)
        """).df()

        orphan_count = len(result)
        total_orphans += orphan_count

        total_batch = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet([{file_list}])
        """).fetchone()[0]
        total_wise += total_batch

        if orphan_count > 0:
            result["nearest_gaia_sep_arcsec"] = np.inf
            chunk_path = OUTPUT_DIR / f"orphans_chunk_{batch_num:03d}.parquet"
            result.to_parquet(chunk_path, compression="snappy", index=False)
            orphan_chunks.append(chunk_path)
            logger.info(f"  Saved {orphan_count:,} orphans")

        orphan_rate = 100 * total_orphans / total_wise if total_wise > 0 else 0
        logger.info(
            f"  Progress: {total_wise:,} WISE, {total_orphans:,} orphans ({orphan_rate:.1f}%)"
        )

    # Combine
    logger.info(f"Combining {len(orphan_chunks)} chunks...")
    chunk_list = ",".join(f"'{str(f)}'" for f in orphan_chunks)

    final_result = con.execute(f"""
        SELECT * FROM read_parquet([{chunk_list}])
    """).df()

    output_path = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    final_result.to_parquet(output_path, compression="snappy", index=False)

    logger.info(f"Saved to {output_path}")
    logger.info(f"Shape: {final_result.shape}")

    for f in orphan_chunks:
        f.unlink()
    logger.info("Cleaned up chunks")

    con.close()

    return final_result


def main():
    result = find_orphans_duckdb()
    logger.info("=" * 60)
    logger.info("DuckDB processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
