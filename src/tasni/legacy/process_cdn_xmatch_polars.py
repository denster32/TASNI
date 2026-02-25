"""Streaming crossmatch - process CDN files in chunks to avoid OOM"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CDN_DIR = _PROJECT_ROOT / "data" / "cdn_xmatch"
WISE_DIR = _PROJECT_ROOT / "data" / "wise"
OUTPUT_DIR = _PROJECT_ROOT / "output"
SEEN_WISE_FILE = OUTPUT_DIR / "matched_wise_designations.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_matched_designations():
    """Build a file of all WISE designations that have Gaia matches"""
    logger.info("Building matched WISE designations from CDN files...")
    xmatch_files = sorted(CDN_DIR.glob("allwiseBestNeighbour*.csv.gz"))
    logger.info(f"Found {len(xmatch_files)} CDN files")

    matched_designations = []
    total_rows = 0

    for i, f in enumerate(xmatch_files, 1):
        # Read just the designation column
        df = pl.read_csv(str(f), columns=["original_ext_source_id"])
        matched_designations.append(df)
        total_rows += len(df)

        if i % 5 == 0 or i == len(xmatch_files):
            logger.info(f"  [{i}/{len(xmatch_files)}] Loaded {total_rows:,} rows")

    # Concat and save unique designations
    logger.info("Concatenating and deduplicating...")
    all_matched = pl.concat(matched_designations).unique()

    logger.info(f"Saving {len(all_matched):,} unique matched designations...")
    all_matched.write_parquet(SEEN_WISE_FILE, compression="snappy")

    return len(all_matched)


def find_orphans_streaming():
    """Find orphans by checking against the matched designations file"""
    logger.info("Finding orphans (streaming mode)...")

    wise_files = sorted(WISE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(wise_files)} WISE tiles")

    BATCH_SIZE = 500
    orphan_chunks = []
    total_wise = 0
    total_orphans = 0

    # Load matched designations once as a set for fast lookup
    # This is more memory efficient than the approach above
    logger.info("Loading matched designations for lookup...")
    matched_df = pl.read_parquet(str(SEEN_WISE_FILE))
    matched_set = set(matched_df["original_ext_source_id"].to_list())
    logger.info(f"Loaded {len(matched_set):,} matched designations into memory")

    for batch_start in range(0, len(wise_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(wise_files))
        batch_files = wise_files[batch_start:batch_end]

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(wise_files) - 1) // BATCH_SIZE + 1

        logger.info(f"Batch {batch_num}/{total_batches} (tiles {batch_start+1}-{batch_end})...")

        batch_dfs = []
        for wf in batch_files:
            df = pl.read_parquet(str(wf))
            batch_dfs.append(df)

        batch_combined = pl.concat(batch_dfs)
        total_wise += len(batch_combined)

        # Filter orphans using Polars isin
        orphans = batch_combined.filter(~pl.col("designation").is_in(matched_set))

        orphan_count = len(orphans)
        total_orphans += orphan_count

        if orphan_count > 0:
            orphans = orphans.with_columns(nearest_gaia_sep_arcsec=np.float64(np.inf))
            chunk_path = OUTPUT_DIR / f"orphans_chunk_{batch_num:03d}.parquet"
            orphans.write_parquet(chunk_path, compression="snappy")
            orphan_chunks.append(chunk_path)
            logger.info(f"  Saved {orphan_count:,} orphans")

        orphan_rate = 100 * total_orphans / total_wise if total_wise > 0 else 0
        logger.info(
            f"  Progress: {total_wise:,} WISE, {total_orphans:,} orphans ({orphan_rate:.1f}%)"
        )

    return orphan_chunks


def combine_orphans(chunk_files):
    logger.info(f"Combining {len(chunk_files)} chunks...")
    dfs = [pl.read_parquet(str(f)) for f in chunk_files]
    combined = pl.concat(dfs)

    output_path = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    combined.write_parquet(output_path, compression="snappy")

    logger.info(f"Saved to {output_path}")
    logger.info(f"Shape: {combined.shape}")

    for f in chunk_files:
        f.unlink()
    logger.info("Cleaned up chunks")

    return combined


def main():
    logger.info("=" * 60)
    logger.info("TASNI: Streaming Crossmatch (Polars)")
    logger.info("=" * 60)

    # First build the matched designations file
    n_matched = build_matched_designations()
    logger.info(f"Built matched designations file with {n_matched:,} entries")

    # Then find orphans
    chunk_files = find_orphans_streaming()
    combined = combine_orphans(chunk_files)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
