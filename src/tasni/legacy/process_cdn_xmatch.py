"""Process CDN crossmatch - fixed version using set membership"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CDN_DIR = _PROJECT_ROOT / "data" / "cdn_xmatch"
WISE_DIR = _PROJECT_ROOT / "data" / "wise"
OUTPUT_DIR = _PROJECT_ROOT / "output"
MATCHED_IDS_FILE = OUTPUT_DIR / "matched_wise_ids.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_matched_ids_set():
    """Load matched IDs as a Python set for O(1) lookup"""
    logger.info(f"Loading matched IDs from {MATCHED_IDS_FILE}...")
    matched_ids = set()
    with open(MATCHED_IDS_FILE) as f:
        for i, line in enumerate(f):
            matched_ids.add(line.strip())
            if (i + 1) % 50_000_000 == 0:
                logger.info(f"  Loaded {i+1:,} IDs...")
    logger.info(f"Total loaded: {len(matched_ids):,} IDs")
    return matched_ids


def is_orphan_mask(wise_ids, matched_set):
    """Check which wise_ids are NOT in matched_set (vectorized approach)"""
    # Use pandas isin which is more memory efficient
    return ~pd.Series(wise_ids).isin(matched_set).values


def find_orphans_chunked(matched_set):
    """Find orphans processing tiles in chunks"""
    logger.info("Finding orphans via anti-join...")
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

        batch_orphans = []
        for wf in batch_files:
            wise_df = pd.read_parquet(wf)
            total_wise += len(wise_df)

            if "designation" not in wise_df.columns:
                continue

            wise_ids = wise_df["designation"].astype(str).values
            is_orphan = is_orphan_mask(wise_ids, matched_set)

            tile_orphans = wise_df[is_orphan].copy()
            if len(tile_orphans) > 0:
                tile_orphans["nearest_gaia_sep_arcsec"] = np.inf
                batch_orphans.append(tile_orphans)
                total_orphans += len(tile_orphans)

        if batch_orphans:
            batch_df = pd.concat(batch_orphans, ignore_index=True)
            chunk_path = OUTPUT_DIR / f"orphans_chunk_{batch_num:03d}.parquet"
            batch_df.to_parquet(chunk_path, compression="snappy", index=False)
            orphan_chunks.append(chunk_path)
            logger.info(f"  Saved {sum(len(o) for o in batch_orphans):,} orphans")

        logger.info(
            f"  Progress: {total_wise:,} WISE, {total_orphans:,} orphans ({100*total_orphans/total_wise:.1f}%)"
        )

    return orphan_chunks


def combine_orphans(chunk_files):
    """Combine all chunks"""
    logger.info(f"Combining {len(chunk_files)} chunks...")
    dfs = [pd.read_parquet(f) for f in chunk_files]
    combined = pd.concat(dfs, ignore_index=True)

    output_path = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    combined.to_parquet(output_path, compression="snappy", index=False)
    logger.info(f"Saved to {output_path}")
    logger.info(f"Shape: {combined.shape}")

    for f in chunk_files:
        f.unlink()
    logger.info("Cleaned up chunk files")
    return combined


def main():
    logger.info("=" * 60)
    logger.info("TASNI: Process CDN Crossmatch (Fixed)")
    logger.info("=" * 60)

    matched_set = load_matched_ids_set()
    chunk_files = find_orphans_chunked(matched_set)
    combined = combine_orphans(chunk_files)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
