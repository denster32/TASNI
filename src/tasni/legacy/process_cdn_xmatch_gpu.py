"""GPU crossmatch - incremental concat to avoid OOM"""

import logging
from pathlib import Path

import cudf
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CDN_DIR = _PROJECT_ROOT / "data" / "cdn_xmatch"
WISE_DIR = _PROJECT_ROOT / "data" / "wise"
OUTPUT_DIR = _PROJECT_ROOT / "output"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_matched_ids_from_cdn():
    logger.info("Loading matched IDs from CDN files...")
    xmatch_files = sorted(CDN_DIR.glob("allwiseBestNeighbour*.csv.gz"))
    logger.info(f"Found {len(xmatch_files)} CDN files")

    all_matched_gdf = None

    for i, f in enumerate(xmatch_files, 1):
        gdf = cudf.read_csv(str(f), usecols=["original_ext_source_id"])

        if all_matched_gdf is None:
            all_matched_gdf = gdf
        else:
            all_matched_gdf = cudf.concat([all_matched_gdf, gdf])
            del gdf

        if i % 5 == 0 or i == len(xmatch_files):
            logger.info(
                f"  [{i}/{len(xmatch_files)}] Accumulated {len(all_matched_gdf):,} rows on GPU"
            )

    logger.info(f"Total matched IDs: {len(all_matched_gdf):,}")
    return all_matched_gdf


def find_orphans_gpu(matched_gdf):
    logger.info("Finding orphans via GPU anti-join...")

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
            wise_gdf = cudf.read_parquet(str(wf))
            total_wise += len(wise_gdf)

            if "designation" not in wise_gdf.columns:
                del wise_gdf
                continue

            is_orphan = ~wise_gdf["designation"].isin(matched_gdf)
            orphans_gdf = wise_gdf[is_orphan].copy()
            del wise_gdf

            if len(orphans_gdf) > 0:
                orphans_gdf["nearest_gaia_sep_arcsec"] = np.inf
                orphans_df = orphans_gdf.to_pandas()
                batch_orphans.append(orphans_df)
                del orphans_gdf

        if batch_orphans:
            batch_df = pd.concat(batch_orphans, ignore_index=True)
            chunk_path = OUTPUT_DIR / f"orphans_chunk_{batch_num:03d}.parquet"
            batch_df.to_parquet(chunk_path, compression="snappy", index=False)
            orphan_chunks.append(chunk_path)

            orphan_count = sum(len(o) for o in batch_orphans)
            logger.info(f"  Saved {orphan_count:,} orphans")

        orphan_rate = 100 * total_orphans / total_wise if total_wise > 0 else 0
        logger.info(
            f"  Progress: {total_wise:,} WISE, {total_orphans:,} orphans ({orphan_rate:.1f}%)"
        )

    return orphan_chunks


def combine_orphans(chunk_files):
    logger.info(f"Combining {len(chunk_files)} chunks...")
    dfs = [pd.read_parquet(f) for f in chunk_files]
    combined = pd.concat(dfs, ignore_index=True)

    output_path = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    combined.to_parquet(output_path, compression="snappy", index=False)

    logger.info(f"Saved to {output_path}")
    logger.info(f"Shape: {combined.shape}")

    for f in chunk_files:
        f.unlink()
    logger.info("Cleaned up chunks")

    return combined


def main():
    logger.info("=" * 60)
    logger.info("TASNI: GPU Crossmatch (Incremental)")
    logger.info("=" * 60)

    matched_gdf = get_matched_ids_from_cdn()
    chunk_files = find_orphans_gpu(matched_gdf)
    combined = combine_orphans(chunk_files)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
