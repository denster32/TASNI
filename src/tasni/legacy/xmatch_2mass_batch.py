#!/usr/bin/env python3
"""
Batch cross-match of TASNI anomalies with 2MASS using astroquery XMatch.

Processes in batches of 10,000 to avoid timeouts and rate limits.
"""

import time
from pathlib import Path

import duckdb
import pandas as pd
from astropy import units as u
from astropy.table import Table
from astroquery.xmatch import XMatch

# Configuration
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"
ANOMALIES_FILE = OUTPUT_DIR / "anomalies_xray_quiet.parquet"
BATCH_SIZE = 10000
MAX_ANOMALIES = 100000  # Process top 100k first

print("=" * 60)
print("TASNI Phase 2: 2MASS Batch Cross-Match")
print("=" * 60)

con = duckdb.connect(":memory:")

# Get total count
total_count = con.execute(f'SELECT COUNT(*) FROM read_parquet("{ANOMALIES_FILE}")').fetchone()[0]
print(f"Total X-ray quiet anomalies: {total_count:,}")

# Get top N by anomaly score
print(f"Loading top {MAX_ANOMALIES} anomalies...")
anomalies = con.execute(f"""
    SELECT * FROM read_parquet("{ANOMALIES_FILE}")
    ORDER BY anomaly_score DESC
    LIMIT {MAX_ANOMALIES}
""").df()

print(f"Loaded {len(anomalies)} anomalies")

# Prepare results storage
all_results = []
all_designations = []
has_2mass = set()

# Process in batches
n_batches = (len(anomalies) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\nProcessing {n_batches} batches of {BATCH_SIZE}...")

for batch_num in range(n_batches):
    start_idx = batch_num * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, len(anomalies))
    batch = anomalies.iloc[start_idx:end_idx]

    print(f"\nBatch {batch_num+1}/{n_batches} (rows {start_idx}-{end_idx})")

    # Convert to astropy Table
    astropy_table = Table.from_pandas(batch[["designation", "ra", "dec"]])

    try:
        result = XMatch.query(
            cat1=astropy_table,
            cat2="vizier:II/246/out",
            max_distance=2 * u.arcsec,
            colRA1="ra",
            colDec1="dec",
        )

        # Convert to pandas
        result_df = result.to_pandas()
        print(f"  Matched: {len(result_df)}/{len(batch)} sources")

        # Track which designations have 2MASS matches
        if len(result_df) > 0:
            matched_designations = set(result_df["designation"].tolist())
            has_2mass.update(matched_designations)

        all_results.append(result_df)

        # Save intermediate progress
        current_has_2mass = pd.DataFrame({"designation": list(has_2mass), "has_2mass": 1})
        progress_path = OUTPUT_DIR / f"2mass_progress_batch{batch_num+1}.parquet"
        current_has_2mass.to_parquet(progress_path, index=False)
        print(f"  Saved progress: {len(has_2mass):,} matches so far")

    except Exception as e:
        print(f"  Error: {e}")
        # Continue anyway

    # Rate limiting sleep
    time.sleep(2)

# Combine all results
print("\n" + "=" * 60)
print("Final Results:")
print("=" * 60)
print(f"Total processed: {len(anomalies):,}")
print(f"Total with 2MASS: {len(has_2mass):,} ({len(has_2mass)/len(anomalies)*100:.1f}%)")
print(
    f"Total without 2MASS: {len(anomalies) - len(has_2mass):,} ({(len(anomalies)-len(has_2mass))/len(anomalies)*100:.1f}%)"
)

# Save final match list
final_matches = pd.DataFrame({"designation": list(has_2mass), "has_2mass": 1})
final_path = OUTPUT_DIR / "2mass_matches.parquet"
final_matches.to_parquet(final_path, index=False)
print(f"Saved: {final_path}")

print("\n" + "=" * 60)
print("2MASS cross-match complete!")
print("=" * 60)

# Show some examples of sources WITHOUT 2MASS (most interesting)
all_matched = set(final_matches["designation"].tolist())
anomalies["has_2mass"] = anomalies["designation"].isin(all_matched).astype(int)

no_2mass = anomalies[anomalies["has_2mass"] == 0].head(10)
print("\nTop 10 anomalies WITHOUT 2MASS (very interesting):")
for i, row in no_2mass.iterrows():
    print(f'  {row["designation"]:20s} | W1-W2: {row["w1_w2_color"]:.2f}')
