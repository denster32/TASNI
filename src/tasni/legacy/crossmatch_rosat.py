#!/usr/bin/env python3
"""
Cross-match TASNI anomalies with ROSAT X-ray sources.

Uses pre-computed AllWISE x ROSAT cross-match to identify which
anomalies have X-ray detections.

Anomalies WITH X-ray detections = likely natural (black holes, accretion)
Anomalies WITHOUT X-ray = more interesting (pure thermal sources)
"""

import os
from pathlib import Path

import duckdb
import pandas as pd

# Configuration
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANOMALIES_FILE = _PROJECT_ROOT / "data" / "processed" / "anomalies_filtered.parquet"
ROSAT_IDS_FILE = _PROJECT_ROOT / "data" / "rosat" / "processed" / "rosat_wise_ids.txt"
OUTPUT_DIR = _PROJECT_ROOT / "output"

def load_rosat_ids() -> set:
    """Load set of AllWISE designations with ROSAT counterparts."""
    if not ROSAT_IDS_FILE.exists():
        raise FileNotFoundError(f"ROSAT IDs file not found: {ROSAT_IDS_FILE}")

    print(f"Loading ROSAT X-ray source IDs from {ROSAT_IDS_FILE}")

    rosat_ids = set()
    with open(ROSAT_IDS_FILE, 'r') as f:
        for line in f:
            design = line.strip()
            if design:
                rosat_ids.add(design)

    print(f"Loaded {len(rosat_ids):,} ROSAT X-ray sources")
    return rosat_ids

def crossmatch_with_rosat():
    """Cross-match anomalies with ROSAT using DuckDB."""

    # Load ROSAT IDs into DuckDB
    print("Creating ROSAT lookup table in DuckDB")
    con = duckdb.connect(':memory:')

    rosat_ids = load_rosat_ids()

    # Create a temporary table with ROSAT IDs
    con.execute("CREATE TEMP TABLE rosat_sources (designation VARCHAR)")

    # Batch insert for speed
    batch_size = 100000
    rosat_list = list(rosat_ids)

    for i in range(0, len(rosat_list), batch_size):
        batch = rosat_list[i:i+batch_size]
        con.execute("INSERT INTO rosat_sources VALUES " +
                    ",".join([f"('{id}')" for id in batch]))
        print(f"\rLoaded {min(i+batch_size, len(rosat_list)):,} / {len(rosat_list):,} ROSAT IDs", end='')

    print()

    # Load anomalies and add ROSAT flag
    print(f"\nLoading anomalies from {ANOMALIES_FILE}")

    result = con.execute(f""
        SELECT
            a.*,
            CASE WHEN r.designation IS NOT NULL THEN 1 ELSE 0 END as has_rosat
        FROM read_parquet('{ANOMALIES_FILE}') a
        LEFT JOIN rosat_sources r ON a.designation = r.designation
    """).df()

    # Count matches
    n_with_rosat = result['has_rosat'].sum()
    n_without_rosat = len(result) - n_with_rosat

    print(f"\n" + "="*60)
    print(f"ROSAT Cross-Match Results:")
    print(f"="*60)
    print(f"Total anomalies: {len(result):,}")
    print(f"With ROSAT X-ray: {n_with_rosat:,} ({n_with_rosat/len(result)*100:.2f}%)")
    print(f"Without ROSAT X-ray: {n_without_rosat:,} ({n_without_rosat/len(result)*100:.2f}%)")
    print(f"="*60)

    # Save results
    output_path = OUTPUT_DIR / "anomalies_with_rosat.parquet"
    print(f"\nSaving to {output_path}")

    result.to_parquet(output_path, compression='snappy', index=False)
    print(f"Saved: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Export X-ray quiet candidates (most interesting)
    xray_quiet = result[result['has_rosat'] == 0].copy()

    xray_quiet_path = OUTPUT_DIR / "anomalies_xray_quet.parquet"
    print(f"\nSaving X-ray quiet anomalies to {xray_quiet_path}")
    xray_quiet.to_parquet(xray_quiet_path, compression='snappy', index=False)

    # Re-rank within X-ray quiet sources
    print("\nRe-calculating anomaly scores for X-ray quiet sources")

    for col in ['w1_w2_color', 'w3mpro', 'w1mpro']:
        if col in xray_quiet.columns:
            mean = xray_quiet[col].mean()
            std = xray_quiet[col].std()
            xray_quiet[f'{col}_zscore'] = (xray_quiet[col] - mean) / std

    # Stealth multiplier: X-ray quiet = 2x score
    if 'anomaly_score' in xray_quiet.columns:
        base_score = xray_quiet['anomaly_score'].values
        # Use zscore components if available
        if 'w1_w2_color_zscore' in xray_quiet.columns:
            xray_quiet['stealth_score'] = (
                xray_quiet['w1_w2_color_zscore'] * 3.0 * 2.0 +  # Red color, X-ray quiet
                xray_quiet.get('w3_zscore', pd.Series(0, index=xray_quiet.index)) * 1.5 +
                xray_quiet.get('w1_zscore', pd.Series(0, index=xray_quiet.index)) * 1.0
            )
        else:
            xray_quiet['stealth_score'] = base_score * 2.0

    # Sort by stealth score
    xray_quiet = xray_quiet.sort_values('stealth_score', ascending=False)

    # Save top ranked
    top_path = OUTPUT_DIR / "xray_quiet_top100k.csv"
    print(f"\nSaving top 100k X-ray quiet anomalies to {top_path}")
    xray_quiet.head(100000).to_csv(top_path, index=False)

    print(f"\n" + "="*60)
    print(f"Top 10 X-Ray Quiet Anomalies:")
    print(f"="*60)
    for i, row in xray_quiet.head(10).iterrows():
        print(f"{row.name+1}. {row['designation']:20s} | W1-W2: {row.get('w1_w2_color', 0):.2f} | Score: {row.get('stealth_score', 0):.2f}")

    return result, xray_quiet

def main():
    print("="*60)
    print("TASNI Phase 2: ROSAT Cross-Match")
    print("="*60)

    result, xray_quiet = crossmatch_with_rosat()

    print("\n" + "="*60)
    print("Phase 2 (ROSAT) Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
