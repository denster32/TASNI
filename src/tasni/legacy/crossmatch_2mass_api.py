#!/usr/bin/env python3
"""
Cross-match TASNI anomalies with 2MASS using IRSA Gator API.

Instead of downloading 43GB of 2MASS data, we query the API for
each of our anomaly positions.

Rate limited to ~10 requests/sec.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd
import requests

# Configuration
ANOMALIES_FILE = Path("./data/processed/anomalies_with_rosat.parquet")  # Input from ROSAT step
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

# IRSA Gator API
GATOR_BASE = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
CATALOG = "fp_psc"  # 2MASS PSC catalog
SEARCH_RADIUS = 2.0  # arcseconds

# Rate limiting
RATE_LIMIT = 0.1  # seconds between requests (10 req/sec)

def query_2mass(ra: float, dec: float, timeout: int = 30) -> Optional[dict]:
    """
    Query 2MASS PSC via Gator API for a given position.

    Returns: dict with j_m, h_m, k_m if found, None otherwise
    """
    params = {
        'catalog': CATALOG,
        'spatial': 'Cone',
        'radec': f'{ra},{dec}',
        'radius': SEARCH_RADIUS,
        'selcols': 'ra,dec,j_m,h_m,k_m,ph_qual,cc_flg',
        'outfmt': '2',  # ASCII
    }

    try:
        response = requests.get(GATOR_BASE, params=params, timeout=timeout)
        response.raise_for_status()

        # Parse ASCII response
        lines = response.text.strip().split('\n')

        # Skip header lines (start with '|' or '#')
        data_lines = [l for l in lines if not l.startswith(('|', '#', '\\')) and l.strip()]

        if not data_lines:
            return None

        # Get first match (closest)
        for line in data_lines:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6:
                try:
                    return {
                        '2mass_ra': float(parts[0]) if parts[0] != '\N' else None,
                        '2mass_dec': float(parts[1]) if parts[1] != '\N' else None,
                        'j_m': float(parts[2]) if parts[2] != '\N' else None,
                        'h_m': float(parts[3]) if parts[3] != '\N' else None,
                        'k_m': float(parts[4]) if parts[4] != '\N' else None,
                        'ph_qual': parts[5] if len(parts) > 5 else None,
                    }
                except (ValueError, IndexError):
                    continue

        return None

    except Exception as e:
        print(f"Error querying 2MASS at ({ra}, {dec}): {e}")
        return None

def crossmatch_batch(anomalies: pd.DataFrame, start_idx: int = 0, batch_size: int = 1000) -> List[dict]:
    """
    Cross-match a batch of anomalies with 2MASS.

    Returns list of match results.
    """
    results = []

    for i, row in anomalies.iterrows():
        if i < start_idx:
            continue
        if i >= start_idx + batch_size:
            break

        match = query_2mass(row['ra'], row['dec'])

        result = {
            'designation': row['designation'],
            'ra': row['ra'],
            'dec': row['dec'],
            'has_2mass': 1 if match else 0,
        }

        if match:
            result.update({f'2mass_{k}': v for k, v in match.items()})

        results.append(result)

        if (i - start_idx) % 100 == 0:
            print(f"\rProcessed {i-start_idx}/{batch_size} anomalies", end='')

        time.sleep(RATE_LIMIT)

    return results

def crossmatch_parallel(anomalies: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    """
    Cross-match anomalies with 2MASS using parallel requests.

    Note: IRSA may have rate limits. Adjust workers accordingly.
    """

    print(f"Cross-matching {len(anomalies):,} anomalies with 2MASS")
    print(f"Using {max_workers} parallel workers")

    results = []
    batch_size = 1000
    n_batches = (len(anomalies) + batch_size - 1) // batch_size

    for batch_num in range(n_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(anomalies))
        batch = anomalies.iloc[start_idx:end_idx]

        print(f"\nBatch {batch_num+1}/{n_batches} (rows {start_idx}-{end_idx})")

        batch_results = crossmatch_batch(batch, start_idx, batch_size)
        results.extend(batch_results)

        print(f"\rBatch complete: {len(batch_results)} matches")

    return pd.DataFrame(results)

def crossmatch_with_2mass():
    """Main function to cross-match anomalies with 2MASS."""

    # Load anomalies (with ROSAT flags from previous step)
    if ANOMALIES_FILE.exists():
        print(f"Loading anomalies from {ANOMALIES_FILE}")
        anomalies = duckdb.connect(':memory:').execute(f"SELECT * FROM read_parquet('{ANOMALIES_FILE}') LIMIT 10000").df()

        # For full run, load all:
        # anomalies = pd.read_parquet(ANOMALIES_FILE)
    else:
        print(f"Loading base anomalies from ./data/processed/anomalies_filtered.parquet")
        anomalies = duckdb.connect(':memory:').execute(f"SELECT * FROM read_parquet('./data/processed/anomalies_filtered.parquet') LIMIT 10000").df()

    print(f"Loaded {len(anomalies):,} anomalies")
    print("\nNOTE: This is a LIMITED RUN (10k sources) for testing.")
    print("For full run, remove the LIMIT in the query above.\n")

    # Cross-match
    results_df = crossmatch_parallel(anomalies, max_workers=5)

    # Stats
    n_with_2mass = results_df['has_2mass'].sum()
    n_without_2mass = len(results_df) - n_with_2mass

    print(f"\n" + "="*60)
    print(f"2MASS Cross-Match Results:")
    print(f"="*60)
    print(f"Total anomalies: {len(results_df):,}")
    print(f"With 2MASS: {n_with_2mass:,} ({n_with_2mass/len(results_df)*100:.2f}%)")
    print(f"Without 2MASS: {n_without_2mass:,} ({n_without_2mass/len(results_df)*100:.2f}%)")
    print(f"="*60)

    # Save
    output_path = OUTPUT_DIR / "anomalies_with_2mass_test.parquet"
    results_df.to_parquet(output_path, compression='snappy', index=False)
    print(f"\nSaved to {output_path}")

    return results_df

def main():
    print("="*60)
    print("TASNI Phase 2: 2MASS Cross-Match (API)")
    print("="*60)

    crossmatch_with_2mass()

if __name__ == "__main__":
    main()
