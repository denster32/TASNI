"""
TASNI: Filter and Rank Anomalies
=================================

Takes WISE sources with no Gaia match and filters/ranks by "weirdness."

Filters:
- Quality flags (remove artifacts)
- Thermal profile (4-band flux ratios)
- Isolation (distance from any known objects)

Ranking:
- Weirdness score based on how much the thermal profile deviates from known object classes

Usage:
    python filter_anomalies.py [--test]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def load_orphans(test=False):
    """Load WISE sources with no Gaia match"""
    suffix = "_test" if test else "_full"
    path = OUTPUT_DIR / f"wise_no_gaia_match{suffix}.parquet"
    print(f"Loading orphans from {path}...")
    return pd.read_parquet(path)


def filter_quality(df):
    """Remove sources with quality issues"""

    original_count = len(df)

    # Remove sources with contamination flags
    if "cc_flags" in df.columns:
        # cc_flags should be '0000' for clean sources
        df = df[df["cc_flags"] == "0000"]

    # Remove extended sources (we want point sources)
    if "ext_flg" in df.columns:
        df = df[df["ext_flg"] == 0]

    # Require good photometry in at least W1 and W2
    if "ph_qual" in df.columns:
        # A or B quality in first two bands
        df = df[df["ph_qual"].str[0].isin(["A", "B"])]
        df = df[df["ph_qual"].str[1].isin(["A", "B"])]

    print(f"Quality filter: {original_count} -> {len(df)} sources")
    return df


def compute_thermal_profile(df):
    """
    Compute thermal profile metrics from 4-band photometry.

    WISE bands:
    - W1: 3.4 μm
    - W2: 4.6 μm
    - W3: 12 μm
    - W4: 22 μm

    Color indices tell us about temperature and composition.
    """

    df = df.copy()

    # Compute color indices (magnitude differences)
    if all(col in df.columns for col in ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]):
        df["w1_w2"] = df["w1mpro"] - df["w2mpro"]
        df["w2_w3"] = df["w2mpro"] - df["w3mpro"]
        df["w3_w4"] = df["w3mpro"] - df["w4mpro"]
        df["w1_w4"] = df["w1mpro"] - df["w4mpro"]  # Overall "redness"

    return df


def compute_weirdness_score(df):
    """
    Compute a "weirdness" score for each source.

    Higher score = more anomalous thermal profile.

    Known natural objects have predictable color ranges:
    - Stars: W1-W2 ≈ 0
    - Brown dwarfs: W1-W2 > 0.5
    - Galaxies: varies but predictable
    - AGN: very red

    We flag sources that don't fit these profiles.
    """

    df = df.copy()
    df["weirdness"] = 0.0

    if "w1_w2" in df.columns:
        # Anomalously cold in W1-W2 (shouldn't happen naturally)
        df.loc[df["w1_w2"] < -0.5, "weirdness"] += 1.0

        # Very isolated (far from nearest Gaia source)
        if "nearest_gaia_sep_arcsec" in df.columns:
            df["weirdness"] += np.log10(df["nearest_gaia_sep_arcsec"]) / 2

        # Detected in long wavelengths but faint in short (warm but not hot)
        if "w3mpro" in df.columns and "w1mpro" in df.columns:
            warm_not_hot = (df["w3mpro"] < 12) & (df["w1mpro"] > 15)
            df.loc[warm_not_hot, "weirdness"] += 2.0

    return df


def rank_anomalies(df):
    """Sort by weirdness score, highest first"""
    return df.sort_values("weirdness", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Filter and rank anomalies")
    parser.add_argument("--test", action="store_true", help="Use test region only")
    args = parser.parse_args()

    # Load orphan sources
    df = load_orphans(test=args.test)

    # Apply filters
    df = filter_quality(df)

    # Compute thermal profile
    df = compute_thermal_profile(df)

    # Compute weirdness score
    df = compute_weirdness_score(df)

    # Rank
    df = rank_anomalies(df)

    # Save
    suffix = "_test" if args.test else "_full"
    output_file = OUTPUT_DIR / f"anomalies_ranked{suffix}.parquet"
    df.to_parquet(output_file)

    # Also save top candidates as CSV for easy viewing
    top_file = OUTPUT_DIR / f"top_anomalies{suffix}.csv"
    df.head(1000).to_csv(top_file, index=False)

    print(f"Saved {len(df)} filtered sources to {output_file}")
    print(f"Top 1000 saved to {top_file}")

    # Summary stats
    print("\n=== SUMMARY ===")
    print(f"Total anomalies: {len(df)}")
    print(f"Weirdness score range: {df['weirdness'].min():.2f} - {df['weirdness'].max():.2f}")
    print(f"Top 10 weirdness scores: {df['weirdness'].head(10).tolist()}")


if __name__ == "__main__":
    main()
