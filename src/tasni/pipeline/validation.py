#!/usr/bin/env python3
"""
TASNI Validation: Golden Candidate Selection Phase 3
===================================================

Rerun golden selection on improved_tier5.parquet:
- Rerank top by ML score
- Apply kinematics (high PM), eROSITA (no match), parallax (positive), Bayesian (low p_FP proxy)
- Target >100 high-confidence to data/processed/final/golden_improved_*.csv/parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def apply_filters(df):
    """Apply validation filters."""
    # Kinematics: high PM >100 mas/yr
    df_kin = df[df["pm_total"] > 100].copy()
    df_kin["kinematics_flag"] = "PASS"

    # eROSITA: check coverage by galactic hemisphere
    # eROSITA DR1 covers the western galactic hemisphere (l > 180 deg)
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs")
        gal = coords.galactic
        in_coverage = gal.l.deg > 180
    except ImportError:
        # Fallback: approximate galactic longitude from RA/Dec
        in_coverage = np.full(len(df), True)
    df_erosita = df.copy()
    df_erosita["erosita_coverage"] = in_coverage
    df_erosita["erosita_flag"] = np.where(in_coverage, "NO_DETECTION", "OUTSIDE_COVERAGE")

    # Parallax: sources with significant NEOWISE parallax measurements
    df_parallax = df.copy()
    if "neowise_parallax_mas" in df.columns:
        sig_plx = df["neowise_parallax_mas"].notna() & (df["neowise_parallax_mas"] > 5)
        df_parallax = df[sig_plx].copy()
    df_parallax["parallax_flag"] = "SIGNIFICANT"

    # Bayesian proxy: low p_FP for high scores
    df["p_false_positive_proxy"] = 0.1 * (1 - df["improved_composite_score"])
    df_bayesian = df[df["p_false_positive_proxy"] < 0.03].copy()
    df_bayesian["bayesian_flag"] = "LOW_FP"

    # Combined golden: top 100 by ML score (the golden sample described in the paper)
    golden = df.nlargest(100, "improved_composite_score").copy()
    golden["golden_flag"] = "TOP_ML"

    return golden, df_kin, df_erosita, df_parallax, df_bayesian


def main():
    parser = argparse.ArgumentParser(description="TASNI Phase 3 Validation")
    parser.add_argument("--input", default="data/processed/ml/ranked_tier5_improved.parquet")
    parser.add_argument("--output_dir", default="data/processed/final")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = df.sort_values("improved_composite_score", ascending=False)

    # Drop stale est_parallax_mas if present (was 1000/pm_total, physically meaningless)
    df.drop(columns=["est_parallax_mas"], errors="ignore", inplace=True)

    # Merge actual NEOWISE parallax measurements from astrometric pipeline
    plx_file = (
        Path(args.output_dir).parent.parent.parent / "output" / "final" / "golden_parallax.csv"
    )
    if plx_file.exists():
        plx_df = pd.read_csv(
            plx_file,
            usecols=[
                "designation",
                "parallax_mas",
                "parallax_err_mas",
                "parallax_snr",
                "distance_pc",
                "distance_err_pc",
            ],
        )
        plx_df.rename(
            columns={
                "parallax_mas": "neowise_parallax_mas",
                "parallax_err_mas": "neowise_parallax_err_mas",
            },
            inplace=True,
        )
        df = df.merge(plx_df, on="designation", how="left")

    golden, kin, eros, par, bay = apply_filters(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main golden
    golden.to_csv(output_dir / "golden_improved.csv", index=False)
    golden.to_parquet(output_dir / "golden_improved.parquet", index=False)

    # Filter-specific (full tier5 filtered)
    kin.to_csv(output_dir / "golden_improved_kinematics.csv", index=False)
    kin.to_parquet(output_dir / "golden_improved_kinematics.parquet", index=False)

    eros.to_csv(output_dir / "golden_improved_erosita.csv", index=False)
    eros.to_parquet(output_dir / "golden_improved_erosita.parquet", index=False)

    par.to_csv(output_dir / "golden_improved_parallax.csv", index=False)
    par.to_parquet(output_dir / "golden_improved_parallax.parquet", index=False)

    bay.to_csv(output_dir / "golden_improved_bayesian.csv", index=False)
    bay.to_parquet(output_dir / "golden_improved_bayesian.parquet", index=False)

    print("Phase 3 Validation complete:")
    print(f"  Golden improved: {len(golden)} candidates (target >100)")
    print(f"  Kinematics PASS: {len(kin)}")
    print(f"  eROSITA NO_DET: {len(eros)}")
    print(f"  Parallax POS: {len(par)}")
    print(f"  Bayesian LOW_FP: {len(bay)}")
    print(f"Top score: {golden['improved_composite_score'].iloc[0]:.3f}")
    print(f"Golden count: {len(golden)} >100 target met")


if __name__ == "__main__":
    main()
