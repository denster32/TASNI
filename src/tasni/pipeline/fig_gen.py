#!/usr/bin/env python3
"""
Generate Phase 3 figures for paper.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ml_scores(input_path, output_path):
    df = pd.read_parquet(input_path)
    plt.figure(figsize=(8, 6))
    plt.hist(df["improved_composite_score"], bins=50, density=True, alpha=0.7, edgecolor="black")
    plt.xlabel("Improved Composite ML Score")
    plt.ylabel("Density")
    plt.title("Distribution of Improved ML Scores (Tier 5, n=4137)")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ML scores plot: {output_path}")


def plot_pop_synth(synth_path, obs_path, output_path):
    synth = pd.read_parquet(synth_path)
    obs = pd.read_csv(obs_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Fading rates
    obs_fading = obs.get("trend_w1", np.zeros(len(obs)))
    ax1.hist(obs_fading, bins=30, alpha=0.5, label="Observed Golden (n=100)", density=True)
    ax1.hist(
        synth["fading_w1_mmag_yr"],
        bins=30,
        alpha=0.5,
        label="Synth Population (n=20k)",
        density=True,
    )
    ax1.set_xlabel("Fading Rate (mmag/yr)")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.set_title("Fading Rate Comparison")

    # Teff
    ax2.hist(synth["teff"], bins=30, alpha=0.7, label="Synth Teff", density=True)
    ax2.set_xlabel("Teff (K)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.set_title("Synthetic Population Teff")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved synth plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_input", default="data/processed/ml/ranked_tier5_improved.parquet")
    parser.add_argument("--synth_path", default="data/processed/synth_fading_new.parquet")
    parser.add_argument("--golden_path", default="data/processed/final/golden_improved.csv")
    parser.add_argument("--figures_dir", default="reports/figures")
    args = parser.parse_args()

    figs_dir = Path(args.figures_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_ml_scores(args.ml_input, figs_dir / "ml_improved_scores.png")
    plot_pop_synth(args.synth_path, args.golden_path, figs_dir / "population_synth_new.png")


if __name__ == "__main__":
    main()
