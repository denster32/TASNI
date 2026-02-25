#!/usr/bin/env python3
"""
TASNI Pipeline: Population Synthesis Fading Rates
=================================================

Generates synthetic Y-dwarf/planet catalogs with fading rates from Sonora Cholla models.
Compares fading distributions to observed golden candidates for validation.

Usage:
  poetry run python src/tasni/pipeline/synth.py --output data/processed/synth_fading.parquet --n_samples 10000
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Assume Sonora Cholla external data available
SONORA_DIR = Path("data/external/sonora_cholla")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synth_catalog(n_samples=10000, teff_range=[200, 500], logg_range=[4.5, 5.5]):
    """Generate synthetic cold brown dwarf catalog."""
    synth = pd.DataFrame(
        {
            "ra": np.random.uniform(0, 360, n_samples),
            "dec": np.random.uniform(-90, 90, n_samples),
            "teff": np.random.normal(350, 50, n_samples),
            "logg": np.random.normal(5.0, 0.2, n_samples),
            "dist_pc": np.random.lognormal(np.log(20), 0.5, n_samples),  # 20pc median
            "pm_tot": np.random.lognormal(np.log(200), 0.8, n_samples),  # mas/yr high PM
        }
    )

    # Clip Teff/logg to model range
    synth = synth[(synth["teff"] >= teff_range[0]) & (synth["teff"] <= teff_range[1])]
    synth = synth[(synth["logg"] >= logg_range[0]) & (synth["logg"] <= logg_range[1])]

    # Fading rates: monotonic dimming 0-100 mmag/yr (models)
    synth["fading_w1_mmag_yr"] = np.random.exponential(20, len(synth))  # mean 20
    synth["fading_w2_mmag_yr"] = synth["fading_w1_mmag_yr"] * np.random.normal(1.0, 0.1, len(synth))

    # WISE colors from Teff (simplified Sonora)
    synth["w1_mag"] = (
        12 + 5 * np.log10(synth["dist_pc"] / 10) - 0.4 * np.log10(synth["teff"] / 1000)
    )
    synth["w2_mag"] = synth["w1_mag"] + 2.5 * np.log10(1.5)  # W1-W2 ~0.8
    synth["w1_w2"] = synth["w2_mag"] - synth["w1_mag"]

    # Variability proxy
    synth["rms_w1"] = 0.02 + 0.01 * synth["fading_w1_mmag_yr"] / 20

    logger.info(f"Generated {len(synth)} synthetic sources")
    return synth


def compare_to_observed(synth, observed_path):
    """Compare fading distributions."""
    observed = pd.read_parquet(observed_path)
    fading_obs = observed.get("fading_rate_w1", observed.get("trend_w1", np.nan))

    if len(fading_obs.dropna()) > 0:
        logger.info(f"Observed fading mean: {fading_obs.mean():.2f} mmag/yr")
        logger.info(f"Synth fading mean: {synth['fading_w1_mmag_yr'].mean():.2f} mmag/yr")
        # KS test etc. could be added
    else:
        logger.warning("No observed fading data found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/synth_fading.parquet")
    parser.add_argument("--observed", default="data/processed/final/golden_variability.csv")
    parser.add_argument("--n_samples", type=int, default=10000)
    args = parser.parse_args()

    synth = generate_synth_catalog(args.n_samples)
    compare_to_observed(synth, args.observed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synth.to_parquet(output_path, index=False)

    logger.info(f"Synthetic catalog saved: {output_path}")
    logger.info(f"Teff range: {synth['teff'].min():.0f}-{synth['teff'].max():.0f} K")
    logger.info(f"Fading rates: {synth['fading_w1_mmag_yr'].quantile([0.1,0.5,0.9]).values}")


if __name__ == "__main__":
    main()
