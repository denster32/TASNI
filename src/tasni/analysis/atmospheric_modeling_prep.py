#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tasni.core.config_env import DATA_ROOT

    _root = DATA_ROOT
except ImportError:
    _root = Path(__file__).resolve().parents[3]
INPUT_FILE = _root / "data" / "processed" / "final" / "golden_improved.csv"
PARALLAX_FILE = _root / "data" / "processed" / "final" / "golden_improved_parallax.csv"
OUTPUT_DIR = Path("./data/processed/spectroscopy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_INPUT_FILE = OUTPUT_DIR / "atmospheric_modeling_input.txt"


def prepare_model_input():
    print("Preparing atmospheric modeling input...")
    df = pd.read_csv(INPUT_FILE)
    df_parallax = pd.read_csv(PARALLAX_FILE)

    # Merge with correct column names
    df = df.merge(
        df_parallax[
            ["designation", "parallax_mas", "parallax_err_mas", "distance_pc", "distance_err_pc"]
        ],
        on="designation",
        how="left",
    )
    top_targets = df.nsmallest(20, "T_eff_K")

    print("\n=== Atmospheric Modeling Parameters ===")
    print(f"{'Target':<25} {'T_eff (K)':<10} {'log g':<10} {'Fe/H':<10} {'Dist (pc)':<10}")
    print("-" * 70)

    with open(MODEL_INPUT_FILE, "w") as f:
        f.write("# Atmospheric Modeling Input for TASNI Golden Targets\n")
        f.write("# Format: Designation  T_eff(K)  log(g)  [Fe/H]  Distance(pc)  W1-W2(mag)\n")
        f.write("# \n")

        for _, row in top_targets.iterrows():
            designation = row["designation"]
            teff = row["T_eff_K"]
            logg = 5.0
            feh = 0.0
            dist = row["distance_pc"] if pd.notna(row["distance_pc"]) else np.nan
            w1_w2 = row["w1mpro"] - row["w2mpro"]
            print(f"{designation:<25} {teff:<10.1f} {logg:<10.2f} {feh:<10.2f} {dist:<10.1f}")
            f.write(
                f"{designation:<25} {teff:<10.1f} {logg:<10.2f} {feh:<10.2f} {dist:<10.1f} {w1_w2:.2f}\n"
            )

    print(f"\nSaved model input to: {MODEL_INPUT_FILE}")


if __name__ == "__main__":
    prepare_model_input()
