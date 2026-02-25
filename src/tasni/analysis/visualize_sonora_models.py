#!/usr/bin/env python3
import glob
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = _PROJECT_ROOT / "data" / "external" / "sonora2018" / "spectra"
TARGETS_FILE = _PROJECT_ROOT / "data" / "processed" / "final" / "golden_improved.csv"
OUTPUT_DIR = _PROJECT_ROOT / "reports" / "figures" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_spectrum(model_file):
    try:
        with gzip.open(model_file, "rt") as f:
            data = np.loadtxt(f)
        wl_cm = data[:, 0]
        flux = data[:, 1]
        wl_micron = wl_cm * 1e4
        return wl_micron, flux
    except Exception:
        return None, None


def visualize_comparison():
    print("Loading Sonora 2018 models...")
    df = pd.read_csv(TARGETS_FILE)
    fading = df[df["variability_flag"] == "FADING"].head(4)

    if not MODELS_DIR.exists():
        print(f"ERROR: {MODELS_DIR} not found")
        return

    model_files = list(glob.glob(str(MODELS_DIR / "sp_t0*.gz")))
    print(f"Found {len(model_files)} model spectra files (T_eff < 1000K)")

    plt.figure(figsize=(14, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(fading)))

    W1_min, W1_max = 3.0, 4.0
    W2_min, W2_max = 4.0, 5.5

    for idx, (target_idx, row) in enumerate(fading.iterrows()):
        teff = row["T_eff_K"]
        target_model_file = None

        for mf in model_files:
            basename = Path(mf).stem
            # sp_t0250g... -> 250
            try:
                temp_str = basename.split("_")[1]  # t0250g...
                model_teff = float(temp_str[1:5])  # 0250 -> 250.0
                if abs(model_teff - teff) < 20:
                    target_model_file = mf
                    break
            except (ValueError, IndexError):
                continue

        if target_model_file:
            wl, flux = load_model_spectrum(target_model_file)
            if wl is not None:
                # Shift flux for plotting clarity
                flux_shifted = flux / np.nanmax(flux) * (idx + 1)
                label = f"{row['designation'][:18]} (Sonora Model: {teff:.0f}K)"
                plt.plot(wl, flux_shifted, color=colors[idx], label=label, lw=2)
        else:
            print(f"No close model for {row['designation']} (Teff={teff})")

    plt.axvspan(W1_min, W1_max, color="red", alpha=0.1)
    plt.text(3.5, 4.5, "W1", color="red", fontsize=12, ha="center")
    plt.axvspan(W2_min, W2_max, color="blue", alpha=0.1)
    plt.text(4.75, 4.5, "W2", color="blue", fontsize=12, ha="center")

    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Normalized Flux (shifted)", fontsize=14)
    plt.title("Sonora 2018 Model Spectra: Fading Thermal Orphans", fontsize=16, weight="bold")
    plt.xlim(0.8, 10.0)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    output_file = OUTPUT_DIR / "sonora_models_fading_orphans.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    visualize_comparison()
