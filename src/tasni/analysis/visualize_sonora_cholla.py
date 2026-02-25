from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = _PROJECT_ROOT / "data" / "external" / "sonora_cholla" / "spectra_files"
TARGETS_FILE = _PROJECT_ROOT / "data" / "processed" / "final" / "golden_improved.csv"
OUTPUT_DIR = _PROJECT_ROOT / "reports" / "figures" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_spectrum(model_file):
    """Load Sonora Cholla spectrum. First column is wavelength in microns (per header)."""
    try:
        data = np.loadtxt(model_file, skiprows=2)
        wl_micron = data[:, 0]
        flux = data[:, 1]
        return wl_micron, flux
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
        return None, None


def visualize_comparison():
    print("Loading Sonora Cholla models...")
    df = pd.read_csv(TARGETS_FILE)
    fading = df[df["variability_flag"] == "FADING"].head(4)

    if not MODELS_DIR.exists():
        print(f"ERROR: {MODELS_DIR} not found")
        return

    # Find model files - avoid glob issues by listing directory
    import os

    all_files = os.listdir(MODELS_DIR)
    model_files = [MODELS_DIR / f for f in all_files if f.endswith(".spec")]
    print(f"Found {len(model_files)} model spectra files.")

    plt.figure(figsize=(14, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(fading)))

    W1_min, W1_max = 3.0, 4.0
    W2_min, W2_max = 4.0, 5.5

    for idx, (target_idx, row) in enumerate(fading.iterrows()):
        teff_target = row["T_eff_K"]

        target_model_file = None
        best_teff_model = None
        best_diff_score = 9999

        for mf in model_files:
            basename = mf.stem
            try:
                parts = basename.split("_")
                teff_str = parts[0]
                logg_str = parts[1]
                teff_model = float(teff_str[:-1])
                logg_model = float(logg_str[:-1])
                score = abs(teff_model - teff_target) * 10 + abs(logg_model - 5.0)
                if score < best_diff_score:
                    best_diff_score = score
                    target_model_file = mf
                    best_teff_model = teff_model
            except (ValueError, IndexError):
                continue

        if target_model_file and best_teff_model is not None:
            wl, flux = load_model_spectrum(target_model_file)
            if wl is not None:
                flux_norm = flux / np.nanmax(flux)
                flux_shifted = flux_norm * (idx + 1)
                teff_label = int(best_teff_model)
                label = f"{row['designation'][:18]} (Model: {teff_label}K)"
                plt.plot(wl, flux_shifted, color=colors[idx], label=label, lw=2)
        else:
            print(f"No matching model found for {row['designation']}")

    plt.axvspan(W1_min, W1_max, color="red", alpha=0.1, label="W1 (3.4 µm)")
    plt.text(3.5, 4.5, "W1", color="red", fontsize=12, ha="center")
    plt.axvspan(W2_min, W2_max, color="blue", alpha=0.1, label="W2 (4.6 µm)")
    plt.text(4.75, 4.5, "W2", color="blue", fontsize=12, ha="center")

    plt.xlabel("Wavelength (µm)", fontsize=14)
    plt.ylabel("Normalized Flux (shifted)", fontsize=14)
    plt.title("Sonora Cholla (2021) Models vs. Fading Thermal Orphans", fontsize=16, weight="bold")
    plt.xlim(0.8, 10.0)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper right", fontsize=10, framealpha=0.9)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    output_file = OUTPUT_DIR / "sonora_cholla_fading_orphans.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    visualize_comparison()
