#!/usr/bin/env python3
"""
TASNI: Population Synthesis Comparison

Compares the golden sample properties against theoretical predictions for
brown dwarf populations in the solar neighborhood.

Key comparisons:
1. Temperature distribution vs Y dwarf models (Kirkpatrick et al. 2021)
2. Proper motion distribution vs disk/halo kinematics
3. Sky distribution vs isotropic expectation
4. Number counts vs population synthesis predictions

References:
- Burgasser (2004): Solar neighborhood BD density
- Ryan & Reid (2017): Cold BD population predictions
- Kirkpatrick et al. (2021): Y dwarf census

Usage:
    python population_synthesis.py [--golden FILE] [--output DIR]
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from tasni.analysis.selection_function import calculate_corrected_space_density
except Exception:
    calculate_corrected_space_density = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [POP-SYN] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Physical constants
KPC_TO_PC = 1000.0

# Brown dwarf population parameters (from literature)
# Burgasser (2004) - local BD space density
BD_SPACE_DENSITY = {
    "L_dwarfs": 0.0038,  # per pc^3
    "T_dwarfs": 0.0021,  # per pc^3
    "Y_dwarfs": 0.001,  # per pc^3 (estimated)
}

# Temperature boundaries (Kirkpatrick 2005, 2021)
SPECTRAL_TYPE_TEMPS = {
    "L0": 2200,
    "L5": 1700,
    "T0": 1400,
    "T5": 1100,
    "T8": 700,
    "Y0": 500,
    "Y1": 400,
    "Y2": 300,  # Estimated
}

# Known Y dwarf statistics (Kirkpatrick 2021)
KNOWN_Y_DWARFS = {
    "count": 30,  # Approximate as of 2021
    "mean_distance_pc": 15,
    "max_distance_pc": 30,
    "mean_teff_K": 400,
    "min_teff_K": 250,
}

# Galactic kinematics (Bensby et al. 2003, 2014)
GALACTIC_KINEMATICS = {
    "thin_disk": {
        "sigma_U": 35,  # km/s
        "sigma_V": 20,
        "sigma_W": 16,
        "fraction": 0.85,
    },
    "thick_disk": {
        "sigma_U": 63,
        "sigma_V": 39,
        "sigma_W": 39,
        "fraction": 0.12,
    },
    "halo": {
        "sigma_U": 160,
        "sigma_V": 90,
        "sigma_W": 90,
        "fraction": 0.03,
    },
}


def load_golden_targets(filepath: str) -> pd.DataFrame:
    """Load golden targets with derived properties."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} golden targets")

    # Ensure required columns exist
    required = ["designation", "w1mpro", "w2mpro", "T_eff_K", "pm_total"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"Missing column: {col}")

    return df


def analyze_temperature_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze temperature distribution and compare to Y dwarf expectations.
    """
    temps = df["T_eff_K"].dropna()

    results = {
        "n_sources": len(temps),
        "mean_T": temps.mean(),
        "std_T": temps.std(),
        "min_T": temps.min(),
        "max_T": temps.max(),
        "median_T": temps.median(),
    }

    # Count by temperature bins
    bins = [0, 250, 300, 350, 400, 500, 600, 1000]
    hist, _ = np.histogram(temps, bins=bins)
    results["T_bins"] = list(zip(bins[:-1], bins[1:], hist, strict=False))

    # Fraction below 300K (extremely cold)
    results["frac_below_300K"] = (temps < 300).mean()

    # Compare to known Y dwarf distribution (deterministic reference sample).
    known_y_temps = np.linspace(250, 500, KNOWN_Y_DWARFS["count"])
    ks_stat, ks_pval = stats.ks_2samp(temps, known_y_temps)
    results["ks_vs_uniform"] = {"statistic": ks_stat, "pvalue": ks_pval}

    logger.info(f"Temperature distribution: {results['mean_T']:.0f} ± {results['std_T']:.0f} K")
    logger.info(f"Fraction below 300K: {results['frac_below_300K']*100:.1f}%")

    return results


def analyze_proper_motion_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze proper motion distribution and compare to disk/halo expectations.
    """
    pm = df["pm_total"].dropna()

    results = {
        "n_sources": len(pm),
        "mean_pm": pm.mean(),
        "std_pm": pm.std(),
        "min_pm": pm.min(),
        "max_pm": pm.max(),
        "median_pm": pm.median(),
    }

    # Proper motion bins
    bins = [0, 50, 100, 200, 300, 500, 1000]
    hist, _ = np.histogram(pm, bins=bins)
    results["pm_bins"] = list(zip(bins[:-1], bins[1:], hist, strict=False))

    # High proper motion fraction (likely nearby)
    results["frac_high_pm"] = (pm > 200).mean()

    # Expected PM for disk stars at 20-50 pc
    # V_tan ~ 30 km/s typical, d = 30 pc -> PM ~ 200 mas/yr
    # Compare to Rayleigh distribution (typical for velocities)
    rayleigh_scale = 100  # mas/yr
    ks_stat, ks_pval = stats.kstest(pm, "rayleigh", args=(0, rayleigh_scale))
    results["ks_vs_rayleigh"] = {"statistic": ks_stat, "pvalue": ks_pval}

    logger.info(
        f"Proper motion distribution: {results['mean_pm']:.0f} ± {results['std_pm']:.0f} mas/yr"
    )
    logger.info(f"High PM fraction (>200 mas/yr): {results['frac_high_pm']*100:.1f}%")

    return results


def analyze_sky_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze sky distribution in galactic coordinates.
    """
    if "gal_l" not in df.columns or "gal_b" not in df.columns:
        logger.warning("No galactic coordinates in data")
        return {}

    gal_l = df["gal_l"].dropna()
    gal_b = df["gal_b"].dropna()

    results = {
        "n_sources": len(gal_l),
        "mean_l": gal_l.mean(),
        "mean_b": gal_b.mean(),
    }

    # Test for galactic plane concentration (expect b ~ 0 for disk)
    # Cold BDs should be relatively nearby, so some plane concentration expected
    results["frac_high_lat"] = (np.abs(gal_b) > 30).mean()
    results["frac_low_lat"] = (np.abs(gal_b) < 10).mean()

    # Compare |b| to expected for isotropic
    # Isotropic: sin(b) is uniform, so |b| peaks at 90
    # Disk: concentrated at b=0
    abs_b = np.abs(gal_b)
    ks_stat, ks_pval = stats.kstest(abs_b, "uniform", args=(0, 90))
    results["ks_vs_isotropic"] = {"statistic": ks_stat, "pvalue": ks_pval}

    logger.info(f"Sky distribution: {results['frac_high_lat']*100:.1f}% at |b|>30°")

    return results


def estimate_space_density(df: pd.DataFrame, parallax_df: pd.DataFrame = None) -> dict:
    """
    Estimate space density from sample.
    """
    results = {}

    # If we have parallax data, use it
    if parallax_df is not None and "distance_pc" in parallax_df.columns:
        distances = parallax_df["distance_pc"].dropna()
        distances = distances[distances > 0]

        if len(distances) > 0:
            # Volume-limited estimate
            d_max = distances.max()
            d_median = distances.median()

            # Spherical volume to median distance
            volume_pc3 = (4 / 3) * np.pi * d_median**3

            # Density (very rough)
            density = len(distances) / volume_pc3

            results["n_with_distance"] = len(distances)
            results["d_median_pc"] = d_median
            results["d_max_pc"] = d_max
            results["density_per_pc3"] = density

            # Compare to known BD density
            results["ratio_to_Y_dwarf_density"] = density / BD_SPACE_DENSITY["Y_dwarfs"]

            logger.info(f"Estimated density: {density:.6f} per pc^3")
            logger.info(
                f"Ratio to expected Y dwarf density: {results['ratio_to_Y_dwarf_density']:.1f}x"
            )

            # Selection-function corrected estimate (if module available and required columns exist).
            if callable(calculate_corrected_space_density):
                try:
                    sf_results = calculate_corrected_space_density(df, max_distance=float(d_median))
                    results["selection_corrected_density_per_pc3"] = sf_results["density"]
                    results["selection_corrected_ci"] = (
                        sf_results["lower_95ci"],
                        sf_results["upper_95ci"],
                    )
                    results["mean_selection_completeness"] = sf_results["mean_completeness"]
                    logger.info(
                        "Selection-corrected density: %.6f per pc^3 (95%% CI %.6f-%.6f)",
                        sf_results["density"],
                        sf_results["lower_95ci"],
                        sf_results["upper_95ci"],
                    )
                except Exception as exc:
                    logger.warning("Selection-corrected density failed: %s", exc)

    return results


def compare_to_known_y_dwarfs(df: pd.DataFrame) -> dict:
    """
    Compare sample to known Y dwarf census.
    """
    temps = df["T_eff_K"].dropna()

    results = {
        "sample_size": len(temps),
        "known_y_count": KNOWN_Y_DWARFS["count"],
    }

    # Temperature comparison
    sample_cold = (temps < 300).sum()
    results["sample_below_300K"] = sample_cold
    results["ratio_to_known"] = len(temps) / KNOWN_Y_DWARFS["count"]

    # Are we finding more cold objects?
    sample_mean_T = temps.mean()
    results["mean_T_difference"] = sample_mean_T - KNOWN_Y_DWARFS["mean_teff_K"]

    logger.info(f"Sample vs known Y dwarfs: {results['ratio_to_known']:.1f}x")

    return results


def generate_figures(df: pd.DataFrame, output_dir: Path, parallax_df: pd.DataFrame = None):
    """
    Generate publication-quality comparison figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Temperature distribution
    ax = axes[0, 0]
    temps = df["T_eff_K"].dropna()
    counts, bin_edges, _ = ax.hist(temps, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    # Overlay with Poisson error bars
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.errorbar(bin_centers, counts, yerr=np.sqrt(counts), fmt="none", color="black", capsize=2)
    ax.axvline(300, color="red", linestyle="--", label="300 K")
    ax.axvline(
        KNOWN_Y_DWARFS["mean_teff_K"],
        color="green",
        linestyle="--",
        label=f'Known Y dwarf mean ({KNOWN_Y_DWARFS["mean_teff_K"]} K)',
    )
    ax.set_xlabel("Effective Temperature (K)")
    ax.set_ylabel("Count")
    ax.set_title("Temperature Distribution")
    ax.legend()

    # 2. Proper motion distribution
    ax = axes[0, 1]
    pm = df["pm_total"].dropna()
    counts, bin_edges, _ = ax.hist(pm, bins=20, edgecolor="black", alpha=0.7, color="coral")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.errorbar(
        bin_centers,
        counts,
        yerr=np.sqrt(np.maximum(counts, 1)),
        fmt="none",
        color="black",
        capsize=2,
    )
    ax.axvline(200, color="red", linestyle="--", label="200 mas/yr")
    ax.set_xlabel("Total Proper Motion (mas/yr)")
    ax.set_ylabel("Count")
    ax.set_title("Proper Motion Distribution")
    ax.legend()

    # 3. W1-W2 color distribution
    ax = axes[1, 0]
    color_data = (
        df["w1_w2_color"].dropna() if "w1_w2_color" in df.columns else df["w1mpro"] - df["w2mpro"]
    )
    counts, bin_edges, _ = ax.hist(
        color_data, bins=20, edgecolor="black", alpha=0.7, color="purple"
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.errorbar(
        bin_centers,
        counts,
        yerr=np.sqrt(np.maximum(counts, 1)),
        fmt="none",
        color="black",
        capsize=2,
    )
    ax.axvline(1.5, color="red", linestyle="--", label="Y dwarf threshold")
    ax.set_xlabel("W1-W2 Color (mag)")
    ax.set_ylabel("Count")
    ax.set_title("WISE Color Distribution")
    ax.legend()

    # 4. Distance distribution (if available)
    ax = axes[1, 1]
    if parallax_df is not None and "distance_pc" in parallax_df.columns:
        distances = parallax_df["distance_pc"].dropna()
        distances = distances[(distances > 0) & (distances < 150)]
        counts, bin_edges, _ = ax.hist(
            distances, bins=15, edgecolor="black", alpha=0.7, color="forestgreen"
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.errorbar(
            bin_centers,
            counts,
            yerr=np.sqrt(np.maximum(counts, 1)),
            fmt="none",
            color="black",
            capsize=2,
        )
        ax.axvline(
            KNOWN_Y_DWARFS["max_distance_pc"],
            color="red",
            linestyle="--",
            label=f'Known Y max ({KNOWN_Y_DWARFS["max_distance_pc"]} pc)',
        )
        ax.set_xlabel("Distance (pc)")
        ax.set_ylabel("Count")
        ax.set_title("Distance Distribution (from parallax)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No parallax data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Distance Distribution")

    plt.tight_layout()
    fig.savefig(output_dir / "population_comparison.png", dpi=150)
    fig.savefig(output_dir / "population_comparison.pdf")
    plt.close()

    logger.info(f"Saved figures to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="TASNI Population Synthesis")
    _project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--golden",
        type=str,
        default=str(_project_root / "data" / "processed" / "final" / "golden_improved.csv"),
        help="Golden targets file",
    )
    parser.add_argument(
        "--parallax",
        type=str,
        default=str(
            _project_root / "data" / "processed" / "final" / "golden_improved_parallax.csv"
        ),
        help="Parallax results file",
    )
    parser.add_argument(
        "--output", type=str, default="./data/processed/figures", help="Output directory"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Population Synthesis Comparison")
    logger.info("=" * 60)

    # Load data
    golden = load_golden_targets(args.golden)

    parallax = None
    if Path(args.parallax).exists():
        parallax = pd.read_csv(args.parallax)
        logger.info(f"Loaded parallax data for {len(parallax)} sources")

    # Run analyses
    results = {}

    logger.info("\n--- Temperature Analysis ---")
    results["temperature"] = analyze_temperature_distribution(golden)

    logger.info("\n--- Proper Motion Analysis ---")
    results["proper_motion"] = analyze_proper_motion_distribution(golden)

    logger.info("\n--- Sky Distribution ---")
    results["sky"] = analyze_sky_distribution(golden)

    logger.info("\n--- Space Density Estimate ---")
    results["density"] = estimate_space_density(golden, parallax)

    logger.info("\n--- Comparison to Known Y Dwarfs ---")
    results["y_dwarf_comparison"] = compare_to_known_y_dwarfs(golden)

    # Generate figures
    output_dir = Path(args.output)
    generate_figures(golden, output_dir, parallax)

    # Summary report
    logger.info("\n" + "=" * 60)
    logger.info("POPULATION SYNTHESIS SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\nSample: {len(golden)} golden targets")

    if "temperature" in results:
        t = results["temperature"]
        logger.info("\nTemperature:")
        logger.info(f"  Mean: {t['mean_T']:.0f} K")
        logger.info(f"  Range: {t['min_T']:.0f} - {t['max_T']:.0f} K")
        logger.info(f"  Below 300K: {t['frac_below_300K']*100:.1f}%")

    if "proper_motion" in results:
        pm = results["proper_motion"]
        logger.info("\nProper Motion:")
        logger.info(f"  Mean: {pm['mean_pm']:.0f} mas/yr")
        logger.info(f"  High PM (>200): {pm['frac_high_pm']*100:.1f}%")

    if "density" in results and results["density"]:
        d = results["density"]
        logger.info("\nSpace Density:")
        logger.info(f"  Median distance: {d.get('d_median_pc', 'N/A'):.1f} pc")
        if "density_per_pc3" in d:
            logger.info(f"  Estimated density: {d['density_per_pc3']:.6f} per pc³")

    if "y_dwarf_comparison" in results:
        y = results["y_dwarf_comparison"]
        logger.info("\nComparison to Known Y Dwarfs:")
        logger.info(f"  Sample size: {y['sample_size']} (known: {y['known_y_count']})")
        logger.info(f"  Ratio: {y['ratio_to_known']:.1f}x")

    logger.info("\n" + "=" * 60)

    # Save results
    import json

    results_file = output_dir / "population_synthesis_results.json"

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(i) for i in obj]
        return obj

    with open(results_file, "w") as f:
        json.dump(convert_types(results), f, indent=2)
    logger.info(f"Saved results to {results_file}")


if __name__ == "__main__":
    main()
