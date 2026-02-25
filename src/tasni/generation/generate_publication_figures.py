#!/usr/bin/env python3
"""
TASNI: Generate Publication Figures

Creates publication-quality figures for the TASNI paper including:
1. All-sky map in Galactic coordinates
2. Color-magnitude diagram (W1-W2 vs W1)
3. Temperature and proper motion distributions
4. Variability summary panels
5. Fading sources light curves
6. Pipeline flowchart

Usage:
    python generate_publication_figures.py [--input FILE] [--output-dir DIR]
"""

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    import astropy.units as u
    from astropy.coordinates import Galactic, SkyCoord

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [PUB-FIGS] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Publication style settings
plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Color scheme
COLORS = {
    "normal": "#1f77b4",  # Blue
    "variable": "#ff7f0e",  # Orange
    "fading": "#d62728",  # Red
    "golden": "#2ca02c",  # Green
}


def load_data(golden_path, variability_path=None, epochs_path=None):
    """Load all required data files."""
    data = {}

    if golden_path.endswith(".parquet"):
        data["golden"] = pd.read_parquet(golden_path)
    else:
        data["golden"] = pd.read_csv(golden_path)
    logger.info(f"Loaded {len(data['golden'])} golden targets")

    if variability_path and Path(variability_path).exists():
        if variability_path.endswith(".parquet"):
            data["variability"] = pd.read_parquet(variability_path)
        else:
            data["variability"] = pd.read_csv(variability_path)

    if epochs_path and Path(epochs_path).exists():
        if epochs_path.endswith(".parquet"):
            data["epochs"] = pd.read_parquet(epochs_path)
        else:
            data["epochs"] = pd.read_csv(epochs_path)
        logger.info(f"Loaded {len(data['epochs'])} epochs")

    return data


def fig1_allsky_galactic(data, output_dir):
    """Figure 1: All-sky distribution in Galactic coordinates."""
    df = data["golden"]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="aitoff")

    if HAS_ASTROPY:
        coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs")
        galactic = coords.galactic
        l = galactic.l.wrap_at(180 * u.deg).deg
        b = galactic.b.deg
    else:
        # Approximate conversion
        l = df["ra"].values - 180
        b = df["dec"].values

    l_rad = np.radians(l)
    b_rad = np.radians(b)

    # Color by variability
    if "variability_flag" in df.columns:
        for flag, color in [
            ("NORMAL", COLORS["normal"]),
            ("VARIABLE", COLORS["variable"]),
            ("FADING", COLORS["fading"]),
        ]:
            mask = df["variability_flag"] == flag
            if mask.sum() > 0:
                size = 150 if flag == "FADING" else 40
                marker = "*" if flag == "FADING" else "o"
                ax.scatter(
                    l_rad[mask],
                    b_rad[mask],
                    c=color,
                    s=size,
                    alpha=0.7,
                    marker=marker,
                    label=f"{flag} ({mask.sum()})",
                    edgecolors="white",
                    linewidths=0.3,
                )
    else:
        ax.scatter(l_rad, b_rad, c=COLORS["golden"], s=40, alpha=0.7)

    ax.set_xlabel("Galactic Longitude")
    ax.set_ylabel("Galactic Latitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.title("TASNI Golden Targets: Galactic Distribution", fontsize=14, pad=20)

    plt.savefig(output_dir / "fig1_allsky_galactic.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig1_allsky_galactic.pdf")
    plt.close()
    logger.info("Saved Figure 1: All-sky Galactic map")


def fig2_color_magnitude(data, output_dir):
    """Figure 2: Color-Magnitude Diagram."""
    df = data["golden"]

    fig, ax = plt.subplots(figsize=(10, 8))

    has_err = "w1sigmpro" in df.columns and "w2sigmpro" in df.columns
    if "variability_flag" in df.columns:
        for flag, color, marker in [
            ("NORMAL", COLORS["normal"], "o"),
            ("VARIABLE", COLORS["variable"], "s"),
            ("FADING", COLORS["fading"], "*"),
        ]:
            mask = df["variability_flag"] == flag
            if mask.sum() > 0:
                size = 150 if flag == "FADING" else 50
                x = df.loc[mask, "w1mpro"]
                y = df.loc[mask, "w1_w2_color"]
                x_err = df.loc[mask, "w1sigmpro"] if has_err else None
                y_err = (
                    np.sqrt(df.loc[mask, "w1sigmpro"] ** 2 + df.loc[mask, "w2sigmpro"] ** 2)
                    if has_err
                    else None
                )
                ax.scatter(
                    x,
                    y,
                    c=color,
                    s=size,
                    alpha=0.7,
                    marker=marker,
                    label=f"{flag} ({mask.sum()})",
                    edgecolors="white",
                )
                if has_err and (x_err.notna().all() or y_err.notna().all()):
                    ax.errorbar(
                        x,
                        y,
                        xerr=x_err,
                        yerr=y_err,
                        fmt="none",
                        color=color,
                        capsize=1.5,
                        elinewidth=1,
                        alpha=0.7,
                    )
    else:
        ax.scatter(
            df["w1mpro"], df["w1_w2_color"], c=COLORS["golden"], s=50, alpha=0.7, edgecolors="white"
        )
        if has_err:
            x_err = df["w1sigmpro"]
            y_err = np.sqrt(df["w1sigmpro"] ** 2 + df["w2sigmpro"] ** 2)
            ax.errorbar(
                df["w1mpro"],
                df["w1_w2_color"],
                xerr=x_err,
                yerr=y_err,
                fmt="none",
                color=COLORS["golden"],
                capsize=1.5,
                elinewidth=1,
                alpha=0.7,
            )

    # Spectral type reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=1.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.text(16.5, 0.3, "L dwarfs", fontsize=9, color="gray")
    ax.text(16.5, 1.8, "T dwarfs", fontsize=9, color="gray")
    ax.text(16.5, 2.8, "Y dwarfs", fontsize=9, color="gray")

    ax.set_xlabel("W1 (mag)")
    ax.set_ylabel("W1 - W2 (mag)")
    ax.set_xlim(12.5, 17)
    ax.set_ylim(0, 4)
    ax.invert_xaxis()
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.title("TASNI Color-Magnitude Diagram", fontsize=14)

    plt.savefig(output_dir / "fig2_color_magnitude.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig2_color_magnitude.pdf")
    plt.close()
    logger.info("Saved Figure 2: Color-magnitude diagram")


def fig3_distributions(data, output_dir):
    """Figure 3: Temperature and proper motion distributions."""
    df = data["golden"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Temperature
    ax = axes[0]
    temp_col = "T_eff_K" if "T_eff_K" in df.columns else "ir_teff"
    if temp_col in df.columns:
        temp_data = df[temp_col].dropna()
        if len(temp_data) > 1:
            lo, hi = temp_data.min(), temp_data.max()
            bins = np.linspace(lo, hi + (1 if hi == lo else 0), 16)
        else:
            bins = 15
        if "variability_flag" in df.columns:
            for flag, color in [
                ("NORMAL", COLORS["normal"]),
                ("VARIABLE", COLORS["variable"]),
                ("FADING", COLORS["fading"]),
            ]:
                mask = df["variability_flag"] == flag
                vals = df.loc[mask, temp_col].dropna()
                if len(vals) > 0:
                    counts, bin_edges = np.histogram(vals, bins=bins)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax.hist(vals, bins=bins, alpha=0.6, color=color, label=flag, edgecolor="white")
                    ax.errorbar(
                        bin_centers,
                        counts,
                        yerr=np.sqrt(np.maximum(counts, 1)),
                        fmt="none",
                        color=color,
                        capsize=2,
                        alpha=0.7,
                    )
        else:
            counts, bin_edges, _ = ax.hist(
                temp_data, bins=25, alpha=0.7, color=COLORS["golden"], edgecolor="white"
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.errorbar(
                bin_centers,
                counts,
                yerr=np.sqrt(np.maximum(counts, 1)),
                fmt="none",
                color="black",
                capsize=2,
                alpha=0.5,
            )

        ax.axvline(x=300, color="red", linestyle="--", linewidth=2, label="300K")
        ax.set_xlabel("Effective Temperature (K)")
        ax.set_ylabel("Number of Sources")
        ax.set_title("(a) Temperature Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Proper motion
    ax = axes[1]
    if "pm_total" in df.columns:
        pm_data = df["pm_total"].dropna()
        if len(pm_data) > 1:
            lo, hi = pm_data.min(), pm_data.max()
            bins_pm = np.linspace(lo, hi + (1 if hi == lo else 0), 16)
        else:
            bins_pm = 15
        if "variability_flag" in df.columns:
            for flag, color in [
                ("NORMAL", COLORS["normal"]),
                ("VARIABLE", COLORS["variable"]),
                ("FADING", COLORS["fading"]),
            ]:
                mask = df["variability_flag"] == flag
                vals = df.loc[mask, "pm_total"].dropna()
                if len(vals) > 0:
                    counts, bin_edges = np.histogram(vals, bins=bins_pm)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax.hist(
                        vals, bins=bins_pm, alpha=0.6, color=color, label=flag, edgecolor="white"
                    )
                    ax.errorbar(
                        bin_centers,
                        counts,
                        yerr=np.sqrt(np.maximum(counts, 1)),
                        fmt="none",
                        color=color,
                        capsize=2,
                        alpha=0.7,
                    )
        else:
            counts, bin_edges, _ = ax.hist(
                pm_data, bins=25, alpha=0.7, color=COLORS["golden"], edgecolor="white"
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.errorbar(
                bin_centers,
                counts,
                yerr=np.sqrt(np.maximum(counts, 1)),
                fmt="none",
                color="black",
                capsize=2,
                alpha=0.5,
            )

        ax.axvline(x=100, color="gray", linestyle="--", label="~60 pc")
        ax.set_xlabel("Total Proper Motion (mas/yr)")
        ax.set_ylabel("Number of Sources")
        ax.set_title("(b) Proper Motion Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_distributions.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig3_distributions.pdf")
    plt.close()
    logger.info("Saved Figure 3: Distributions")


def fig4_variability(data, output_dir):
    """Figure 4: Variability analysis summary."""
    df = data["golden"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A: Pie chart
    ax = axes[0, 0]
    if "variability_flag" in df.columns:
        counts = df["variability_flag"].value_counts()
        colors_pie = [COLORS.get(k.lower(), "gray") for k in counts.index]
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors_pie,
            explode=[0.05 if k == "FADING" else 0 for k in counts.index],
        )
        ax.set_title("(a) Variability Ranking")

    # B: RMS vs W1
    ax = axes[0, 1]
    if "rms_w1" in df.columns:
        if "variability_flag" in df.columns:
            for flag, color, marker in [
                ("NORMAL", COLORS["normal"], "o"),
                ("VARIABLE", COLORS["variable"], "s"),
                ("FADING", COLORS["fading"], "*"),
            ]:
                mask = (df["variability_flag"] == flag) & df["rms_w1"].notna()
                size = 100 if flag == "FADING" else 30
                ax.scatter(
                    df.loc[mask, "w1mpro"],
                    df.loc[mask, "rms_w1"],
                    c=color,
                    s=size,
                    alpha=0.7,
                    marker=marker,
                    label=flag,
                )
        ax.set_xlabel("W1 (mag)")
        ax.set_ylabel("W1 RMS (mag)")
        ax.set_title("(b) Variability vs Brightness")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # C: Trend histogram
    ax = axes[1, 0]
    if "trend_w1" in df.columns:
        trends = df["trend_w1"].dropna() * 1000
        ax.hist(trends, bins=30, alpha=0.7, color=COLORS["golden"], edgecolor="white")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
        if "variability_flag" in df.columns:
            fading_trends = df.loc[df["variability_flag"] == "FADING", "trend_w1"].dropna() * 1000
            for t in fading_trends:
                ax.axvline(x=t, color=COLORS["fading"], linestyle="--", linewidth=2)
        ax.set_xlabel("W1 Trend (mmag yr$^{-1}$)")
        ax.set_ylabel("Count")
        ax.set_title("(c) Brightness Trends")
        ax.grid(True, alpha=0.3)

    # D: Epochs vs baseline
    ax = axes[1, 1]
    if "n_epochs" in df.columns and "baseline_years" in df.columns:
        if "variability_flag" in df.columns:
            for flag, color, marker in [
                ("NORMAL", COLORS["normal"], "o"),
                ("VARIABLE", COLORS["variable"], "s"),
                ("FADING", COLORS["fading"], "*"),
            ]:
                mask = df["variability_flag"] == flag
                size = 100 if flag == "FADING" else 30
                ax.scatter(
                    df.loc[mask, "baseline_years"],
                    df.loc[mask, "n_epochs"],
                    c=color,
                    s=size,
                    alpha=0.7,
                    marker=marker,
                    label=flag,
                )
        ax.set_xlabel("Baseline (years)")
        ax.set_ylabel("N Epochs")
        ax.set_title("(d) Temporal Coverage")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("TASNI Variability Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_variability.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig4_variability.pdf")
    plt.close()
    logger.info("Saved Figure 4: Variability summary")


def fig5_fading_lightcurves(data, output_dir):
    """Figure 5: Fading sources light curves."""
    df = data["golden"]

    if "variability_flag" not in df.columns or "epochs" not in data:
        logger.warning("Missing data for fading light curves")
        return

    fading = df[df["variability_flag"] == "FADING"].copy()
    fading = fading[fading["designation"] != "J044024.40-731441.6"]

    if len(fading) == 0:
        return

    epochs = data["epochs"]
    n = len(fading)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for idx, (_, source) in enumerate(fading.iterrows()):
        ax = axes[idx]
        designation = source["designation"]
        src_epochs = epochs[epochs["designation"] == designation].copy()

        if len(src_epochs) == 0:
            continue

        src_epochs["year"] = 2000 + (src_epochs["mjd"] - 51544.5) / 365.25

        ax.errorbar(
            src_epochs["year"],
            src_epochs["w1mpro_ep"],
            yerr=src_epochs.get("w1sigmpro_ep", None),
            fmt="o",
            color="#1f77b4",
            markersize=3,
            alpha=0.5,
            label="W1",
            elinewidth=0.5,
        )

        if "w2mpro_ep" in src_epochs.columns:
            ax.errorbar(
                src_epochs["year"],
                src_epochs["w2mpro_ep"],
                yerr=src_epochs.get("w2sigmpro_ep", None),
                fmt="s",
                color="#d62728",
                markersize=3,
                alpha=0.5,
                label="W2",
                elinewidth=0.5,
            )

        # Trend line
        if "trend_w1" in source and pd.notna(source["trend_w1"]):
            years = np.array([src_epochs["year"].min(), src_epochs["year"].max()])
            mjd_range = np.array([src_epochs["mjd"].min(), src_epochs["mjd"].max()])
            w1_trend = source["w1mpro"] + source["trend_w1"] * (mjd_range - mjd_range.mean())
            ax.plot(
                years,
                w1_trend,
                "b--",
                linewidth=2,
                alpha=0.8,
                label=f'Trend: {source["trend_w1"]*1000:.1f} mmag/yr',
            )

        ax.set_ylabel("Magnitude")
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        temp_col = "T_eff_K" if "T_eff_K" in source else "ir_teff"
        title = f'{designation}  |  W1-W2={source["w1_w2_color"]:.2f}  |  '
        title += f'PM={source["pm_total"]:.0f} mas/yr  |  T={source.get(temp_col, 0):.0f}K'
        ax.set_title(title, fontsize=10)

    axes[-1].set_xlabel("Year")
    plt.suptitle("TASNI Fading Thermal Orphans: 10-Year Light Curves", fontsize=14, y=1.02)
    plt.tight_layout()

    plt.savefig(output_dir / "fig5_fading_lightcurves.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig5_fading_lightcurves.pdf")
    plt.close()
    logger.info("Saved Figure 5: Fading light curves")


def fig6_pipeline_flowchart(data, output_dir):
    """Figure 6: Pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    stages = [
        ("AllWISE Catalog", "747,634,026", "#e6f2ff"),
        ("No Gaia Optical", "406,387,755", "#e6f2ff"),
        ("Thermal Selection\n(W1-W2 > 0.5)", "~1,000,000", "#e6f2ff"),
        ("No 2MASS NIR", "62,856", "#fff2e6"),
        ("No Pan-STARRS", "39,188", "#fff2e6"),
        ("No Legacy DR10", "39,151", "#fff2e6"),
        ("Radio Silent (NVSS)", "4,137", "#e6ffe6"),
        ("Golden Targets", "100", "#e6ffe6"),
        ("Fading Sources", "4", "#ffe6e6"),
    ]

    box_width = 6
    box_height = 0.9
    x_center = 5

    for i, (label, count, color) in enumerate(stages):
        y = 13 - i * 1.4

        box = FancyBboxPatch(
            (x_center - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x_center,
            y,
            f"{label}\n({count})",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_center, y - box_height / 2 - 0.1),
                xytext=(x_center, y - box_height / 2 - 0.4),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2),
            )

    # Veto labels
    vetoes = [(1, "Gaia DR3"), (3, "2MASS"), (4, "Pan-STARRS"), (5, "Legacy DR10"), (6, "NVSS")]
    for stage, label in vetoes:
        y = 13 - stage * 1.4
        ax.text(
            x_center + box_width / 2 + 0.3,
            y,
            f"✗ {label}",
            ha="left",
            va="center",
            fontsize=9,
            color="gray",
        )

    plt.title("TASNI Pipeline Flowchart", fontsize=14, pad=20)
    plt.savefig(output_dir / "fig6_pipeline_flowchart.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig6_pipeline_flowchart.pdf")
    plt.close()
    logger.info("Saved Figure 6: Pipeline flowchart")


def create_summary_table(data, output_dir):
    """Create summary statistics."""
    df = data["golden"]

    lines = ["TASNI Pipeline Summary Statistics", "=" * 50, ""]
    lines.append(f"Total golden targets: {len(df)}")
    lines.append(f"Mean W1 magnitude: {df['w1mpro'].mean():.2f} ± {df['w1mpro'].std():.2f}")
    lines.append(
        f"Mean W1-W2 color: {df['w1_w2_color'].mean():.2f} ± {df['w1_w2_color'].std():.2f}"
    )

    temp_col = "T_eff_K" if "T_eff_K" in df.columns else "ir_teff"
    if temp_col in df.columns:
        lines.append(f"Mean T_eff: {df[temp_col].mean():.0f} ± {df[temp_col].std():.0f} K")

    if "pm_total" in df.columns:
        lines.append(
            f"Mean proper motion: {df['pm_total'].mean():.0f} ± {df['pm_total'].std():.0f} mas/yr"
        )

    if "variability_flag" in df.columns:
        lines.append("")
        lines.append("Variability Ranking:")
        for flag in ["NORMAL", "VARIABLE", "FADING"]:
            count = (df["variability_flag"] == flag).sum()
            lines.append(f"  {flag}: {count} ({100*count/len(df):.1f}%)")

    if "baseline_years" in df.columns:
        lines.append(f"\nMean baseline: {df['baseline_years'].mean():.1f} years")
    if "n_epochs" in df.columns:
        lines.append(f"Mean epochs: {df['n_epochs'].mean():.0f}")

    with open(output_dir / "summary_statistics.txt", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    logger.info("Saved summary statistics")


def main():
    parser = argparse.ArgumentParser(description="TASNI Publication Figures")
    parser.add_argument("--golden", "-g", default="./data/processed/final/golden_improved.csv")
    parser.add_argument("--variability", "-v", default="./data/processed/golden_variability.csv")
    parser.add_argument("--epochs", "-e", default="./data/processed/neowise_epochs.parquet")
    parser.add_argument("--output", "-o", default="./data/processed/figures")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Publication Figure Generation")
    logger.info("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.golden, args.variability, args.epochs)

    logger.info("Generating figures...")
    fig1_allsky_galactic(data, output_dir)
    fig2_color_magnitude(data, output_dir)
    fig3_distributions(data, output_dir)
    fig4_variability(data, output_dir)
    fig5_fading_lightcurves(data, output_dir)
    fig6_pipeline_flowchart(data, output_dir)
    create_summary_table(data, output_dir)

    logger.info("=" * 60)
    logger.info(f"All figures saved to: {output_dir}")

    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
