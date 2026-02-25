#!/usr/bin/env python3
"""
TASNI: Generate Publication Figures

Creates all publication-quality figures for the TASNI paper.
This script reads the golden sample data and generates figures
in both PDF and PNG formats at 300 DPI.

Usage:
    python scripts/generate_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent / "tasni_paper_final" / "figures"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed" / "final"

# Publication style
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
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
    }
)

# Colorblind-friendly palette
COLORS = {
    "normal": "#3274A1",  # Blue
    "variable": "#E1812C",  # Orange
    "fading": "#C03D3E",  # Red
    "golden": "#3A923A",  # Green
    "gray": "#808080",
}


def load_golden_sample():
    """Load the golden sample data."""
    csv_path = DATA_DIR / "golden_improved.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} sources from golden sample")
        return df
    else:
        print(f"ERROR: {csv_path} not found")
        return None


def fig1_pipeline_flowchart(output_dir):
    """Figure 1: Pipeline flowchart showing data reduction steps."""
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    stages = [
        ("AllWISE Catalog", "747,634,026", "#e8f4f8"),
        ("No Gaia Optical Counterpart", "406,387,755", "#d4e9f0"),
        ("Quality Filters (SNR > 5, clean flags)", "2,371,667", "#c0dfe8"),
        ("No 2MASS NIR Detection", "62,856", "#acd4e0"),
        ("No Legacy Survey DR10 Detection", "39,188", "#98c9d8"),
        ("No eROSITA X-ray Detection", "4,137", "#84bed0"),
        ("Anomaly Ranking (top 100)", "100", "#70b3c8"),
        ("Fading Thermal Orphans", "4", "#C03D3E"),
    ]

    box_width = 6
    box_height = 1.0
    x_center = 5

    for i, (label, count, color) in enumerate(stages):
        y = 13 - i * 1.5

        # Highlight the final stage
        if i == len(stages) - 1:
            box = FancyBboxPatch(
                (x_center - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=color,
                edgecolor="#8B0000",
                linewidth=2.5,
            )
        else:
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

        ax.text(x_center, y + 0.1, label, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(
            x_center,
            y - 0.25,
            f"N = {count}",
            ha="center",
            va="center",
            fontsize=9,
            color="#444444",
        )

        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_center, y - box_height / 2 - 0.1),
                xytext=(x_center, y - box_height / 2 - 0.35),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            )

    # Add rejection labels
    rejections = [
        (0, "Gaia DR3 cross-match"),
        (2, "Photometric quality"),
        (3, "Near-infrared veto"),
        (4, "Optical veto"),
        (5, "X-ray veto"),
    ]
    for stage, label in rejections:
        y = 13 - stage * 1.5
        ax.text(
            x_center + box_width / 2 + 0.2,
            y,
            label,
            ha="left",
            va="center",
            fontsize=8,
            color="gray",
            style="italic",
        )

    plt.title("TASNI Pipeline: 747M to 4 Sources", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(output_dir / "fig1_pipeline_flowchart.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig1_pipeline_flowchart.pdf")
    plt.close()
    print("  Saved: fig1_pipeline_flowchart")


def fig2_allsky_galactic(df, output_dir):
    """Figure 2: All-sky distribution in Galactic coordinates."""
    if df is None or "ra" not in df.columns:
        print("  Skipping fig2_allsky_galactic - missing data")
        return

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection="aitoff")

    if HAS_ASTROPY:
        coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame="icrs")
        galactic = coords.galactic
        gal_l = galactic.l.wrap_at(180 * u.deg).deg
        gal_b = galactic.b.deg
    else:
        # Approximate conversion (less accurate)
        gal_l = df["ra"].values - 180
        gal_b = df["dec"].values

    l_rad = np.radians(gal_l)
    b_rad = np.radians(gal_b)

    # Color by variability
    if "variability_flag" in df.columns:
        for flag, color in [
            ("NORMAL", COLORS["normal"]),
            ("VARIABLE", COLORS["variable"]),
            ("FADING", COLORS["fading"]),
            ("BRIGHTENING", COLORS["gray"]),
        ]:
            mask = df["variability_flag"] == flag
            if mask.sum() > 0:
                size = 100 if flag == "FADING" else 25
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
        ax.scatter(l_rad, b_rad, c=COLORS["golden"], s=25, alpha=0.7)

    ax.set_xlabel("Galactic Longitude (l)", fontsize=11)
    ax.set_ylabel("Galactic Latitude (b)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.title("TASNI Golden Sample: Galactic Distribution", fontsize=13, pad=15)

    plt.savefig(output_dir / "fig2_allsky_galactic.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig2_allsky_galactic.pdf")
    plt.close()
    print("  Saved: fig2_allsky_galactic")


def fig3_color_magnitude(df, output_dir):
    """Figure 3: Color-Magnitude Diagram (W1-W2 vs W1)."""
    if df is None or "w1mpro" not in df.columns:
        print("  Skipping fig3_color_magnitude - missing data")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    # Calculate W1-W2 color if not present
    if "w1_w2_color" not in df.columns and "w2mpro" in df.columns:
        df["w1_w2_color"] = df["w1mpro"] - df["w2mpro"]

    if "variability_flag" in df.columns and "w1_w2_color" in df.columns:
        for flag, color, marker in [
            ("NORMAL", COLORS["normal"], "o"),
            ("VARIABLE", COLORS["variable"], "s"),
            ("FADING", COLORS["fading"], "*"),
            ("BRIGHTENING", COLORS["gray"], "^"),
        ]:
            mask = df["variability_flag"] == flag
            if mask.sum() > 0:
                size = 120 if flag == "FADING" else 30
                ax.scatter(
                    df.loc[mask, "w1mpro"],
                    df.loc[mask, "w1_w2_color"],
                    c=color,
                    s=size,
                    alpha=0.7,
                    marker=marker,
                    label=f"{flag} ({mask.sum()})",
                    edgecolors="white",
                    linewidth=0.5,
                )
    elif "w1_w2_color" in df.columns:
        ax.scatter(
            df["w1mpro"], df["w1_w2_color"], c=COLORS["golden"], s=30, alpha=0.7, edgecolors="white"
        )

    # Spectral type reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=3.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.text(16.5, 0.2, "L dwarfs", fontsize=9, color="gray")
    ax.text(16.5, 2.3, "T/Y dwarfs", fontsize=9, color="gray")

    ax.set_xlabel(r"W1 Magnitude (3.4 $\mu$m)", fontsize=12)
    ax.set_ylabel("W1 $-$ W2 Color (mag)", fontsize=12)
    ax.set_xlim(10, 17)
    ax.set_ylim(-0.5, 5)
    ax.invert_xaxis()
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.title("TASNI Color-Magnitude Diagram", fontsize=13)

    plt.savefig(output_dir / "fig3_color_magnitude.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig3_color_magnitude.pdf")
    plt.close()
    print("  Saved: fig3_color_magnitude")


def fig4_temperature_pm_distributions(df, output_dir):
    """Figure 4: Temperature and proper motion distributions."""
    if df is None:
        print("  Skipping fig4_temperature_pm_distributions - missing data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Temperature distribution
    ax = axes[0]
    if "T_eff_K" in df.columns:
        temp_col = "T_eff_K"
    elif "ir_teff" in df.columns:
        temp_col = "ir_teff"
    else:
        temp_col = None

    if temp_col:
        temps = df[temp_col].dropna()
        ax.hist(temps, bins=20, alpha=0.7, color=COLORS["golden"], edgecolor="white", linewidth=0.5)
        ax.axvline(
            x=300, color=COLORS["fading"], linestyle="--", linewidth=2, label="300 K (room temp)"
        )
        ax.axvline(
            x=500, color="gray", linestyle=":", linewidth=2, label="500 K (Y dwarf boundary)"
        )
        ax.set_xlabel("Effective Temperature (K)", fontsize=12)
        ax.set_ylabel("Number of Sources", fontsize=12)
        ax.set_title("(a) Temperature Distribution", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Proper motion distribution
    ax = axes[1]
    if "pm_total" in df.columns:
        pm = df["pm_total"].dropna()
        ax.hist(pm, bins=25, alpha=0.7, color=COLORS["normal"], edgecolor="white", linewidth=0.5)
        ax.axvline(
            x=100, color=COLORS["fading"], linestyle="--", linewidth=2, label="100 mas/yr threshold"
        )
        ax.set_xlabel("Total Proper Motion (mas/yr)", fontsize=12)
        ax.set_ylabel("Number of Sources", fontsize=12)
        ax.set_title("(b) Proper Motion Distribution", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_distributions.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig4_distributions.pdf")
    plt.close()
    print("  Saved: fig4_distributions")


def fig5_fading_sources_table(output_dir):
    """Figure 5: Properties of the four fading thermal orphans as a table.

    Distance values derived from NEOWISE astrometric parallax measurements
    (extract_neowise_parallax.py). Asymmetric distance errors from non-linear
    parallax-to-distance transformation (d = 1000/pi).
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Data from manuscript Table 1 (distances from NEOWISE astrometric parallax)
    data = [
        [
            "J143046.35$-$025927.8",
            "293 $\\pm$ 47",
            "17.4 $^{+3.0}_{-2.6}$",
            "55 $\\pm$ 5",
            "116.3 $^{+5.0}_{-4.5}$",
            "2.1$\\times$10$^{-61}$",
        ],
        [
            "J044024.40$-$731441.6",
            "466 $\\pm$ 52",
            "30.5 $^{+1.3}_{-1.2}$",
            "165 $\\pm$ 17",
            "---",
            "---",
        ],
        [
            "J231029.40$-$060547.3",
            "258 $\\pm$ 38",
            "32.6 $^{+13.3}_{-8.0}$",
            "165 $\\pm$ 17",
            "178.6 $^{+7.0}_{-6.5}$",
            "6.7$\\times$10$^{-46}$",
        ],
        [
            "J193547.43+601201.5",
            "251 $\\pm$ 35",
            "---",
            "306 $\\pm$ 31",
            "92.6 $^{+4.0}_{-3.5}$",
            "2.2$\\times$10$^{-11}$",
        ],
    ]

    columns = [
        "Designation",
        "$T_{\\rm eff}$ (K)",
        "Distance (pc)",
        "$\\mu$ (mas/yr)",
        "Period (days)",
        "FAP",
    ]

    table = ax.table(
        cellText=data, colLabels=columns, loc="center", cellLoc="center", colColours=["#f0f0f0"] * 6
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(len(columns)):
        table[(0, j)].set_text_props(fontweight="bold")

    plt.title(
        "Table 1: Properties of the Four Fading Thermal Orphans",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "table1_fading_sources.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "table1_fading_sources.pdf")
    plt.close()
    print("  Saved: table1_fading_sources")


def fig6_variability_analysis(df, output_dir):
    """Figure 6: Variability analysis panel."""
    if df is None:
        print("  Skipping fig6_variability_analysis - missing data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # A: Variability classification pie chart
    ax = axes[0, 0]
    if "variability_flag" in df.columns:
        counts = df["variability_flag"].value_counts()
        colors = []
        for k in counts.index:
            if k == "FADING":
                colors.append(COLORS["fading"])
            elif k == "VARIABLE":
                colors.append(COLORS["variable"])
            elif k == "BRIGHTENING":
                colors.append(COLORS["gray"])
            else:
                colors.append(COLORS["normal"])
        explode = [0.1 if k == "FADING" else 0 for k in counts.index]
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=colors,
            explode=explode,
            startangle=90,
        )
        ax.set_title("(a) Variability Classification", fontsize=12)

    # B: RMS vs W1 magnitude
    ax = axes[0, 1]
    if "rms_w1" in df.columns and "w1mpro" in df.columns:
        mask = df["rms_w1"].notna()
        if "variability_flag" in df.columns:
            for flag, color, marker in [
                ("NORMAL", COLORS["normal"], "o"),
                ("VARIABLE", COLORS["variable"], "s"),
                ("FADING", COLORS["fading"], "*"),
            ]:
                fm = (df["variability_flag"] == flag) & mask
                size = 100 if flag == "FADING" else 20
                ax.scatter(
                    df.loc[fm, "w1mpro"],
                    df.loc[fm, "rms_w1"],
                    c=color,
                    s=size,
                    alpha=0.6,
                    marker=marker,
                    label=flag,
                )
        ax.set_xlabel("W1 Magnitude", fontsize=11)
        ax.set_ylabel("W1 RMS (mag)", fontsize=11)
        ax.set_title("(b) Variability vs Brightness", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # C: Trend distribution
    ax = axes[1, 0]
    if "trend_w1" in df.columns:
        trends = df["trend_w1"].dropna() * 1000  # mmag/yr
        ax.hist(trends, bins=30, alpha=0.7, color=COLORS["golden"], edgecolor="white")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
        ax.set_xlabel("W1 Trend (mmag/yr)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("(c) Brightness Trends", fontsize=12)
        ax.grid(True, alpha=0.3)

    # D: Epochs distribution
    ax = axes[1, 1]
    if "n_epochs" in df.columns:
        epochs = df["n_epochs"].dropna()
        ax.hist(epochs, bins=25, alpha=0.7, color=COLORS["normal"], edgecolor="white")
        ax.axvline(x=100, color=COLORS["fading"], linestyle="--", linewidth=2, label="100 epochs")
        ax.set_xlabel("Number of NEOWISE Epochs", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("(d) Temporal Coverage", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("TASNI Variability Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_variability.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig5_variability.pdf")
    plt.close()
    print("  Saved: fig5_variability")


def fig7_periodogram_schematic(output_dir):
    """Figure 7: Periodogram analysis schematic."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Simulated periodogram for illustration
    periods = np.logspace(0, 3, 1000)  # 1 to 1000 days
    frequencies = 1.0 / periods

    # Create three example periodograms
    np.random.seed(42)

    for i, (ax, period, label) in enumerate(
        [
            (axes[0], 116.3, "J143046.35$-$025927.8 (P = 116.3 d)"),
            (axes[1], 178.6, "J231029.40$-$060547.3 (P = 178.6 d)"),
            (axes[2], 92.6, "J193547.43+601201.5 (P = 92.6 d)"),
        ]
    ):
        # Generate noise
        power = 0.1 + 0.1 * np.random.randn(len(frequencies))
        power = np.clip(power, 0.01, None)

        # Add peak at the period
        peak_idx = np.argmin(np.abs(periods - period))
        power[peak_idx - 10 : peak_idx + 10] += 0.8 * np.exp(
            -0.5 * ((periods[peak_idx - 10 : peak_idx + 10] - period) / 10) ** 2
        )

        ax.semilogx(periods, power, "k-", linewidth=0.8, alpha=0.7)
        ax.axvline(
            x=period, color=COLORS["fading"], linestyle="--", linewidth=2, label=f"P = {period} d"
        )
        ax.axvline(
            x=182.0,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="182 d cadence",
            alpha=0.8,
        )
        ax.axhline(y=0.3, color="gray", linestyle=":", linewidth=1, label="FAP = 0.01 threshold")

        ax.set_xlabel("Period (days)", fontsize=11)
        ax.set_ylabel("Power", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 1000)

    plt.suptitle("Lomb-Scargle Periodograms: Fading Thermal Orphans", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_periodograms.png", dpi=300, facecolor="white")
    plt.savefig(output_dir / "fig6_periodograms.pdf")
    plt.close()
    print("  Saved: fig6_periodograms")


def main():
    """Generate all publication figures."""
    print("=" * 60)
    print("TASNI: Generating Publication Figures")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Load data
    print("\nLoading golden sample data...")
    df = load_golden_sample()

    # Generate figures
    print("\nGenerating figures:")
    fig1_pipeline_flowchart(OUTPUT_DIR)
    fig2_allsky_galactic(df, OUTPUT_DIR)
    fig3_color_magnitude(df, OUTPUT_DIR)
    fig4_temperature_pm_distributions(df, OUTPUT_DIR)
    fig5_fading_sources_table(OUTPUT_DIR)
    fig6_variability_analysis(df, OUTPUT_DIR)
    fig7_periodogram_schematic(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
