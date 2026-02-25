#!/usr/bin/env python3
"""
TASNI: Prepare Spectroscopy Targets for Telescope Proposals

Generates:
1. Target list with coordinates, magnitudes, observability
2. Finding charts with annotations
3. Visibility plots (airmass vs time) for major facilities
4. Exposure time estimates
5. LaTeX table for telescope proposals

Usage:
    python prepare_spectroscopy_targets.py [--input FILE] [--output-dir DIR]
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [SPEC-PREP] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Telescope facilities
FACILITIES = {
    "Keck": {
        "name": "W. M. Keck Observatory",
        "location": EarthLocation.of_site("Keck Observatory") if HAS_ASTROPY else None,
        "lat": 19.8260,
        "lon": -155.4747,
        "elev": 4145,
        "instruments": ["NIRES", "MOSFIRE"],
        "dec_range": (-35, 90),
        "best_months": [1, 2, 3, 9, 10, 11, 12],
    },
    "Gemini-N": {
        "name": "Gemini North",
        "location": EarthLocation.of_site("Gemini North") if HAS_ASTROPY else None,
        "lat": 19.8238,
        "lon": -155.4690,
        "elev": 4213,
        "instruments": ["GNIRS", "NIRI"],
        "dec_range": (-37, 90),
        "best_months": [1, 2, 3, 9, 10, 11, 12],
    },
    "Gemini-S": {
        "name": "Gemini South",
        "location": EarthLocation.of_site("Cerro Pachon") if HAS_ASTROPY else None,
        "lat": -30.2407,
        "lon": -70.7367,
        "elev": 2722,
        "instruments": ["Flamingos-2", "GNIRS"],
        "dec_range": (-90, 28),
        "best_months": [1, 2, 3, 10, 11, 12],
    },
    "VLT": {
        "name": "Very Large Telescope",
        "location": EarthLocation.of_site("Paranal Observatory") if HAS_ASTROPY else None,
        "lat": -24.6253,
        "lon": -70.4033,
        "elev": 2635,
        "instruments": ["KMOS", "X-shooter", "CRIRES+"],
        "dec_range": (-90, 25),
        "best_months": [1, 2, 3, 10, 11, 12],
    },
}

# Exclude LMC contamination
EXCLUDE_DESIGNATIONS = ["J044024.40-731441.6"]  # MSX LMC 1152


def load_fading_targets(input_file: str) -> pd.DataFrame:
    """Load fading targets from variability analysis."""
    if input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    # Filter to fading sources
    if "trend_type" in df.columns:
        fading = df[df["trend_type"] == "fading"].copy()
    elif "variability_flag" in df.columns:
        fading = df[df["variability_flag"] == "FADING"].copy()
    else:
        logger.warning("No trend_type or variability_flag column, using all sources")
        fading = df.copy()

    # Exclude known contamination
    fading = fading[~fading["designation"].isin(EXCLUDE_DESIGNATIONS)]

    logger.info(
        f"Loaded {len(fading)} fading targets (excluding {len(EXCLUDE_DESIGNATIONS)} known contaminants)"
    )
    return fading


def compute_observability(targets: pd.DataFrame) -> pd.DataFrame:
    """Compute observability from each facility."""
    if not HAS_ASTROPY:
        logger.warning("astropy not available, skipping observability calculation")
        return targets

    for facility_name, facility in FACILITIES.items():
        col_name = f'observable_{facility_name.lower().replace("-", "_")}'
        observable = []

        dec_min, dec_max = facility["dec_range"]

        for _, row in targets.iterrows():
            dec = row["dec"]
            # Simple declination check
            is_observable = dec_min <= dec <= dec_max
            observable.append(is_observable)

        targets[col_name] = observable

    return targets


def compute_airmass_curve(ra: float, dec: float, facility: dict, date: datetime = None) -> tuple:
    """Compute airmass vs time for a target from a facility."""
    if not HAS_ASTROPY:
        return None, None

    if date is None:
        # Use a date 6 months from now (typical proposal timeline)
        date = datetime.now() + timedelta(days=180)

    # Create location
    if facility["location"] is not None:
        location = facility["location"]
    else:
        location = EarthLocation(
            lat=facility["lat"] * u.deg, lon=facility["lon"] * u.deg, height=facility["elev"] * u.m
        )

    # Create target coordinates
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # Generate times throughout the night
    midnight = Time(date.strftime("%Y-%m-%d") + " 06:00:00")  # Approx midnight UTC-6
    times = midnight + np.linspace(-6, 6, 49) * u.hour

    # Compute alt-az
    altaz_frame = AltAz(obstime=times, location=location)
    altaz = target.transform_to(altaz_frame)

    # Compute airmass (only when above horizon)
    airmass = altaz.secz.value
    airmass[altaz.alt.deg < 10] = np.nan  # Below horizon or too low

    hours = np.linspace(-6, 6, 49)
    return hours, airmass


def create_visibility_plot(targets: pd.DataFrame, output_dir: Path):
    """Create visibility plot for all targets at each facility."""
    if not HAS_MATPLOTLIB or not HAS_ASTROPY:
        logger.warning("matplotlib or astropy not available, skipping visibility plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))

    for ax_idx, (facility_name, facility) in enumerate(FACILITIES.items()):
        ax = axes[ax_idx]

        for idx, (_, target) in enumerate(targets.iterrows()):
            hours, airmass = compute_airmass_curve(target["ra"], target["dec"], facility)

            if hours is not None:
                label = target["designation"].replace("J", "")[:15]
                ax.plot(hours, airmass, color=colors[idx], label=label, linewidth=2)

        ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, label="Airmass=2")
        ax.set_xlim(-6, 6)
        ax.set_ylim(1, 3)
        ax.set_xlabel("Hours from Midnight")
        ax.set_ylabel("Airmass")
        ax.set_title(f'{facility["name"]}')
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    plt.suptitle("TASNI Fading Sources: Observability from Major Facilities", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / "visibility_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visibility plot to {output_path}")


def create_finding_charts(targets: pd.DataFrame, cutout_dir: Path, output_dir: Path):
    """Create annotated finding charts for each target."""
    if not HAS_MATPLOTLIB or not HAS_PIL:
        logger.warning("matplotlib or PIL not available, skipping finding charts")
        return

    finding_chart_dir = output_dir / "finding_charts"
    finding_chart_dir.mkdir(exist_ok=True)

    for _, target in targets.iterrows():
        designation = target["designation"]

        # Convert designation to cutout filename format
        # dr9_J143046.35m025927.8.jpg (+ -> p, - -> m)
        cutout_name = designation.replace("+", "p").replace("-", "m")

        # Find existing cutout
        cutout_patterns = [
            cutout_dir / f"dr9_{cutout_name}.jpg",
            cutout_dir / f"{designation}_legacy.jpg",
            cutout_dir / f"{designation}_wise.jpg",
            cutout_dir / f"{designation}.jpg",
        ]

        cutout_path = None
        for pattern in cutout_patterns:
            if pattern.exists():
                cutout_path = pattern
                break

        if cutout_path is None:
            logger.warning(f"No cutout found for {designation}")
            continue

        # Create annotated finding chart
        fig, ax = plt.subplots(figsize=(8, 8))

        img = Image.open(cutout_path)
        ax.imshow(img)

        # Add crosshairs at center
        cx, cy = img.size[0] // 2, img.size[1] // 2
        ax.axhline(y=cy, color="lime", linewidth=1, alpha=0.7)
        ax.axvline(x=cx, color="lime", linewidth=1, alpha=0.7)

        # Add circle at center
        circle = plt.Circle((cx, cy), 15, fill=False, color="lime", linewidth=2)
        ax.add_patch(circle)

        # Add compass
        ax.annotate("N", xy=(cx, 20), fontsize=14, color="white", ha="center", fontweight="bold")
        ax.annotate("E", xy=(20, cy), fontsize=14, color="white", ha="center", fontweight="bold")

        # Title with target info
        title = f"{designation}\n"
        title += f"RA={target['ra']:.6f}° Dec={target['dec']:.6f}°\n"
        title += f"W1={target.get('w1mpro', 'N/A'):.2f} W2={target.get('w2mpro', 'N/A'):.2f} "
        title += f"W1-W2={target.get('w1_w2_color', target.get('w1mpro', 0) - target.get('w2mpro', 0)):.2f}\n"
        title += f"PM={target.get('pm_total', 'N/A'):.1f} mas/yr  T={target.get('T_eff_K', target.get('ir_teff', 'N/A')):.0f}K"

        ax.set_title(title, fontsize=11, pad=10)
        ax.axis("off")

        # Add scale bar (approximate - 1 arcmin if 256px = 4 arcmin)
        scale_px = img.size[0] / 4  # ~1 arcmin
        ax.plot(
            [20, 20 + scale_px], [img.size[1] - 20, img.size[1] - 20], color="white", linewidth=3
        )
        ax.text(20 + scale_px / 2, img.size[1] - 30, "1'", color="white", fontsize=12, ha="center")

        output_path = finding_chart_dir / f"{designation}_finding_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="black", edgecolor="none")
        plt.close()

        logger.info(f"Created finding chart: {output_path}")


def estimate_exposure_time(w1mag: float, instrument: str = "NIRES") -> dict:
    """Estimate exposure time based on W1 magnitude."""
    # Rough estimates based on instrument sensitivity
    # These are approximate and should be refined for actual proposals

    estimates = {
        "NIRES": {  # Keck NIRES
            "snr_target": 10,
            "base_time_s": 300,  # for W1=14
            "reference_mag": 14.0,
        },
        "GNIRS": {  # Gemini GNIRS
            "snr_target": 10,
            "base_time_s": 600,
            "reference_mag": 14.0,
        },
        "Flamingos-2": {  # Gemini-S
            "snr_target": 10,
            "base_time_s": 900,
            "reference_mag": 14.0,
        },
        "KMOS": {  # VLT
            "snr_target": 10,
            "base_time_s": 600,
            "reference_mag": 14.0,
        },
    }

    if instrument not in estimates:
        instrument = "NIRES"

    est = estimates[instrument]

    # Scale by magnitude difference
    mag_diff = w1mag - est["reference_mag"]
    time_factor = 10 ** (0.4 * mag_diff)  # 2.5x per magnitude

    exp_time_s = est["base_time_s"] * time_factor

    return {
        "instrument": instrument,
        "snr_target": est["snr_target"],
        "exposure_s": int(exp_time_s),
        "exposure_min": round(exp_time_s / 60, 1),
        "n_exposures": max(1, int(exp_time_s / 900)),  # 15-min max per exposure
    }


def generate_latex_table(targets: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for telescope proposal."""
    latex_lines = []

    # Header
    latex_lines.append(r"\begin{table*}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{TASNI Fading Thermal Orphans: Spectroscopy Targets}")
    latex_lines.append(r"\label{tab:spectroscopy_targets}")
    latex_lines.append(r"\begin{tabular}{lcccccccl}")
    latex_lines.append(r"\hline\hline")
    latex_lines.append(
        r"Designation & RA & Dec & W1 & W2 & W1$-$W2 & $\mu$ & T$_{\rm eff}$ & Best Facility \\"
    )
    latex_lines.append(r" & (deg) & (deg) & (mag) & (mag) & (mag) & (mas/yr) & (K) & \\")
    latex_lines.append(r"\hline")

    # Sort by priority (e.g., highest W1-W2 color first)
    targets_sorted = targets.sort_values("w1_w2_color", ascending=False)

    for _, row in targets_sorted.iterrows():
        designation = row["designation"]
        ra = row["ra"]
        dec = row["dec"]
        w1 = row.get("w1mpro", 0)
        w2 = row.get("w2mpro", 0)
        w1_w2 = row.get("w1_w2_color", w1 - w2)
        pm = row.get("pm_total", 0)
        teff = row.get("T_eff_K", row.get("ir_teff", 0))

        # Determine best facility
        if dec > 0:
            best_facility = "Keck/NIRES"
        elif dec > -35:
            best_facility = "Gemini-N/GNIRS"
        else:
            best_facility = "VLT/KMOS"

        line = f"{designation} & {ra:.4f} & {dec:.4f} & {w1:.2f} & {w2:.2f} & {w1_w2:.2f} & {pm:.0f} & {teff:.0f} & {best_facility} \\\\"
        latex_lines.append(line)

    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(
        r"\tablecomments{Targets selected from TASNI pipeline as fading thermal orphans."
    )
    latex_lines.append(
        r"W1$-$W2 colors suggest Y/T dwarf candidates. Proper motions ($\mu$) indicate nearby objects.}"
    )
    latex_lines.append(r"\end{table*}")

    output_path = output_dir / "spectroscopy_targets.tex"
    with open(output_path, "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"Saved LaTeX table to {output_path}")


def generate_target_list(targets: pd.DataFrame, output_dir: Path):
    """Generate human-readable target list."""
    lines = []
    lines.append("=" * 80)
    lines.append("TASNI: Fading Thermal Orphans - Spectroscopy Target List")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Sort by priority
    targets_sorted = targets.sort_values("w1_w2_color", ascending=False)

    for priority, (_, row) in enumerate(targets_sorted.iterrows(), 1):
        designation = row["designation"]
        ra = row["ra"]
        dec = row["dec"]
        w1 = row.get("w1mpro", 0)
        w2 = row.get("w2mpro", 0)
        w1_w2 = row.get("w1_w2_color", w1 - w2)
        pm = row.get("pm_total", 0)
        teff = row.get("T_eff_K", row.get("ir_teff", 0))
        fade_rate = row.get("trend_w1", 0)

        lines.append(f"Priority {priority}: {designation}")
        lines.append("-" * 60)
        lines.append(f"  Coordinates:  RA = {ra:.6f}°  Dec = {dec:.6f}°")
        lines.append(f"  WISE mags:    W1 = {w1:.3f}  W2 = {w2:.3f}")
        lines.append(f"  Color:        W1-W2 = {w1_w2:.3f} mag")
        lines.append(f"  Proper motion: {pm:.1f} mas/yr")
        lines.append(f"  Temperature:  T_eff ~ {teff:.0f} K")
        lines.append(f"  Fade rate:    {fade_rate*1000:.1f} mmag/yr in W1")

        # Estimate distance from proper motion (rough approximation)
        # Assume typical tangential velocity ~30 km/s for disk objects
        if pm > 0:
            dist_pc = 30 / (4.74 * pm / 1000)  # v_tan = 4.74 * mu * d
            lines.append(f"  Est. distance: ~{dist_pc:.0f} pc (assuming v_tan=30 km/s)")

        # Exposure estimates
        exp = estimate_exposure_time(w1, "NIRES")
        lines.append(
            f"  Exp. estimate: {exp['exposure_min']} min (Keck/NIRES, SNR={exp['snr_target']})"
        )

        # Observability
        if dec > 0:
            lines.append("  Best facility: Keck (Mauna Kea) - NIRES or MOSFIRE")
        elif dec > -35:
            lines.append("  Best facility: Gemini-N or Keck - GNIRS or NIRES")
        else:
            lines.append("  Best facility: VLT (Paranal) or Gemini-S - KMOS or F2")

        lines.append("")

    lines.append("=" * 80)
    lines.append("Notes:")
    lines.append("- All targets are optically invisible (no Gaia, 2MASS, Pan-STARRS counterparts)")
    lines.append("- W1-W2 colors suggest extremely cold objects (Y/T dwarf candidates)")
    lines.append("- Fading behavior is unusual and warrants spectroscopic investigation")
    lines.append("- Primary spectral features to detect: CH4 (1.6, 2.2 μm), H2O, NH3")
    lines.append("=" * 80)

    output_path = output_dir / "spectroscopy_targets.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved target list to {output_path}")

    # Also print to console
    print("\n".join(lines))


def generate_proposal_summary(targets: pd.DataFrame, output_dir: Path):
    """Generate summary text for telescope proposal."""
    n_targets = len(targets)

    # Calculate statistics
    w1_w2_mean = targets["w1_w2_color"].mean() if "w1_w2_color" in targets.columns else 0
    w1_w2_max = targets["w1_w2_color"].max() if "w1_w2_color" in targets.columns else 0
    pm_mean = targets["pm_total"].mean() if "pm_total" in targets.columns else 0

    summary = f"""
TASNI Spectroscopy Proposal: Summary
=====================================

Scientific Rationale:
---------------------
We propose near-infrared spectroscopy of {n_targets} "fading thermal orphans"
identified by the TASNI (Thermal Anomaly Search for Non-communicating Intelligence)
pipeline. These sources exhibit:

1. EXTREME W1-W2 COLORS: Mean W1-W2 = {w1_w2_mean:.2f} mag (max {w1_w2_max:.2f} mag),
   consistent with Y/T dwarf temperatures (<300K)

2. OPTICAL INVISIBILITY: No counterparts in Gaia DR3, 2MASS, Pan-STARRS DR1,
   or Legacy Survey DR10 - they are detectable ONLY in mid-infrared

3. SYSTEMATIC FADING: All {n_targets} targets show monotonic dimming over 10+ years
   of NEOWISE monitoring, at rates of 0.02-0.05 mag/yr

4. HIGH PROPER MOTION: Mean PM = {pm_mean:.0f} mas/yr, indicating nearby (<100 pc) objects

Proposed Observations:
----------------------
- Instrument: Keck/NIRES or VLT/KMOS (depending on declination)
- Wavelength: 1.0-2.5 μm (J, H, K bands)
- Spectral features: CH4 (1.6, 2.2 μm), H2O (1.4, 1.9 μm), NH3 (if Y dwarf)
- Total time: ~{n_targets * 30} minutes (including overheads)

Expected Outcomes:
------------------
- Confirm/refute Y/T dwarf classification
- Measure spectral types and effective temperatures
- Constrain atmospheric composition
- Potentially discover new extremely cold brown dwarfs

Alternative Hypotheses:
-----------------------
If NOT brown dwarfs, these objects could be:
- Dust-obscured transients (but 10-year timescale is unusual)
- Cooling stellar remnants (but colors are too red)
- Something genuinely anomalous (SETI consideration, low probability)

Spectroscopy is essential to resolve these possibilities.
"""

    output_path = output_dir / "proposal_summary.txt"
    with open(output_path, "w") as f:
        f.write(summary)

    logger.info(f"Saved proposal summary to {output_path}")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="TASNI Spectroscopy Target Preparation")
    parser.add_argument(
        "--input",
        "-i",
        default="./data/processed/golden_improved.csv",
        help="Input file with targets",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./data/processed/spectroscopy",
        help="Output directory for spectroscopy materials",
    )
    parser.add_argument(
        "--cutout-dir",
        "-c",
        default="./data/processed/cutouts",
        help="Directory with cutout images",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Spectroscopy Target Preparation")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fading targets
    targets = load_fading_targets(args.input)

    if len(targets) == 0:
        logger.error("No fading targets found!")
        return

    # Compute observability
    targets = compute_observability(targets)

    # Generate outputs
    logger.info("Generating target list...")
    generate_target_list(targets, output_dir)

    logger.info("Generating LaTeX table...")
    generate_latex_table(targets, output_dir)

    logger.info("Generating proposal summary...")
    generate_proposal_summary(targets, output_dir)

    logger.info("Creating visibility plot...")
    create_visibility_plot(targets, output_dir)

    logger.info("Creating finding charts...")
    cutout_dir = Path(args.cutout_dir)
    if cutout_dir.exists():
        create_finding_charts(targets, cutout_dir, output_dir)
    else:
        logger.warning(f"Cutout directory not found: {cutout_dir}")

    # Save full target data
    output_csv = output_dir / "fading_targets_full.csv"
    targets.to_csv(output_csv, index=False)
    logger.info(f"Saved full target data to {output_csv}")

    logger.info("=" * 60)
    logger.info("Spectroscopy preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
