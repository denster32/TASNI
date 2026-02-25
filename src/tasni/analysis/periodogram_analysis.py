#!/usr/bin/env python3
"""
TASNI: Periodogram Analysis of NEOWISE Light Curves

Computes Lomb-Scargle periodograms to search for periodic signals in
multi-epoch NEOWISE photometry.

Scientific motivation:
- Binary brown dwarfs: orbital periods (days to years)
- Rotation: cloud modulation (hours to days)
- Eclipsing systems: periodic dimming
- Artificial signals: regular modulation (SETI consideration)

Usage:
    python periodogram_analysis.py [--epochs FILE] [--output DIR]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from astropy.timeseries import LombScargle

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

try:
    from tasni.analysis.periodogram_significance import assess_alias_probability
except Exception:
    assess_alias_probability = None

try:
    from tasni.analysis.statistical_analysis import fdr_correction
except Exception:
    fdr_correction = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [PERIODOGRAM] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Period search range
MIN_PERIOD_DAYS = 0.5  # 12 hours (rotation minimum)
MAX_PERIOD_DAYS = 1000.0  # ~3 years (long-term variability)
N_PERIODS = 10000  # Number of trial periods

# Significance threshold
FAP_THRESHOLD = 0.01  # False Alarm Probability < 1%
ALIAS_PROB_THRESHOLD = 0.6


def compute_periodogram(times: np.ndarray, mags: np.ndarray, mag_errs: np.ndarray = None) -> dict:
    """
    Compute Lomb-Scargle periodogram for a light curve.

    Args:
        times: MJD times
        mags: Magnitudes
        mag_errs: Magnitude uncertainties (optional)

    Returns:
        Dictionary with periodogram results
    """
    if not HAS_ASTROPY:
        logger.error("astropy not installed")
        return {}

    # Remove NaN values
    mask = np.isfinite(mags) & np.isfinite(times)
    if mag_errs is not None:
        mask &= np.isfinite(mag_errs)

    times = times[mask]
    mags = mags[mask]
    if mag_errs is not None:
        mag_errs = mag_errs[mask]

    if len(times) < 10:
        return {"n_points": len(times), "status": "insufficient_data"}

    # Create frequency grid
    min_freq = 1.0 / MAX_PERIOD_DAYS
    max_freq = 1.0 / MIN_PERIOD_DAYS
    frequencies = np.linspace(min_freq, max_freq, N_PERIODS)

    # Compute Lomb-Scargle periodogram
    if mag_errs is not None and len(mag_errs) > 0:
        ls = LombScargle(times, mags, mag_errs)
    else:
        ls = LombScargle(times, mags)

    power = ls.power(frequencies)

    # Find best period
    best_idx = np.argmax(power)
    best_freq = frequencies[best_idx]
    best_period = 1.0 / best_freq
    best_power = power[best_idx]

    # Compute False Alarm Probability
    fap = ls.false_alarm_probability(best_power)

    # Find secondary peaks (avoiding aliases of best period)
    secondary_periods = []
    for i in range(3):
        # Mask around existing peaks
        mask = np.ones(len(power), dtype=bool)
        for p in [best_period] + secondary_periods:
            f = 1.0 / p
            mask &= np.abs(frequencies - f) > 0.1 * f  # Exclude ±10% of peak
            # Also exclude harmonics
            mask &= np.abs(frequencies - 2 * f) > 0.1 * f
            mask &= np.abs(frequencies - 0.5 * f) > 0.05 * f

        if mask.sum() > 0:
            masked_power = power.copy()
            masked_power[~mask] = 0
            sec_idx = np.argmax(masked_power)
            sec_period = 1.0 / frequencies[sec_idx]
            sec_power = power[sec_idx]
            if sec_power > 0.3 * best_power:  # Only if significant
                secondary_periods.append(sec_period)

    return {
        "n_points": len(times),
        "baseline_days": times.max() - times.min(),
        "frequencies": frequencies,
        "power": power,
        "best_period_days": best_period,
        "best_power": best_power,
        "fap": fap,
        "is_significant": fap < FAP_THRESHOLD,
        "secondary_periods": secondary_periods,
        "status": "success",
    }


def _bh_fdr(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal Benjamini-Hochberg fallback when shared utility import is unavailable.
    """
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    order = np.argsort(p)
    ranked = p[order]
    critical = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= critical
    rejected = np.zeros(n, dtype=bool)
    if np.any(below):
        k = int(np.max(np.where(below)[0]))
        rejected[order[: k + 1]] = True

    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        prev = val
        adjusted[order[i]] = val
    return rejected, np.clip(adjusted, 0.0, 1.0)


def apply_multiple_testing_correction(
    results: list[dict], alpha: float = FAP_THRESHOLD
) -> list[dict]:
    """
    Apply global FDR correction (across all analyzed sources) per band.
    """
    for band in ("w1", "w2"):
        entries: list[tuple[int, float]] = []
        for idx, item in enumerate(results):
            fap = item.get(f"{band}_fap")
            if fap is not None and np.isfinite(fap):
                entries.append((idx, float(fap)))

        if not entries:
            continue

        pvals = np.array([value for _, value in entries], dtype=float)
        if callable(fdr_correction):
            rejected, adjusted = fdr_correction(pvals, alpha=alpha, method="bh")
        else:
            rejected, adjusted = _bh_fdr(pvals, alpha)

        for local_idx, (result_idx, raw_p) in enumerate(entries):
            item = results[result_idx]
            item[f"{band}_raw_fap"] = raw_p
            item[f"{band}_fdr_p"] = float(adjusted[local_idx])
            item[f"{band}_is_periodic_raw"] = bool(raw_p < alpha)
            item[f"{band}_is_periodic_fdr"] = bool(rejected[local_idx])
            alias_flag = bool(item.get(f"{band}_alias_flag", False))
            item[f"{band}_is_periodic"] = bool(rejected[local_idx] and not alias_flag)

    # Consistency check should be based on corrected significance.
    for item in results:
        if item.get("w1_is_periodic") and item.get("w2_is_periodic"):
            p1 = item.get("w1_best_period")
            p2 = item.get("w2_best_period")
            if p1 and p2 and np.isfinite(p1) and np.isfinite(p2):
                ratio = p1 / p2
                item["consistent_period"] = bool(0.9 < ratio < 1.1)
                if item["consistent_period"]:
                    item["mean_period"] = float((p1 + p2) / 2.0)
            else:
                item["consistent_period"] = False
        else:
            item["consistent_period"] = False

    return results


def analyze_source(designation: str, epochs: pd.DataFrame) -> dict:
    """
    Analyze periodogram for a single source.

    Args:
        designation: Source designation
        epochs: DataFrame with epochs for this source

    Returns:
        Dictionary with analysis results
    """
    result = {"designation": designation, "n_epochs": len(epochs)}

    if len(epochs) < 10:
        result["status"] = "insufficient_epochs"
        return result

    # Get times and magnitudes
    times = epochs["mjd"].values

    # Analyze W1 band
    if "w1mpro_ep" in epochs.columns:
        w1_mags = epochs["w1mpro_ep"].values
        w1_errs = epochs["w1sigmpro_ep"].values if "w1sigmpro_ep" in epochs.columns else None

        w1_result = compute_periodogram(times, w1_mags, w1_errs)

        if w1_result.get("status") == "success":
            if callable(assess_alias_probability):
                alias_prob = assess_alias_probability(w1_result["best_period_days"], time=times)
            else:
                alias_prob = 0.0

            result["w1_best_period"] = w1_result["best_period_days"]
            result["w1_best_power"] = w1_result["best_power"]
            result["w1_fap"] = w1_result["fap"]
            result["w1_is_periodic_raw"] = w1_result["is_significant"]
            result["w1_alias_probability"] = float(alias_prob)
            result["w1_alias_flag"] = bool(alias_prob >= ALIAS_PROB_THRESHOLD)
            result["w1_periodogram"] = w1_result

    # Analyze W2 band
    if "w2mpro_ep" in epochs.columns:
        w2_mags = epochs["w2mpro_ep"].values
        w2_errs = epochs["w2sigmpro_ep"].values if "w2sigmpro_ep" in epochs.columns else None

        w2_result = compute_periodogram(times, w2_mags, w2_errs)

        if w2_result.get("status") == "success":
            if callable(assess_alias_probability):
                alias_prob = assess_alias_probability(w2_result["best_period_days"], time=times)
            else:
                alias_prob = 0.0

            result["w2_best_period"] = w2_result["best_period_days"]
            result["w2_best_power"] = w2_result["best_power"]
            result["w2_fap"] = w2_result["fap"]
            result["w2_is_periodic_raw"] = w2_result["is_significant"]
            result["w2_alias_probability"] = float(alias_prob)
            result["w2_alias_flag"] = bool(alias_prob >= ALIAS_PROB_THRESHOLD)
            result["w2_periodogram"] = w2_result

    result["status"] = "success"
    return result


def plot_periodogram(result: dict, output_dir: Path):
    """Plot periodogram for a source."""
    if not HAS_MATPLOTLIB:
        return

    designation = result["designation"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # W1 periodogram
    if "w1_periodogram" in result:
        w1 = result["w1_periodogram"]
        w1_sig = result.get("w1_is_periodic", w1.get("is_significant", False))
        w1_fdr = result.get("w1_fdr_p", np.nan)
        w1_alias = result.get("w1_alias_probability", np.nan)
        periods = 1.0 / w1["frequencies"]

        ax = axes[0, 0]
        ax.semilogx(periods, w1["power"], "b-", linewidth=0.5)
        ax.axvline(
            w1["best_period_days"],
            color="red",
            linestyle="--",
            label=f"Best: {w1['best_period_days']:.2f} d",
        )
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Lomb-Scargle Power")
        ax.set_title(f'W1 Periodogram (raw FAP={w1["fap"]:.2e}, FDR={w1_fdr:.2e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase-folded light curve
        ax = axes[1, 0]
        # Would need original data to plot phase-folded
        ax.text(
            0.5,
            0.5,
            f'Best Period: {w1["best_period_days"]:.2f} days\n'
            f'Power: {w1["best_power"]:.3f}\n'
            f'raw FAP: {w1["fap"]:.2e}\n'
            f"FDR p: {w1_fdr:.2e}\n"
            f"alias prob: {w1_alias:.2f}\n"
            f"Significant (corrected): {w1_sig}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("W1 Period Analysis")
        ax.axis("off")

    # W2 periodogram
    if "w2_periodogram" in result:
        w2 = result["w2_periodogram"]
        w2_sig = result.get("w2_is_periodic", w2.get("is_significant", False))
        w2_fdr = result.get("w2_fdr_p", np.nan)
        w2_alias = result.get("w2_alias_probability", np.nan)
        periods = 1.0 / w2["frequencies"]

        ax = axes[0, 1]
        ax.semilogx(periods, w2["power"], "r-", linewidth=0.5)
        ax.axvline(
            w2["best_period_days"],
            color="blue",
            linestyle="--",
            label=f"Best: {w2['best_period_days']:.2f} d",
        )
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Lomb-Scargle Power")
        ax.set_title(f'W2 Periodogram (raw FAP={w2["fap"]:.2e}, FDR={w2_fdr:.2e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.text(
            0.5,
            0.5,
            f'Best Period: {w2["best_period_days"]:.2f} days\n'
            f'Power: {w2["best_power"]:.3f}\n'
            f'raw FAP: {w2["fap"]:.2e}\n'
            f"FDR p: {w2_fdr:.2e}\n"
            f"alias prob: {w2_alias:.2f}\n"
            f"Significant (corrected): {w2_sig}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("W2 Period Analysis")
        ax.axis("off")

    plt.suptitle(
        f"{designation} - Periodogram Analysis\n" f'({result["n_epochs"]} epochs)', fontsize=14
    )
    plt.tight_layout()

    output_path = output_dir / f"{designation}_periodogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved periodogram plot: {output_path}")


def plot_combined_periodograms(results: list, output_dir: Path):
    """Create combined periodogram plot for fading sources."""
    if not HAS_MATPLOTLIB:
        return

    # Filter to sources with valid periodograms
    valid_results = [r for r in results if "w1_periodogram" in r]

    if len(valid_results) == 0:
        logger.warning("No valid periodograms to plot")
        return

    n_sources = len(valid_results)
    fig, axes = plt.subplots(n_sources, 2, figsize=(14, 4 * n_sources))

    if n_sources == 1:
        axes = axes.reshape(1, -1)

    for idx, result in enumerate(valid_results):
        designation = result["designation"]

        # W1
        if "w1_periodogram" in result:
            w1 = result["w1_periodogram"]
            w1_sig_corr = bool(result.get("w1_is_periodic", False))
            w1_fdr = result.get("w1_fdr_p", np.nan)
            w1_alias = result.get("w1_alias_probability", np.nan)
            periods = 1.0 / w1["frequencies"]

            ax = axes[idx, 0]
            ax.semilogx(periods, w1["power"], "b-", linewidth=0.5)
            ax.axvline(w1["best_period_days"], color="red", linestyle="--", alpha=0.7)

            # Mark significance
            if w1_sig_corr:
                ax.set_facecolor("#ffe6e6")
                sig_text = f"SIGNIFICANT (FDR={w1_fdr:.1e}, alias={w1_alias:.2f})"
            else:
                sig_text = f"Not significant (FDR={w1_fdr:.1e}, alias={w1_alias:.2f})"

            ax.set_xlabel("Period (days)")
            ax.set_ylabel("Power")
            ax.set_title(f'{designation} W1\nBest: {w1["best_period_days"]:.1f}d, {sig_text}')
            ax.grid(True, alpha=0.3)

        # W2
        if "w2_periodogram" in result:
            w2 = result["w2_periodogram"]
            w2_sig_corr = bool(result.get("w2_is_periodic", False))
            w2_fdr = result.get("w2_fdr_p", np.nan)
            w2_alias = result.get("w2_alias_probability", np.nan)
            periods = 1.0 / w2["frequencies"]

            ax = axes[idx, 1]
            ax.semilogx(periods, w2["power"], "r-", linewidth=0.5)
            ax.axvline(w2["best_period_days"], color="blue", linestyle="--", alpha=0.7)

            if w2_sig_corr:
                ax.set_facecolor("#ffe6e6")
                sig_text = f"SIGNIFICANT (FDR={w2_fdr:.1e}, alias={w2_alias:.2f})"
            else:
                sig_text = f"Not significant (FDR={w2_fdr:.1e}, alias={w2_alias:.2f})"

            ax.set_xlabel("Period (days)")
            ax.set_ylabel("Power")
            ax.set_title(f'{designation} W2\nBest: {w2["best_period_days"]:.1f}d, {sig_text}')
            ax.grid(True, alpha=0.3)

    plt.suptitle("TASNI Fading Sources: Periodogram Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / "fading_periodograms.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved combined periodogram plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TASNI Periodogram Analysis")
    parser.add_argument(
        "--epochs",
        "-e",
        default="./data/processed/neowise_epochs.parquet",
        help="Input file with NEOWISE epochs",
    )
    parser.add_argument(
        "--targets",
        "-t",
        default="./data/processed/golden_targets.csv",
        help="Target list (to identify fading sources)",
    )
    parser.add_argument(
        "--output", "-o", default="./data/processed/periodogram", help="Output directory"
    )
    parser.add_argument("--fading-only", action="store_true", help="Only analyze fading sources")
    parser.add_argument("--all", action="store_true", help="Analyze all golden targets")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Periodogram Analysis")
    logger.info("=" * 60)

    if not HAS_ASTROPY:
        logger.error("astropy not installed. Run: pip install astropy")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load epochs
    logger.info(f"Loading epochs from {args.epochs}")
    if args.epochs.endswith(".parquet"):
        epochs = pd.read_parquet(args.epochs)
    else:
        epochs = pd.read_csv(args.epochs)

    logger.info(f"Loaded {len(epochs)} epochs for {epochs['designation'].nunique()} sources")

    # Load targets to identify fading sources
    logger.info(f"Loading targets from {args.targets}")
    if args.targets.endswith(".parquet"):
        targets = pd.read_parquet(args.targets)
    else:
        targets = pd.read_csv(args.targets)

    # Identify fading sources
    if "variability_flag" in targets.columns:
        fading_designations = targets[targets["variability_flag"] == "FADING"][
            "designation"
        ].tolist()
    elif "trend_type" in targets.columns:
        fading_designations = targets[targets["trend_type"] == "fading"]["designation"].tolist()
    else:
        fading_designations = []

    # Note: J044024.40-731441.6 is identified as an LMC member and excluded
    # from the confirmed fading thermal orphans in the paper, but we still
    # analyze it here for completeness

    logger.info(f"Identified {len(fading_designations)} fading sources")

    # Determine which sources to analyze
    if args.fading_only or (not args.all):
        designations_to_analyze = fading_designations
        logger.info("Analyzing fading sources only")
    else:
        designations_to_analyze = epochs["designation"].unique().tolist()
        logger.info(f"Analyzing all {len(designations_to_analyze)} sources")

    # Run periodogram analysis
    results = []
    for idx, designation in enumerate(designations_to_analyze):
        source_epochs = epochs[epochs["designation"] == designation]

        logger.info(
            f"[{idx+1}/{len(designations_to_analyze)}] Analyzing {designation} "
            f"({len(source_epochs)} epochs)"
        )

        result = analyze_source(designation, source_epochs)
        results.append(result)

    # Global multiple-testing correction (per band) and alias filtering.
    results = apply_multiple_testing_correction(results, alpha=FAP_THRESHOLD)

    # Create combined plot for fading sources
    fading_results = [r for r in results if r["designation"] in fading_designations]
    for r in fading_results:
        if r.get("status") == "success":
            plot_periodogram(r, output_dir)
    if fading_results:
        plot_combined_periodograms(fading_results, output_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("Periodogram Analysis Summary")
    logger.info("=" * 60)

    n_periodic_w1 = sum(1 for r in results if r.get("w1_is_periodic", False))
    n_periodic_w2 = sum(1 for r in results if r.get("w2_is_periodic", False))
    n_consistent = sum(1 for r in results if r.get("consistent_period", False))

    logger.info(f"Sources analyzed: {len(results)}")
    n_periodic_w1_raw = sum(1 for r in results if r.get("w1_is_periodic_raw", False))
    n_periodic_w2_raw = sum(1 for r in results if r.get("w2_is_periodic_raw", False))
    n_periodic_w1_fdr = sum(1 for r in results if r.get("w1_is_periodic_fdr", False))
    n_periodic_w2_fdr = sum(1 for r in results if r.get("w2_is_periodic_fdr", False))
    n_alias_w1 = sum(1 for r in results if r.get("w1_alias_flag", False))
    n_alias_w2 = sum(1 for r in results if r.get("w2_alias_flag", False))

    logger.info(f"Significant W1 periodicity (raw): {n_periodic_w1_raw}")
    logger.info(f"Significant W2 periodicity (raw): {n_periodic_w2_raw}")
    logger.info(f"Significant W1 periodicity (FDR): {n_periodic_w1_fdr}")
    logger.info(f"Significant W2 periodicity (FDR): {n_periodic_w2_fdr}")
    logger.info(f"Likely cadence aliases (W1/W2): {n_alias_w1}/{n_alias_w2}")
    logger.info(f"Significant W1 periodicity (FDR + alias-filtered): {n_periodic_w1}")
    logger.info(f"Significant W2 periodicity (FDR + alias-filtered): {n_periodic_w2}")
    logger.info(f"Consistent period (both bands): {n_consistent}")

    # Print details for periodic sources
    periodic_sources = [r for r in results if r.get("w1_is_periodic") or r.get("w2_is_periodic")]
    if periodic_sources:
        logger.info("\nSources with significant periodicity:")
        for r in periodic_sources:
            logger.info(f"  {r['designation']}:")
            if r.get("w1_is_periodic"):
                logger.info(
                    f"    W1: P={r['w1_best_period']:.2f}d, "
                    f"raw_FAP={r['w1_fap']:.2e}, FDR_p={r.get('w1_fdr_p', np.nan):.2e}, "
                    f"alias_prob={r.get('w1_alias_probability', np.nan):.2f}"
                )
            if r.get("w2_is_periodic"):
                logger.info(
                    f"    W2: P={r['w2_best_period']:.2f}d, "
                    f"raw_FAP={r['w2_fap']:.2e}, FDR_p={r.get('w2_fdr_p', np.nan):.2e}, "
                    f"alias_prob={r.get('w2_alias_probability', np.nan):.2f}"
                )
    else:
        logger.info("\nNo sources with significant periodicity detected")

    # Print fading source summary
    logger.info("\nFading Sources Periodogram Summary:")
    for r in fading_results:
        designation = r["designation"]
        w1_period = r.get("w1_best_period", 0)
        w1_fap = r.get("w1_fap", 1)
        w1_sig = "YES" if r.get("w1_is_periodic") else "no"
        w1_fdr = r.get("w1_fdr_p", np.nan)
        w1_alias = r.get("w1_alias_probability", np.nan)
        w2_period = r.get("w2_best_period", 0)
        w2_fap = r.get("w2_fap", 1)
        w2_sig = "YES" if r.get("w2_is_periodic") else "no"
        w2_fdr = r.get("w2_fdr_p", np.nan)
        w2_alias = r.get("w2_alias_probability", np.nan)

        logger.info(f"  {designation}:")
        logger.info(
            "    W1: Best P=%.1fd, raw_FAP=%.2e, FDR_p=%.2e, alias_prob=%.2f, Significant=%s",
            w1_period,
            w1_fap,
            w1_fdr,
            w1_alias,
            w1_sig,
        )
        logger.info(
            "    W2: Best P=%.1fd, raw_FAP=%.2e, FDR_p=%.2e, alias_prob=%.2f, Significant=%s",
            w2_period,
            w2_fap,
            w2_fdr,
            w2_alias,
            w2_sig,
        )

    logger.info("=" * 60)

    # Save results
    summary_data = []
    for r in results:
        row = {
            "designation": r["designation"],
            "n_epochs": r.get("n_epochs", 0),
            "w1_best_period": r.get("w1_best_period", np.nan),
            "w1_best_power": r.get("w1_best_power", np.nan),
            "w1_fap": r.get("w1_fap", np.nan),
            "w1_fdr_p": r.get("w1_fdr_p", np.nan),
            "w1_alias_probability": r.get("w1_alias_probability", np.nan),
            "w1_alias_flag": r.get("w1_alias_flag", False),
            "w1_is_periodic_raw": r.get("w1_is_periodic_raw", False),
            "w1_is_periodic_fdr": r.get("w1_is_periodic_fdr", False),
            "w1_is_periodic": r.get("w1_is_periodic", False),
            "w2_best_period": r.get("w2_best_period", np.nan),
            "w2_best_power": r.get("w2_best_power", np.nan),
            "w2_fap": r.get("w2_fap", np.nan),
            "w2_fdr_p": r.get("w2_fdr_p", np.nan),
            "w2_alias_probability": r.get("w2_alias_probability", np.nan),
            "w2_alias_flag": r.get("w2_alias_flag", False),
            "w2_is_periodic_raw": r.get("w2_is_periodic_raw", False),
            "w2_is_periodic_fdr": r.get("w2_is_periodic_fdr", False),
            "w2_is_periodic": r.get("w2_is_periodic", False),
            "consistent_period": r.get("consistent_period", False),
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "periodogram_results.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved results to {summary_path}")

    return results


if __name__ == "__main__":
    main()
