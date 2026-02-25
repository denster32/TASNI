#!/usr/bin/env python3
"""
Binary Model Analysis for Fading Thermal Orphans

Models the periodicity observed in fading sources to determine whether:
1. Rotational modulation (single object with surface inhomogeneities)
2. Eclipsing binary (two objects in orbit)

Key discriminants:
- Eclipsing binaries: symmetric dimming, depth related to size ratio
- Rotational modulation: asymmetric, related to cloud coverage
- Period distribution: BD binaries typically > 1 day periods

This analysis follows up on the periodogram results showing P=40-400 days.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tasni.core.config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class BinaryModelResult:
    """Result of binary model fitting."""

    designation: str
    period_days: float
    period_uncertainty: float
    depth_mag: float
    duration_fraction: float
    is_eclipsing_candidate: bool
    is_rotational_candidate: bool
    model_type: str
    chi_squared: float
    bayes_factor: float
    notes: str


def load_periodogram_results() -> pd.DataFrame:
    """Load periodogram results for fading sources."""
    periodogram_file = OUTPUT_DIR / "periodogram" / "periodogram_results.csv"
    if periodogram_file.exists():
        return pd.read_csv(periodogram_file)

    # Try alternate location
    periodogram_file = OUTPUT_DIR / "final" / "periodogram_results.csv"
    if periodogram_file.exists():
        return pd.read_csv(periodogram_file)

    # Return empty DataFrame if not found
    log.warning("Periodogram results not found")
    return pd.DataFrame()


def eclipsing_binary_model(
    times: np.ndarray,
    period: float,
    t0: float,
    depth: float,
    duration: float,
    baseline: float = 0.0,
) -> np.ndarray:
    """
    Simple eclipsing binary model.

    Args:
        times: Time array
        period: Orbital period
        t0: Time of primary eclipse
        depth: Eclipse depth (magnitudes)
        duration: Eclipse duration as fraction of period
        baseline: Baseline magnitude

    Returns:
        Model magnitudes
    """
    phase = ((times - t0) % period) / period
    # Wrap phase to [-0.5, 0.5]
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # In eclipse when |phase| < duration/2
    half_duration = duration / 2.0
    in_eclipse = np.abs(phase) < half_duration

    magnitudes = np.full_like(times, baseline)

    # Simple box model for eclipse
    magnitudes[in_eclipse] = baseline + depth

    return magnitudes


def rotational_modulation_model(
    times: np.ndarray,
    period: float,
    t0: float,
    amplitude: float,
    phase_offset: float = 0.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """
    Sinusoidal rotational modulation model.

    Args:
        times: Time array
        period: Rotation period
        t0: Reference time
        amplitude: Modulation amplitude (magnitudes)
        phase_offset: Phase offset
        baseline: Baseline magnitude

    Returns:
        Model magnitudes
    """
    phase = 2 * np.pi * (times - t0) / period + phase_offset
    magnitudes = baseline + amplitude * np.sin(phase)
    return magnitudes


def fit_binary_model(
    times: np.ndarray,
    mags: np.ndarray,
    mag_errs: np.ndarray | None = None,
    initial_period: float = 100.0,
) -> dict[str, Any]:
    """
    Fit both eclipsing binary and rotational models.

    Args:
        times: Time array (MJD)
        mags: Magnitude array
        mag_errs: Magnitude uncertainties
        initial_period: Initial period guess

    Returns:
        Dictionary with fitting results
    """
    results = {}

    # Remove NaN
    mask = np.isfinite(times) & np.isfinite(mags)
    if mag_errs is not None:
        mask &= np.isfinite(mag_errs)
        mag_errs = mag_errs[mask]

    times = times[mask]
    mags = mags[mask]

    if len(times) < 10:
        return {"status": "insufficient_data", "n_points": len(times)}

    baseline = np.median(mags)
    amplitude = (np.max(mags) - np.min(mags)) / 2.0

    # Model 1: Rotational (sinusoidal)
    try:

        def rot_residuals(params):
            period, phase, amp = params
            model = rotational_modulation_model(times, period, times[0], amp, phase, baseline)
            if mag_errs is not None:
                return np.sum(((mags - model) / mag_errs) ** 2)
            return np.sum((mags - model) ** 2)

        # Simple grid search for period
        periods = np.linspace(10, 500, 50)
        best_chi2 = np.inf
        best_period = initial_period

        for p in periods:
            chi2 = rot_residuals([p, 0, amplitude])
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_period = p

        results["rotational"] = {
            "period": best_period,
            "amplitude": amplitude,
            "chi_squared": best_chi2,
            "status": "success",
        }
    except Exception as e:
        results["rotational"] = {"status": "failed", "error": str(e)}

    # Model 2: Eclipsing binary (box model)
    try:

        def eclipse_residuals(params):
            period, t0, depth, duration = params
            if duration < 0.01 or duration > 0.5:  # Physical constraints
                return np.inf
            model = eclipsing_binary_model(times, period, t0, depth, duration, baseline)
            if mag_errs is not None:
                return np.sum(((mags - model) / mag_errs) ** 2)
            return np.sum((mags - model) ** 2)

        # Grid search
        best_chi2_ecl = np.inf
        best_params = None

        for p in [initial_period, initial_period * 2, initial_period / 2]:
            for dur in [0.05, 0.1, 0.2]:
                chi2 = eclipse_residuals([p, times[0], amplitude, dur])
                if chi2 < best_chi2_ecl:
                    best_chi2_ecl = chi2
                    best_params = (p, times[0], amplitude, dur)

        if best_params is not None:
            results["eclipsing"] = {
                "period": best_params[0],
                "depth": best_params[2],
                "duration_fraction": best_params[3],
                "chi_squared": best_chi2_ecl,
                "status": "success",
            }
        else:
            results["eclipsing"] = {"status": "failed"}

    except Exception as e:
        results["eclipsing"] = {"status": "failed", "error": str(e)}

    # Compare models
    if "rotational" in results and "eclipsing" in results:
        if (
            results["rotational"]["status"] == "success"
            and results["eclipsing"]["status"] == "success"
        ):
            chi2_rot = results["rotational"]["chi_squared"]
            chi2_ecl = results["eclipsing"]["chi_squared"]

            # Bayes factor approximation (simplified)
            delta_chi2 = chi2_ecl - chi2_rot
            bayes_factor = np.exp(-0.5 * delta_chi2)

            results["comparison"] = {
                "preferred_model": "rotational" if chi2_rot < chi2_ecl else "eclipsing",
                "delta_chi_squared": delta_chi2,
                "bayes_factor": bayes_factor,
            }

    results["n_points"] = len(times)
    results["baseline_mag"] = baseline
    results["amplitude_mag"] = amplitude

    return results


def analyze_fading_sources(
    fading_designations: list[str], epochs_file: Path | None = None
) -> list[BinaryModelResult]:
    """
    Analyze all fading sources for binary vs rotational classification.

    Args:
        fading_designations: List of fading source designations
        epochs_file: Path to NEOWISE epochs file

    Returns:
        List of BinaryModelResult objects
    """
    results = []

    # Load epochs
    if epochs_file is None:
        epochs_file = OUTPUT_DIR / "final" / "neowise_epochs.parquet"

    if not epochs_file.exists():
        log.error(f"Epochs file not found: {epochs_file}")
        return results

    epochs = pd.read_parquet(epochs_file)
    log.info(f"Loaded {len(epochs)} epoch records")

    for designation in fading_designations:
        log.info(f"Analyzing {designation}...")

        # Get epochs for this source
        source_epochs = epochs[epochs["designation"] == designation].copy()

        if len(source_epochs) < 10:
            log.warning(f"Insufficient epochs for {designation}: {len(source_epochs)}")
            continue

        # Get columns
        w1_col = "w1mpro_ep" if "w1mpro_ep" in source_epochs.columns else "w1mpro"

        times = source_epochs["mjd"].values
        mags = source_epochs[w1_col].values
        mag_errs = source_epochs.get(
            f'{w1_col.replace("mpro", "sigmpro")}', source_epochs.get("w1sigmpro_ep", None)
        )

        if mag_errs is not None:
            mag_errs = mag_errs.values if hasattr(mag_errs, "values") else mag_errs

        # Fit models
        fit_results = fit_binary_model(times, mags, mag_errs)

        # Create result
        if "comparison" in fit_results:
            model_type = fit_results["comparison"]["preferred_model"]
            is_eclipse = model_type == "eclipsing"
            is_rot = model_type == "rotational"
            bayes = fit_results["comparison"]["bayes_factor"]
            chi2 = fit_results[model_type]["chi_squared"]
        else:
            model_type = "unknown"
            is_eclipse = False
            is_rot = False
            bayes = 0.0
            chi2 = 0.0

        period = fit_results.get("rotational", {}).get("period", 0)
        if period == 0:
            period = fit_results.get("eclipsing", {}).get("period", 0)

        result = BinaryModelResult(
            designation=designation,
            period_days=period,
            period_uncertainty=period * 0.1,  # 10% heuristic; use periodogram CI when available
            depth_mag=fit_results.get("amplitude_mag", 0),
            duration_fraction=fit_results.get("eclipsing", {}).get("duration_fraction", 0),
            is_eclipsing_candidate=is_eclipse,
            is_rotational_candidate=is_rot,
            model_type=model_type,
            chi_squared=chi2,
            bayes_factor=bayes,
            notes=f"N={fit_results.get('n_points', 0)} epochs",
        )
        results.append(result)

    return results


def run_binary_analysis():
    """Run complete binary model analysis."""
    print("=" * 70)
    print("TASNI Binary Model Analysis")
    print("=" * 70)

    # Load periodogram results to get fading sources
    periodogram_df = load_periodogram_results()

    if len(periodogram_df) == 0:
        # Use known fading sources
        fading_sources = [
            "J143046.35-025927.8",
            "J231029.40-060547.3",
            "J193547.43+601201.5",
            "J060501.01-545944.5",
        ]
        print(f"\nUsing known fading sources: {fading_sources}")
    else:
        # Filter for periodic sources
        fading_sources = periodogram_df[
            (periodogram_df["w1_is_periodic"] == True) | (periodogram_df["w2_is_periodic"] == True)
        ]["designation"].tolist()
        print(f"\nFound {len(fading_sources)} periodic fading sources")

    # Run analysis
    results = analyze_fading_sources(fading_sources)

    # Print results
    print("\n" + "=" * 70)
    print("BINARY MODEL ANALYSIS RESULTS")
    print("=" * 70)

    for r in results:
        print(f"\n{r.designation}:")
        print(f"  Period: {r.period_days:.1f} +/- {r.period_uncertainty:.1f} days")
        print(f"  Model: {r.model_type}")
        print(f"  Eclipsing candidate: {r.is_eclipsing_candidate}")
        print(f"  Rotational candidate: {r.is_rotational_candidate}")
        print(f"  Chi-squared: {r.chi_squared:.2f}")
        print(f"  Bayes factor: {r.bayes_factor:.2f}")
        print(f"  Notes: {r.notes}")

    # Save results
    results_df = pd.DataFrame(
        [
            {
                "designation": r.designation,
                "period_days": r.period_days,
                "period_unc": r.period_uncertainty,
                "depth_mag": r.depth_mag,
                "duration_frac": r.duration_fraction,
                "is_eclipsing": r.is_eclipsing_candidate,
                "is_rotational": r.is_rotational_candidate,
                "model_type": r.model_type,
                "chi_squared": r.chi_squared,
                "bayes_factor": r.bayes_factor,
                "notes": r.notes,
            }
            for r in results
        ]
    )

    output_file = OUTPUT_DIR / "analysis" / "binary_model_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Summary
    n_eclipsing = sum(1 for r in results if r.is_eclipsing_candidate)
    n_rotational = sum(1 for r in results if r.is_rotational_candidate)
    print("\nSummary:")
    print(f"  Eclipsing binary candidates: {n_eclipsing}")
    print(f"  Rotational modulation candidates: {n_rotational}")
    print(f"  Total analyzed: {len(results)}")

    return results


if __name__ == "__main__":
    run_binary_analysis()
