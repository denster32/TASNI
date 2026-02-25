"""
Periodogram Significance Analysis for TASNI

This module calculates False Alarm Probabilities (FAP) for Lomb-Scargle
periodograms and assesses aliasing effects from NEOWISE cadence.

Author: Dennis Palucki
Date: February 4, 2026
"""

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle


def calculate_lomb_scargle_periodogram(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Lomb-Scargle periodogram.

    Parameters:
    -----------
    time : np.ndarray
        Time array (in days)
    flux : np.ndarray
        Flux array (in magnitudes)
    flux_err : np.ndarray, optional
        Flux error array

    Returns:
    --------
    frequency : np.ndarray
        Frequency array
    power : np.ndarray
        Power array
    """
    if flux_err is None:
        flux_err = np.ones_like(flux) * np.median(flux) * 0.01

    ls = LombScargle(
        time,
        flux,
        flux_err,
        normalization="standard",
        minimum_frequency=1 / 365.25,  # 1 year period
        maximum_frequency=1 / 10.0,  # 10 day period
        samples_per_peak=10,
    )

    return ls.frequency, ls.power


def calculate_fap_analytical(power: np.ndarray, n_independent: int) -> np.ndarray:
    """
    Calculate analytical False Alarm Probability using Horne & Baliunas (1986).

    Parameters:
    -----------
    power : np.ndarray
        Power spectrum from periodogram
    n_independent : int
        Number of independent frequencies

    Returns:
    --------
    fap : np.ndarray
        False Alarm Probability for each power value
    """
    # Normalize power
    z = power / np.mean(power)

    # Calculate FAP using exponential distribution
    fap = 1.0 - (1.0 - np.exp(-z)) ** n_independent

    return fap


def bootstrap_periodogram_fap(
    time: np.ndarray, flux: np.ndarray, n_bootstrap: int = 1000
) -> dict[str, np.ndarray]:
    """
    Calculate FAP using bootstrap resampling of light curves.

    Parameters:
    -----------
    time : np.ndarray
        Time array (in days)
    flux : np.ndarray
        Flux array (in magnitudes)
    n_bootstrap : int
        Number of bootstrap iterations

    Returns:
    --------
    results : dict
        Dictionary with FAP arrays and statistics
    """
    n_freq = 1000
    bootstrap_power = np.zeros((n_bootstrap, n_freq))

    for i in range(n_bootstrap):
        # Resample light curve with replacement
        indices = np.random.choice(len(flux), size=len(flux), replace=True)
        flux_sample = flux[indices]

        # Calculate periodogram
        freq, power = calculate_lomb_scargle_periodogram(time, flux_sample)
        bootstrap_power[i, :] = power

    # Calculate FAP as fraction of bootstrap power exceeding observed power
    freq, observed_power = calculate_lomb_scargle_periodogram(time, flux)
    fap = np.sum(bootstrap_power > observed_power[np.newaxis, :], axis=0) / n_bootstrap

    results = {"frequency": freq, "fap_bootstrap": fap, "n_bootstrap": n_bootstrap}

    return results


def assess_6_month_aliasing(period: float) -> float:
    """
    Assess whether a detected period is likely a 6-month alias.

    NEOWISE has a 6-month observing cadence, which can create
    aliases at harmonics of the true period.

    Parameters:
    -----------
    period : float
        Detected period in days

    Returns:
    --------
    alias_probability : float
        Probability that period is a 6-month alias
    """
    # 6-month cadence in days
    cadence = 182.625

    # Check for harmonics
    harmonics = [cadence / n for n in range(1, 10)]

    # Calculate probability of being an alias
    alias_probability = 0.0
    for h in harmonics:
        if abs(period - h) / period < 0.1:  # Within 10%
            alias_probability = max(alias_probability, 1.0 / (h + 1))

    return alias_probability


def compute_window_function(
    time: np.ndarray,
    min_period: float = 10.0,
    max_period: float = 400.0,
    n_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the sampling window-function periodogram for uneven cadence.

    This captures cadence-driven periodic structure independent of photometric values.
    """
    if len(time) < 5:
        return np.array([]), np.array([])

    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 5:
        return np.array([]), np.array([])

    # Normalize to avoid very large absolute times impacting numerical conditioning.
    t = t - np.nanmin(t)
    freq = np.linspace(1.0 / max_period, 1.0 / min_period, n_samples)

    # Window function: all observations have unit weight/flux.
    ls = LombScargle(t, np.ones_like(t), fit_mean=False, center_data=False)
    window_power = ls.power(freq)
    return freq, window_power


def assess_alias_probability(
    period: float,
    time: np.ndarray | None = None,
    cadence_days: float = 182.625,
    tolerance_fraction: float = 0.10,
) -> float:
    """
    Estimate probability that a period is cadence-induced aliasing.

    Combines:
    1) harmonic proximity to cadence aliases (P = cadence/n), and
    2) window-function power at candidate/alias frequencies when time sampling is provided.
    """
    if not np.isfinite(period) or period <= 0:
        return 0.0

    harmonics = np.array([cadence_days / n for n in range(1, 10)], dtype=float)
    rel_sep = np.abs(period - harmonics) / harmonics
    nearest_rel_sep = float(np.min(rel_sep))

    # 1. Harmonic proximity score
    harmonic_score = float(np.exp(-((nearest_rel_sep / max(tolerance_fraction, 1e-6)) ** 2)))

    # 2. Window-function support score
    window_score = 0.0
    if time is not None and len(time) >= 5:
        freq, power = compute_window_function(np.asarray(time))
        if len(freq) > 0 and len(power) > 0 and np.nanmax(power) > 0:
            candidate_freq = 1.0 / period
            harmonic_freq = 1.0 / harmonics[np.argmin(rel_sep)]

            idx_candidate = int(np.argmin(np.abs(freq - candidate_freq)))
            idx_harmonic = int(np.argmin(np.abs(freq - harmonic_freq)))

            # Normalize by global max to get [0, 1]
            denom = float(np.nanmax(power))
            window_score = float(max(power[idx_candidate], power[idx_harmonic]) / denom)

    # Weighted fusion with clipping
    alias_probability = float(np.clip(0.65 * harmonic_score + 0.35 * window_score, 0.0, 1.0))
    return alias_probability


def find_significant_periods(
    time: np.ndarray,
    flux: np.ndarray,
    fap_threshold: float = 0.01,
    min_period: float = 10.0,
    max_period: float = 400.0,
) -> pd.DataFrame:
    """
    Find all significant periods in a light curve.

    Parameters:
    -----------
    time : np.ndarray
        Time array (in days)
    flux : np.ndarray
        Flux array (in magnitudes)
    fap_threshold : float
        FAP threshold for significance (default: 0.01 for 1%)
    min_period : float
        Minimum period to search (days)
    max_period : float
        Maximum period to search (days)

    Returns:
    --------
    periods : pd.DataFrame
        DataFrame with detected periods and their properties
    """
    # Calculate periodogram
    freq, power = calculate_lomb_scargle_periodogram(time, flux)
    period = 1.0 / freq

    # Filter by period range
    mask = (period >= min_period) & (period <= max_period)
    period = period[mask]
    power = power[mask]

    # Calculate FAP
    n_independent = len(period)
    fap = calculate_fap_analytical(power, n_independent)

    # Find significant periods
    significant_mask = fap < fap_threshold

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "period_days": period[significant_mask],
            "power": power[significant_mask],
            "fap": fap[significant_mask],
            "significance_sigma": -np.log10(fap[significant_mask]),
        }
    )

    if not results.empty:
        results["alias_probability"] = results["period_days"].apply(
            lambda p: assess_alias_probability(float(p), time=time)
        )
        results["is_likely_alias"] = results["alias_probability"] >= 0.6

    # Sort by power
    results = results.sort_values("power", ascending=False)

    return results


if __name__ == "__main__":
    # Example usage
    print("TASNI Periodogram Significance Analysis")
    print("=" * 50)

    # Simulate a light curve with 387 epochs over 9.2 years
    np.random.seed(42)
    time = np.linspace(0, 9.2 * 365.25, 387)
    flux = np.random.normal(0, 0.1, 387) + 0.01 * np.sin(2 * np.pi * time / 116)

    # Find significant periods
    periods = find_significant_periods(time, flux)
    print(f"\nDetected {len(periods)} significant periods:")
    print(periods.to_string(index=False))
