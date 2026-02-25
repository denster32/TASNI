"""
Comprehensive Statistical Analysis Framework for TASNI

This module provides functions for calculating p-values, confidence intervals,
and performing Monte Carlo simulations for TASNI results.

Author: Dennis Palucki
Date: February 4, 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from tasni.core.seeds import DEFAULT_RANDOM_SEED, make_rng


def calculate_p_value(n_observed: int, n_expected: float, n_trials: int = 1) -> float:
    """
    Calculate p-value for observing n_observed or more events
    when n_expected events are expected.

    Parameters:
    -----------
    n_observed : int
        Number of events actually observed
    n_expected : float
        Expected number of events from null hypothesis
    n_trials : int
        Number of independent trials (default: 1)

    Returns:
    --------
    p_value : float
        Probability of observing >= n_observed events under H0
    """
    # Poisson probability of observing >= n_observed
    p_value = 1.0 - stats.poisson.cdf(n_observed - 1, n_expected)

    # Adjust for multiple testing
    p_value_corrected = 1.0 - (1.0 - p_value) ** n_trials

    return p_value_corrected


def calculate_confidence_interval(
    value: float, error: float, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Calculate confidence interval for a measurement with Gaussian errors.

    Parameters:
    -----------
    value : float
        Central value of measurement
    error : float
        Standard error of measurement
    confidence : float
        Confidence level (default: 0.95 for 95% CI)

    Returns:
    --------
    lower : float
        Lower bound of confidence interval
    upper : float
        Upper bound of confidence interval
    """
    z_score = norm.ppf((1 + confidence) / 2)
    lower = value - z_score * error
    upper = value + z_score * error
    return lower, upper


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a dataset.

    Parameters:
    -----------
    data : np.ndarray
        Array of data values
    n_bootstrap : int
        Number of bootstrap iterations (default: 10000)
    confidence : float
        Confidence level (default: 0.95 for 95% CI)

    Returns:
    --------
    lower : float
        Lower bound of confidence interval
    upper : float
        Upper bound of confidence interval
    """
    rng = make_rng(random_seed)
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(sample)

    lower = np.percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    return lower, upper


def monte_carlo_fading_simulation(
    n_sources: int,
    n_epochs: int = 387,
    noise_std: float = 0.1,
    fade_threshold: float = 15.0,
    n_simulations: int = 10000,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, float]:
    """
    Monte Carlo simulation to estimate expected number of fading sources.

    Parameters:
    -----------
    n_sources : int
        Number of sources to simulate (e.g., 100 golden targets)
    n_epochs : int
        Number of epochs per light curve (default: 387 from TASNI)
    noise_std : float
        Standard deviation of measurement noise (mag)
    fade_threshold : float
        Minimum fade rate to be classified as fading (mmag/yr)
    n_simulations : int
        Number of Monte Carlo iterations

    Returns:
    --------
    results : dict
        Dictionary with p-value and statistics
    """
    rng = make_rng(random_seed)
    fading_counts = np.zeros(n_simulations)

    for i in range(n_simulations):
        # Simulate random light curves
        light_curves = rng.normal(0, noise_std, (n_sources, n_epochs))

        # Calculate linear trends
        epochs = np.arange(n_epochs)
        fade_rates = []
        for j in range(n_sources):
            slope, _ = np.polyfit(epochs, light_curves[j, :], 1)
            fade_rates.append(slope * 365.25 * 1000)  # Convert to mmag/yr

        # Count fading sources
        fading_counts[i] = np.sum(np.abs(np.array(fade_rates)) >= fade_threshold)

    # Calculate p-value for observed 4 fading sources
    n_observed = 4
    p_value = np.sum(fading_counts >= n_observed) / n_simulations

    results = {
        "p_value": p_value,
        "expected_fading": np.mean(fading_counts),
        "std_fading": np.std(fading_counts),
        "median_fading": np.median(fading_counts),
        "n_simulations": n_simulations,
    }

    return results


def ks_test_distribution(sample1: np.ndarray, sample2: np.ndarray) -> tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test to compare two distributions.

    Parameters:
    -----------
    sample1 : np.ndarray
        First sample (e.g., TASNI temperatures)
    sample2 : np.ndarray
        Second sample (e.g., known Y dwarf temperatures)

    Returns:
    --------
    statistic : float
        KS test statistic
    p_value : float
        P-value for two samples being drawn from same distribution
    """
    statistic, p_value = stats.ks_2samp(sample1, sample2)
    return statistic, p_value


def chi2_goodness_of_fit(
    observed: np.ndarray, expected: np.ndarray, ddof: int
) -> tuple[float, float]:
    """
    Calculate chi-squared goodness of fit.

    Parameters:
    -----------
    observed : np.ndarray
        Observed values
    expected : np.ndarray
        Expected values from model
    ddof : int
        Degrees of freedom

    Returns:
    --------
    chi2 : float
        Chi-squared statistic
    p_value : float
        P-value for goodness of fit
    """
    chi2_val = np.sum((observed - expected) ** 2 / expected)
    p_value = 1.0 - stats.chi2.cdf(chi2_val, ddof)
    return chi2_val, p_value


def calculate_space_density(
    sources: pd.DataFrame, max_distance: float = 100.0, confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Calculate space density with confidence intervals using volume method.

    Parameters:
    -----------
    sources : pd.DataFrame
        DataFrame with parallax measurements
    max_distance : float
        Maximum distance for volume calculation (pc)
    confidence : float
        Confidence level (default: 0.95)

    Returns:
    --------
    density : float
        Space density (pc^-3)
    lower : float
        Lower bound of confidence interval
    upper : float
        Upper bound of confidence interval
    """
    # Calculate volume for each source
    volumes = (4 / 3) * np.pi * max_distance**3

    # Bootstrap space density
    densities = []
    for _ in range(10000):
        sample = sources.sample(frac=1, replace=True)
        density = len(sample) / volumes
        densities.append(density)

    lower = np.percentile(densities, (1 - confidence) / 2 * 100)
    upper = np.percentile(densities, (1 + confidence) / 2 * 100)
    density = np.mean(densities)

    return density, lower, upper


def fdr_correction(
    p_values: np.ndarray, alpha: float = 0.05, method: str = "bh"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply False Discovery Rate (FDR) correction for multiple testing.

    Uses Benjamini-Hochberg procedure to control the expected proportion
    of false discoveries among rejected hypotheses.

    Parameters:
    -----------
    p_values : np.ndarray
        Array of p-values from multiple tests
    alpha : float
        Desired FDR level (default: 0.05)
    method : str
        Correction method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)

    Returns:
    --------
    rejected : np.ndarray
        Boolean array indicating which hypotheses are rejected
    adjusted_p : np.ndarray
        FDR-adjusted p-values
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([])

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Calculate critical values
    if method == "bh":
        # Benjamini-Hochberg
        critical = (np.arange(1, n + 1) / n) * alpha
    elif method == "by":
        # Benjamini-Yekutieli (more conservative)
        c = np.sum(1.0 / np.arange(1, n + 1))
        critical = (np.arange(1, n + 1) / (n * c)) * alpha
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find largest p-value that is still significant
    below_threshold = sorted_p <= critical
    if not np.any(below_threshold):
        # No significant results
        return np.zeros(n, dtype=bool), np.ones(n)

    # Find the threshold
    max_idx = np.max(np.where(below_threshold)[0])
    threshold = sorted_p[max_idx]

    # Create rejection array
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices[: max_idx + 1]] = True

    # Calculate adjusted p-values
    adjusted_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        rank = i + 1
        adjusted_p[idx] = sorted_p[i] * n / rank
        if i > 0:
            # Ensure monotonicity
            adjusted_p[idx] = min(adjusted_p[idx], adjusted_p[sorted_indices[i - 1]])

    # Cap at 1.0
    adjusted_p = np.clip(adjusted_p, 0, 1)

    return rejected, adjusted_p


def power_analysis(
    effect_size: float, alpha: float = 0.05, n_samples: int = None, power: float = None
) -> dict[str, float]:
    """
    Calculate power analysis for sample size justification.

    Either provide n_samples to get power, or power to get required n_samples.

    Parameters:
    -----------
    effect_size : float
        Cohen's d effect size (standardized)
    alpha : float
        Significance level (default: 0.05)
    n_samples : int, optional
        Sample size (if calculating power)
    power : float, optional
        Desired power (if calculating sample size)

    Returns:
    --------
    results : dict
        Dictionary with power analysis results
    """
    from scipy.stats import norm

    if n_samples is None and power is None:
        raise ValueError("Must provide either n_samples or power")

    # Z-scores for alpha and power
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed

    if n_samples is not None:
        # Calculate power from sample size
        z_effect = effect_size * np.sqrt(n_samples / 2)
        z_power = z_effect - z_alpha
        calculated_power = norm.cdf(z_power)

        return {
            "n_samples": n_samples,
            "effect_size": effect_size,
            "alpha": alpha,
            "power": calculated_power,
            "z_alpha": z_alpha,
            "z_effect": z_effect,
        }
    else:
        # Calculate required sample size from power
        z_power = norm.ppf(power)
        n_required = 2 * ((z_alpha + z_power) / effect_size) ** 2

        return {
            "n_required": int(np.ceil(n_required)),
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "z_alpha": z_alpha,
            "z_power": z_power,
        }


if __name__ == "__main__":
    # Example usage
    print("TASNI Statistical Analysis Framework")
    print("=" * 50)

    # Test p-value calculation
    p_val = calculate_p_value(n_observed=4, n_expected=0.5, n_trials=1)
    print(f"\nP-value for 4 fading sources (expected 0.5): {p_val:.4f}")

    # Test confidence interval
    value, error = 293, 30
    lower, upper = calculate_confidence_interval(value, error)
    print(f"\n95% CI for T_eff = {value} ± {error} K: [{lower:.1f}, {upper:.1f}] K")

    # Test Monte Carlo simulation
    results = monte_carlo_fading_simulation(n_sources=100)
    print("\nMonte Carlo fading simulation:")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Expected fading: {results['expected_fading']:.2f} ± {results['std_fading']:.2f}")
