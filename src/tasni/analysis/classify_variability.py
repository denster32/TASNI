"""
Variability Classification Framework for TASNI

This module provides quantitative criteria for classifying TASNI sources
as NORMAL, VARIABLE, or FADING based on NEOWISE light curves.

Author: Dennis Palucki
Date: February 4, 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import norm

# Import standardized thresholds from config
try:
    from ..core.config import (
        CHI2_VARIABILITY_THRESHOLD,
        FADE_P_VALUE_THRESHOLD,
        FADE_RATE_THRESHOLD_MMAG_YR,
        TREND_THRESHOLD_MAG_YR,
    )

    DEFAULT_FADE_THRESHOLD = FADE_RATE_THRESHOLD_MMAG_YR  # 15.0 mmag/yr
    DEFAULT_CHI2_THRESHOLD = CHI2_VARIABILITY_THRESHOLD  # 3.0
    DEFAULT_FADE_P_THRESHOLD = FADE_P_VALUE_THRESHOLD  # 0.01
except ImportError:
    DEFAULT_FADE_THRESHOLD = 15.0
    DEFAULT_CHI2_THRESHOLD = 3.0
    DEFAULT_FADE_P_THRESHOLD = 0.01


def linear_fit_with_uncertainty(
    time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None
) -> tuple[float, float, float]:
    """
    Fit linear model to light curve with uncertainty estimation.

    Parameters:
    -----------
    time : np.ndarray
        Time array (in years)
    flux : np.ndarray
        Flux array (in magnitudes)
    flux_err : np.ndarray, optional
        Flux error array

    Returns:
    --------
    slope : float
        Linear slope (mmag/yr)
    slope_err : float
        Uncertainty in slope
    """
    if flux_err is None:
        flux_err = np.ones_like(flux) * np.median(flux) * 0.01

    def linear_model(t, a, b):
        return a * t + b

    # Fit linear model
    popt, pcov = curve_fit(linear_model, time, flux, sigma=flux_err, absolute_sigma=True)

    slope, intercept = popt
    slope_err = np.sqrt(pcov[0, 0])

    return slope, slope_err


def calculate_chi2(
    time: np.ndarray, flux: np.ndarray, model: np.ndarray, flux_err: np.ndarray | None = None
) -> tuple[float, int]:
    """
    Calculate chi-squared statistic for light curve fit.

    Uses flux_err^2 as variance (standard chi-squared). If flux_err is None,
    falls back to model^2 for backward compatibility (deprecated).

    Parameters:
    -----------
    time : np.ndarray
        Time array
    flux : np.ndarray
        Flux array
    model : np.ndarray
        Model values
    flux_err : np.ndarray, optional
        Flux uncertainty (mag). If None, uses model^2 as variance (legacy).

    Returns:
    --------
    chi2 : float
        Chi-squared statistic
    dof : int
        Degrees of freedom
    """
    residuals = flux - model
    if flux_err is not None and np.all(flux_err > 0):
        variance = np.maximum(flux_err**2, 1e-20)  # Avoid div by zero
        chi2 = np.sum(residuals**2 / variance)
    else:
        # Legacy fallback when no errors available
        variance = np.maximum(model**2, 1e-20)
        chi2 = np.sum(residuals**2 / variance)
    dof = len(time) - 2  # Linear model has 2 parameters

    return chi2, dof


def classify_variability(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    chi2_threshold: float = None,
    fade_threshold: float = None,
    fade_p_threshold: float = None,
) -> dict[str, any]:
    """
    Classify source variability based on quantitative criteria.

    Thresholds are imported from config.py by default:
    - chi2_threshold: 3.0 (CHI2_VARIABILITY_THRESHOLD)
    - fade_threshold: 15.0 mmag/yr (FADE_RATE_THRESHOLD_MMAG_YR)
    - fade_p_threshold: 0.01 (FADE_P_VALUE_THRESHOLD)

    Parameters:
    -----------
    time : np.ndarray
        Time array (in years)
    flux : np.ndarray
        Flux array (in magnitudes)
    flux_err : np.ndarray, optional
        Flux error array
    chi2_threshold : float
        Chi-squared threshold for variability (default: from config)
    fade_threshold : float
        Minimum fade rate for FADING classification (mmag/yr, default: from config)
    fade_p_threshold : float
        P-value threshold for significant fading (default: from config)

    Returns:
    --------
    classification : dict
        Dictionary with classification results
    """
    # Use config defaults if not specified
    if chi2_threshold is None:
        chi2_threshold = DEFAULT_CHI2_THRESHOLD
    if fade_threshold is None:
        fade_threshold = DEFAULT_FADE_THRESHOLD
    if fade_p_threshold is None:
        fade_p_threshold = DEFAULT_FADE_P_THRESHOLD

    # Default flux_err if not provided
    if flux_err is None:
        flux_err = np.ones_like(flux) * np.median(np.abs(flux)) * 0.01

    # Calculate linear fit
    slope, slope_err = linear_fit_with_uncertainty(time, flux, flux_err)
    model = slope * time + np.mean(flux)

    # Calculate chi-squared (uses flux_err^2 as variance)
    chi2, dof = calculate_chi2(time, flux, model, flux_err)

    # Calculate p-value for slope (two-tailed test)
    if slope_err > 0:
        z_score = abs(slope) / slope_err
        p_slope = 2 * (1 - norm.cdf(z_score))
    else:
        p_slope = 1.0

    # Classification criteria
    if chi2 > chi2_threshold:
        # Source shows variability
        if abs(slope) >= fade_threshold and p_slope < fade_p_threshold:
            classification = "FADING"
        else:
            classification = "VARIABLE"
    else:
        # Source is stable
        classification = "NORMAL"

    # Calculate variability metrics
    rms = np.std(flux)
    std_err = np.std(flux_err) if flux_err is not None else 0.0

    results = {
        "classification": classification,
        "slope_mmag_yr": slope,
        "slope_err": slope_err,
        "p_slope": p_slope,
        "chi2": chi2,
        "dof": dof,
        "rms_mag": rms,
        "std_err_mag": std_err,
    }

    return results


def batch_classify_variability(
    light_curves: dict[str, np.ndarray],
    chi2_threshold: float = 3.0,
    fade_threshold: float = 15.0,
    fade_p_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Classify variability for multiple light curves.

    Parameters:
    -----------
    light_curves : dict
        Dictionary mapping source IDs to (time, flux, flux_err) tuples
    chi2_threshold : float
        Chi-squared threshold for variability
    fade_threshold : float
        Minimum fade rate for FADING classification
    fade_p_threshold : float
        P-value threshold for significant fading

    Returns:
    --------
    classifications : pd.DataFrame
        DataFrame with classification results for all sources
    """
    results = []

    for designation, (time, flux, flux_err) in light_curves.items():
        classification = classify_variability(
            time, flux, flux_err, chi2_threshold, fade_threshold, fade_p_threshold
        )
        results.append({"designation": designation, **classification})

    return pd.DataFrame(results)


def calculate_cross_correlation(
    w1_time: np.ndarray, w1_flux: np.ndarray, w2_time: np.ndarray, w2_flux: np.ndarray
) -> tuple[float, float]:
    """
    Calculate cross-correlation between W1 and W2 light curves.

    Parameters:
    -----------
    w1_time : np.ndarray
        W1 time array
    w1_flux : np.ndarray
        W1 flux array
    w2_time : np.ndarray
        W2 time array
    w2_flux : np.ndarray
        W2 flux array

    Returns:
    --------
    correlation : float
        Pearson correlation coefficient
    p_value : float
        P-value for correlation
    """
    # Interpolate W2 to W1 time points
    w2_interp = interp1d(w2_time, w2_flux, kind="linear", fill_value="extrapolate")
    w2_at_w1 = w2_interp(w1_time)

    # Calculate correlation
    correlation, p_value = stats.pearsonr(w1_flux, w2_at_w1)

    return correlation, p_value


if __name__ == "__main__":
    # Example usage
    print("TASNI Variability Classification Framework")
    print("=" * 50)

    # Simulate a fading light curve
    np.random.seed(42)
    time = np.linspace(0, 10, 387)
    flux = -0.025 * time + np.random.normal(0, 0.1, 387)
    flux_err = np.ones_like(flux) * 0.01

    # Classify variability
    classification = classify_variability(time, flux, flux_err)
    print(f"\nClassification: {classification['classification']}")
    print(
        f"  Slope: {classification['slope_mmag_yr']:.2f} ± {classification['slope_err']:.2f} mmag/yr"
    )
    print(f"  P-value: {classification['p_slope']:.4f}")
    print(f"  χ²/ν: {classification['chi2']:.2f}")
