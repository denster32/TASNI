"""
Selection Function Analysis for TASNI

This module calculates the survey selection function and completeness
corrections for the TASNI pipeline.

Author: Dennis Palucki
Date: February 4, 2026
"""

import numpy as np
import pandas as pd

from tasni.core.seeds import DEFAULT_RANDOM_SEED


def _logistic_completeness(
    magnitudes: np.ndarray,
    m50: float,
    width: float,
    max_completeness: float,
) -> np.ndarray:
    """
    Smooth completeness curve parameterization.

    Completeness drops from ~max_completeness at bright magnitudes to ~0 at faint end.
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = np.where(np.isfinite(mags), mags, np.nan)
    comp = max_completeness / (1.0 + np.exp((mags - m50) / max(width, 1e-6)))
    return np.clip(comp, 0.0, 1.0)


def calculate_wise_completeness(
    w1_mags: np.ndarray, w1_limits: tuple[float, float] = (14.0, 17.0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate WISE completeness as a function of W1 magnitude.

    Parameters:
    -----------
    w1_mags : np.ndarray
        Array of W1 magnitudes
    w1_limits : tuple
        (min_mag, max_mag) for completeness calculation

    Returns:
    --------
    mag_bins : np.ndarray
        Magnitude bins
    completeness : np.ndarray
        Completeness at each magnitude bin
    """
    mags = np.asarray(w1_mags, dtype=float)
    # Approximate W1 50% completeness near 16.3 (survey-depth dependent).
    completeness = _logistic_completeness(
        mags,
        m50=np.mean(w1_limits) + 0.8,
        width=0.35,
        max_completeness=0.99,
    )
    return mags, completeness


def calculate_gaia_completeness(
    g_mags: np.ndarray, g_limits: tuple[float, float] = (15.0, 21.0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Gaia completeness as a function of G magnitude.

    Parameters:
    -----------
    g_mags : np.ndarray
        Array of G magnitudes
    g_limits : tuple
        (min_mag, max_mag) for completeness calculation

    Returns:
    --------
    mag_bins : np.ndarray
        Magnitude bins
    completeness : np.ndarray
        Completeness at each magnitude bin
    """
    mags = np.asarray(g_mags, dtype=float)
    completeness = _logistic_completeness(
        mags,
        m50=np.mean(g_limits) + 1.5,
        width=0.5,
        max_completeness=0.98,
    )
    return mags, completeness


def calculate_2mass_completeness(
    k_mags: np.ndarray, k_limits: tuple[float, float] = (13.0, 16.0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate 2MASS completeness as a function of K magnitude.

    Parameters:
    -----------
    k_mags : np.ndarray
        Array of K magnitudes
    k_limits : tuple
        (min_mag, max_mag) for completeness calculation

    Returns:
    --------
    mag_bins : np.ndarray
        Magnitude bins
    completeness : np.ndarray
        Completeness at each magnitude bin
    """
    mags = np.asarray(k_mags, dtype=float)
    completeness = _logistic_completeness(
        mags,
        m50=np.mean(k_limits) + 0.5,
        width=0.3,
        max_completeness=0.97,
    )
    return mags, completeness


def calculate_combined_selection_function(sources: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate combined selection function for TASNI sources.

    Parameters:
    -----------
    sources : pd.DataFrame
        DataFrame with source magnitudes

    Returns:
    --------
    selection_function : pd.DataFrame
        DataFrame with completeness for each source
    """
    if "w1mpro" not in sources.columns:
        raise KeyError("Selection-function calculation requires 'w1mpro' column")

    out = pd.DataFrame(index=sources.index)
    out["designation"] = sources.get("designation", sources.index.astype(str))
    out["w1"] = sources["w1mpro"].to_numpy()

    _, c_wise = calculate_wise_completeness(out["w1"].to_numpy())
    out["c_wise"] = np.clip(c_wise, 1e-3, 1.0)

    # Missing magnitudes are treated as "not part of this catalog selection term"
    # and therefore use neutral completeness 1.0 for multiplicative combination.
    if "gaia_g_mag" in sources.columns:
        _, c_gaia = calculate_gaia_completeness(sources["gaia_g_mag"].to_numpy())
        out["c_gaia"] = np.where(np.isfinite(sources["gaia_g_mag"]), c_gaia, 1.0)
    else:
        out["c_gaia"] = 1.0

    if "twomass_k_mag" in sources.columns:
        _, c_2mass = calculate_2mass_completeness(sources["twomass_k_mag"].to_numpy())
        out["c_2mass"] = np.where(np.isfinite(sources["twomass_k_mag"]), c_2mass, 1.0)
    else:
        out["c_2mass"] = 1.0

    out["c_combined"] = np.clip(out["c_wise"] * out["c_gaia"] * out["c_2mass"], 1e-3, 1.0)

    # Approximate systematic model uncertainty per catalog completeness curve.
    out["c_wise_err"] = 0.05 * out["c_wise"]
    out["c_gaia_err"] = 0.05 * out["c_gaia"]
    out["c_2mass_err"] = 0.05 * out["c_2mass"]
    rel_err_sq = (
        (out["c_wise_err"] / out["c_wise"]) ** 2
        + (out["c_gaia_err"] / out["c_gaia"]) ** 2
        + (out["c_2mass_err"] / out["c_2mass"]) ** 2
    )
    out["c_combined_err"] = out["c_combined"] * np.sqrt(rel_err_sq)
    return out


def calculate_survey_volume(max_distance: float, completeness: float) -> float:
    """
    Calculate effective survey volume for a given completeness.

    Parameters:
    -----------
    max_distance : float
        Maximum distance for volume calculation (pc)
    completeness : float
        Completeness fraction (0-1)

    Returns:
    --------
    volume : float
        Effective survey volume (pc^3)
    """
    volume = (4 / 3) * np.pi * max_distance**3 * float(completeness)
    return volume


def calculate_corrected_space_density(
    sources: pd.DataFrame,
    max_distance: float = 100.0,
    n_bootstrap: int = 10000,
    rng_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, float]:
    """
    Calculate space density with selection function corrections.

    Parameters:
    -----------
    sources : pd.DataFrame
        DataFrame with parallax measurements
    max_distance : float
        Maximum distance for volume calculation (pc)

    Returns:
    --------
    results : dict
        Dictionary with corrected space density and confidence intervals
    """
    sf = calculate_combined_selection_function(sources)
    n_sources = len(sf)
    survey_volume = (4 / 3) * np.pi * max_distance**3

    # Horvitz-Thompson style correction: each detected source contributes 1/c.
    inv_prob_weights = 1.0 / np.clip(sf["c_combined"].to_numpy(), 1e-3, 1.0)
    corrected_density = float(np.sum(inv_prob_weights) / survey_volume)
    raw_density = float(n_sources / survey_volume)

    rng = np.random.default_rng(rng_seed)
    boot_densities = np.empty(n_bootstrap, dtype=float)
    c_mean = sf["c_combined"].to_numpy()
    c_err = sf["c_combined_err"].to_numpy()

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_sources, size=n_sources)
        c_sample = np.clip(rng.normal(c_mean[idx], c_err[idx]), 1e-3, 1.0)
        boot_densities[i] = float(np.sum(1.0 / c_sample) / survey_volume)

    lower = float(np.percentile(boot_densities, 2.5))
    upper = float(np.percentile(boot_densities, 97.5))

    return {
        "density": corrected_density,
        "raw_density": raw_density,
        "lower_95ci": lower,
        "upper_95ci": upper,
        "n_sources": int(n_sources),
        "survey_volume": float(survey_volume),
        "mean_completeness": float(np.mean(c_mean)),
        "median_completeness": float(np.median(c_mean)),
    }


if __name__ == "__main__":
    # Example usage
    print("TASNI Selection Function Analysis")
    print("=" * 50)

    # Test completeness functions
    w1_mags = np.linspace(14.0, 17.0, 100)
    mag_bins, c_wise = calculate_wise_completeness(w1_mags)
    print(f"\nWISE completeness at W1=15.5 mag: {c_wise[50]:.2f}")
