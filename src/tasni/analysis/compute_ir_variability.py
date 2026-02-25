#!/usr/bin/env python3
"""
TASNI: Compute IR Variability Metrics from Multi-Epoch Photometry

Computes variability statistics from NEOWISE multi-epoch photometry:
- RMS scatter
- Reduced chi-squared
- Stetson J index (correlated variability)
- Long-term trends
- Periodogram (if sufficient epochs)

Scoring Logic:
- STABLE_BONUS: 14-year stable source = genuine thermal emission (interesting)
- VARIABLE_PENALTY: Variable = likely astrophysical (natural)
- FADING_BONUS: Fading over time = unusual (very interesting)
- BRIGHTENING_PENALTY: Brightening = likely transient (natural)

Usage:
    python compute_ir_variability.py [--epochs FILE] [--output FILE]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Import standardized thresholds from config
try:
    from ..core.config import (
        CHI2_VARIABILITY_THRESHOLD,
        FADE_P_VALUE_THRESHOLD,
        FADE_RATE_THRESHOLD_MMAG_YR,
        MIN_BASELINE_YEARS,
        MIN_EPOCHS_VARIABILITY,
        TREND_THRESHOLD_MAG_YR,
    )
except ImportError:
    # Fallback if relative import fails
    FADE_RATE_THRESHOLD_MMAG_YR = 15.0
    TREND_THRESHOLD_MAG_YR = 0.015
    CHI2_VARIABILITY_THRESHOLD = 3.0
    FADE_P_VALUE_THRESHOLD = 0.01
    MIN_BASELINE_YEARS = 2.0
    MIN_EPOCHS_VARIABILITY = 10

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [IR-VAR] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Variability thresholds
MIN_EPOCHS = MIN_EPOCHS_VARIABILITY  # Minimum epochs for reliable variability
RMS_STABLE_THRESHOLD = 0.05  # mag - RMS < 0.05 is stable
RMS_VARIABLE_THRESHOLD = 0.15  # mag - RMS > 0.15 is variable
CHI2_STABLE_THRESHOLD = 2.0  # Reduced chi-squared
CHI2_VARIABLE_THRESHOLD = CHI2_VARIABILITY_THRESHOLD

# Trend thresholds (mag/year) - use standardized value from config
TREND_THRESHOLD = TREND_THRESHOLD_MAG_YR  # 0.015 mag/yr = 15 mmag/yr

# Scoring constants
STABLE_BONUS = 10.0  # 14-year stable IR emission
VARIABLE_PENALTY = -10.0  # Significant variability
FADING_BONUS = 25.0  # Fading over time (very unusual)
BRIGHTENING_PENALTY = -5.0  # Brightening (transient-like)


def compute_rms(values: np.ndarray, errors: np.ndarray = None) -> float:
    """Compute RMS scatter."""
    if len(values) < 2:
        return np.nan
    return np.std(values, ddof=1)


def compute_reduced_chi2(values: np.ndarray, errors: np.ndarray) -> float:
    """
    Compute reduced chi-squared for constant model.

    χ² = Σ((m - <m>) / σ)²
    χ²_red = χ² / (N - 1)
    """
    if len(values) < 2 or errors is None or np.all(np.isnan(errors)):
        return np.nan

    mean = np.nanmean(values)
    valid = ~np.isnan(values) & ~np.isnan(errors) & (errors > 0)

    if valid.sum() < 2:
        return np.nan

    chi2 = np.sum(((values[valid] - mean) / errors[valid]) ** 2)
    dof = valid.sum() - 1
    return chi2 / dof


def compute_stetson_j(
    values1: np.ndarray, errors1: np.ndarray, values2: np.ndarray, errors2: np.ndarray
) -> float:
    """
    Compute Stetson J index for correlated variability.

    J = Σ sign(P_i) * sqrt(|P_i|)
    where P_i = δ_1 * δ_2 and δ = (m - <m>) / σ
    """
    if len(values1) < 2:
        return np.nan

    valid = (
        ~np.isnan(values1)
        & ~np.isnan(values2)
        & ~np.isnan(errors1)
        & ~np.isnan(errors2)
        & (errors1 > 0)
        & (errors2 > 0)
    )

    if valid.sum() < 2:
        return np.nan

    mean1 = np.nanmean(values1[valid])
    mean2 = np.nanmean(values2[valid])

    delta1 = (values1[valid] - mean1) / errors1[valid]
    delta2 = (values2[valid] - mean2) / errors2[valid]

    p = delta1 * delta2
    j = np.sum(np.sign(p) * np.sqrt(np.abs(p)))

    # Normalize by number of pairs
    return j / np.sqrt(valid.sum())


def compute_trend(mjd: np.ndarray, values: np.ndarray, errors: np.ndarray = None) -> dict:
    """
    Compute linear trend in light curve.

    Returns:
        Dictionary with slope, slope_err, p_value
    """
    valid = ~np.isnan(mjd) & ~np.isnan(values)
    if valid.sum() < 3:
        return {"slope": np.nan, "slope_err": np.nan, "p_value": np.nan}

    # Convert MJD to years
    mjd_valid = mjd[valid]
    years = (mjd_valid - mjd_valid.min()) / 365.25
    values_valid = values[valid]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values_valid)

    return {
        "slope": slope,  # mag/year
        "slope_err": std_err,
        "p_value": p_value,
        "r_squared": r_value**2,
    }


def classify_variability(
    rms_w1: float, rms_w2: float, chi2_w1: float, chi2_w2: float, trend_w1: float, trend_w2: float
) -> dict:
    """
    Classify variability type and compute score.

    Returns:
        Dictionary with is_variable, is_stable, trend_type, variability_score
    """
    result = {
        "is_variable": False,
        "is_stable": False,
        "trend_type": "none",
        "variability_score": 0.0,
        "variability_flag": "UNKNOWN",
    }

    # Check for stable source
    if (
        rms_w1 < RMS_STABLE_THRESHOLD
        and rms_w2 < RMS_STABLE_THRESHOLD
        and chi2_w1 < CHI2_STABLE_THRESHOLD
        and chi2_w2 < CHI2_STABLE_THRESHOLD
    ):
        result["is_stable"] = True
        result["variability_score"] += STABLE_BONUS
        result["variability_flag"] = "STABLE"

    # Check for variable source
    if (
        rms_w1 > RMS_VARIABLE_THRESHOLD
        or rms_w2 > RMS_VARIABLE_THRESHOLD
        or chi2_w1 > CHI2_VARIABLE_THRESHOLD
        or chi2_w2 > CHI2_VARIABLE_THRESHOLD
    ):
        result["is_variable"] = True
        result["variability_score"] += VARIABLE_PENALTY
        result["variability_flag"] = "VARIABLE"

    # Check for trends
    # Fading (getting fainter = positive slope in magnitudes)
    if trend_w1 > TREND_THRESHOLD or trend_w2 > TREND_THRESHOLD:
        result["trend_type"] = "fading"
        result["variability_score"] += FADING_BONUS
        result["variability_flag"] = "FADING"

    # Brightening (getting brighter = negative slope in magnitudes)
    if trend_w1 < -TREND_THRESHOLD or trend_w2 < -TREND_THRESHOLD:
        result["trend_type"] = "brightening"
        result["variability_score"] += BRIGHTENING_PENALTY
        if result["variability_flag"] == "UNKNOWN":
            result["variability_flag"] = "BRIGHTENING"

    if result["variability_flag"] == "UNKNOWN":
        result["variability_flag"] = "NORMAL"

    return result


def analyze_source(epochs: pd.DataFrame) -> dict:
    """
    Analyze variability for a single source.

    Args:
        epochs: DataFrame with epoch photometry for one source

    Returns:
        Dictionary with variability metrics
    """
    result = {
        "n_epochs": len(epochs),
        "baseline_years": np.nan,
        "rms_w1": np.nan,
        "rms_w2": np.nan,
        "chi2_w1": np.nan,
        "chi2_w2": np.nan,
        "stetson_j": np.nan,
        "trend_w1": np.nan,
        "trend_w2": np.nan,
        "trend_p_w1": np.nan,
        "trend_p_w2": np.nan,
        "is_variable": False,
        "is_stable": False,
        "trend_type": "none",
        "variability_score": 0.0,
        "variability_flag": "INSUFFICIENT_DATA",
    }

    if len(epochs) < MIN_EPOCHS:
        return result

    # Time baseline
    if "mjd" in epochs.columns:
        mjd = epochs["mjd"].values
        result["baseline_years"] = (mjd.max() - mjd.min()) / 365.25

    # W1 variability
    if "w1mpro_ep" in epochs.columns:
        w1 = epochs["w1mpro_ep"].values
        w1_err = epochs.get("w1sigmpro_ep", pd.Series([0.1] * len(epochs))).values
        result["rms_w1"] = compute_rms(w1)
        result["chi2_w1"] = compute_reduced_chi2(w1, w1_err)

        trend = compute_trend(mjd, w1, w1_err)
        result["trend_w1"] = trend["slope"]
        result["trend_p_w1"] = trend["p_value"]

    # W2 variability
    if "w2mpro_ep" in epochs.columns:
        w2 = epochs["w2mpro_ep"].values
        w2_err = epochs.get("w2sigmpro_ep", pd.Series([0.1] * len(epochs))).values
        result["rms_w2"] = compute_rms(w2)
        result["chi2_w2"] = compute_reduced_chi2(w2, w2_err)

        trend = compute_trend(mjd, w2, w2_err)
        result["trend_w2"] = trend["slope"]
        result["trend_p_w2"] = trend["p_value"]

    # Stetson J (correlated variability)
    if "w1mpro_ep" in epochs.columns and "w2mpro_ep" in epochs.columns:
        result["stetson_j"] = compute_stetson_j(
            epochs["w1mpro_ep"].values,
            epochs.get("w1sigmpro_ep", pd.Series([0.1] * len(epochs))).values,
            epochs["w2mpro_ep"].values,
            epochs.get("w2sigmpro_ep", pd.Series([0.1] * len(epochs))).values,
        )

    # Classify variability
    classification = classify_variability(
        result["rms_w1"],
        result["rms_w2"],
        result["chi2_w1"],
        result["chi2_w2"],
        result["trend_w1"],
        result["trend_w2"],
    )
    result.update(classification)

    return result


def analyze_all_sources(epochs: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze variability for all sources.

    Args:
        epochs: DataFrame with all epoch photometry

    Returns:
        DataFrame with variability metrics per source
    """
    logger.info(f"Analyzing variability for {epochs['designation'].nunique()} sources")

    results = []
    for designation, source_epochs in epochs.groupby("designation"):
        result = analyze_source(source_epochs)
        result["designation"] = designation
        results.append(result)

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print variability analysis summary."""
    logger.info("=" * 60)
    logger.info("IR Variability Analysis Summary")
    logger.info("=" * 60)

    # Data coverage
    has_data = df["n_epochs"] >= MIN_EPOCHS
    logger.info(f"Sources with sufficient data (>={MIN_EPOCHS} epochs): {has_data.sum()}/{len(df)}")

    if has_data.sum() == 0:
        logger.warning("No sources with sufficient epochs for analysis!")
        return

    # Baseline
    logger.info(f"Median baseline: {df.loc[has_data, 'baseline_years'].median():.1f} years")

    # Variability distribution
    logger.info("\nVariability Classification:")
    flag_counts = df.loc[has_data, "variability_flag"].value_counts()
    for flag, count in flag_counts.items():
        pct = 100 * count / has_data.sum()
        logger.info(f"  {flag}: {count} ({pct:.1f}%)")

    # Score distribution
    logger.info("\nVariability Score:")
    logger.info(f"  Mean: {df.loc[has_data, 'variability_score'].mean():.2f}")
    logger.info(f"  Min: {df.loc[has_data, 'variability_score'].min():.2f}")
    logger.info(f"  Max: {df.loc[has_data, 'variability_score'].max():.2f}")

    # RMS distribution
    logger.info("\nW1 RMS distribution:")
    logger.info(f"  Median: {df.loc[has_data, 'rms_w1'].median():.3f} mag")
    logger.info(f"  Max: {df.loc[has_data, 'rms_w1'].max():.3f} mag")

    # Trends
    n_fading = (df["trend_type"] == "fading").sum()
    n_brightening = (df["trend_type"] == "brightening").sum()
    logger.info("\nTrends:")
    logger.info(f"  Fading: {n_fading}")
    logger.info(f"  Brightening: {n_brightening}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TASNI IR Variability Analysis")
    parser.add_argument(
        "--epochs",
        "-e",
        default="./data/processed/neowise_epochs.parquet",
        help="Input parquet file with epoch photometry",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./data/processed/golden_variability.parquet",
        help="Output parquet file with variability metrics",
    )
    parser.add_argument(
        "--golden",
        "-g",
        default="./data/processed/golden_targets.csv",
        help="Golden targets file to merge with",
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Update golden_targets.csv with variability scores",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: IR Variability Analysis")
    logger.info("=" * 60)
    logger.info(f"Epochs file: {args.epochs}")
    logger.info(f"Output: {args.output}")

    # Check if epochs file exists
    epochs_path = Path(args.epochs)
    if not epochs_path.exists():
        logger.error(f"Epochs file not found: {args.epochs}")
        logger.error("Run: python query_neowise_variability.py first")
        return

    # Load epochs
    epochs = pd.read_parquet(args.epochs)
    logger.info(f"Loaded {len(epochs)} epochs for {epochs['designation'].nunique()} sources")

    # Analyze variability
    variability = analyze_all_sources(epochs)

    # Print summary
    print_summary(variability)

    # Save results
    output_path = Path(args.output)
    variability.to_parquet(output_path, index=False)
    logger.info(f"Saved variability metrics to {output_path}")

    # Also save as CSV for easy viewing
    csv_path = output_path.with_suffix(".csv")
    variability.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")

    # Optionally update golden targets
    if args.update_golden:
        golden_path = Path(args.golden)
        if golden_path.exists():
            golden = pd.read_csv(golden_path)

            # Merge variability data
            var_cols = [
                "designation",
                "n_epochs",
                "baseline_years",
                "rms_w1",
                "rms_w2",
                "chi2_w1",
                "chi2_w2",
                "trend_w1",
                "trend_w2",
                "is_variable",
                "is_stable",
                "variability_score",
                "variability_flag",
            ]
            var_cols = [c for c in var_cols if c in variability.columns]

            golden = golden.merge(variability[var_cols], on="designation", how="left")

            # Update score
            golden["score"] = golden["score"] + golden["variability_score"].fillna(0)

            # Re-sort
            golden = golden.sort_values("score", ascending=False)
            golden.to_csv(golden_path, index=False)
            logger.info(f"Updated {golden_path} with variability scores")

    return variability


if __name__ == "__main__":
    main()
