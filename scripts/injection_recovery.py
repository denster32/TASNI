#!/usr/bin/env python3
"""
TASNI Injection-Recovery Validation

Generates synthetic Y-dwarf-like fading light curves, runs the same
trend and classification logic as the pipeline, and reports the
recovery fraction at >3 sigma. Used to support the Methods claim
that the pipeline recovers a high fraction of injected fading signals.

Usage:
    python scripts/injection_recovery.py [--n N] [--seed SEED]
"""

import argparse

import numpy as np
from scipy import stats

# Pipeline thresholds (from config / compute_ir_variability)
TREND_THRESHOLD_MAG_YR = 0.015  # 15 mmag/yr = fading threshold
MIN_BASELINE_YEARS = 2.0
MIN_EPOCHS = 10


def generate_synthetic_light_curve(
    n_epochs: int,
    baseline_years: float,
    fade_rate_mmag_yr: float,
    w1_mean: float = 14.0,
    w2_mean: float = 11.0,
    mag_sigma: float = 0.03,
    seed: int = None,
) -> tuple:
    """
    Generate synthetic W1, W2 light curves with a linear fading trend.

    Fade rate in mmag/yr; positive = getting fainter (magnitude increases).
    Returns (mjd, w1, w2, w1_err, w2_err).
    """
    rng = np.random.default_rng(seed)
    mjd0 = 55500.0  # ~2010.7
    mjd = mjd0 + rng.uniform(0, baseline_years * 365.25, size=n_epochs)
    mjd = np.sort(mjd)
    years = (mjd - mjd.min()) / 365.25
    # Fade rate: mag/year (positive = fainter)
    fade_mag_yr = fade_rate_mmag_yr / 1000.0
    w1 = w1_mean + fade_mag_yr * years + rng.normal(0, mag_sigma, size=n_epochs)
    w2 = w2_mean + fade_mag_yr * years + rng.normal(0, mag_sigma, size=n_epochs)
    err = np.full(n_epochs, mag_sigma)
    return mjd, w1, w2, err, err


def compute_trend(years: np.ndarray, values: np.ndarray) -> dict:
    """Linear trend (mag/yr); same as compute_ir_variability.compute_trend."""
    if len(years) < 3:
        return {"slope": np.nan, "slope_err": np.nan}
    slope, _, _, _, std_err = stats.linregress(years, values)
    return {"slope": slope, "slope_err": std_err}


def is_recovered(
    mjd: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w1_err: np.ndarray,
    w2_err: np.ndarray,
    sigma_threshold: float = 3.0,
) -> bool:
    """
    True if (1) baseline and epochs meet minimums, (2) classified as FADING
    (slope_w1 or slope_w2 > TREND_THRESHOLD), and (3) trend significant at
    > sigma_threshold (slope / slope_err >= sigma_threshold).
    """
    years = (mjd - mjd.min()) / 365.25
    baseline_years = years.max() - years.min()
    if baseline_years < MIN_BASELINE_YEARS or len(mjd) < MIN_EPOCHS:
        return False
    t1 = compute_trend(years, w1)
    t2 = compute_trend(years, w2)
    slope_w1, err_w1 = t1["slope"], t1["slope_err"]
    slope_w2, err_w2 = t2["slope"], t2["slope_err"]
    is_fading = slope_w1 > TREND_THRESHOLD_MAG_YR or slope_w2 > TREND_THRESHOLD_MAG_YR
    if not is_fading:
        return False
    # Require at least one band with trend significant at >3 sigma
    sig_w1 = slope_w1 / err_w1 if err_w1 and err_w1 > 0 else 0
    sig_w2 = slope_w2 / err_w2 if err_w2 and err_w2 > 0 else 0
    return sig_w1 >= sigma_threshold or sig_w2 >= sigma_threshold


def main():
    parser = argparse.ArgumentParser(
        description="TASNI injection-recovery: synthetic Y-dwarf fading signals"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of synthetic sources to inject (default 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--fade_min",
        type=float,
        default=20.0,
        help="Minimum fade rate mmag/yr (default 20)",
    )
    parser.add_argument(
        "--fade_max",
        type=float,
        default=50.0,
        help="Maximum fade rate mmag/yr (default 50)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Recovery significance threshold (default 3)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    recovered = 0
    for i in range(args.n):
        fade_rate = rng.uniform(args.fade_min, args.fade_max)
        n_epochs = int(rng.integers(50, 350))
        baseline_years = float(rng.uniform(7.0, 10.5))
        mjd, w1, w2, w1_err, w2_err = generate_synthetic_light_curve(
            n_epochs=n_epochs,
            baseline_years=baseline_years,
            fade_rate_mmag_yr=fade_rate,
            mag_sigma=0.03,
            seed=args.seed + i,
        )
        if is_recovered(mjd, w1, w2, w1_err, w2_err, sigma_threshold=args.sigma):
            recovered += 1

    frac = recovered / args.n
    print("Injection-recovery (synthetic Y-dwarf fading light curves):")
    print(f"  Injected: {args.n}")
    print(f"  Recovered (FADING and trend >{args.sigma} sigma): {recovered}")
    print(f"  Recovery fraction: {frac:.1%}")
    print(
        f"  Thresholds: fade rate {args.fade_min}-{args.fade_max} mmag/yr, "
        f"trend > {TREND_THRESHOLD_MAG_YR * 1000:.0f} mmag/yr."
    )
    if frac >= 0.90:
        print("  Result suitable for Methods: pipeline recovers ≥90% at >3 sigma.")


if __name__ == "__main__":
    main()
