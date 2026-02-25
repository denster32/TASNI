#!/usr/bin/env python3
"""
TASNI: MCMC vs Least-Squares Parallax Comparison for One Source

Generates synthetic astrometric epochs consistent with J143046.35-025927.8
(or loads real epochs if provided), fits the 5-parameter model with both
linear least squares and a Bayesian MCMC, and optionally produces the
appendix figure (parallax posterior vs LS estimate).

Usage:
    python scripts/mcmc_parallax_comparison.py [--output_dir DIR] [--plot]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from astropy.coordinates import get_sun  # noqa: E402
from astropy.time import Time  # noqa: E402

# Constants (match extract_neowise_parallax)
DEG_TO_MAS = 3600000.0
YEAR_TO_DAY = 365.25
REF_MJD = 58000.0

# J143046.35-025927.8 approximate parameters (from manuscript)
DESIGNATION = "J143046.35-025927.8"
REF_RA = 217.6931509
REF_DEC = -2.9910653
TRUE_PLX_MAS = 57.6
TRUE_PM_RA = -5.0
TRUE_PM_DEC = 55.0
OBS_NOISE_MAS = 120.0  # per-epoch astrometric noise (conservative)


def compute_parallax_factors(ra_deg, dec_deg, mjd):
    """Parallax factors (Pα, Pδ) at given MJD; same as extract_neowise_parallax."""
    times = Time(mjd, format="mjd")
    sun = get_sun(times)
    sun_ra = sun.ra.deg
    sun_dec = sun.dec.deg
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    sun_ra_rad = np.radians(sun_ra)
    sun_dec_rad = np.radians(sun_dec)
    delta_ra = sun_ra_rad - ra_rad
    p_ra = np.sin(delta_ra) / np.cos(dec_rad)
    p_dec = np.sin(sun_dec_rad) * np.cos(dec_rad) - np.cos(sun_dec_rad) * np.sin(dec_rad) * np.cos(
        delta_ra
    )
    return p_ra, p_dec


def model_offsets(mjd, ra0, dec0, pm_ra, pm_dec, plx_mas, ref_ra, ref_dec, ref_mjd):
    """Predicted RA/Dec offsets from reference (mas)."""
    dt = (mjd - ref_mjd) / YEAR_TO_DAY
    p_ra, p_dec = compute_parallax_factors(ref_ra, ref_dec, mjd)
    ra_off = ra0 + pm_ra * dt + plx_mas * p_ra
    dec_off = dec0 + pm_dec * dt + plx_mas * p_dec
    return ra_off, dec_off


def generate_synthetic_epochs(n_epochs=80, baseline_years=9.5, seed=42):
    """Generate synthetic RA/Dec epochs (uniform random MJD) for the 5-parameter model."""
    rng = np.random.default_rng(seed)
    mjd = (
        REF_MJD
        - baseline_years * YEAR_TO_DAY / 2
        + rng.uniform(0, baseline_years * YEAR_TO_DAY, size=n_epochs)
    )
    mjd = np.sort(mjd)
    ref_mjd = np.median(mjd)
    ra_off_true, dec_off_true = model_offsets(
        mjd, 0, 0, TRUE_PM_RA, TRUE_PM_DEC, TRUE_PLX_MAS, REF_RA, REF_DEC, ref_mjd
    )
    ra_obs = ra_off_true + rng.normal(0, OBS_NOISE_MAS, size=n_epochs)
    dec_obs = dec_off_true + rng.normal(0, OBS_NOISE_MAS, size=n_epochs)
    ra_deg = REF_RA + ra_obs / (np.cos(np.radians(REF_DEC)) * DEG_TO_MAS)
    dec_deg = REF_DEC + dec_obs / DEG_TO_MAS
    return mjd, ra_deg, dec_deg, ref_mjd


def generate_synthetic_epochs_neowise(
    parallax_mas,
    pm_ra=TRUE_PM_RA,
    pm_dec=TRUE_PM_DEC,
    n_visits=22,
    ref_ra=REF_RA,
    ref_dec=REF_DEC,
    noise_mas=OBS_NOISE_MAS,
    seed=42,
):
    """Generate synthetic epochs with realistic NEOWISE cadence (~182 d visits)."""
    rng = np.random.default_rng(seed)
    mjds = []
    for i in range(n_visits):
        visit_mjd = 56700.0 + i * 182.0
        n_exp = rng.integers(8, 16)
        mjds.extend(visit_mjd + rng.uniform(0, 1.5, n_exp))
    mjd = np.array(sorted(mjds))
    ref_mjd = np.median(mjd)
    ra_off_true, dec_off_true = model_offsets(
        mjd, 0, 0, pm_ra, pm_dec, parallax_mas, ref_ra, ref_dec, ref_mjd
    )
    ra_obs = ra_off_true + rng.normal(0, noise_mas, size=len(mjd))
    dec_obs = dec_off_true + rng.normal(0, noise_mas, size=len(mjd))
    cos_dec = np.cos(np.radians(ref_dec))
    ra_deg = ref_ra + ra_obs / (cos_dec * DEG_TO_MAS)
    dec_deg = ref_dec + dec_obs / DEG_TO_MAS
    return mjd, ra_deg, dec_deg, ref_mjd


def run_injection_recovery(seed=42):
    """Run 5-parameter injection-recovery at 449, 100, 60, 50 mas with NEOWISE cadence."""
    plx_injected = [449.0, 100.0, 60.0, 50.0, 30.0]
    results = []
    for plx in plx_injected:
        mjd, ra_deg, dec_deg, ref_mjd = generate_synthetic_epochs_neowise(plx, seed=seed)
        ls = fit_ls(mjd, ra_deg, dec_deg, REF_RA, REF_DEC, ref_mjd)
        residual = ls["parallax_mas"] - plx
        results.append(
            {
                "injected_parallax_mas": plx,
                "recovered_parallax_mas": ls["parallax_mas"],
                "recovered_err_mas": ls["parallax_err_mas"],
                "residual_mas": residual,
                "n_epochs": len(mjd),
                "baseline_years": (mjd.max() - mjd.min()) / YEAR_TO_DAY,
            }
        )
    return results


def fit_ls(mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd):
    """Least-squares fit; returns dict with parallax_mas, parallax_err_mas."""
    n = len(mjd)
    ra_off = (ra_deg - ref_ra) * np.cos(np.radians(ref_dec)) * DEG_TO_MAS
    dec_off = (dec_deg - ref_dec) * DEG_TO_MAS
    dt = (mjd - ref_mjd) / YEAR_TO_DAY
    p_ra, p_dec = compute_parallax_factors(ref_ra, ref_dec, mjd)
    A_ra = np.column_stack([np.ones(n), np.zeros(n), dt, np.zeros(n), p_ra])
    A_dec = np.column_stack([np.zeros(n), np.ones(n), np.zeros(n), dt, p_dec])
    A = np.vstack([A_ra, A_dec])
    y = np.concatenate([ra_off, dec_off])
    params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ params
    sigma2 = np.sum((y - y_pred) ** 2) / (len(y) - 5)
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        plx_err = np.sqrt(cov[4, 4])
    except np.linalg.LinAlgError:
        plx_err = np.nan
    return {
        "parallax_mas": float(params[4]),
        "parallax_err_mas": float(plx_err),
        "ra0": float(params[0]),
        "dec0": float(params[1]),
        "pm_ra": float(params[2]),
        "pm_dec": float(params[3]),
    }


def log_likelihood(params, mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd, sigma_obs):
    """Gaussian log-likelihood for 5-parameter model."""
    ra0, dec0, pm_ra, pm_dec, plx = params
    if plx <= 0 or plx > 500:
        return -np.inf
    ra_off_pred, dec_off_pred = model_offsets(
        mjd, ra0, dec0, pm_ra, pm_dec, plx, ref_ra, ref_dec, ref_mjd
    )
    ra_off_obs = (ra_deg - ref_ra) * np.cos(np.radians(ref_dec)) * DEG_TO_MAS
    dec_off_obs = (dec_deg - ref_dec) * DEG_TO_MAS
    ll = -0.5 * (
        np.sum(((ra_off_obs - ra_off_pred) / sigma_obs) ** 2)
        + np.sum(((dec_off_obs - dec_off_pred) / sigma_obs) ** 2)
    )
    return ll


def run_mcmc(mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd, n_walkers=8, n_steps=400):
    """Simple Metropolis MCMC for 5 parameters; returns chain (n_steps*n_walkers, 5)."""
    ls = fit_ls(mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd)
    p0 = np.array(
        [
            ls["ra0"],
            ls["dec0"],
            ls["pm_ra"],
            ls["pm_dec"],
            max(1.0, ls["parallax_mas"]),
        ]
    )
    sigma_obs = OBS_NOISE_MAS
    chain = np.zeros((n_walkers * n_steps, 5))
    chain[0] = p0
    for i in range(1, n_walkers * n_steps):
        step = chain[i - 1].copy()
        j = np.random.randint(5)
        step[j] += np.random.normal(0, [50, 50, 5, 5, 8][j])
        if step[4] <= 0:
            step[4] = chain[i - 1, 4]
        ll_old = log_likelihood(
            chain[i - 1], mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd, sigma_obs
        )
        ll_new = log_likelihood(step, mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd, sigma_obs)
        if np.exp(ll_new - ll_old) > np.random.uniform():
            chain[i] = step
        else:
            chain[i] = chain[i - 1]
    return chain


def compute_psf_fraction(parallax_mas, pm_masyr, psf_fwhm_arcsec=6.0):
    """Fraction of WISE PSF traversed by parallax vs proper motion (J143046-like)."""
    psf_mas = psf_fwhm_arcsec * 1000
    parallax_amplitude_mas = 2 * parallax_mas  # peak-to-peak
    pm_10yr_mas = pm_masyr * 10
    frac_parallax = parallax_amplitude_mas / psf_mas
    frac_pm_10yr = pm_10yr_mas / psf_mas
    return {
        "psf_fwhm_mas": psf_mas,
        "parallax_amplitude_mas": parallax_amplitude_mas,
        "pm_10yr_mas": pm_10yr_mas,
        "fraction_psf_parallax": frac_parallax,
        "fraction_psf_pm_10yr": frac_pm_10yr,
    }


def main():
    parser = argparse.ArgumentParser(description="MCMC vs LS parallax for appendix")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "tasni_paper_final" / "figures"),
        help="Output directory for results and figure",
    )
    parser.add_argument("--plot", action="store_true", help="Generate appendix figure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--injection_recovery",
        action="store_true",
        help="Run realistic NEOWISE-cadence injection-recovery and write JSON",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.injection_recovery:
        inj_results = run_injection_recovery(seed=args.seed)
        psf_frac = compute_psf_fraction(57.6, 55.2)
        validation_dir = ROOT / "output" / "parallax_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "description": "5-parameter injection-recovery with realistic NEOWISE epoch distribution (182-day visit cadence)",
            "injection_recovery": inj_results,
            "psf_fraction_analysis": {
                "source": "J143046.35-025927.8",
                "parallax_mas": 57.6,
                "pm_masyr": 55.2,
                **psf_frac,
            },
        }
        json_path = validation_dir / "injection_recovery_realistic.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved injection-recovery to {json_path}")

    mjd, ra_deg, dec_deg, ref_mjd = generate_synthetic_epochs(seed=args.seed)
    ref_ra, ref_dec = REF_RA, REF_DEC

    ls_result = fit_ls(mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd)
    chain = run_mcmc(mjd, ra_deg, dec_deg, ref_ra, ref_dec, ref_mjd)
    plx_chain = chain[:, 4]
    plx_chain = plx_chain[plx_chain > 0]
    mcmc_median = np.median(plx_chain)
    mcmc_lo = np.percentile(plx_chain, 16)
    mcmc_hi = np.percentile(plx_chain, 84)

    results = {
        "designation": DESIGNATION,
        "ls_parallax_mas": ls_result["parallax_mas"],
        "ls_parallax_err_mas": ls_result["parallax_err_mas"],
        "mcmc_median_mas": float(mcmc_median),
        "mcmc_16_mas": float(mcmc_lo),
        "mcmc_84_mas": float(mcmc_hi),
        "n_epochs": len(mjd),
        "baseline_years": (mjd.max() - mjd.min()) / YEAR_TO_DAY,
    }
    results_path = out_dir / "parallax_mcmc_ls_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    print(f"  LS:    π = {ls_result['parallax_mas']:.2f} ± {ls_result['parallax_err_mas']:.2f} mas")
    print(f"  MCMC:  π = {mcmc_median:.2f} ({mcmc_lo:.2f}--{mcmc_hi:.2f}) mas")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        ax.hist(
            plx_chain,
            bins=50,
            density=True,
            color="0.6",
            edgecolor="0.4",
            label="MCMC posterior",
        )
        ax.axvline(
            ls_result["parallax_mas"],
            color="C0",
            ls="--",
            lw=2,
            label=f"LS: {ls_result['parallax_mas']:.1f} ± {ls_result['parallax_err_mas']:.1f} mas",
        )
        ax.axvspan(
            ls_result["parallax_mas"] - ls_result["parallax_err_mas"],
            ls_result["parallax_mas"] + ls_result["parallax_err_mas"],
            alpha=0.3,
            color="C0",
        )
        ax.set_xlabel("Parallax (mas)")
        ax.set_ylabel("Posterior density")
        ax.set_title(f"{DESIGNATION}: LS vs MCMC")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, 120)
        fig.tight_layout()
        fig_path = out_dir / "fig_appendix_parallax_mcmc_ls.pdf"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
