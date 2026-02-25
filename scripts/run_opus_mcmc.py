import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasni.analysis.extract_neowise_parallax import compute_parallax_factors, fit_astrometry

ROOT = Path("/home/server/tasni")


def model_offsets(p_ra, p_dec, mjd, ra0, dec0, pm_ra, pm_dec, plx_mas, ref_ra, ref_dec, ref_mjd):
    from tasni.analysis.extract_neowise_parallax import YEAR_TO_DAY

    dt = (mjd - ref_mjd) / YEAR_TO_DAY
    ra_off = ra0 + pm_ra * dt + plx_mas * p_ra
    dec_off = dec0 + pm_dec * dt + plx_mas * p_dec
    return ra_off, dec_off


def log_prior(plx):
    if plx <= 0 or plx > 1000:
        return -np.inf
    # Uniform volume density prior: p(r) ~ r^2 => p(plx) ~ 1/plx^4
    return -4.0 * np.log(plx)


def log_likelihood(
    params, p_ra, p_dec, mjd, ra_obs, dec_obs, ref_ra, ref_dec, ref_mjd, sigma_ra, sigma_dec
):
    ra0, dec0, pm_ra, pm_dec, plx = params

    prior = log_prior(plx)
    if not np.isfinite(prior):
        return -np.inf

    ra_off_pred, dec_off_pred = model_offsets(
        p_ra, p_dec, mjd, ra0, dec0, pm_ra, pm_dec, plx, ref_ra, ref_dec, ref_mjd
    )
    DEG_TO_MAS = 3600000.0
    ra_off_obs = (ra_obs - ref_ra) * np.cos(np.radians(ref_dec)) * DEG_TO_MAS
    dec_off_obs = (dec_obs - ref_dec) * DEG_TO_MAS

    ll = -0.5 * (
        np.sum(((ra_off_obs - ra_off_pred) / sigma_ra) ** 2)
        + np.sum(((dec_off_obs - dec_off_pred) / sigma_dec) ** 2)
    )
    return ll + prior


def run_mcmc(mjd, ra_obs, dec_obs, ref_ra, ref_dec, ref_mjd, ls_result, n_walkers=16, n_steps=2000):
    p0 = np.array(
        [
            ls_result["ra0"],
            ls_result["dec0"],
            ls_result["pm_ra_fit"],
            ls_result["pm_dec_fit"],
            max(1.0, ls_result["parallax_mas"]),
        ]
    )

    # Use empirical scatter as sigma
    sigma_ra = max(60.0, ls_result["rms_ra_mas"])
    sigma_dec = max(60.0, ls_result["rms_dec_mas"])

    chain = np.zeros((n_walkers * n_steps, 5))
    chain[0] = p0

    p_ra, p_dec = compute_parallax_factors(ref_ra, ref_dec, mjd)

    accepted = 0
    # Simple Metropolis
    for i in range(1, n_walkers * n_steps):
        step = chain[i - 1].copy()

        # Propose jump
        j = np.random.randint(5)
        step[j] += np.random.normal(0, [20, 20, 5, 5, 5][j])

        ll_old = log_likelihood(
            chain[i - 1],
            p_ra,
            p_dec,
            mjd,
            ra_obs,
            dec_obs,
            ref_ra,
            ref_dec,
            ref_mjd,
            sigma_ra,
            sigma_dec,
        )
        ll_new = log_likelihood(
            step, p_ra, p_dec, mjd, ra_obs, dec_obs, ref_ra, ref_dec, ref_mjd, sigma_ra, sigma_dec
        )

        if np.exp(ll_new - ll_old) > np.random.uniform():
            chain[i] = step
            accepted += 1
        else:
            chain[i] = chain[i - 1]

    # Burn-in 50%
    burn_in = n_walkers * n_steps // 2
    return chain[burn_in:]


def process_source(desig, epochs_df, ax):
    df = epochs_df[epochs_df["designation"] == desig].copy()
    df = df[~df["ra"].isna() & ~df["dec"].isna() & ~df["mjd"].isna()]

    ref_ra = df["target_ra"].iloc[0]
    ref_dec = df["target_dec"].iloc[0]

    mjd = df["mjd"].values
    ra_obs = df["ra"].values
    dec_obs = df["dec"].values

    ls_result = fit_astrometry(ra_obs, dec_obs, mjd, ref_ra, ref_dec)

    chain = run_mcmc(mjd, ra_obs, dec_obs, ref_ra, ref_dec, ls_result["ref_mjd"], ls_result)
    plx_chain = chain[:, 4]

    plx_med = np.median(plx_chain)
    plx_16 = np.percentile(plx_chain, 16)
    plx_84 = np.percentile(plx_chain, 84)

    ax.hist(plx_chain, bins=40, density=True, color="0.6", edgecolor="0.4", label="MCMC Posterior")
    ax.axvline(
        ls_result["parallax_mas"],
        color="C0",
        ls="--",
        lw=2,
        label=f"LS: {ls_result['parallax_mas']:.1f} +- {ls_result['parallax_err_mas']:.1f}",
    )
    ax.axvspan(
        ls_result["parallax_mas"] - ls_result["parallax_err_mas"],
        ls_result["parallax_mas"] + ls_result["parallax_err_mas"],
        alpha=0.3,
        color="C0",
    )

    ax.axvline(
        plx_med,
        color="C1",
        ls="-",
        lw=2,
        label=f"MCMC: {plx_med:.1f}^{{+{plx_84-plx_med:.1f}}}_{{{plx_16-plx_med:.1f}}}",
    )

    ax.set_xlabel("Parallax (mas)")
    ax.set_ylabel("Density")
    ax.set_title(desig)
    ax.legend(loc="upper right", fontsize=8)

    return plx_med, plx_84 - plx_med, plx_med - plx_16, ls_result


def main():
    epochs_path = ROOT / "output/final/neowise_epochs.parquet"
    epochs_df = pd.read_parquet(epochs_path)

    sources = ["J143046.35-025927.8", "J231029.40-060547.3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    results = {}

    for i, desig in enumerate(sources):
        med, err_hi, err_lo, ls_result = process_source(desig, epochs_df, axes[i])
        results[desig] = {
            "mcmc_plx": med,
            "mcmc_err_hi": err_hi,
            "mcmc_err_lo": err_lo,
            "ls_plx": ls_result["parallax_mas"],
            "ls_err": ls_result["parallax_err_mas"],
        }

    fig.tight_layout()
    output_dir = ROOT / "tasni_paper_final/figures"
    fig.savefig(output_dir / "fig_appendix_parallax_mcmc_ls.pdf")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
