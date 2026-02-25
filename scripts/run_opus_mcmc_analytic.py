import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasni.analysis.extract_neowise_parallax import fit_astrometry

ROOT = Path("/home/server/tasni")


def compute_analytic_posterior(ls_plx, ls_err, plx_grid):
    # Gaussian likelihood from marginalized linear model
    like = np.exp(-0.5 * ((plx_grid - ls_plx) / ls_err) ** 2)
    # L-Z volume uniform prior with exponential decrease: p(d) ~ d^2 exp(-d/L)
    # L = 30 pc for local brown dwarfs. d = 1000/plx
    L = 30.0
    prior = (1.0 / (plx_grid**4)) * np.exp(-1000.0 / (L * plx_grid))

    post = like * prior
    post /= np.trapezoid(post, plx_grid)

    cdf = cumulative_trapezoid(post, plx_grid, initial=0)
    cdf /= cdf[-1]

    med = np.interp(0.5, cdf, plx_grid)
    lo = np.interp(0.16, cdf, plx_grid)
    hi = np.interp(0.84, cdf, plx_grid)

    # Compute the mode (MAP)
    mode = plx_grid[np.argmax(post)]

    return post, med, hi - med, med - lo, mode


def process_source(desig, epochs_df, ax):
    df = epochs_df[epochs_df["designation"] == desig].copy()
    df = df[~df["ra"].isna() & ~df["dec"].isna() & ~df["mjd"].isna()]

    ref_ra = df["target_ra"].iloc[0]
    ref_dec = df["target_dec"].iloc[0]

    mjd = df["mjd"].values
    ra_obs = df["ra"].values
    dec_obs = df["dec"].values

    ls_result = fit_astrometry(ra_obs, dec_obs, mjd, ref_ra, ref_dec)

    ls_plx = ls_result["parallax_mas"]
    ls_err = ls_result["parallax_err_mas"]

    # Evaluate on a fine grid from 0.1 to 150 mas
    plx_grid = np.linspace(0.1, 150, 10000)
    post, med, err_hi, err_lo, mode = compute_analytic_posterior(ls_plx, ls_err, plx_grid)

    ax.plot(plx_grid, post, color="0.4", lw=2, label="Analytic Posterior")
    ax.fill_between(plx_grid, 0, post, color="0.6", alpha=0.5)

    ax.axvline(ls_plx, color="C0", ls="--", lw=2, label=f"LS: {ls_plx:.1f} ± {ls_err:.1f} mas")
    ax.axvspan(ls_plx - ls_err, ls_plx + ls_err, alpha=0.2, color="C0")

    ax.axvline(
        med, color="C1", ls="-", lw=2, label=f"Bayes: {med:.1f}^{{+{err_hi:.1f}}}_{{{err_lo:.1f}}}"
    )

    ax.set_xlabel("Parallax (mas)")
    ax.set_ylabel("Posterior Density")
    ax.set_title(desig)
    ax.legend(loc="upper right", fontsize=8)

    # Make sure limits are sensible
    ax.set_xlim(0, max(120, ls_plx + 3 * ls_err))

    return med, err_hi, err_lo, mode, ls_result


def main():
    epochs_path = ROOT / "output/final/neowise_epochs.parquet"
    if not epochs_path.exists():
        epochs_path = ROOT / "data/processed/final/neowise_epochs.parquet"
    epochs_df = pd.read_parquet(epochs_path)

    sources = ["J143046.35-025927.8", "J231029.40-060547.3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    results = {}

    for i, desig in enumerate(sources):
        med, err_hi, err_lo, mode, ls_result = process_source(desig, epochs_df, axes[i])
        results[desig] = {
            "bayes_plx": float(med),
            "bayes_err_hi": float(err_hi),
            "bayes_err_lo": float(err_lo),
            "bayes_mode": float(mode),
            "ls_plx": float(ls_result["parallax_mas"]),
            "ls_err": float(ls_result["parallax_err_mas"]),
        }

    fig.tight_layout()
    output_dir = ROOT / "tasni_paper_final/figures"
    fig.savefig(output_dir / "fig_appendix_parallax_mcmc_ls.pdf")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
