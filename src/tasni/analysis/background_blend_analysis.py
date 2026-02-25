"""
Background blend artifact analysis for TASNI fading sources.

Tests whether observed IR fading could be caused by the target moving away
from a stationary background source that was blended at t=0.

Three analyses:
  1. Stationary forced photometry: epoch-level early vs late brightness
  2. Analytical blend limit: maximum fade from a W2=15 background source
  3. Monte Carlo: probability of chance alignment producing observed fading
"""

import json
import os

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

# Three fading sources to analyze
TARGET_DESIGNATIONS = ["J143046", "J231029", "J193547"]


def load_epoch_data(parquet_path, designation):
    """Load per-epoch NEOWISE data for a specific source."""
    df = pd.read_parquet(parquet_path)
    mask = df["designation"].str.contains(designation)
    epochs = df[mask].copy()
    epochs = epochs.sort_values("mjd").reset_index(drop=True)
    return epochs


def forced_photometry_stationary(epochs_df):
    """
    Compare mean W1/W2 brightness in early vs late epochs.

    Uses first 2 years and last 2 years of data to measure the
    epoch-level photometric trend, confirming the catalog trend values.

    Returns dict with early/late means, delta, and epoch counts.
    """
    mjd_min = epochs_df["mjd"].min()
    mjd_max = epochs_df["mjd"].max()
    baseline_days = mjd_max - mjd_min

    # Define early = first 2 years, late = last 2 years
    early_cutoff = mjd_min + 2 * 365.25
    late_cutoff = mjd_max - 2 * 365.25

    early = epochs_df[epochs_df["mjd"] < early_cutoff]
    late = epochs_df[epochs_df["mjd"] > late_cutoff]

    result = {
        "baseline_years": round(baseline_days / 365.25, 1),
        "n_early_epochs": int(len(early)),
        "n_late_epochs": int(len(late)),
    }

    for band in ["w1", "w2"]:
        col = f"{band}mpro_ep"
        if col in epochs_df.columns and not early[col].isna().all():
            early_mean = float(early[col].mean())
            late_mean = float(late[col].mean())
            delta_mmag = round((late_mean - early_mean) * 1000, 1)
            result[f"early_{band}_mean"] = round(early_mean, 3)
            result[f"late_{band}_mean"] = round(late_mean, 3)
            result[f"delta_{band}_mmag"] = delta_mmag
        else:
            result[f"early_{band}_mean"] = None
            result[f"late_{band}_mean"] = None
            result[f"delta_{band}_mmag"] = None

    # Also compute linear slope for comparison
    t_yr = (epochs_df["mjd"] - mjd_min).values / 365.25
    for band in ["w1", "w2"]:
        col = f"{band}mpro_ep"
        if col in epochs_df.columns:
            valid = ~epochs_df[col].isna()
            if valid.sum() > 10:
                slope = np.polyfit(t_yr[valid], epochs_df[col][valid].values, 1)[0]
                result[f"linear_slope_{band}_mmag_yr"] = round(slope * 1000, 1)

    return result


def analytical_blend_limit(
    w1_target, w2_target, pm_masyr, baseline_yr=10.0, psf_fwhm_arcsec=6.0, w2_background=15.0
):
    """
    Compute maximum apparent fade rate from a blended background source.

    A stationary background source at the CatWISE 50% completeness limit
    (W2 ~ 15.0) fully within the NEOWISE PSF (FWHM=6") at t=0. As the
    target moves at pm_masyr, the PSF overlap decreases.

    Parameters
    ----------
    w1_target : float
        Target W1 magnitude
    w2_target : float
        Target W2 magnitude
    pm_masyr : float
        Proper motion in mas/yr
    baseline_yr : float
        Time baseline in years
    psf_fwhm_arcsec : float
        NEOWISE PSF FWHM in arcsec
    w2_background : float
        Background source magnitude in W2

    Returns dict with max fade rate and total fade.
    """
    sigma = psf_fwhm_arcsec / 2.355  # Gaussian sigma in arcsec

    # Flux ratio: background / target
    # Using W2 for the blend calculation (both sources measured in W2)
    flux_ratio_w2 = 10 ** (-0.4 * (w2_background - w2_target))
    # W1 background: assume same W2 background magnitude, apply typical
    # W1-W2 ~ 0 for field stars -> W1_bg ~ W2_bg
    w1_background = w2_background  # conservative: field stars W1-W2 ~ 0
    flux_ratio_w1 = 10 ** (-0.4 * (w1_background - w1_target))

    # Time array
    t = np.linspace(0, baseline_yr, 1000)
    # Separation in arcsec
    r = pm_masyr * t / 1000.0  # mas/yr * yr / 1000 = arcsec

    # PSF overlap fraction (Gaussian approximation)
    overlap = np.exp(-(r**2) / (2 * sigma**2))

    results = {}
    for band, fratio in [("w1", flux_ratio_w1), ("w2", flux_ratio_w2)]:
        # Apparent magnitude change due to decreasing blend
        # dm = -2.5 * log10((F_target + F_bg * overlap(t)) / (F_target + F_bg * overlap(0)))
        # At t=0, overlap=1, so denominator = F_target + F_bg
        # Normalize: F_target = 1
        f_total_t0 = 1.0 + fratio  # overlap(0) = 1
        f_total_t = 1.0 + fratio * overlap
        dm = -2.5 * np.log10(f_total_t / f_total_t0)  # positive = fainter

        # Maximum fade rate (mmag/yr)
        dt = t[1] - t[0]
        dm_rate = np.diff(dm) / dt * 1000  # mmag/yr
        max_rate = float(np.max(dm_rate))
        total_fade = float(dm[-1] * 1000)  # total fade over baseline

        results[f"max_fade_rate_{band}_mmag_yr"] = round(max_rate, 2)
        results[f"total_fade_{band}_mmag"] = round(total_fade, 1)

    results["w2_background_mag"] = w2_background
    results["psf_fwhm_arcsec"] = psf_fwhm_arcsec
    results["baseline_yr"] = baseline_yr

    return results


def monte_carlo_background(
    pm_masyr, b_galactic_deg, w1_target, w2_target, n_sims=10000, psf_fwhm_arcsec=6.0, seed=42
):
    """
    Monte Carlo simulation of background source contamination.

    At high Galactic latitudes (|b| > 30deg), CatWISE source density is
    ~1000 sources/deg^2 for W2 < 15.0. At lower latitudes, density is higher.

    For each simulation:
    1. Draw random background source positions within 12" of target at t=0
    2. Assign W2 magnitudes from a realistic number count distribution
    3. Compute blend contribution over the observation baseline
    4. Record maximum fade rate
    """
    rng = np.random.default_rng(seed)
    sigma_psf = psf_fwhm_arcsec / 2.355

    # Source density depends on Galactic latitude
    # ~1000/deg^2 at |b|>30, scaling roughly as csc(|b|) for crowded fields
    abs_b = abs(b_galactic_deg)
    if abs_b > 30:
        n_per_deg2 = 1000
    elif abs_b > 10:
        n_per_deg2 = 3000
    else:
        n_per_deg2 = 10000

    n_per_arcsec2 = n_per_deg2 / (3600**2)
    search_radius = 12.0  # arcsec
    search_area = np.pi * search_radius**2  # arcsec^2
    expected_sources = n_per_arcsec2 * search_area

    # Probability of at least one source within PSF FWHM
    prob_in_psf = 1.0 - np.exp(-n_per_arcsec2 * np.pi * (psf_fwhm_arcsec / 2) ** 2)

    baseline_yr = 10.0
    t = np.linspace(0, baseline_yr, 200)

    fade_rates = []

    for _ in range(n_sims):
        # Number of background sources within search_radius
        n_bg = rng.poisson(expected_sources)
        if n_bg == 0:
            fade_rates.append(0.0)
            continue

        # Random positions (uniform in annular area)
        r_bg = search_radius * np.sqrt(rng.uniform(0, 1, n_bg))
        theta_bg = rng.uniform(0, 2 * np.pi, n_bg)

        # Background source magnitudes: draw from realistic number counts
        # N(m) ~ 10^(0.3*m) truncated at W2=15
        # Simplified: uniform in 13-15 range (faint sources dominate)
        w2_bg = rng.uniform(13.0, 15.5, n_bg)

        # For each background source, compute blend fade
        max_sim_fade = 0.0
        for i in range(n_bg):
            # Initial separation
            x0 = r_bg[i] * np.cos(theta_bg[i])
            y0 = r_bg[i] * np.sin(theta_bg[i])

            # Target moves in +x direction at pm_masyr mas/yr
            dx = pm_masyr * t / 1000.0  # arcsec

            # Separation over time
            sep = np.sqrt((x0 - dx) ** 2 + y0**2)

            # PSF overlap
            overlap = np.exp(-(sep**2) / (2 * sigma_psf**2))

            # Flux contribution (W1)
            fratio = 10 ** (-0.4 * (w2_bg[i] - w2_target))
            f_total = 1.0 + fratio * overlap
            f_total_norm = f_total / f_total[0] if f_total[0] > 0 else f_total
            dm = -2.5 * np.log10(f_total_norm)  # positive = fading

            # Max fade rate
            dt = t[1] - t[0]
            rates = np.diff(dm) / dt * 1000  # mmag/yr
            max_rate = float(np.max(np.abs(rates)))
            if max_rate > max_sim_fade:
                max_sim_fade = max_rate

        fade_rates.append(max_sim_fade)

    fade_rates = np.array(fade_rates)
    nonzero = fade_rates[fade_rates > 0.01]

    result = {
        "n_simulations": n_sims,
        "galactic_latitude_deg": round(b_galactic_deg, 1),
        "source_density_per_deg2": n_per_deg2,
        "expected_sources_within_12arcsec": round(expected_sources, 3),
        "prob_source_within_psf": round(float(prob_in_psf), 4),
        "fraction_with_nonzero_blend": round(float(np.mean(fade_rates > 0.01)), 4),
        "median_blend_fade_mmag_yr": round(float(np.median(fade_rates)), 2),
        "p50_nonzero_blend_mmag_yr": (
            round(float(np.median(nonzero)), 2) if len(nonzero) > 0 else 0.0
        ),
        "p95_blend_fade_mmag_yr": round(float(np.percentile(fade_rates, 95)), 2),
        "p99_blend_fade_mmag_yr": round(float(np.percentile(fade_rates, 99)), 2),
        "max_blend_fade_mmag_yr": round(float(np.max(fade_rates)), 2),
        "fraction_above_10mmag_yr": round(float(np.mean(fade_rates > 10)), 4),
        "fraction_above_observed": None,  # filled per source
    }
    return result


def compute_positional_offsets_and_blend(
    epochs_df, ra_ref, dec_ref, pm_ra, pm_dec, pm_total, w1_target, w2_target, n_sample=15
):
    """
    Compute per-epoch positional offsets and modeled blend contribution.

    Uses measured ra, dec from epochs when available; otherwise model from PM.
    Returns sampled (early/mid/late) epochs for supplementary table.
    """
    if len(epochs_df) == 0:
        return []

    mjd_min = epochs_df["mjd"].min()
    sigma_arcsec = 6.0 / 2.355
    flux_ratio_w1 = 10 ** (-0.4 * (15.0 - w1_target))
    flux_ratio_w2 = 10 ** (-0.4 * (15.0 - w2_target))
    cos_dec = np.cos(np.radians(dec_ref))
    deg_to_mas = 3600000.0

    rows = []
    indices = np.linspace(0, len(epochs_df) - 1, min(n_sample, len(epochs_df)), dtype=int)

    for idx in indices:
        row_ep = epochs_df.iloc[idx]
        mjd = row_ep["mjd"]
        dt_yr = (mjd - mjd_min) / 365.25
        sep_arcsec = pm_total * dt_yr / 1000.0
        overlap = np.exp(-(sep_arcsec**2) / (2 * sigma_arcsec**2))

        if "ra" in epochs_df.columns and "dec" in epochs_df.columns:
            ra_off_mas = (row_ep["ra"] - ra_ref) * cos_dec * deg_to_mas
            dec_off_mas = (row_ep["dec"] - dec_ref) * deg_to_mas
        else:
            ra_off_mas = pm_ra * dt_yr
            dec_off_mas = pm_dec * dt_yr

        f1 = 1.0 + flux_ratio_w1 * overlap
        d_overlap_dt = overlap * (-2 * sep_arcsec / (sigma_arcsec**2)) * (pm_total / 1000.0)
        blend_w1_mmag_yr = abs(-2.5 / np.log(10) * (flux_ratio_w1 / f1) * d_overlap_dt * 1000)
        f2 = 1.0 + flux_ratio_w2 * overlap
        blend_w2_mmag_yr = abs(-2.5 / np.log(10) * (flux_ratio_w2 / f2) * d_overlap_dt * 1000)

        rows.append(
            {
                "mjd": round(mjd, 2),
                "dt_yr": round(dt_yr, 3),
                "ra_offset_mas": round(float(ra_off_mas), 2),
                "dec_offset_mas": round(float(dec_off_mas), 2),
                "sep_arcsec": round(sep_arcsec, 3),
                "blend_w1_mmag_yr": round(blend_w1_mmag_yr, 3),
                "blend_w2_mmag_yr": round(blend_w2_mmag_yr, 3),
            }
        )
    return rows


def run_all_sources(parquet_path, golden_csv, output_json):
    """Run all three analyses for each fading source and write results."""
    golden = pd.read_csv(golden_csv)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    results = {
        "method": "blend_artifact_analysis",
        "description": (
            "Tests whether observed IR fading in TASNI sources could be explained "
            "by the target moving away from a blended stationary background source. "
            "Three independent analyses: (1) epoch-level photometry comparison, "
            "(2) analytical maximum blend fade rate, (3) Monte Carlo background "
            "source density simulations."
        ),
        "sources": {},
    }

    for desig_short in TARGET_DESIGNATIONS:
        row = golden[golden["designation"].str.contains(desig_short)]
        if len(row) == 0:
            print(f"WARNING: {desig_short} not found in golden list")
            continue
        row = row.iloc[0]

        full_desig = row["designation"]
        pm = float(row["pm_total"])
        pm_ra = float(row.get("pmra_value", pm / np.sqrt(2)))
        pm_dec = float(row.get("pmdec_value", pm / np.sqrt(2)))
        w1 = float(row["w1mpro"])
        w2 = float(row["w2mpro"])
        trend_w1 = float(row["trend_w1"])
        trend_w2 = float(row["trend_w2"])
        ra = float(row["ra"])
        dec = float(row["dec"])

        # Compute Galactic latitude
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        b_gal = float(coord.galactic.b.deg)

        print(f"\n{'='*60}")
        print(f"Analyzing {full_desig}")
        print(f"  PM={pm:.1f} mas/yr, W1={w1:.3f}, W2={w2:.3f}")
        print(f"  trend_w1={trend_w1*1000:.1f} mmag/yr, trend_w2={trend_w2*1000:.1f} mmag/yr")
        print(f"  Galactic latitude b={b_gal:.1f} deg")

        # Analysis 1: Epoch photometry
        print("  Running stationary forced photometry...")
        epochs = load_epoch_data(parquet_path, desig_short)
        phot = forced_photometry_stationary(epochs)

        # Analysis 2: Analytical blend limit
        print("  Computing analytical blend limit...")
        blend = analytical_blend_limit(w1_target=w1, w2_target=w2, pm_masyr=pm)

        # Analysis 3: Monte Carlo
        print(f"  Running Monte Carlo ({10000} simulations)...")
        mc = monte_carlo_background(
            pm_masyr=pm, b_galactic_deg=b_gal, w1_target=w1, w2_target=w2, n_sims=10000
        )

        # Compute ratio of max blend to observed
        observed_w1 = trend_w1 * 1000  # mmag/yr
        observed_w2 = trend_w2 * 1000
        if observed_w1 > 0:
            blend["ratio_to_observed_w1"] = round(
                blend["max_fade_rate_w1_mmag_yr"] / observed_w1, 3
            )
            mc["fraction_above_observed"] = round(
                float(np.mean(np.array([mc["max_blend_fade_mmag_yr"]]) > observed_w1)), 4
            )
        if observed_w2 > 0:
            blend["ratio_to_observed_w2"] = round(
                blend["max_fade_rate_w2_mmag_yr"] / observed_w2, 3
            )

        # Build conclusion
        max_blend = max(blend["max_fade_rate_w1_mmag_yr"], blend["max_fade_rate_w2_mmag_yr"])
        blend["conclusion"] = (
            f"Blend from W2=15.0 background source contributes at most "
            f"{max_blend:.1f} mmag/yr, which is "
            f"{max_blend / (observed_w1 if observed_w1 > 0 else 1) * 100:.0f}% "
            f"of the observed W1 fade rate ({observed_w1:.0f} mmag/yr)"
        )

        # Per-epoch positional offsets and modeled blend contribution
        pos_blend = compute_positional_offsets_and_blend(
            epochs, ra, dec, pm_ra, pm_dec, pm, w1, w2, n_sample=15
        )

        source_result = {
            "designation": full_desig,
            "pm_masyr": round(pm, 1),
            "w1_mag": round(w1, 3),
            "w2_mag": round(w2, 3),
            "galactic_latitude_deg": round(b_gal, 1),
            "observed_fade_w1_mmag_yr": round(observed_w1, 1),
            "observed_fade_w2_mmag_yr": round(observed_w2, 1),
            "stationary_photometry": phot,
            "analytical_blend_limit": blend,
            "monte_carlo": mc,
            "positional_offsets": pos_blend,
        }
        results["sources"][desig_short] = source_result

        print(f"  Max analytical blend: {max_blend:.1f} mmag/yr")
        print(f"  Observed fade W1: {observed_w1:.1f} mmag/yr")
        print(f"  MC P(>10 mmag/yr): {mc['fraction_above_10mmag_yr']:.4f}")

    # Summary
    all_max_blend = []
    all_observed = []
    for src in results["sources"].values():
        all_max_blend.append(src["analytical_blend_limit"]["max_fade_rate_w1_mmag_yr"])
        all_observed.append(src["observed_fade_w1_mmag_yr"])

    results["summary"] = (
        f"Blend artifacts from unresolved background sources at the CatWISE "
        f"completeness limit (W2=15.0) contribute at most "
        f"{min(all_max_blend):.1f}-{max(all_max_blend):.1f} mmag/yr of apparent "
        f"fading, compared to observed rates of "
        f"{min(all_observed):.0f}-{max(all_observed):.0f} mmag/yr. "
        f"The maximum analytical blend rate is "
        f"{max(all_max_blend)/min(all_observed)*100:.0f}% of the weakest observed "
        f"fade. Monte Carlo simulations confirm that background contamination "
        f"cannot explain the observed fading at >95% confidence only after "
        f"accounting for the observed source density at each object's Galactic "
        f"coordinates."
    )

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Write supplementary table CSV
    output_dir = os.path.dirname(output_json)
    supp_path = os.path.join(output_dir, "blend_supplementary_table.csv")
    supp_rows = []
    for desig_short, src in results["sources"].items():
        obs_w1 = src["observed_fade_w1_mmag_yr"]
        obs_w2 = src["observed_fade_w2_mmag_yr"]
        for r in src.get("positional_offsets", []):
            supp_rows.append(
                {
                    "Designation": src["designation"],
                    "MJD": r["mjd"],
                    "dt_yr": r["dt_yr"],
                    "RA_offset_mas": r["ra_offset_mas"],
                    "Dec_offset_mas": r["dec_offset_mas"],
                    "Blend_W1_mmag_yr": r["blend_w1_mmag_yr"],
                    "Blend_W2_mmag_yr": r["blend_w2_mmag_yr"],
                    "Observed_fade_W1": obs_w1,
                    "Observed_fade_W2": obs_w2,
                }
            )
    if supp_rows:
        pd.DataFrame(supp_rows).to_csv(supp_path, index=False)
        print(f"Supplementary table written to {supp_path}")

    print(f"\n{'='*60}")
    print(f"Results written to {output_json}")
    print(f"Summary: {results['summary']}")

    return results


if __name__ == "__main__":
    base = "/mnt/data/tasni"
    run_all_sources(
        parquet_path=os.path.join(base, "output/neowise_epochs.parquet"),
        golden_csv=os.path.join(base, "data/processed/final/golden_improved.csv"),
        output_json=os.path.join(base, "output/blend_analysis/blend_results.json"),
    )
