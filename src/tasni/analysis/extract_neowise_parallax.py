#!/usr/bin/env python3
"""
TASNI: Extract Trigonometric Parallax from NEOWISE Multi-Epoch Astrometry

Uses the ~10-year baseline of NEOWISE observations to fit a 5-parameter
astrometric model and extract parallax for golden targets.

Model:
    RA(t)  = RA₀  + μα* × Δt + π × Pα(t)
    Dec(t) = Dec₀ + μδ  × Δt + π × Pδ(t)

where:
    - (RA₀, Dec₀) = reference position at reference epoch
    - (μα*, μδ) = proper motion (μα* = μα × cos(Dec))
    - π = parallax
    - (Pα, Pδ) = parallax factors (from Earth's orbital motion)

Usage:
    python extract_neowise_parallax.py [--epochs FILE] [--output FILE]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import get_sun
from astropy.time import Time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [PARALLAX] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MAS_TO_DEG = 1.0 / 3600000.0  # milliarcsec to degrees
DEG_TO_MAS = 3600000.0
YEAR_TO_DAY = 365.25

# Reference epoch (midpoint of NEOWISE data)
REFERENCE_MJD = 58000.0  # ~2017.5

# Minimum requirements for parallax fit
MIN_EPOCHS = 20
MIN_BASELINE_YEARS = 2.0
MIN_PARALLAX_SNR = 2.0  # For "significant" detection


def asymmetric_distance_errors(parallax_mas, parallax_err_mas):
    """
    Compute asymmetric distance errors from the non-linear parallax-to-distance
    transformation d = 1000/pi.

    NOTE: This function is intentionally unused by the main pipeline, which uses
    symmetric first-order error propagation (distance_err = d * sigma_plx / plx)
    in fit_astrometry() for consistency with the published table values. This
    function is retained for reference and potential future use with MCMC posteriors.

    Parameters:
        parallax_mas: Parallax in milliarcseconds
        parallax_err_mas: Parallax uncertainty in milliarcseconds

    Returns:
        d: Distance in parsecs
        d_upper: Upper (positive) distance error (d_far - d)
        d_lower: Lower (positive) distance error (d - d_near)
    """
    if parallax_mas <= 0 or parallax_mas <= parallax_err_mas:
        return np.nan, np.nan, np.nan
    d = 1000.0 / parallax_mas
    d_upper = 1000.0 / (parallax_mas - parallax_err_mas) - d
    d_lower = d - 1000.0 / (parallax_mas + parallax_err_mas)
    return d, d_upper, d_lower


def compute_parallax_factors(ra_deg, dec_deg, mjd):
    """
    Compute parallax factors (Pα, Pδ) at given times.

    The parallax ellipse in equatorial coordinates depends on
    the Sun's position relative to the target.

    Parameters:
        ra_deg: Target RA in degrees
        dec_deg: Target Dec in degrees
        mjd: Array of MJD times

    Returns:
        p_ra: Parallax factor in RA direction (dimensionless, multiply by π for mas offset)
        p_dec: Parallax factor in Dec direction
    """
    # Convert MJD to astropy Time
    times = Time(mjd, format="mjd")

    # Get Sun position at each epoch
    sun = get_sun(times)
    sun_ra = sun.ra.deg
    sun_dec = sun.dec.deg

    # Target position
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    # Sun position in radians
    sun_ra_rad = np.radians(sun_ra)
    sun_dec_rad = np.radians(sun_dec)

    # Parallax factors in equatorial coordinates
    # These project the Earth-Sun baseline onto the sky at the target position
    # See e.g., Green (1985) "Spherical Astronomy" or Lindegren (2012)

    # Simplified parallax factors (ignoring higher-order terms)
    # Pα = sin(sun_ra - target_ra) / cos(target_dec)
    # Pδ = sin(sun_dec)*cos(target_dec) - cos(sun_dec)*sin(target_dec)*cos(sun_ra - target_ra)

    delta_ra = sun_ra_rad - ra_rad

    p_ra = np.sin(delta_ra) / np.cos(dec_rad)
    p_dec = np.sin(sun_dec_rad) * np.cos(dec_rad) - np.cos(sun_dec_rad) * np.sin(dec_rad) * np.cos(
        delta_ra
    )

    return p_ra, p_dec


def astrometric_model(t_mjd, ra0, dec0, pm_ra, pm_dec, parallax, ref_ra, ref_dec, ref_mjd):
    """
    5-parameter astrometric model.

    Parameters:
        t_mjd: MJD times
        ra0: Reference RA (mas offset from ref_ra)
        dec0: Reference Dec (mas offset from ref_dec)
        pm_ra: Proper motion in RA (mas/yr), already includes cos(dec)
        pm_dec: Proper motion in Dec (mas/yr)
        parallax: Parallax in mas
        ref_ra, ref_dec: Reference position (degrees)
        ref_mjd: Reference epoch

    Returns:
        ra_offset: RA offset from reference (mas)
        dec_offset: Dec offset from reference (mas)
    """
    # Time since reference epoch in years
    dt_years = (t_mjd - ref_mjd) / YEAR_TO_DAY

    # Parallax factors
    p_ra, p_dec = compute_parallax_factors(ref_ra, ref_dec, t_mjd)

    # Model: position = reference + proper motion + parallax
    ra_offset = ra0 + pm_ra * dt_years + parallax * p_ra
    dec_offset = dec0 + pm_dec * dt_years + parallax * p_dec

    return ra_offset, dec_offset


def fit_astrometry(ra_obs, dec_obs, mjd, ref_ra, ref_dec, pm_prior=None):
    """
    Fit 5-parameter astrometric model to observations.

    Parameters:
        ra_obs: Observed RA (degrees)
        dec_obs: Observed Dec (degrees)
        mjd: MJD of observations
        ref_ra, ref_dec: Reference position (degrees)
        pm_prior: Prior proper motion [pmra, pmdec] in mas/yr (optional)

    Returns:
        dict with fitted parameters and uncertainties
    """
    n_obs = len(mjd)

    # Convert positions to offsets from reference (in mas)
    ra_offset = (ra_obs - ref_ra) * np.cos(np.radians(ref_dec)) * DEG_TO_MAS
    dec_offset = (dec_obs - ref_dec) * DEG_TO_MAS

    # Reference epoch (middle of data)
    ref_mjd = np.median(mjd)

    # Baseline in years
    baseline_years = (mjd.max() - mjd.min()) / YEAR_TO_DAY

    # Compute parallax factors
    p_ra, p_dec = compute_parallax_factors(ref_ra, ref_dec, mjd)

    # Time since reference in years
    dt_years = (mjd - ref_mjd) / YEAR_TO_DAY

    # Build design matrix for linear least squares
    # Model: [ra_offset, dec_offset] = A @ [ra0, dec0, pm_ra, pm_dec, parallax]
    # But RA and Dec are coupled through parallax, so we solve jointly

    # For RA: ra_offset = ra0 + pm_ra*dt + parallax*p_ra
    # For Dec: dec_offset = dec0 + pm_dec*dt + parallax*p_dec

    # Design matrix for RA
    A_ra = np.column_stack(
        [
            np.ones(n_obs),  # ra0
            np.zeros(n_obs),  # dec0 (not in RA equation)
            dt_years,  # pm_ra
            np.zeros(n_obs),  # pm_dec (not in RA equation)
            p_ra,  # parallax
        ]
    )

    # Design matrix for Dec
    A_dec = np.column_stack(
        [
            np.zeros(n_obs),  # ra0 (not in Dec equation)
            np.ones(n_obs),  # dec0
            np.zeros(n_obs),  # pm_ra (not in Dec equation)
            dt_years,  # pm_dec
            p_dec,  # parallax
        ]
    )

    # Stack into single system
    A = np.vstack([A_ra, A_dec])
    y = np.concatenate([ra_offset, dec_offset])

    # Weighted least squares (assuming uniform errors for now)
    # Could add position errors if available
    try:
        # Solve using SVD for numerical stability
        params, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

        ra0_fit, dec0_fit, pm_ra_fit, pm_dec_fit, parallax_fit = params

        # Compute residuals
        y_pred = A @ params
        ra_resid = ra_offset - y_pred[:n_obs]
        dec_resid = dec_offset - y_pred[n_obs:]

        # RMS residuals (mas)
        rms_ra = np.std(ra_resid)
        rms_dec = np.std(dec_resid)
        rms_total = np.sqrt(rms_ra**2 + rms_dec**2)

        # Estimate parameter covariance
        # Cov = σ² × (A^T A)^(-1)
        sigma2 = np.sum((y - y_pred) ** 2) / (len(y) - 5)
        try:
            cov = sigma2 * np.linalg.inv(A.T @ A)
            param_errors = np.sqrt(np.diag(cov))
            _ra0_err, _dec0_err, _pm_ra_err, _pm_dec_err, parallax_err = param_errors
        except np.linalg.LinAlgError:
            parallax_err = np.nan

        # Parallax SNR
        parallax_snr = abs(parallax_fit) / parallax_err if parallax_err > 0 else 0

        # Distance estimate (pc)
        if parallax_fit > 0:
            distance_pc = 1000.0 / parallax_fit
            distance_err_pc = distance_pc * (parallax_err / parallax_fit)
        else:
            distance_pc = np.nan
            distance_err_pc = np.nan

        return {
            "ra0": ra0_fit,
            "dec0": dec0_fit,
            "pm_ra_fit": pm_ra_fit,
            "pm_dec_fit": pm_dec_fit,
            "parallax_mas": parallax_fit,
            "parallax_err_mas": parallax_err,
            "parallax_snr": parallax_snr,
            "distance_pc": distance_pc,
            "distance_err_pc": distance_err_pc,
            "rms_ra_mas": rms_ra,
            "rms_dec_mas": rms_dec,
            "rms_total_mas": rms_total,
            "n_epochs": n_obs,
            "baseline_years": baseline_years,
            "ref_mjd": ref_mjd,
            "fit_success": True,
        }

    except Exception as e:
        logger.warning(f"Fit failed: {e}")
        return {
            "parallax_mas": np.nan,
            "parallax_err_mas": np.nan,
            "parallax_snr": 0,
            "distance_pc": np.nan,
            "distance_err_pc": np.nan,
            "rms_total_mas": np.nan,
            "n_epochs": n_obs,
            "baseline_years": baseline_years,
            "fit_success": False,
        }


def extract_parallax_for_source(source_epochs, designation):
    """
    Extract parallax for a single source.

    Parameters:
        source_epochs: DataFrame with ra, dec, mjd for this source
        designation: Source designation

    Returns:
        dict with parallax results
    """
    # Filter valid epochs
    valid = (
        ~source_epochs["ra"].isna() & ~source_epochs["dec"].isna() & ~source_epochs["mjd"].isna()
    )

    epochs = source_epochs[valid].copy()
    n_epochs = len(epochs)

    if n_epochs < MIN_EPOCHS:
        logger.debug(f"{designation}: Only {n_epochs} epochs, need {MIN_EPOCHS}")
        return {
            "designation": designation,
            "parallax_mas": np.nan,
            "parallax_snr": 0,
            "n_epochs": n_epochs,
            "fit_success": False,
            "skip_reason": "insufficient_epochs",
        }

    # Check baseline
    baseline_years = (epochs["mjd"].max() - epochs["mjd"].min()) / YEAR_TO_DAY
    if baseline_years < MIN_BASELINE_YEARS:
        logger.debug(f"{designation}: Baseline {baseline_years:.1f} yr < {MIN_BASELINE_YEARS}")
        return {
            "designation": designation,
            "parallax_mas": np.nan,
            "parallax_snr": 0,
            "n_epochs": n_epochs,
            "fit_success": False,
            "skip_reason": "insufficient_baseline",
        }

    # Reference position (use target position from data)
    ref_ra = epochs["target_ra"].iloc[0]
    ref_dec = epochs["target_dec"].iloc[0]

    # Fit astrometry
    result = fit_astrometry(
        ra_obs=epochs["ra"].values,
        dec_obs=epochs["dec"].values,
        mjd=epochs["mjd"].values,
        ref_ra=ref_ra,
        ref_dec=ref_dec,
    )

    result["designation"] = designation
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract parallax from NEOWISE astrometry")
    _project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--epochs",
        type=str,
        default=str(_project_root / "data" / "processed" / "final" / "neowise_epochs.parquet"),
        help="NEOWISE epochs file",
    )
    parser.add_argument(
        "--golden",
        type=str,
        default=str(_project_root / "data" / "processed" / "final" / "golden_targets.csv"),
        help="Golden targets file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_project_root / "data" / "processed" / "final" / "golden_parallax.csv"),
        help="Output parallax file",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    logger.info(f"Loading epochs from {args.epochs}")
    epochs_df = pd.read_parquet(args.epochs)
    logger.info(f"Loaded {len(epochs_df)} epochs")

    logger.info(f"Loading golden targets from {args.golden}")
    golden_df = pd.read_csv(args.golden)
    logger.info(f"Loaded {len(golden_df)} golden targets")

    # Get unique designations
    designations = epochs_df["designation"].unique()
    logger.info(f"Found {len(designations)} unique sources in epochs data")

    # Extract parallax for each source
    results = []
    significant_detections = 0

    for i, desig in enumerate(designations):
        source_epochs = epochs_df[epochs_df["designation"] == desig]

        result = extract_parallax_for_source(source_epochs, desig)
        results.append(result)

        if result.get("fit_success", False):
            plx = result["parallax_mas"]
            snr = result["parallax_snr"]

            if snr >= MIN_PARALLAX_SNR:
                significant_detections += 1
                dist = result["distance_pc"]
                logger.info(
                    f"{desig}: π = {plx:.2f} ± {result['parallax_err_mas']:.2f} mas "
                    f"(SNR={snr:.1f}, d={dist:.1f} pc)"
                )

        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i+1}/{len(designations)} sources")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Merge with golden targets to add context
    results_df = results_df.merge(
        golden_df[["designation", "w1_w2_color", "T_eff_K", "pm_total", "variability_flag"]],
        on="designation",
        how="left",
    )

    # Sort by parallax SNR (most significant first)
    results_df = results_df.sort_values("parallax_snr", ascending=False)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved parallax results to {output_path}")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("PARALLAX EXTRACTION SUMMARY")
    logger.info("=" * 60)

    successful = results_df[results_df["fit_success"].eq(True)]
    logger.info(f"Successful fits: {len(successful)}/{len(results_df)}")
    logger.info(f"Significant detections (SNR >= {MIN_PARALLAX_SNR}): {significant_detections}")

    if significant_detections > 0:
        sig = results_df[results_df["parallax_snr"] >= MIN_PARALLAX_SNR]
        logger.info("\nSignificant parallax detections:")
        for _, row in sig.iterrows():
            logger.info(
                f"  {row['designation']}: "
                f"π = {row['parallax_mas']:.2f} ± {row['parallax_err_mas']:.2f} mas, "
                f"d = {row['distance_pc']:.1f} ± {row['distance_err_pc']:.1f} pc"
            )

    # Check fading sources specifically
    fading = results_df[results_df["variability_flag"] == "FADING"]
    if len(fading) > 0:
        logger.info("\nFading sources parallax:")
        for _, row in fading.iterrows():
            if row["fit_success"]:
                logger.info(
                    f"  {row['designation']}: "
                    f"π = {row['parallax_mas']:.2f} ± {row['parallax_err_mas']:.2f} mas "
                    f"(SNR={row['parallax_snr']:.1f})"
                )
            else:
                logger.info(f"  {row['designation']}: fit failed")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
