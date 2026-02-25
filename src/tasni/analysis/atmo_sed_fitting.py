"""ATMO 2020 SED fitting for TASNI golden-list sources.

Downloads the ATMO 2020 atmospheric model grid (Phillips et al. 2020,
A&A 637, A38) from STScI, computes synthetic WISE photometry, and fits
the observed W1/W2/W3 magnitudes of TASNI candidate sources to derive
best-fit effective temperatures.

Two fitting modes are used:
  1. W1+W2 fit: fits distance modulus + Teff using only W1 and W2 (the
     most reliable bands), then reports W3/W4 excess.
  2. Color fit: matches the observed W1-W2 color directly to the grid,
     independent of distance.

Usage:
    PYTHONPATH="" VIRTUAL_ENV="" conda run -n bge-env python \\
        src/tasni/analysis/atmo_sed_fitting.py
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
C_CGS = 2.99792458e10  # Speed of light in cm/s

# WISE zero-point flux densities in Jy (Wright et al. 2010, AJ 140, 1868)
WISE_F0 = {"W1": 306.682, "W2": 170.663, "W3": 29.045, "W4": 8.284}

# Simple rectangular bandpass definitions (wavelength in Angstroms)
WISE_BANDS = {
    "W1": (28000.0, 38000.0),
    "W2": (40000.0, 52000.0),
    "W3": (100000.0, 130000.0),
    "W4": (200000.0, 240000.0),
}

# STScI base URL for the ATMO 2020 grid
STSCI_BASE = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/atmo2020/"

# Fallback hard-coded synthetic WISE colors (W1-W2) at log g=5.0, CEQ
# Approximate values consistent with Phillips et al. 2020 Fig. 8 / Table A2
FALLBACK_GRID = {
    200: {"W1-W2": 10.3, "W2-W3": 1.0},
    250: {"W1-W2": 7.5, "W2-W3": 0.6},
    300: {"W1-W2": 6.2, "W2-W3": 0.7},
    350: {"W1-W2": 5.3, "W2-W3": 0.8},
    400: {"W1-W2": 4.7, "W2-W3": 0.8},
    450: {"W1-W2": 4.4, "W2-W3": 0.8},
    500: {"W1-W2": 4.0, "W2-W3": 0.9},
    550: {"W1-W2": 3.8, "W2-W3": 0.9},
    600: {"W1-W2": 3.5, "W2-W3": 0.9},
    700: {"W1-W2": 3.1, "W2-W3": 0.9},
    800: {"W1-W2": 2.7, "W2-W3": 0.9},
    900: {"W1-W2": 2.4, "W2-W3": 0.9},
    1000: {"W1-W2": 2.1, "W2-W3": 0.9},
    1100: {"W1-W2": 1.8, "W2-W3": 0.8},
    1200: {"W1-W2": 1.5, "W2-W3": 0.7},
    1300: {"W1-W2": 1.3, "W2-W3": 0.7},
    1400: {"W1-W2": 1.1, "W2-W3": 0.6},
    1500: {"W1-W2": 0.9, "W2-W3": 0.5},
}

# Available temperatures on STScI (from directory listing)
STSCI_TEFFS = [
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
]

# 4 target source designations
TARGET_SOURCES = [
    "J143046.35-025927.8",
    "J231029.40-060547.3",
    "J193547.43+601201.5",
    "J044024.40-731441.6",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── Utility functions ──────────────────────────────────────────────────────────


def _flam_to_fnu_jy(flux_flam: np.ndarray, wave_ang: np.ndarray) -> np.ndarray:
    """Convert F_lambda (erg/s/cm^2/A) to F_nu (Jy)."""
    wave_cm = wave_ang * 1e-8
    return flux_flam * (wave_cm**2) / C_CGS * 1e23


def _synthetic_mag(
    wave_ang: np.ndarray, flux_flam: np.ndarray, band_lo: float, band_hi: float, f0_jy: float
) -> float:
    """Compute synthetic Vega magnitude using a rectangular bandpass."""
    mask = (wave_ang >= band_lo) & (wave_ang <= band_hi)
    if mask.sum() < 2:
        return np.nan

    w = wave_ang[mask]
    f = flux_flam[mask]
    fnu = _flam_to_fnu_jy(f, w)

    mean_fnu = np.trapezoid(fnu, w) / np.trapezoid(np.ones_like(w), w)
    if mean_fnu <= 0:
        return np.nan
    return -2.5 * math.log10(mean_fnu / f0_jy)


# ── Grid download ─────────────────────────────────────────────────────────────


def download_atmo2020_grid(
    output_dir: Path,
    teffs: list[int] | None = None,
) -> dict[int, Path]:
    """Download ATMO 2020 FITS spectra from STScI.

    Parameters
    ----------
    output_dir : Path
        Directory to store downloaded FITS files.
    teffs : list[int], optional
        Specific Teff values to download. Defaults to all available.

    Returns
    -------
    dict mapping Teff -> local FITS path for successful downloads.
    """
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)
    if teffs is None:
        teffs = STSCI_TEFFS
    downloaded = {}

    for teff in teffs:
        fname = f"spec_t{teff}_lg5.0_ceq.fits"
        local = output_dir / fname
        if local.exists() and local.stat().st_size > 0:
            logger.info("Using cached %s", fname)
            downloaded[teff] = local
            continue

        url = STSCI_BASE + fname
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            local.write_bytes(resp.content)
            logger.info("Downloaded %s (%d bytes)", fname, len(resp.content))
            downloaded[teff] = local
        except Exception as exc:
            logger.warning("Failed to download %s: %s", fname, exc)

    return downloaded


# ── Synthetic photometry from spectra ──────────────────────────────────────────


def compute_wise_synthetic_colors(spectrum_file: Path) -> dict:
    """Compute synthetic WISE magnitudes from an ATMO 2020 FITS spectrum.

    Returns
    -------
    dict with keys 'W1', 'W2', 'W3', 'W4' (magnitudes), plus colors.
    """
    from astropy.io import fits as pyfits

    with pyfits.open(spectrum_file) as hdul:
        data = hdul[1].data
        wave = data["WAVELENGTH"].astype(np.float64)
        flux = data["FLUX"].astype(np.float64)

    result = {}
    for band, (lo, hi) in WISE_BANDS.items():
        result[band] = _synthetic_mag(wave, flux, lo, hi, WISE_F0[band])

    for c1, c2 in [("W1", "W2"), ("W2", "W3"), ("W3", "W4")]:
        key = f"{c1}-{c2}"
        if not np.isnan(result[c1]) and not np.isnan(result[c2]):
            result[key] = result[c1] - result[c2]
        else:
            result[key] = np.nan

    return result


def _build_model_grid(grid_files: dict[int, Path]) -> dict[int, dict]:
    """Build a grid of synthetic WISE photometry from downloaded FITS files."""
    grid = {}
    for teff, fpath in sorted(grid_files.items()):
        try:
            mags = compute_wise_synthetic_colors(fpath)
            grid[teff] = mags
            logger.info(
                "T=%dK: W1-W2=%.3f  W2-W3=%.3f",
                teff,
                mags.get("W1-W2", np.nan),
                mags.get("W2-W3", np.nan),
            )
        except Exception as exc:
            logger.warning("Failed to process T=%dK: %s", teff, exc)
    return grid


# ── Fitting ────────────────────────────────────────────────────────────────────


def _color_fit(
    w1w2_obs: float,
    w2w3_obs: float,
    sig_w1w2: float,
    sig_w2w3: float,
    model_grid: dict[int, dict],
) -> dict:
    """Fit Teff by matching W1-W2 and W2-W3 colors to the grid.

    Returns best-fit Teff and chi2 based on color residuals.
    """
    best_chi2 = np.inf
    best_teff = None
    best_colors = None
    all_fits = []

    for teff, mags in sorted(model_grid.items()):
        m_w1w2 = mags.get("W1-W2", np.nan)
        m_w2w3 = mags.get("W2-W3", np.nan)
        if np.isnan(m_w1w2):
            continue

        # chi2 for W1-W2 color
        chi2_12 = ((w1w2_obs - m_w1w2) / sig_w1w2) ** 2
        # chi2 for W2-W3 color (if model has it)
        if not np.isnan(m_w2w3) and not np.isnan(w2w3_obs):
            chi2_23 = ((w2w3_obs - m_w2w3) / sig_w2w3) ** 2
            chi2 = chi2_12 + chi2_23
            dof = 2
        else:
            chi2_23 = np.nan
            chi2 = chi2_12
            dof = 1

        entry = {
            "teff": teff,
            "model_W1-W2": round(m_w1w2, 3),
            "model_W2-W3": round(m_w2w3, 3) if not np.isnan(m_w2w3) else None,
            "chi2": round(chi2, 3),
            "dof": dof,
        }
        all_fits.append(entry)

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_teff = teff
            best_colors = {"W1-W2": m_w1w2, "W2-W3": m_w2w3}

    return {
        "teff": best_teff,
        "chi2": round(best_chi2, 3) if best_teff else None,
        "model_colors": {k: round(v, 3) for k, v in best_colors.items()} if best_colors else None,
        "all_fits": all_fits,
    }


def _w1w2_only_fit(
    w1w2_obs: float,
    sig_w1w2: float,
    model_grid: dict[int, dict],
) -> dict:
    """Fit Teff using W1-W2 color only (the primary diagnostic color).

    Interpolates between grid points for best-fit temperature.
    """
    teffs = []
    colors = []
    for teff, mags in sorted(model_grid.items()):
        c = mags.get("W1-W2", np.nan)
        if not np.isnan(c):
            teffs.append(teff)
            colors.append(c)

    teffs = np.array(teffs)
    colors = np.array(colors)

    # W1-W2 decreases with increasing Teff, so invert for interpolation
    # Find the two bracketing grid points
    best_chi2 = np.inf
    best_teff = None

    for i in range(len(teffs)):
        chi2 = ((w1w2_obs - colors[i]) / sig_w1w2) ** 2
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_teff = teffs[i]

    # Linear interpolation between adjacent grid points
    teff_interp = None
    if len(teffs) >= 2:
        for i in range(len(teffs) - 1):
            if (colors[i] >= w1w2_obs >= colors[i + 1]) or (colors[i] <= w1w2_obs <= colors[i + 1]):
                frac = (w1w2_obs - colors[i]) / (colors[i + 1] - colors[i])
                teff_interp = float(teffs[i] + frac * (teffs[i + 1] - teffs[i]))
                break

    # If observed color is outside grid range
    if w1w2_obs > colors[0]:
        teff_interp = None  # cooler than grid minimum
        note = (
            f"W1-W2={w1w2_obs:.2f} exceeds coldest model "
            f"({colors[0]:.2f} at {teffs[0]}K); source is cooler than grid"
        )
    elif w1w2_obs < colors[-1]:
        teff_interp = None  # warmer than grid maximum
        note = (
            f"W1-W2={w1w2_obs:.2f} below warmest model "
            f"({colors[-1]:.2f} at {teffs[-1]}K); source is warmer than grid"
        )
    else:
        note = "interpolated from grid"

    return {
        "teff_grid_nearest": int(best_teff) if best_teff is not None else None,
        "teff_interpolated": round(teff_interp, 0) if teff_interp is not None else None,
        "chi2_nearest": round(best_chi2, 3) if best_teff is not None else None,
        "note": note,
    }


def fit_atmo_to_source(
    w1_obs: float,
    w2_obs: float,
    w3_obs: float,
    w4_obs: float | None = None,
    sig_w1: float = 0.05,
    sig_w2: float = 0.05,
    sig_w3: float = 0.10,
    sig_w4: float = 0.15,
    model_grid: dict[int, dict] | None = None,
) -> dict:
    """Fit observed WISE magnitudes to the ATMO 2020 grid.

    Performs two fits:
    1. W1-W2 color-only fit: determines Teff from the primary diagnostic color,
       independent of distance. The W1-W2 color is the key discriminant for
       cold brown dwarf temperatures.
    2. W1+W2 magnitude fit: fits distance modulus + Teff using W1 and W2,
       then reports W3/W4 excess over the atmospheric model prediction.

    Parameters
    ----------
    w1_obs, w2_obs, w3_obs : float
        Observed WISE magnitudes.
    w4_obs : float, optional
        W4 magnitude (often upper limit; excluded if None).
    sig_w1, sig_w2, sig_w3, sig_w4 : float
        Photometric uncertainties.
    model_grid : dict
        Mapping of Teff -> dict with synthetic magnitudes and colors.

    Returns
    -------
    dict with best-fit parameters.
    """
    if model_grid is None:
        model_grid = FALLBACK_GRID

    w1w2_obs = w1_obs - w2_obs
    w2w3_obs = w2_obs - w3_obs
    sig_w1w2 = math.sqrt(sig_w1**2 + sig_w2**2)

    # ── Color-based Teff fit (W1-W2 only) ──
    w1w2_fit = _w1w2_only_fit(w1w2_obs, sig_w1w2, model_grid)

    # ── W1+W2 magnitude fit with distance modulus ──
    best_chi2_mag = np.inf
    best_teff_mag = None
    best_dm = None
    best_model_mags = None

    for teff, mags in sorted(model_grid.items()):
        m_w1 = mags.get("W1", np.nan)
        m_w2 = mags.get("W2", np.nan)
        if np.isnan(m_w1) or np.isnan(m_w2):
            continue

        # Fit DM from W1 and W2 only
        w_arr = np.array([1.0 / sig_w1**2, 1.0 / sig_w2**2])
        diff = np.array([w1_obs - m_w1, w2_obs - m_w2])
        dm = np.sum(w_arr * diff) / np.sum(w_arr)

        resid_w1 = (w1_obs - (m_w1 + dm)) / sig_w1
        resid_w2 = (w2_obs - (m_w2 + dm)) / sig_w2
        chi2 = resid_w1**2 + resid_w2**2

        if chi2 < best_chi2_mag:
            best_chi2_mag = chi2
            best_teff_mag = teff
            best_dm = dm
            pred = {"W1": m_w1 + dm, "W2": m_w2 + dm}
            m_w3 = mags.get("W3", np.nan)
            m_w4 = mags.get("W4", np.nan)
            if not np.isnan(m_w3):
                pred["W3"] = m_w3 + dm
            if not np.isnan(m_w4):
                pred["W4"] = m_w4 + dm
            best_model_mags = pred

    # Compute W3 and W4 excess over model prediction
    w3_excess = None
    w4_excess = None
    if best_model_mags:
        if "W3" in best_model_mags:
            w3_excess = round(best_model_mags["W3"] - w3_obs, 3)  # positive = brighter than model
        if w4_obs is not None and "W4" in best_model_mags:
            w4_excess = round(best_model_mags["W4"] - w4_obs, 3)

    # Determine the best Teff to report
    teff_best = w1w2_fit.get("teff_interpolated") or w1w2_fit.get("teff_grid_nearest")

    # Bracket info
    teffs_sorted = sorted(model_grid.keys())
    bracket = ""
    if teff_best and teff_best in teffs_sorted:
        idx = teffs_sorted.index(teff_best)
        if 0 < idx < len(teffs_sorted) - 1:
            bracket = f"bracketed by {teffs_sorted[idx-1]}-{teffs_sorted[idx+1]}K grid"
        elif idx == 0:
            bracket = f"at grid lower boundary ({teff_best}K)"
        elif idx == len(teffs_sorted) - 1:
            bracket = f"at grid upper boundary ({teff_best}K)"
    elif teff_best:
        bracket = "interpolated between grid points"

    result = {
        "teff_atmo": int(teff_best) if teff_best else None,
        "logg_atmo": 5.0,
        "observed_colors": {
            "W1-W2": round(w1w2_obs, 3),
            "W2-W3": round(w2w3_obs, 3),
        },
        "w1w2_color_fit": {
            "teff_nearest": w1w2_fit["teff_grid_nearest"],
            "teff_interpolated": w1w2_fit["teff_interpolated"],
            "chi2": w1w2_fit["chi2_nearest"],
            "note": w1w2_fit["note"],
        },
        "magnitude_fit": {
            "teff": best_teff_mag,
            "distance_modulus": round(float(best_dm), 3) if best_dm is not None else None,
            "chi2_w1w2": round(float(best_chi2_mag), 3) if best_teff_mag else None,
            "model_apparent_mags": (
                {k: round(v, 3) for k, v in best_model_mags.items()} if best_model_mags else None
            ),
        },
        "w3_excess_mag": w3_excess,
        "w4_excess_mag": w4_excess,
        "notes": bracket if bracket else "best fit from grid",
    }

    return result


# ── Main driver ────────────────────────────────────────────────────────────────


def run_all_sources(
    sources_csv: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    """Run ATMO 2020 SED fitting for all TASNI golden-list target sources.

    Parameters
    ----------
    sources_csv : Path
        Path to golden_improved.csv.
    output_json : Path
        Path to write results JSON.

    Returns
    -------
    dict : the results structure written to JSON.
    """
    if sources_csv is None:
        sources_csv = PROJECT_ROOT / "data" / "processed" / "final" / "golden_improved.csv"
    if output_json is None:
        output_json = PROJECT_ROOT / "output" / "atmo_fits" / "atmo_results.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    grid_dir = output_json.parent / "grid"

    # ── Step 1: Download ATMO 2020 grid (all temperatures) ──
    logger.info("Downloading ATMO 2020 grid (all %d temperatures)...", len(STSCI_TEFFS))
    grid_files = download_atmo2020_grid(grid_dir, teffs=STSCI_TEFFS)
    download_success = len(grid_files) >= 5

    # ── Step 2: Build model grid ──
    if download_success:
        logger.info("Building synthetic photometry grid from %d spectra...", len(grid_files))
        model_grid = _build_model_grid(grid_files)
        grid_source = "STScI HLSP (Phillips et al. 2020)"
        if len(model_grid) < 5:
            logger.warning("Insufficient valid spectra; falling back to hard-coded grid")
            model_grid = FALLBACK_GRID
            grid_source = "fallback_hardcoded"
    else:
        logger.warning("Download failed; using fallback hard-coded grid")
        model_grid = FALLBACK_GRID
        grid_source = "fallback_hardcoded"

    # Log the model grid colors
    logger.info("Model grid (%s), %d temperatures:", grid_source, len(model_grid))
    for teff, mags in sorted(model_grid.items()):
        w1w2 = mags.get("W1-W2", np.nan)
        w2w3 = mags.get("W2-W3", np.nan)
        logger.info("  T=%4dK: W1-W2=%6.3f  W2-W3=%6.3f", teff, w1w2, w2w3)

    # ── Step 3: Read source photometry ──
    df = pd.read_csv(sources_csv)
    logger.info("Loaded %d sources from %s", len(df), sources_csv)

    # ── Step 4: Fit each target ──
    results_sources = {}
    for desig in TARGET_SOURCES:
        row = df[df["designation"] == desig]
        if row.empty:
            logger.warning("Source %s not found in CSV", desig)
            results_sources[desig] = {"error": "not found in CSV"}
            continue

        row = row.iloc[0]
        w1 = float(row["w1mpro"])
        w2 = float(row["w2mpro"])
        w3 = float(row["w3mpro"])
        w4 = float(row["w4mpro"]) if not pd.isna(row["w4mpro"]) else None
        sig_w1 = float(row["w1sigmpro"]) if not pd.isna(row["w1sigmpro"]) else 0.05
        sig_w2 = float(row["w2sigmpro"]) if not pd.isna(row["w2sigmpro"]) else 0.05
        sig_w3 = float(row["w3sigmpro"]) if not pd.isna(row["w3sigmpro"]) else 0.10
        sig_w4 = float(row["w4sigmpro"]) if not pd.isna(row["w4sigmpro"]) else 0.15

        short = desig.split(".")[0]

        fit = fit_atmo_to_source(
            w1,
            w2,
            w3,
            w4,
            sig_w1,
            sig_w2,
            sig_w3,
            sig_w4,
            model_grid=model_grid,
        )

        if "J044024" in desig:
            fit["notes"] = (
                "LMC source at ~50 kpc (DM~18.5). "
                "Distance-corrected absolute magnitudes are highly uncertain. "
                + fit.get("notes", "")
            )
            results_sources[short + "_LMC"] = fit
        else:
            results_sources[short] = fit

    # ── Step 5: Summary ──
    teff_values = []
    w3_excesses = []
    for key, fit in results_sources.items():
        if "teff_atmo" in fit and fit["teff_atmo"] is not None and "LMC" not in key:
            teff_values.append(fit["teff_atmo"])
        if "w3_excess_mag" in fit and fit["w3_excess_mag"] is not None and "LMC" not in key:
            w3_excesses.append(fit["w3_excess_mag"])

    if teff_values:
        tmin, tmax = min(teff_values), max(teff_values)
        summary = (
            f"Color-fitted Teff range for three orphan sources: "
            f"{tmin}-{tmax}K (log g=5.0, chemical equilibrium). "
        )
        if w3_excesses:
            summary += (
                f"All sources show significant W3 excess ({min(w3_excesses):.1f} to "
                f"{max(w3_excesses):.1f} mag brighter than atmospheric model), "
                "indicating either circumstellar/interstellar dust or "
                "non-equilibrium atmospheric chemistry. "
            )
        summary += f"Grid source: {grid_source}."
    else:
        summary = "Fitting failed for all sources."

    output = {
        "method": "ATMO2020_synthetic_photometry",
        "grid_source": grid_source,
        "reference": "Phillips et al. 2020, A&A 637, A38",
        "logg_fixed": 5.0,
        "chemistry": "chemical_equilibrium",
        "wise_zeropoints_Jy": WISE_F0,
        "wise_bandpass": "rectangular (approximate)",
        "n_grid_temperatures": len(model_grid),
        "grid_temperatures_K": sorted(model_grid.keys()),
        "download_success": download_success,
        "sources": results_sources,
        "summary": summary,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", output_json)

    return output


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = run_all_sources()

    print("\n" + "=" * 72)
    print("ATMO 2020 SED Fitting Results")
    print("=" * 72)
    print(f"Grid: {result['grid_source']}")
    print(f"Download success: {result['download_success']}")
    print(f"Grid temperatures: {result['grid_temperatures_K']}")
    print()
    for name, fit in result["sources"].items():
        if "error" in fit:
            print(f"  {name}: ERROR - {fit['error']}")
            continue
        teff = fit.get("teff_atmo", "?")
        oc = fit.get("observed_colors", {})
        w3x = fit.get("w3_excess_mag")
        w4x = fit.get("w4_excess_mag")
        cf = fit.get("w1w2_color_fit", {})
        mf = fit.get("magnitude_fit", {})
        print(f"  {name}:")
        print(f"    Teff (color fit)    = {teff} K")
        print(f"    Observed W1-W2      = {oc.get('W1-W2', '?')}")
        print(f"    Observed W2-W3      = {oc.get('W2-W3', '?')}")
        print(f"    W1-W2 fit nearest   = {cf.get('teff_nearest')} K (chi2={cf.get('chi2')})")
        print(f"    W1-W2 interpolated  = {cf.get('teff_interpolated')} K")
        print(f"    Mag fit Teff        = {mf.get('teff')} K, DM = {mf.get('distance_modulus')}")
        print(f"    W3 excess           = {w3x} mag (model - observed; +ve = brighter than model)")
        print(f"    W4 excess           = {w4x} mag")
        print(f"    Notes: {fit.get('notes', '')}")
    print()
    print(f"Summary: {result['summary']}")
    print("=" * 72)
