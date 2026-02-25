import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Configuration
INPUT_FILE = "./data/processed/tier4_prime.parquet"
OUTPUT_FILE = "./data/processed/tier4_physics.parquet"

# WISE Zero Points (Jarrett et al. 2011) in Jy
# W1, W2, W3, W4
ZERO_POINTS = {"w1": 309.540, "w2": 171.787, "w3": 31.674, "w4": 8.363}
# Effective Wavelengths (microns)
WAVELENGTHS = {"w1": 3.4, "w2": 4.6, "w3": 12.0, "w4": 22.0}


def mag_to_flux(mag, band):
    # F_nu = F_0 * 10^(-m/2.5)
    f0 = ZERO_POINTS[band]
    return f0 * 10 ** (-mag / 2.5)


def mag_err_to_flux_err(mag, mag_err, band):
    # dF/dmag = -F * ln(10) / 2.5
    f = mag_to_flux(mag, band)
    return np.abs(f * np.log(10) / 2.5 * mag_err)


def planck_nu(nu, T, scale):
    # B_nu(T) = (2h nu^3 / c^2) * (1 / (exp(h nu / kT) - 1))
    # We add a scaling factor 'scale' representing solid angle (R/d)^2

    # Constants in SI
    h_val = 6.626e-34
    c_val = 3.0e8
    k_val = 1.38e-23

    # Avoid overflow
    try:
        expert = (h_val * nu) / (k_val * T)
        # Clip expert to avoid overflow in exp
        expert = np.clip(expert, 1e-10, 700)
        bb = (2 * h_val * nu**3 / c_val**2) * (1 / (np.exp(expert) - 1))
        return scale * bb * 1e26  # Convert SI (W/m^2/Hz) to Jy (10^-26 W/m^2/Hz)? No, 1 Jy = 10^-26
    except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
        return 0.0


def fit_temperature(row):
    # Extract photometry with magnitude errors for proper weighting
    fluxes = []
    freqs = []
    flux_errors = []

    bands = ["w1", "w2", "w3", "w4"]

    for b in bands:
        mag = row.get(f"{b}mpro")
        mag_err = row.get(f"{b}sigmpro")
        snr = row.get(f"{b}snr")

        # Only use reliable detections (SNR > 3 usually, but lets match what we have)
        if pd.notnull(mag) and pd.notnull(snr) and snr > 3:
            flux = mag_to_flux(mag, b)
            # freq = c / lambda
            freq = 3.0e8 / (WAVELENGTHS[b] * 1e-6)
            # Convert mag uncertainty to flux uncertainty; use 5% if missing
            if pd.notnull(mag_err) and mag_err > 0:
                flux_err = mag_err_to_flux_err(mag, mag_err, b)
            else:
                flux_err = flux * 0.05
            fluxes.append(flux)
            freqs.append(freq)
            flux_errors.append(flux_err)

    if len(fluxes) < 2:
        return np.nan, np.nan, np.nan  # T_eff, Scale, T_eff_err

    try:
        # Weight by flux errors; use absolute_sigma for physical uncertainties
        popt, pcov = curve_fit(
            planck_nu,
            freqs,
            fluxes,
            sigma=flux_errors,
            absolute_sigma=True,
            p0=[500, 1e-20],
            bounds=([10, 0], [5000, 1e-10]),
        )
        t_eff_err = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.nan
        return popt[0], popt[1], t_eff_err  # T_eff, Scale, T_eff_err
    except (ValueError, RuntimeError, TypeError):
        return np.nan, np.nan, np.nan


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    print(f"Calculating T_eff for {len(df)} sources...")

    # Vectorizing curve_fit is hard, applying row-wise
    # Parallelize?
    # For 4000 sources, serial is fine (seconds).

    results = df.apply(fit_temperature, axis=1, result_type="expand")
    df["T_eff_K"] = results[0]
    df["SolidAngle_scale"] = results[1]
    df["T_eff_err_K"] = results[2]

    # Save
    df.to_parquet(OUTPUT_FILE)
    print(f"Saved physics results to {OUTPUT_FILE}")

    # Quick stats
    valid = df.dropna(subset=["T_eff_K"])
    print(f"\nSuccessfully fitted {len(valid)} sources.")
    print(valid["T_eff_K"].describe())

    # Potential Alien/Dyson Zone: Broad range [200, 500] K
    dyson_candidates = valid[(valid["T_eff_K"] > 250) & (valid["T_eff_K"] < 350)]
    print("\n--- ROOM TEMPERATURE OBJECTS (250-350K) ---")
    print(f"Count: {len(dyson_candidates)}")
    if len(dyson_candidates) > 0:
        print(dyson_candidates[["designation", "T_eff_K", "w1mpro"]].head())
        dyson_candidates.to_csv("./data/processed/room_temp_anomalies.csv")


if __name__ == "__main__":
    main()
