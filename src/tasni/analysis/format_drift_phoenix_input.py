#!/usr/bin/env python3
"""
Converts TASNI atmospheric input to DRIFT-PHOENIX batch file format.

DRIFT-PHOENIX typically requires a text file with:
Teff [K] logg [cgs] [M/H] [alpha/M] micro-turbulence velocity [km/s]
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = (
    _PROJECT_ROOT / "data" / "processed" / "spectroscopy" / "atmospheric_modeling_input.txt"
)
OUTPUT_FILE = _PROJECT_ROOT / "data" / "processed" / "spectroscopy" / "drift_phoenix_batch.inp"


def format_drift_batch():
    print("Formatting input for DRIFT-PHOENIX...")

    # Read prepared input (skip comments)
    lines = []
    with open(INPUT_FILE) as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    lines.append(parts)

    # DRIFT-PHOENIX Batch Format
    # Columns: Teff, log(g), [M/H], [alpha/M], xi (micro-turbulence)
    # Assumptions: Solar metallicity [M/H]=0.0, Solar alpha [alpha/M]=0.0, xi=2.0 km/s

    batch_content = "# DRIFT-PHOENIX Batch Input for TASNI Targets\n"
    batch_content += "# Format: Teff[K]  log(g[cgs])  [M/H]  [alpha/M]  xi[km/s]\n"
    batch_content += "# Generated from TASNI pipeline\n"
    batch_content += "# \n"

    for line in lines:
        designation = line[0]
        teff = float(line[1])
        logg = float(line[2])  # From our input (assumed 5.0)
        feh = float(line[3])  # From our input (assumed 0.0)
        dist = line[4]

        # Fixed parameters for cold brown dwarfs
        alpha = 0.0  # Solar alpha enhancement
        xi = 2.0  # Micro-turbulence velocity (km/s)

        # Format line for DRIFT-PHOENIX
        dp_line = (
            f"{teff:>7.1f}  {logg:>5.2f}  {feh:>6.2f}  {alpha:>6.2f}  {xi:>4.1f}  ! {designation}\n"
        )
        batch_content += dp_line

    # Save batch file
    with open(OUTPUT_FILE, "w") as f:
        f.write(batch_content)

    print(f"Saved DRIFT-PHOENIX batch file to: {OUTPUT_FILE}")
    print("\nTo run with DRIFT-PHOENIX (IDL environment):")
    print("1. Ensure DRIFT-PHOENIX is installed")
    print("2. Load model grids")
    print("3. Run batch processing on this file")

    return OUTPUT_FILE


if __name__ == "__main__":
    format_drift_batch()
