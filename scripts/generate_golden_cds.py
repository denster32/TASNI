#!/usr/bin/env python3
"""
Generate the TASNI Golden Sample catalog in CDS byte-by-byte format.

Reads data/processed/final/golden_improved.csv and writes
tasni_paper_final/golden_sample_cds.txt with full parallax columns.
Keeps the CDS file in sync with the pipeline output.

Usage:
    python scripts/generate_golden_cds.py
"""

from pathlib import Path

import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "processed" / "final" / "golden_improved.csv"
OUT_PATH = ROOT / "tasni_paper_final" / "golden_sample_cds.txt"

# CDS fixed-width layout (bytes 1-based, inclusive)
# Columns: Desig(1-22), RA(24-35), DE(37-48), W1(50-55), W2(57-62), W1-W2(64-69),
#          Teff(71-77), pmTotal(79-86), VarFlag(88-97), MLscore(99-104), Rank(106-108),
#          plx(110-117), e_plx(119-126), dist(128-134)
WIDTHS = [
    (1, 22, "s", "designation"),
    (24, 35, "f", "ra"),
    (37, 48, "f", "dec"),
    (50, 55, "f", "w1mpro"),
    (57, 62, "f", "w2mpro"),
    (64, 69, "f", "w1_w2_color"),
    (71, 77, "f", "T_eff_K"),
    (79, 86, "f", "pm_total"),
    (88, 97, "s", "variability_flag"),
    (99, 104, "f", "ml_ensemble_score"),
    (106, 108, "d", "rank"),
    (110, 117, "f", "neowise_parallax_mas"),
    (119, 126, "f", "neowise_parallax_err_mas"),
    (128, 134, "f", "distance_pc"),
]


def format_cds_value(val, width: int, fmt: str) -> str:
    """Format a value for CDS fixed-width; use blank for NaN."""
    if fmt == "s":
        return str(val)[:width].ljust(width)
    if fmt == "d":
        if pd.isna(val):
            return " " * width
        return str(int(val)).rjust(width)
    if fmt == "f":
        if pd.isna(val):
            return " " * width
        if width >= 8:
            return f"{float(val):{width}.2f}"[:width]
        return f"{float(val):{width}.1f}"[:width]
    return " " * width


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("rank").reset_index(drop=True)

    header = """Title: TASNI Golden Sample Catalog
Authors: Palucki, D.
Table: Golden Sample of 100 Thermal Anomaly Candidates
==============================================================================
Byte-by-byte Description of file: golden_sample_cds.txt
------------------------------------------------------------------------------
   Bytes  Format  Units    Label       Explanations
------------------------------------------------------------------------------
   1- 22  A22     ---      Desig       WISE designation (Jhhmmss.ss+ddmmss.s)
  24- 35  F12.7   deg      RAdeg       Right Ascension (J2000)
  37- 48  F12.7   deg      DEdeg       Declination (J2000)
  50- 55  F6.3    mag      W1mag       WISE W1 magnitude
  57- 62  F6.3    mag      W2mag       WISE W2 magnitude
  64- 69  F6.3    mag      W1-W2       W1-W2 color index
  71- 77  F7.2    K        Teff        Effective temperature from SED fitting
  79- 86  F8.2    mas/yr   pmTotal     Total proper motion
  88- 97  A10     ---      VarFlag     Variability classification
  99-104  F6.4    ---      MLscore     ML ensemble anomaly score
 106-108  I3      ---      Rank        Ranking by composite score
 110-117  F8.2    mas      plx        NEOWISE parallax (blank if not detected)
 119-126  F8.2    mas      e_plx      Parallax uncertainty
 128-134  F7.1    pc       dist       Distance 1000/plx (blank if plx missing)
------------------------------------------------------------------------------
Note: 100 sources from the TASNI golden sample.
      4 sources are classified as FADING (3 confirmed + 1 LMC member).
      Teff estimated via Planck blackbody SED fitting to WISE photometry.
      Systematic uncertainty of +/-100K should be added in quadrature.
------------------------------------------------------------------------------
"""

    lines = [header.strip()]
    max_col = 134
    for _, row in df.iterrows():
        arr = [" "] * max_col
        for start, end, fmt, col in WIDTHS:
            val = row.get(col)
            if col == "designation":
                val = str(val).strip()
            elif col == "variability_flag":
                val = str(val).strip()[:10]
            width = end - start + 1
            s = format_cds_value(val, width, fmt)
            if len(s) > width:
                s = s[:width]
            for i, c in enumerate(s):
                idx = start - 1 + i
                if idx < max_col:
                    arr[idx] = c
        line = "".join(arr).rstrip()
        lines.append(line)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
