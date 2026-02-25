"""
TASNI: Multi-Wavelength Anomaly Scoring
========================================

Enhanced anomaly scoring that incorporates data from multiple wavelengths:
- WISE (primary IR)
- 2MASS (near-IR)
- Spitzer (mid-IR)
- Gaia (optical)
- Radio (VLASS/NVSS)
- X-ray (Chandra)

Higher score = more anomalous:
- Strong IR signal across multiple bands
- NO optical counterpart
- NO radio/X-ray emission
- Isolated (not near known sources)

Usage:
    python multi_wavelength_scoring.py [--input orphans.parquet] [--output scored.parquet]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from tasni.core.config import LOG_DIR, OUTPUT_DIR, ensure_dirs

ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SCORE] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "scoring.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_multi_wavelength_score(row):
    """
    Compute enhanced anomaly score using multi-wavelength data

    Scoring formula:
    score = ir_brightness - optical_penalty - other_wavelength_penalty + isolation_bonus

    Components:
    - ir_brightness: Sum of WISE + 2MASS + Spitzer magnitudes (inverted)
    - optical_penalty: Penalty for having optical counterpart
    - other_penalty: Penalty for radio/X-ray detection
    - isolation_bonus: Bonus for being isolated
    """

    score = 0.0
    components = {}

    # === WISE IR brightness ===
    w1 = row.get("w1mpro", row.get("w1", 99))
    w2 = row.get("w2mpro", row.get("w2", 99))
    w3 = row.get("w3mpro", row.get("w3", 99))
    w4 = row.get("w4mpro", row.get("w4", 99))

    # Invert magnitudes (brighter = lower mag = higher score)
    if not pd.isna(w1) and w1 < 90:
        wise_brightness = 20 - w1  # Bright sources get positive score
    else:
        wise_brightness = 0

    components["wise_brightness"] = wise_brightness
    score += wise_brightness

    # === W1-W2 color (red = cold = interesting) ===
    if not pd.isna(w1) and not pd.isna(w2) and w1 < 90 and w2 < 90:
        w1_w2 = w1 - w2
        if w1_w2 > 0.5:  # Red source
            color_bonus = (w1_w2 - 0.5) * 2
            components["color_bonus"] = color_bonus
            score += color_bonus

    # === 2MASS near-IR ===
    j_mag = row.get("j_m", 99)
    h_mag = row.get("h_m", 99)
    ks_mag = row.get("ks_m", 99)

    if not pd.isna(j_mag) and j_mag < 90:
        twomass_brightness = (17 - j_mag) * 0.5
        components["2mass_brightness"] = twomass_brightness
        score += twomass_brightness

    # === Spitzer mid-IR ===
    i1_mag = row.get("i1_mag", 99)
    i2_mag = row.get("i2_mag", 99)

    if not pd.isna(i1_mag) and i1_mag < 90:
        spitzer_brightness = (17 - i1_mag) * 0.3
        components["spitzer_brightness"] = spitzer_brightness
        score += spitzer_brightness

    # === Optical penalty ===
    # Has Gaia match within 3 arcsec?
    nearest_gaia = row.get("nearest_gaia_sep_arcsec", np.inf)
    if nearest_gaia < 3.0:
        # Strong penalty for optical counterpart
        optical_penalty = -10
        components["optical_penalty"] = optical_penalty
        score += optical_penalty
    elif nearest_gaia < 10.0:
        # Mild penalty for nearby optical source
        optical_penalty = -2
        components["optical_penalty"] = optical_penalty
        score += optical_penalty
    else:
        components["optical_penalty"] = 0

    # === Radio/X-ray penalty ===
    has_radio = row.get("radio_match", False)
    has_xray = row.get("xray_match", False)

    if has_radio:
        score -= 5
        components["radio_penalty"] = -5
    if has_xray:
        score -= 3
        components["xray_penalty"] = -3

    # === W3/W4 detection (cold objects are faint in W3/W4) ===
    w3_faint = pd.isna(w3) or w3 > 90 or w3 > 14
    w4_faint = pd.isna(w4) or w4 > 90 or w4 > 10

    if w3_faint and w1_w2 > 1.0:  # Red and faint in W3
        components["w3_faint_bonus"] = 1
        score += 1

    # === Quality filter ===
    ph_qual = row.get("ph_qual", "U")
    cc_flags = row.get("cc_flags", "9999")

    if ph_qual not in ["A", "B"]:
        score -= 2  # Penalize poor photometry
        components["ph_qual_penalty"] = -2

    if cc_flags != "0000":
        score -= 1  # Penalize contamination
        components["cc_flags_penalty"] = -1

    # === LAMOST Spectroscopy ===
    # If LAMOST data is available, apply spectral-based scoring
    lamost_score = row.get("lamost_score", 0.0)
    if lamost_score != 0:
        score += lamost_score
        components["lamost_score"] = lamost_score

    # Additional bonus for LAMOST temperature mismatch
    if row.get("lamost_temp_mismatch", False):
        components["lamost_temp_mismatch"] = True

    # Flag known IR types
    if row.get("lamost_is_known_ir", False):
        components["lamost_known_ir"] = True

    return score, components


def score_orphans(input_file=None, output_file=None):
    """Apply multi-wavelength scoring to orphan catalog"""

    if input_file is None:
        input_file = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    if output_file is None:
        output_file = OUTPUT_DIR / "orphans_multiwavelength_scored.parquet"

    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading orphans from {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} orphans")

    # Compute scores
    logger.info("Computing multi-wavelength anomaly scores...")

    scores = []
    all_components = []

    for idx, row in df.iterrows():
        score, components = compute_multi_wavelength_score(row)
        scores.append(score)
        all_components.append(components)

        if (idx + 1) % 10000 == 0:
            logger.info(f"Scored {idx+1:,}/{len(df):,}")

    df["multiwave_score"] = scores
    df["score_components"] = all_components

    # Normalize scores (z-score)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    if std_score > 0:
        df["multiwave_zscore"] = (scores - mean_score) / std_score
    else:
        df["multiwave_zscore"] = 0

    # Rank
    df = df.sort_values("multiwave_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    # Statistics
    logger.info("=" * 60)
    logger.info("Scoring Statistics:")
    logger.info(f"  Mean score: {mean_score:.2f}")
    logger.info(f"  Std score: {std_score:.2f}")
    logger.info(f"  Max score: {np.max(scores):.2f}")
    logger.info(f"  Min score: {np.min(scores):.2f}")
    logger.info(f"  Top 1% threshold: {np.percentile(scores, 99):.2f}")
    logger.info(f"  Top 0.1% threshold: {np.percentile(scores, 99.9):.2f}")
    logger.info("=" * 60)

    # Save
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved scored catalog to {output_file}")

    # Save top anomalies
    top_file = output_file.parent / "top_multiwavelength_anomalies.csv"
    df.head(10000).to_csv(top_file, index=False)
    logger.info(f"Saved top 10,000 to {top_file}")

    # Save tier 5 (top 0.1%)
    tier5_threshold = np.percentile(scores, 99.9)
    tier5 = df[df["multiwave_score"] >= tier5_threshold].head(1000)
    tier5_file = output_file.parent / "tier5_candidates.csv"
    tier5.to_csv(tier5_file, index=False)
    logger.info(f"Saved tier 5 candidates ({len(tier5)} sources) to {tier5_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Multi-wavelength anomaly scoring")
    parser.add_argument("--input", type=str, help="Input orphan catalog")
    parser.add_argument("--output", type=str, help="Output scored catalog")
    parser.add_argument(
        "--with-lamost",
        action="store_true",
        help="Run LAMOST cross-matching first (requires local LAMOST catalog)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Multi-Wavelength Anomaly Scoring")
    logger.info("=" * 60)

    input_file = args.input

    # Optional LAMOST cross-matching
    if args.with_lamost:
        logger.info("Running LAMOST cross-matching...")
        try:
            from crossmatch_lamost import crossmatch_tier4_with_lamost

            lamost_output = OUTPUT_DIR / "temp_lamost_scored.parquet"
            result = crossmatch_tier4_with_lamost(
                input_file=input_file, output_file=lamost_output, use_local=True
            )
            if result is not None:
                input_file = str(lamost_output)
                logger.info(f"Using LAMOST-enhanced catalog: {input_file}")
        except ImportError as e:
            logger.warning(f"Could not import LAMOST crossmatch: {e}")
        except Exception as e:
            logger.warning(f"LAMOST crossmatch failed: {e}")

    score_orphans(input_file, args.output)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
