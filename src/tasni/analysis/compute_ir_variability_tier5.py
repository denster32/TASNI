#!/usr/bin/env python3
"""
TASNI: Compute IR Variability for Full Tier5 Sample

Queries NEOWISE multi-epoch photometry and computes variability metrics
for all ~4,137 Tier5 radio-silent sources, looking for additional fading
sources beyond the golden sample.

This is a long-running task (~2-4 hours for full sample).

Usage:
    python compute_ir_variability_tier5.py [--limit N] [--batch-size N]
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import standardized thresholds from config
try:
    from ..core.config import (
        CHI2_VARIABILITY_THRESHOLD,
        FADE_RATE_THRESHOLD_MMAG_YR,
        MIN_EPOCHS_VARIABILITY,
        TREND_THRESHOLD_MAG_YR,
    )
except ImportError:
    FADE_RATE_THRESHOLD_MMAG_YR = 15.0
    TREND_THRESHOLD_MAG_YR = 0.015
    CHI2_VARIABILITY_THRESHOLD = 3.0
    MIN_EPOCHS_VARIABILITY = 10

try:
    import pyvo

    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [TIER5-VAR] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# IRSA TAP Service
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# Search parameters
SEARCH_RADIUS_ARCSEC = 3.0
MIN_EPOCHS = MIN_EPOCHS_VARIABILITY  # Minimum epochs for variability analysis
QUERY_TIMEOUT = 60  # seconds

# Variability thresholds (use standardized values from config)
RMS_STABLE_THRESHOLD = 0.05
RMS_VARIABLE_THRESHOLD = 0.15
CHI2_STABLE_THRESHOLD = 2.0
CHI2_VARIABLE_THRESHOLD = CHI2_VARIABILITY_THRESHOLD
TREND_THRESHOLD = TREND_THRESHOLD_MAG_YR  # mag/year - threshold for fading detection

# Checkpoint settings
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N sources


def query_neowise_single(
    ra: float, dec: float, radius_arcsec: float = 3.0, max_retries: int = 3
) -> pd.DataFrame:
    """Query NEOWISE-R epochs for a single source."""
    if not HAS_PYVO:
        return pd.DataFrame()

    query = f"""
    SELECT TOP 500
        ra, dec, mjd,
        w1mpro as w1mpro_ep, w1sigmpro as w1sigmpro_ep,
        w2mpro as w2mpro_ep, w2sigmpro as w2sigmpro_ep,
        qual_frame
    FROM neowiser_p1bs_psd
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0})) = 1
    ORDER BY mjd
    """

    for attempt in range(max_retries):
        try:
            service = pyvo.dal.TAPService(IRSA_TAP_URL)
            result = service.run_sync(query, timeout=QUERY_TIMEOUT)
            return result.to_table().to_pandas()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                logger.debug(f"Query failed for ({ra:.4f}, {dec:.4f}): {e}")

    return pd.DataFrame()


def compute_variability_metrics(epochs: pd.DataFrame) -> dict:
    """
    Compute variability metrics from multi-epoch photometry.

    Returns dict with RMS, chi2, trend, and classification.
    """
    if len(epochs) < MIN_EPOCHS:
        return {
            "n_epochs": len(epochs),
            "skip_reason": "insufficient_epochs",
            "variability_flag": "UNKNOWN",
        }

    # Filter valid epochs
    valid = ~epochs["w1mpro_ep"].isna() & ~epochs["w2mpro_ep"].isna() & ~epochs["mjd"].isna()
    epochs = epochs[valid].copy()

    if len(epochs) < MIN_EPOCHS:
        return {
            "n_epochs": len(epochs),
            "skip_reason": "insufficient_valid",
            "variability_flag": "UNKNOWN",
        }

    # Time baseline in years
    mjd = epochs["mjd"].values
    baseline_years = (mjd.max() - mjd.min()) / 365.25

    # W1 and W2 magnitudes
    w1 = epochs["w1mpro_ep"].values
    w2 = epochs["w2mpro_ep"].values
    w1_err = epochs["w1sigmpro_ep"].values if "w1sigmpro_ep" in epochs else np.full_like(w1, 0.03)
    w2_err = epochs["w2sigmpro_ep"].values if "w2sigmpro_ep" in epochs else np.full_like(w2, 0.03)

    # RMS scatter
    rms_w1 = np.std(w1, ddof=1) if len(w1) > 1 else np.nan
    rms_w2 = np.std(w2, ddof=1) if len(w2) > 1 else np.nan

    # Reduced chi-squared
    mean_w1 = np.mean(w1)
    mean_w2 = np.mean(w2)

    valid_w1 = w1_err > 0
    valid_w2 = w2_err > 0

    if valid_w1.sum() > 1:
        chi2_w1 = np.sum(((w1[valid_w1] - mean_w1) / w1_err[valid_w1]) ** 2) / (valid_w1.sum() - 1)
    else:
        chi2_w1 = np.nan

    if valid_w2.sum() > 1:
        chi2_w2 = np.sum(((w2[valid_w2] - mean_w2) / w2_err[valid_w2]) ** 2) / (valid_w2.sum() - 1)
    else:
        chi2_w2 = np.nan

    # Linear trend (mag/year)
    t_years = (mjd - mjd.min()) / 365.25

    try:
        slope_w1, intercept_w1 = np.polyfit(t_years, w1, 1)
        slope_w2, intercept_w2 = np.polyfit(t_years, w2, 1)
    except (ValueError, np.linalg.LinAlgError):
        slope_w1 = slope_w2 = np.nan

    # Classification
    is_variable = (
        rms_w1 > RMS_VARIABLE_THRESHOLD
        or rms_w2 > RMS_VARIABLE_THRESHOLD
        or chi2_w1 > CHI2_VARIABLE_THRESHOLD
        or chi2_w2 > CHI2_VARIABLE_THRESHOLD
    )

    # Fading: positive slope (getting fainter) above threshold
    is_fading = (slope_w1 > TREND_THRESHOLD or slope_w2 > TREND_THRESHOLD) and baseline_years > 2.0

    # Brightening: negative slope
    is_brightening = (
        slope_w1 < -TREND_THRESHOLD or slope_w2 < -TREND_THRESHOLD
    ) and baseline_years > 2.0

    if is_fading:
        variability_flag = "FADING"
        variability_score = 15.0
    elif is_brightening:
        variability_flag = "BRIGHTENING"
        variability_score = -5.0
    elif is_variable:
        variability_flag = "VARIABLE"
        variability_score = -10.0
    else:
        variability_flag = "NORMAL"
        variability_score = 0.0

    return {
        "n_epochs": len(epochs),
        "baseline_years": baseline_years,
        "rms_w1": rms_w1,
        "rms_w2": rms_w2,
        "chi2_w1": chi2_w1,
        "chi2_w2": chi2_w2,
        "trend_w1": slope_w1,
        "trend_w2": slope_w2,
        "is_variable": is_variable,
        "is_fading": is_fading,
        "variability_flag": variability_flag,
        "variability_score": variability_score,
    }


def process_tier5_sample(
    tier5_path: str, output_path: str, checkpoint_path: str, limit: int = None, start_from: int = 0
):
    """
    Process full Tier5 sample for variability.

    Args:
        tier5_path: Path to tier5_radio_silent.parquet
        output_path: Output path for results
        checkpoint_path: Path for checkpoint saves
        limit: Optional limit on number of sources
        start_from: Resume from this index
    """
    if not HAS_PYVO:
        logger.error("pyvo not installed. Run: pip install pyvo")
        return

    # Load Tier5 sample
    logger.info(f"Loading Tier5 from {tier5_path}")
    tier5 = pd.read_parquet(tier5_path)
    logger.info(f"Loaded {len(tier5)} Tier5 sources")

    if limit:
        tier5 = tier5.head(limit)
        logger.info(f"Limited to {len(tier5)} sources")

    # Load existing checkpoint if resuming
    results = []
    if start_from > 0 and Path(checkpoint_path).exists():
        existing = pd.read_parquet(checkpoint_path)
        results = existing.to_dict("records")
        logger.info(f"Resuming from checkpoint with {len(results)} existing results")

    n_sources = len(tier5)
    n_fading = 0
    start_time = datetime.now()

    for idx in range(start_from, n_sources):
        row = tier5.iloc[idx]
        designation = row["designation"]
        ra, dec = row["ra"], row["dec"]

        # Query NEOWISE epochs
        epochs = query_neowise_single(ra, dec, SEARCH_RADIUS_ARCSEC)

        # Compute variability metrics
        if len(epochs) >= MIN_EPOCHS:
            metrics = compute_variability_metrics(epochs)
        else:
            metrics = {
                "n_epochs": len(epochs),
                "skip_reason": "insufficient_epochs",
                "variability_flag": "UNKNOWN",
            }

        # Add source info
        result = {
            "designation": designation,
            "ra": ra,
            "dec": dec,
            "w1mpro": row.get("w1mpro", np.nan),
            "w2mpro": row.get("w2mpro", np.nan),
            "w1_w2_color": row.get("w1_w2_color", np.nan),
            **metrics,
        }
        results.append(result)

        # Track fading sources
        if metrics.get("variability_flag") == "FADING":
            n_fading += 1
            logger.info(
                f"*** FADING DETECTED: {designation} "
                f"(trend_w1={metrics.get('trend_w1', 0):.4f}, "
                f"trend_w2={metrics.get('trend_w2', 0):.4f} mag/yr)"
            )

        # Progress logging
        if (idx + 1) % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (idx - start_from + 1) / elapsed * 60  # sources/min
            remaining = (n_sources - idx - 1) / rate if rate > 0 else 0  # minutes
            logger.info(
                f"Progress: {idx + 1}/{n_sources} ({(idx+1)/n_sources*100:.1f}%) "
                f"| {n_fading} fading found | ETA: {remaining:.0f} min"
            )

        # Save checkpoint
        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_parquet(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved at {idx + 1} sources")

        # Rate limiting
        time.sleep(0.3)

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_parquet(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TIER5 VARIABILITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total sources processed: {len(results_df)}")
    logger.info(f"Sources with sufficient epochs: {(results_df['n_epochs'] >= MIN_EPOCHS).sum()}")

    flag_counts = results_df["variability_flag"].value_counts()
    for flag, count in flag_counts.items():
        logger.info(f"  {flag}: {count}")

    if n_fading > 0:
        fading = results_df[results_df["variability_flag"] == "FADING"]
        logger.info(f"\nFading sources found: {n_fading}")
        for _, row in fading.iterrows():
            logger.info(
                f"  {row['designation']}: "
                f"trend_w1={row.get('trend_w1', 0):.4f}, "
                f"trend_w2={row.get('trend_w2', 0):.4f} mag/yr"
            )

    logger.info("=" * 60)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="TASNI Tier5 Variability Analysis")
    _project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--input",
        type=str,
        default=str(_project_root / "data" / "processed" / "final" / "tier5_radio_silent.parquet"),
        help="Input Tier5 parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_project_root / "data" / "processed" / "final" / "tier5_variability.parquet"),
        help="Output variability results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            _project_root / "data" / "interim" / "checkpoints" / "tier5_var_checkpoint.parquet"
        ),
        help="Checkpoint file for resuming",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of sources (for testing)"
    )
    parser.add_argument("--resume", type=int, default=0, help="Resume from source index")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: Tier5 Variability Analysis")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Ensure directories exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    process_tier5_sample(
        args.input, args.output, args.checkpoint, limit=args.limit, start_from=args.resume
    )


if __name__ == "__main__":
    main()
