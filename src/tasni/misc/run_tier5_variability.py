#!/usr/bin/env python3
"""
TASNI: Full Tier5 Variability Analysis with Async Queries
=========================================================

Runs complete variability analysis on the cleaned Tier5 sample using
async NEOWISE queries for maximum performance.

Pipeline:
1. Load cleaned Tier5 sample
2. Fetch NEOWISE multi-epoch photometry (async, 10-50x faster)
3. Compute variability metrics (RMS, chi-squared, Stetson J, trends)
4. Classify sources (NORMAL, VARIABLE, FADING)
5. Generate output catalog and summary

Expected runtime: ~2-4 hours for ~3000 sources with async queries
(vs. 10-20 hours with sequential queries)

Usage:
    python run_tier5_variability.py [--input FILE] [--output FILE]
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tasni.core.config import CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR, ensure_dirs

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "tier5_variability.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Variability classification thresholds
RMS_THRESHOLD = 0.05  # 5% RMS variation
CHI2_THRESHOLD = 2.0  # Reduced chi-squared > 2
STETSON_J_THRESHOLD = 0.5  # Stetson J > 0.5
FADE_RATE_THRESHOLD = 0.010  # 10 mmag/year fading
MIN_EPOCHS = 20  # Minimum epochs for reliable variability


def compute_variability_metrics(epochs_df, source_id):
    """
    Compute variability metrics for a single source.

    Args:
        epochs_df: DataFrame of NEOWISE epochs for this source
        source_id: Source identifier

    Returns:
        dict with variability metrics
    """
    if len(epochs_df) < MIN_EPOCHS:
        return {
            "source_id": source_id,
            "n_epochs": len(epochs_df),
            "var_class": "INSUFFICIENT",
            "w1_rms": np.nan,
            "w2_rms": np.nan,
            "w1_chi2": np.nan,
            "w2_chi2": np.nan,
            "stetson_j": np.nan,
            "w1_trend": np.nan,
            "w2_trend": np.nan,
            "fade_rate": np.nan,
        }

    result = {"source_id": source_id, "n_epochs": len(epochs_df)}

    # W1 metrics
    if "w1mpro" in epochs_df.columns:
        w1 = epochs_df["w1mpro"].dropna()
        w1_err = epochs_df["w1sigmpro"].dropna() if "w1sigmpro" in epochs_df.columns else None

        if len(w1) >= MIN_EPOCHS:
            result["w1_mean"] = w1.mean()
            result["w1_rms"] = w1.std()

            # Reduced chi-squared
            if w1_err is not None and len(w1_err) == len(w1):
                chi2 = np.sum(((w1 - w1.mean()) / w1_err) ** 2)
                result["w1_chi2"] = chi2 / (len(w1) - 1)
            else:
                result["w1_chi2"] = np.nan

            # Linear trend (fading detection)
            if "mjd" in epochs_df.columns:
                mjd = epochs_df.loc[w1.index, "mjd"]
                years = (mjd - mjd.min()) / 365.25
                if len(years) == len(w1):
                    slope, intercept, r, p, se = stats.linregress(years, w1)
                    result["w1_trend"] = slope  # mag/year
                    result["w1_trend_r2"] = r**2
                    result["w1_trend_p"] = p

    # W2 metrics
    if "w2mpro" in epochs_df.columns:
        w2 = epochs_df["w2mpro"].dropna()
        w2_err = epochs_df["w2sigmpro"].dropna() if "w2sigmpro" in epochs_df.columns else None

        if len(w2) >= MIN_EPOCHS:
            result["w2_mean"] = w2.mean()
            result["w2_rms"] = w2.std()

            if w2_err is not None and len(w2_err) == len(w2):
                chi2 = np.sum(((w2 - w2.mean()) / w2_err) ** 2)
                result["w2_chi2"] = chi2 / (len(w2) - 1)
            else:
                result["w2_chi2"] = np.nan

            if "mjd" in epochs_df.columns:
                mjd = epochs_df.loc[w2.index, "mjd"]
                years = (mjd - mjd.min()) / 365.25
                if len(years) == len(w2):
                    slope, intercept, r, p, se = stats.linregress(years, w2)
                    result["w2_trend"] = slope
                    result["w2_trend_r2"] = r**2
                    result["w2_trend_p"] = p

    # Stetson J (correlated variability between bands)
    if "w1mpro" in epochs_df.columns and "w2mpro" in epochs_df.columns:
        w1 = epochs_df["w1mpro"].dropna()
        w2 = epochs_df["w2mpro"].dropna()
        common_idx = w1.index.intersection(w2.index)

        if len(common_idx) >= MIN_EPOCHS:
            w1_norm = (w1.loc[common_idx] - w1.mean()) / w1.std()
            w2_norm = (w2.loc[common_idx] - w2.mean()) / w2.std()
            stetson_j = np.sum(
                np.sign(w1_norm * w2_norm) * np.sqrt(np.abs(w1_norm * w2_norm))
            ) / len(common_idx)
            result["stetson_j"] = stetson_j

    # Average fade rate (both bands)
    fade_rates = []
    if result.get("w1_trend") is not None:
        fade_rates.append(result["w1_trend"])
    if result.get("w2_trend") is not None:
        fade_rates.append(result["w2_trend"])
    result["fade_rate"] = np.mean(fade_rates) if fade_rates else np.nan

    # Classification
    result["var_class"] = classify_variability(result)

    return result


def classify_variability(metrics):
    """
    Classify source variability based on metrics.

    Classes:
    - FADING: Systematic dimming (technosignature candidate)
    - VARIABLE: Significant variability (likely astrophysical)
    - NORMAL: No significant variability
    - INSUFFICIENT: Not enough epochs
    """
    # Check for fading
    fade_rate = metrics.get("fade_rate", np.nan)
    if not np.isnan(fade_rate) and fade_rate > FADE_RATE_THRESHOLD:
        # Verify trend is significant
        p_w1 = metrics.get("w1_trend_p", 1.0)
        p_w2 = metrics.get("w2_trend_p", 1.0)
        if p_w1 < 0.01 or p_w2 < 0.01:
            return "FADING"

    # Check for general variability
    w1_rms = metrics.get("w1_rms", 0)
    w2_rms = metrics.get("w2_rms", 0)
    w1_chi2 = metrics.get("w1_chi2", 0)
    w2_chi2 = metrics.get("w2_chi2", 0)
    stetson_j = metrics.get("stetson_j", 0)

    if (
        w1_rms > RMS_THRESHOLD
        or w2_rms > RMS_THRESHOLD
        or w1_chi2 > CHI2_THRESHOLD
        or w2_chi2 > CHI2_THRESHOLD
        or abs(stetson_j) > STETSON_J_THRESHOLD
    ):
        return "VARIABLE"

    return "NORMAL"


async def run_variability_analysis(sources_df, epochs_df=None):
    """
    Run full variability analysis on sources.

    Args:
        sources_df: DataFrame of sources with ra, dec
        epochs_df: Optional pre-loaded epochs (if None, will fetch)

    Returns:
        DataFrame with variability metrics
    """
    # Fetch epochs if not provided
    if epochs_df is None:
        from async_neowise_query import fetch_all_neowise_epochs

        logger.info("Fetching NEOWISE epochs with async queries...")
        checkpoint = CHECKPOINT_DIR / "tier5_variability_checkpoint.json"
        epochs_df = await fetch_all_neowise_epochs(
            sources_df, max_concurrent=20, rate_limit=50, checkpoint_file=checkpoint
        )

    if len(epochs_df) == 0:
        logger.error("No epochs retrieved!")
        return pd.DataFrame()

    logger.info(f"Total epochs: {len(epochs_df):,}")

    # Group epochs by source position
    # Round coordinates to match sources
    epochs_df["source_key"] = (
        epochs_df["source_ra"].round(5).astype(str)
        + "_"
        + epochs_df["source_dec"].round(5).astype(str)
    )

    sources_df = sources_df.copy()
    sources_df["source_key"] = (
        sources_df["ra"].round(5).astype(str) + "_" + sources_df["dec"].round(5).astype(str)
    )

    # Compute metrics for each source
    logger.info("Computing variability metrics...")
    results = []

    for idx, row in sources_df.iterrows():
        source_epochs = epochs_df[epochs_df["source_key"] == row["source_key"]]

        if len(source_epochs) > 0:
            source_id = row.get("designation", f"src_{idx}")
            metrics = compute_variability_metrics(source_epochs, source_id)
            metrics["ra"] = row["ra"]
            metrics["dec"] = row["dec"]
            results.append(metrics)

        if len(results) % 100 == 0:
            logger.info(f"  Processed {len(results)}/{len(sources_df)} sources")

    results_df = pd.DataFrame(results)

    # Summary
    logger.info("")
    logger.info("=== VARIABILITY SUMMARY ===")
    logger.info(f"Total sources analyzed: {len(results_df)}")

    for var_class in ["FADING", "VARIABLE", "NORMAL", "INSUFFICIENT"]:
        count = (results_df["var_class"] == var_class).sum()
        logger.info(f"  {var_class}: {count}")

    # Highlight fading sources
    fading = results_df[results_df["var_class"] == "FADING"]
    if len(fading) > 0:
        logger.info("")
        logger.info("=== FADING SOURCES (technosignature candidates) ===")
        for idx, row in fading.iterrows():
            logger.info(f"  {row['source_id']}: fade_rate={row['fade_rate']*1000:.1f} mmag/yr")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run full Tier5 variability analysis")
    parser.add_argument(
        "--input",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_cleaned.parquet"),
        help="Input cleaned Tier5 file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "final" / "tier5_variability.parquet"),
        help="Output variability catalog",
    )
    parser.add_argument("--epochs-file", type=str, help="Optional pre-downloaded epochs file")
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: Full Tier5 Variability Analysis")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Load sources
    input_path = Path(args.input)

    # Try cleaned Tier5 first, fall back to original
    if not input_path.exists():
        input_path = OUTPUT_DIR / "final" / "tier5_radio_silent.parquet"
        logger.warning(f"Cleaned Tier5 not found, using original: {input_path}")

    if input_path.suffix == ".parquet":
        sources = pd.read_parquet(input_path)
    else:
        sources = pd.read_csv(input_path)

    logger.info(f"Loaded {len(sources)} sources")

    # Load pre-downloaded epochs if provided
    epochs_df = None
    if args.epochs_file:
        epochs_path = Path(args.epochs_file)
        if epochs_path.exists():
            logger.info(f"Loading pre-downloaded epochs from {epochs_path}")
            epochs_df = pd.read_parquet(epochs_path)
            logger.info(f"Loaded {len(epochs_df)} epochs")

    # Run analysis
    start_time = time.time()
    results = asyncio.run(run_variability_analysis(sources, epochs_df))
    elapsed = time.time() - start_time

    logger.info(f"Analysis completed in {elapsed/60:.1f} minutes")

    # Save results
    if len(results) > 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

        # Save CSV of interesting sources
        interesting = results[results["var_class"].isin(["FADING", "VARIABLE"])]
        if len(interesting) > 0:
            csv_path = output_path.parent / "tier5_variable_sources.csv"
            interesting.to_csv(csv_path, index=False)
            logger.info(f"Variable sources: {csv_path}")

        # Save fading sources separately
        fading = results[results["var_class"] == "FADING"]
        if len(fading) > 0:
            fading_path = output_path.parent / "tier5_fading_candidates.csv"
            fading.to_csv(fading_path, index=False)
            logger.info(f"Fading candidates: {fading_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
