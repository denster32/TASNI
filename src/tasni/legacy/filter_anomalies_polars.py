"""
TASNI: Filter and Rank Anomalies (Polars version - memory efficient)
=====================================================================

Uses Polars instead of Pandas for better memory handling.
"""

import logging
from datetime import datetime

import polars as pl

from tasni.core.config import (
    CLEAN_CC_FLAGS,
    GOOD_PH_QUAL,
    ISOLATION_WEIGHT,
    LOG_DIR,
    MIN_SNR_W1,
    MIN_SNR_W2,
    OUTPUT_DIR,
    W1_FAINT_THRESHOLD,
    W1_W2_ANOMALY_THRESHOLD,
    W3_BRIGHT_THRESHOLD,
    ensure_dirs,
)

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "filter_anomalies.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("TASNI: Filter and Rank Anomalies (Polars)")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    input_path = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    logger.info(f"Loading orphans from {input_path}...")

    # Use lazy evaluation for memory efficiency
    df = pl.scan_parquet(input_path)

    # Get initial count
    initial_count = pl.scan_parquet(input_path).select(pl.count()).collect().item()
    logger.info(f"Total orphan sources: {initial_count:,}")

    # === QUALITY FILTERS ===
    logger.info("")
    logger.info("Applying quality filters...")

    # Clean contamination flags
    if "cc_flags" in df.collect_schema().names():
        df = df.filter(pl.col("cc_flags") == CLEAN_CC_FLAGS)

    # Point sources only
    if "ext_flg" in df.collect_schema().names():
        df = df.filter(pl.col("ext_flg") == 0)

    # SNR thresholds
    if "w1snr" in df.collect_schema().names():
        df = df.filter(pl.col("w1snr") >= MIN_SNR_W1)
    if "w2snr" in df.collect_schema().names():
        df = df.filter(pl.col("w2snr") >= MIN_SNR_W2)

    # Photometric quality (A or B in W1 and W2)
    if "ph_qual" in df.collect_schema().names():
        df = df.filter(
            pl.col("ph_qual").str.slice(0, 1).is_in(GOOD_PH_QUAL)
            & pl.col("ph_qual").str.slice(1, 1).is_in(GOOD_PH_QUAL)
        )

    # === COMPUTE THERMAL PROFILES ===
    logger.info("Computing thermal profiles...")

    cols = df.collect_schema().names()
    if all(c in cols for c in ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]):
        df = df.with_columns(
            [
                (pl.col("w1mpro") - pl.col("w2mpro")).alias("w1_w2"),
                (pl.col("w2mpro") - pl.col("w3mpro")).alias("w2_w3"),
                (pl.col("w3mpro") - pl.col("w4mpro")).alias("w3_w4"),
                (pl.col("w1mpro") - pl.col("w4mpro")).alias("w1_w4"),
            ]
        )

    # === COMPUTE WEIRDNESS SCORE ===
    logger.info("Computing weirdness scores...")

    # Start with base score of 0
    df = df.with_columns(pl.lit(0.0).alias("weirdness"))
    df = df.with_columns(pl.lit("").alias("anomaly_flags"))

    # Blue W1-W2 anomalies (+3 points)
    if "w1_w2" in df.collect_schema().names():
        df = df.with_columns(
            [
                pl.when(pl.col("w1_w2") < W1_W2_ANOMALY_THRESHOLD)
                .then(pl.col("weirdness") + 3.0)
                .otherwise(pl.col("weirdness"))
                .alias("weirdness"),
                pl.when(pl.col("w1_w2") < W1_W2_ANOMALY_THRESHOLD)
                .then(pl.col("anomaly_flags") + "BLUE_W1W2,")
                .otherwise(pl.col("anomaly_flags"))
                .alias("anomaly_flags"),
            ]
        )

    # Warm but not hot (+2 points)
    cols = df.collect_schema().names()
    if all(c in cols for c in ["w1mpro", "w3mpro"]):
        df = df.with_columns(
            [
                pl.when(
                    (pl.col("w3mpro") < W3_BRIGHT_THRESHOLD)
                    & (pl.col("w1mpro") > W1_FAINT_THRESHOLD)
                )
                .then(pl.col("weirdness") + 2.0)
                .otherwise(pl.col("weirdness"))
                .alias("weirdness"),
                pl.when(
                    (pl.col("w3mpro") < W3_BRIGHT_THRESHOLD)
                    & (pl.col("w1mpro") > W1_FAINT_THRESHOLD)
                )
                .then(pl.col("anomaly_flags") + "WARM_NOT_HOT,")
                .otherwise(pl.col("anomaly_flags"))
                .alias("anomaly_flags"),
            ]
        )

    # Isolation score (log10 of separation)
    if "nearest_gaia_sep_arcsec" in df.collect_schema().names():
        df = df.with_columns(
            [
                (pl.col("nearest_gaia_sep_arcsec").clip(1, None).log10() * ISOLATION_WEIGHT).alias(
                    "isolation_score"
                ),
            ]
        )
        df = df.with_columns(
            [
                (pl.col("weirdness") + pl.col("isolation_score")).alias("weirdness"),
            ]
        )

    # Very red W1-W4 (+1 point)
    if "w1_w4" in df.collect_schema().names():
        df = df.with_columns(
            [
                pl.when(pl.col("w1_w4") > 5.0)
                .then(pl.col("weirdness") + 1.0)
                .otherwise(pl.col("weirdness"))
                .alias("weirdness"),
                pl.when(pl.col("w1_w4") > 5.0)
                .then(pl.col("anomaly_flags") + "VERY_RED,")
                .otherwise(pl.col("anomaly_flags"))
                .alias("anomaly_flags"),
            ]
        )

    # Very blue W1-W4 (+1.5 points)
    if "w1_w4" in df.collect_schema().names():
        df = df.with_columns(
            [
                pl.when(pl.col("w1_w4") < -1.0)
                .then(pl.col("weirdness") + 1.5)
                .otherwise(pl.col("weirdness"))
                .alias("weirdness"),
                pl.when(pl.col("w1_w4") < -1.0)
                .then(pl.col("anomaly_flags") + "VERY_BLUE,")
                .otherwise(pl.col("anomaly_flags"))
                .alias("anomaly_flags"),
            ]
        )

    # Clean up flags
    df = df.with_columns(
        [
            pl.col("anomaly_flags").str.strip_chars_end(",").alias("anomaly_flags"),
        ]
    )

    # === COLLECT AND SORT ===
    logger.info("Collecting and sorting results...")
    result = df.sort("weirdness", descending=True).collect()

    # Add rank
    result = result.with_row_index("rank", offset=1)

    logger.info(f"After filtering: {len(result):,} sources")

    # === EXPORT ===
    logger.info("")
    logger.info("Exporting results...")

    # Full catalog
    full_path = OUTPUT_DIR / "anomalies_ranked.parquet"
    result.write_parquet(full_path)
    logger.info(f"Full catalog: {full_path}")

    # Top 10000 as CSV
    top_cols = [
        "rank",
        "designation",
        "ra",
        "dec",
        "w1mpro",
        "w2mpro",
        "w3mpro",
        "w4mpro",
        "w1_w2",
        "w1_w4",
        "nearest_gaia_sep_arcsec",
        "isolation_score",
        "weirdness",
        "anomaly_flags",
    ]
    top_cols = [c for c in top_cols if c in result.columns]

    top_path = OUTPUT_DIR / "top_anomalies.csv"
    result.head(10000).select(top_cols).write_csv(top_path)
    logger.info(f"Top 10000: {top_path}")

    # Extreme anomalies (weirdness > 5)
    extreme = result.filter(pl.col("weirdness") > 5.0)
    if len(extreme) > 0:
        extreme_path = OUTPUT_DIR / "extreme_anomalies.csv"
        extreme.select(top_cols).write_csv(extreme_path)
        logger.info(f"Extreme anomalies (>5.0): {extreme_path} ({len(extreme):,} sources)")

    # === SUMMARY ===
    logger.info("")
    logger.info("=== SUMMARY ===")
    logger.info(f"Total filtered: {len(result):,}")
    logger.info(f"Weirdness > 3: {result.filter(pl.col('weirdness') > 3).height:,}")
    logger.info(f"Weirdness > 5: {result.filter(pl.col('weirdness') > 5).height:,}")
    logger.info(f"Weirdness > 7: {result.filter(pl.col('weirdness') > 7).height:,}")

    # Flag breakdown
    logger.info("")
    logger.info("Anomaly flag counts:")
    for flag in ["BLUE_W1W2", "WARM_NOT_HOT", "VERY_RED", "VERY_BLUE"]:
        count = result.filter(pl.col("anomaly_flags").str.contains(flag)).height
        logger.info(f"  {flag}: {count:,}")

    # Top 5 weirdest
    logger.info("")
    logger.info("Top 5 weirdest sources:")
    top5 = result.head(5).select(["rank", "designation", "ra", "dec", "weirdness", "anomaly_flags"])
    for row in top5.iter_rows(named=True):
        logger.info(
            f"  #{row['rank']}: {row['designation']} ({row['ra']:.4f}, {row['dec']:.4f}) - score {row['weirdness']:.2f} [{row['anomaly_flags']}]"
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
