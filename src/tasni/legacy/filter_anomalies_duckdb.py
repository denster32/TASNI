"""
DuckDB-based anomaly filter - handles large data with spillover
"""

import logging
from pathlib import Path

import duckdb

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "output"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def filter_and_rank():
    logger.info("=" * 60)
    logger.info("TASNI: Anomaly Filter (DuckDB)")
    logger.info("=" * 60)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='20GB'")

    orphan_file = OUTPUT_DIR / "wise_no_gaia_match.parquet"
    logger.info(f"Loading orphans from {orphan_file}...")

    # Quality filters - WISE cc_flags format: 'cc00' where cc are hex digits
    # cc = '00' means clean (no artifact flags)
    logger.info("Applying quality filters...")

    filtered = con.execute(f"""
        SELECT
            designation, ra, dec,
            w1mpro, w2mpro, w3mpro, w4mpro,
            w1sigmpro, w2sigmpro, w3sigmpro, w4sigmpro,
            w1snr, w2snr, w3snr, w4snr,
            pmra, pmdec, ph_qual,
            -- W1-W2 color (larger = redder/cooler)
            (w1mpro - w2mpro) as w1_w2_color,
            -- W3 magnitude
            w3mpro,
            -- W1 magnitude
            w1mpro,
            -- SNR checks
            w1snr * w2snr as snr_w1w2
        FROM read_parquet('{orphan_file}')
        WHERE substring(cc_flags, 1, 2) = '00'  -- Clean sources
          AND ext_flg = '0'  -- Point sources only
          AND w1snr >= 7
          AND w2snr >= 5
          AND w3snr >= 3
          AND w4snr >= 3
    """).df()

    logger.info(f"After quality filters: {len(filtered):,} sources")

    # Compute anomaly scores
    logger.info("Computing anomaly scores...")

    filtered = filtered.copy()

    # W1-W2 color: redder sources = potentially more interesting
    # High positive W1-W2 means W1 is much brighter than W2 (cool/obscured)
    w1_w2_mean = filtered["w1_w2_color"].mean()
    w1_w2_std = filtered["w1_w2_color"].std()
    filtered["w1_w2_zscore"] = (filtered["w1_w2_color"] - w1_w2_mean) / w1_w2_std

    # W3 brightness: bright W3 = warm (potentially interesting)
    w3_mean = filtered["w3mpro"].mean()
    w3_std = filtered["w3mpro"].std()
    filtered["w3_zscore"] = (w3_mean - filtered["w3mpro"]) / w3_std

    # W1 faintness: very faint W1 could be interesting
    w1_mean = filtered["w1mpro"].mean()
    w1_std = filtered["w1mpro"].std()
    filtered["w1_zscore"] = (w1_mean - filtered["w1mpro"]) / w1_std

    # Combined anomaly score
    # Priority: very red W1-W2 + bright W3 + decent SNR
    filtered["anomaly_score"] = (
        filtered["w1_w2_zscore"] * 3.0
        + filtered["w3_zscore"] * 1.5  # Red color is key indicator
        + filtered["w1_zscore"] * 1.0  # Bright W3 = warm  # Faint W1
    )

    # Sort by anomaly score
    filtered = filtered.sort_values("anomaly_score", ascending=False)

    logger.info(f'Scores computed. Top score: {filtered["anomaly_score"].iloc[0]:.2f}')

    # Save outputs
    output_path = OUTPUT_DIR / "anomalies_filtered.parquet"
    filtered.to_parquet(output_path, compression="snappy", index=False)
    logger.info(f"Saved to {output_path}")

    # Save top 1M
    top_n = min(1000000, len(filtered))
    top = filtered.head(top_n)
    top_path = OUTPUT_DIR / "anomalies_top1M.parquet"
    top.to_parquet(top_path, compression="snappy", index=False)
    logger.info(f"Saved top {top_n:,} to {top_path}")

    # Save top 100k CSV
    extreme = filtered.head(100000)
    extreme_path = OUTPUT_DIR / "extreme_anomalies.csv"
    extreme.to_csv(extreme_path, index=False)
    logger.info(f"Saved top 100k to {extreme_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("Input orphans: 406,387,755")
    logger.info(f"After quality filters: {len(filtered):,}")
    logger.info("")
    logger.info("Top 10 anomalies:")
    print(
        filtered[["designation", "ra", "dec", "w1_w2_color", "w3mpro", "anomaly_score"]]
        .head(10)
        .to_string()
    )
    logger.info("=" * 60)

    con.close()
    return filtered


if __name__ == "__main__":
    filter_and_rank()
