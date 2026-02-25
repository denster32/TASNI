#!/usr/bin/env python3
"""
TASNI: Cross-match with AST3-2 Antarctic Time-Domain Survey

AST3-2 at Dome A, Antarctica provides continuous time-series photometry
with polar night allowing months of continuous monitoring.

Data Access:
- NADC: https://nadc.china-vo.org/data/
- AST3-2 Catalog/Image/Light Curve available via NADC portal

Scientific Value for SETI:
- Variability detection (artificial structures might modulate periodically)
- Transient identification (sudden IR brightening could indicate activity)
- Long baseline (polar night allows months of continuous monitoring)
- Coverage: Southern polar sky (Dec < -30°)

Usage:
    python crossmatch_ast3.py [--input FILE] [--output FILE]
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

try:
    import pyvo

    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [AST3-XMATCH] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NADC TAP Service (if available)
NADC_TAP_URL = "https://nadc.china-vo.org/tap"

# AST3 coverage (approximate)
AST3_DEC_MIN = -90.0  # South celestial pole
AST3_DEC_MAX = -30.0  # Northern limit of coverage

# Search parameters
SEARCH_RADIUS_ARCSEC = 5.0  # Match radius

# Scoring constants
AST3_PERIODIC_BONUS = 15.0  # Periodic variability is interesting
AST3_TRANSIENT_BONUS = 20.0  # Transient behavior is very interesting
AST3_STABLE_BONUS = 5.0  # Stable source confirms thermal origin
AST3_NO_MATCH_NEUTRAL = 0.0  # No data doesn't penalize


def check_ast3_coverage(dec: float) -> bool:
    """Check if declination is within AST3 coverage area."""
    return AST3_DEC_MIN <= dec <= AST3_DEC_MAX


def query_nadc_tap(
    ra: float, dec: float, radius_arcsec: float = 5.0, table: str = "ast3.sources"
) -> pd.DataFrame:
    """
    Query NADC TAP service for AST3 data.

    Note: NADC TAP interface availability may vary.
    This function attempts TAP query and falls back gracefully.
    """
    if not HAS_PYVO:
        logger.warning("pyvo not installed")
        return pd.DataFrame()

    query = f"""
    SELECT *
    FROM {table}
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                     CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600.0}))
    """

    try:
        service = pyvo.dal.TAPService(NADC_TAP_URL)
        result = service.run_sync(query, timeout=60)
        return result.to_table().to_pandas()
    except Exception as e:
        logger.debug(f"NADC TAP query failed: {e}")
        return pd.DataFrame()


def check_nadc_catalog_api(ra: float, dec: float, radius_arcsec: float = 5.0) -> dict:
    """
    Check NADC catalog via REST API.

    NADC provides various APIs for catalog access.
    This attempts to query the AST3 catalog.
    """
    if not HAS_REQUESTS:
        return {}

    # NADC Cone Search endpoint (may vary)
    base_url = "https://nadc.china-vo.org/conesearch/ast3"

    params = {"RA": ra, "DEC": dec, "SR": radius_arcsec / 3600.0}  # degrees

    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code == 200:
            # Parse VOTable or JSON response
            return {"status": "found", "data": response.text}
        return {"status": "not_found"}
    except Exception as e:
        logger.debug(f"NADC API query failed: {e}")
        return {"status": "error", "error": str(e)}


def crossmatch_with_ast3(sources: pd.DataFrame, radius_arcsec: float = 5.0) -> pd.DataFrame:
    """
    Cross-match sources with AST3 catalog.

    Args:
        sources: DataFrame with ra, dec, designation columns
        radius_arcsec: Search radius

    Returns:
        DataFrame with AST3 match information
    """
    results = []

    n_in_coverage = 0
    n_checked = 0
    n_matched = 0

    for idx, row in sources.iterrows():
        ra, dec = row["ra"], row["dec"]
        designation = row["designation"]

        result = {
            "designation": designation,
            "ast3_coverage": False,
            "ast3_match": False,
            "ast3_n_epochs": 0,
            "ast3_variability": None,
            "ast3_score": 0.0,
        }

        # Check coverage
        if pd.notna(dec) and check_ast3_coverage(dec):
            result["ast3_coverage"] = True
            n_in_coverage += 1

            # Try TAP query
            matches = query_nadc_tap(ra, dec, radius_arcsec)

            if len(matches) > 0:
                result["ast3_match"] = True
                result["ast3_n_epochs"] = len(matches)
                n_matched += 1

                # Compute variability if we have multi-epoch data
                # (This would be enhanced with actual light curve analysis)
                result["ast3_score"] = AST3_STABLE_BONUS

            n_checked += 1

            # Rate limiting
            if n_checked % 10 == 0:
                logger.info(f"Checked {n_checked}/{n_in_coverage} sources in AST3 coverage")
                time.sleep(0.5)

        results.append(result)

    logger.info(f"AST3 coverage: {n_in_coverage}/{len(sources)} sources")
    logger.info(f"AST3 matches: {n_matched}/{n_in_coverage} checked")

    return pd.DataFrame(results)


def simulate_ast3_coverage(sources: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate AST3 coverage check without actual API calls.

    Useful for planning and testing when NADC API is unavailable.
    """
    results = []

    for idx, row in sources.iterrows():
        dec = row["dec"]
        designation = row["designation"]

        result = {
            "designation": designation,
            "ast3_coverage": check_ast3_coverage(dec) if pd.notna(dec) else False,
            "ast3_match": False,  # No actual query
            "ast3_n_epochs": 0,
            "ast3_variability": None,
            "ast3_score": 0.0,
            "ast3_status": "simulated",
        }

        results.append(result)

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, total_sources: int):
    """Print AST3 cross-match summary."""
    logger.info("=" * 60)
    logger.info("AST3 Cross-match Summary")
    logger.info("=" * 60)

    n_coverage = df["ast3_coverage"].sum()
    n_match = df["ast3_match"].sum()

    logger.info(f"Total sources: {total_sources}")
    logger.info(
        f"In AST3 coverage (Dec < -30°): {n_coverage} ({100*n_coverage/total_sources:.1f}%)"
    )
    logger.info(f"AST3 matches: {n_match}")

    if n_match > 0:
        mean_epochs = df.loc[df["ast3_match"], "ast3_n_epochs"].mean()
        logger.info(f"Mean epochs per match: {mean_epochs:.1f}")

    # Score distribution
    if "ast3_score" in df.columns:
        logger.info("Score distribution:")
        logger.info(f"  Mean: {df['ast3_score'].mean():.2f}")
        logger.info(f"  Max: {df['ast3_score'].max():.2f}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TASNI AST3 Cross-match")
    parser.add_argument(
        "--input",
        "-i",
        default="./data/processed/golden_targets.csv",
        help="Input CSV with candidates",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./data/processed/golden_with_ast3.parquet",
        help="Output file with AST3 data",
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Simulate coverage check without API calls"
    )
    parser.add_argument(
        "--update-golden", action="store_true", help="Update golden_targets.csv with AST3 data"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: AST3 Antarctic Survey Cross-match")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Mode: {'Simulation' if args.simulate else 'Live query'}")

    # Load sources
    if args.input.endswith(".parquet"):
        sources = pd.read_parquet(args.input)
    else:
        sources = pd.read_csv(args.input)

    logger.info(f"Loaded {len(sources)} sources")

    # Check AST3 coverage distribution
    in_coverage = sources["dec"].apply(lambda d: check_ast3_coverage(d) if pd.notna(d) else False)
    logger.info(f"Sources in AST3 coverage (Dec < -30°): {in_coverage.sum()}")

    # Run cross-match
    if args.simulate:
        logger.info("Running coverage simulation (no API calls)...")
        results = simulate_ast3_coverage(sources)
    else:
        logger.info("Querying NADC for AST3 matches...")
        logger.info("Note: NADC TAP availability may vary. Consider --simulate for testing.")
        results = crossmatch_with_ast3(sources)

    # Print summary
    print_summary(results, len(sources))

    # Save results
    output_path = Path(args.output)
    results.to_parquet(output_path, index=False)
    logger.info(f"Saved AST3 data to {output_path}")

    # CSV copy
    csv_path = output_path.with_suffix(".csv")
    results.to_csv(csv_path, index=False)

    # Update golden targets if requested
    if args.update_golden:
        golden_path = Path("./data/processed/golden_targets.csv")
        if golden_path.exists():
            golden = pd.read_csv(golden_path)

            # Merge AST3 data
            ast3_cols = [
                "designation",
                "ast3_coverage",
                "ast3_match",
                "ast3_n_epochs",
                "ast3_score",
            ]
            ast3_cols = [c for c in ast3_cols if c in results.columns]

            golden = golden.merge(results[ast3_cols], on="designation", how="left")

            # Update score
            golden["score"] = golden["score"] + golden["ast3_score"].fillna(0)

            # Re-sort
            golden = golden.sort_values("score", ascending=False)
            golden.to_csv(golden_path, index=False)
            logger.info(f"Updated {golden_path} with AST3 data")

    return results


if __name__ == "__main__":
    main()
