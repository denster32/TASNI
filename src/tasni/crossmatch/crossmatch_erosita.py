#!/usr/bin/env python3
"""
TASNI: Cross-match with eROSITA DR1 X-ray Catalog

eROSITA (extended ROentgen Survey with an Imaging Telescope Array) provides
the deepest all-sky X-ray survey. DR1 was released December 2023 and covers
the western galactic hemisphere.

Key advantages over ROSAT:
- ~30x deeper flux limit
- Better spatial resolution (16" vs 25")
- Newer data (2020 vs 1990s)

Purpose: Strengthen X-ray veto for Tier5 sources. Any detection would
suggest AGN/binary/stellar activity rather than cold thermal source.

Data access:
- eROSITA-DE DR1: https://erosita.mpe.mpg.de/dr1/
- TAP service: https://erosita.mpe.mpg.de/dr1/erodat/tap/

Usage:
    python crossmatch_erosita.py [--input FILE] [--output FILE]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyvo

    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [EROSITA] - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# eROSITA Simple Cone Search service (DR1 uses SCS, not TAP)
EROSITA_SCS_URL = "https://erosita.mpe.mpg.de/dr1/erodat/catalogue/SCS"

# Search parameters
SEARCH_RADIUS_ARCSEC = 30.0  # 30" match radius (eROSITA PSF is ~16")
BATCH_SIZE = 100  # Sources per TAP query


def check_erosita_coverage(ra: float, dec: float) -> bool:
    """
    Check if position is in eROSITA DR1 coverage.

    eROSITA DR1 covers the western galactic hemisphere (l > 180).
    """
    if not HAS_ASTROPY:
        return True  # Can't check, assume covered

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    gal = coord.galactic

    # Western hemisphere: galactic longitude > 180
    return gal.l.deg > 180


def query_erosita_scs(
    ra: float, dec: float, radius_arcsec: float = 30.0, max_retries: int = 3
) -> pd.DataFrame:
    """
    Query eROSITA DR1 main catalog via Simple Cone Search (SCS).

    Returns detected X-ray sources within radius.
    """
    from io import BytesIO

    import requests

    radius_deg = radius_arcsec / 3600.0

    # SCS query URL
    url = f"{EROSITA_SCS_URL}?CAT=DR1_Main&RA={ra}&DEC={dec}&SR={radius_deg}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Parse VOTable response
            from astropy.io.votable import parse

            votable = parse(BytesIO(response.content))
            table = votable.get_first_table()

            if table is not None and len(table.array) > 0:
                df = table.to_table().to_pandas()
                # Rename columns to consistent names
                col_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if col_lower == "ra" or col_lower == "ra_deg":
                        col_mapping[col] = "erosita_ra"
                    elif col_lower == "dec" or col_lower == "dec_deg":
                        col_mapping[col] = "erosita_dec"
                    elif "flux" in col_lower and "0.5" in col_lower:
                        col_mapping[col] = "flux_05_2kev"
                    elif col_lower == "iauname" or col_lower == "name":
                        col_mapping[col] = "iauname"
                    elif "rate" in col_lower and "0.5" in col_lower:
                        col_mapping[col] = "rate_05_2kev"
                    elif "det_like" in col_lower or col_lower == "det_like":
                        col_mapping[col] = "det_likelihood"
                if col_mapping:
                    df = df.rename(columns=col_mapping)
                # Ensure we have ra/dec columns
                if "erosita_ra" not in df.columns:
                    for col in df.columns:
                        if "ra" in col.lower():
                            df["erosita_ra"] = df[col]
                            break
                if "erosita_dec" not in df.columns:
                    for col in df.columns:
                        if "dec" in col.lower():
                            df["erosita_dec"] = df[col]
                            break
                return df
            return pd.DataFrame()

        except Exception as e:
            if attempt < max_retries - 1:
                import time

                time.sleep((attempt + 1) * 2)
            else:
                logger.debug(f"eROSITA SCS query failed: {e}")

    return pd.DataFrame()


def crossmatch_erosita_bulk(
    sources: pd.DataFrame, radius_arcsec: float = 30.0, batch_size: int = 100
) -> pd.DataFrame:
    """
    Cross-match sources against eROSITA DR1 using Simple Cone Search.

    Args:
        sources: DataFrame with ra, dec, designation columns
        radius_arcsec: Search radius
        batch_size: Not used (SCS is per-source)

    Returns:
        DataFrame with eROSITA matches
    """
    import time

    all_matches = []
    n_sources = len(sources)
    n_with_xray = 0

    # Filter to eROSITA coverage first
    in_coverage = sources.apply(lambda row: check_erosita_coverage(row["ra"], row["dec"]), axis=1)
    sources_in_coverage = sources[in_coverage].copy()
    n_in_coverage = len(sources_in_coverage)

    logger.info(f"Sources in eROSITA DR1 coverage: {n_in_coverage}/{n_sources}")

    if n_in_coverage == 0:
        logger.warning("No sources in eROSITA coverage (need galactic l > 180)")
        return pd.DataFrame()

    # Query each source individually via SCS
    for idx, (_, row) in enumerate(sources_in_coverage.iterrows()):
        if pd.notna(row["ra"]) and pd.notna(row["dec"]):
            matches = query_erosita_scs(row["ra"], row["dec"], radius_arcsec)

            if len(matches) > 0:
                matches["target_designation"] = row["designation"]
                matches["target_ra"] = row["ra"]
                matches["target_dec"] = row["dec"]
                all_matches.append(matches)
                n_with_xray += 1
                logger.info(f"X-ray detection: {row['designation']}")

        # Progress every 10 sources
        if (idx + 1) % 10 == 0:
            logger.info(
                f"Progress: {idx + 1}/{n_in_coverage} sources, {n_with_xray} X-ray detections"
            )

        time.sleep(0.3)  # Rate limiting

    if all_matches:
        matches_df = pd.concat(all_matches, ignore_index=True)
        return matches_df

    return pd.DataFrame()


def match_back_to_sources(
    matches: pd.DataFrame, sources: pd.DataFrame, radius_arcsec: float = 30.0
) -> pd.DataFrame:
    """
    Match eROSITA detections back to input sources.
    """
    if len(matches) == 0:
        # No matches - all sources are X-ray quiet
        result = sources[["designation", "ra", "dec"]].copy()
        result["has_erosita"] = False
        result["erosita_flux"] = np.nan
        result["erosita_separation"] = np.nan
        return result

    result_rows = []

    for _, source in sources.iterrows():
        desig = source["designation"]
        ra, dec = source["ra"], source["dec"]

        # Find matches within radius
        if HAS_ASTROPY:
            source_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            match_coords = SkyCoord(
                ra=matches["erosita_ra"].values * u.deg, dec=matches["erosita_dec"].values * u.deg
            )
            separations = source_coord.separation(match_coords).arcsec
            within_radius = separations <= radius_arcsec
        else:
            # Simple distance calculation
            separations = (
                np.sqrt(
                    (matches["erosita_ra"] - ra) ** 2 * np.cos(np.radians(dec)) ** 2
                    + (matches["erosita_dec"] - dec) ** 2
                )
                * 3600
            )
            within_radius = separations <= radius_arcsec

        if within_radius.any():
            # Found X-ray counterpart
            closest_idx = separations.argmin()
            closest_match = matches.iloc[closest_idx]

            result_rows.append(
                {
                    "designation": desig,
                    "ra": ra,
                    "dec": dec,
                    "has_erosita": True,
                    "erosita_name": closest_match.get("iauname", ""),
                    "erosita_flux": closest_match.get("flux_05_2kev", np.nan),
                    "erosita_rate": closest_match.get("rate_05_2kev", np.nan),
                    "erosita_det_like": closest_match.get("det_likelihood", np.nan),
                    "erosita_separation": separations[closest_idx],
                }
            )
        else:
            result_rows.append(
                {
                    "designation": desig,
                    "ra": ra,
                    "dec": dec,
                    "has_erosita": False,
                    "erosita_name": "",
                    "erosita_flux": np.nan,
                    "erosita_rate": np.nan,
                    "erosita_det_like": np.nan,
                    "erosita_separation": np.nan,
                }
            )

    return pd.DataFrame(result_rows)


def main():
    parser = argparse.ArgumentParser(description="TASNI eROSITA DR1 Cross-match")
    _project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--input",
        "-i",
        default=str(_project_root / "data" / "processed" / "final" / "golden_targets.csv"),
        help="Input file with targets",
    )
    parser.add_argument(
        "--tier5",
        default=str(_project_root / "data" / "processed" / "final" / "tier5_radio_silent.parquet"),
        help="Tier5 file for full cross-match",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(_project_root / "data" / "processed" / "final" / "golden_erosita.csv"),
        help="Output file",
    )
    parser.add_argument(
        "--full", action="store_true", help="Cross-match full Tier5 instead of golden"
    )
    parser.add_argument("--radius", type=float, default=30.0, help="Search radius in arcseconds")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: eROSITA DR1 Cross-match")
    logger.info("=" * 60)

    # Load input
    if args.full:
        input_path = args.tier5
        output_path = args.output.replace("golden_erosita", "tier5_erosita")
        sources = pd.read_parquet(input_path)
    else:
        input_path = args.input
        output_path = args.output
        sources = pd.read_csv(input_path)

    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Sources: {len(sources)}")
    logger.info(f"Search radius: {args.radius} arcsec")

    # Check dependencies
    if not HAS_PYVO:
        logger.error("pyvo not installed. Run: pip install pyvo")
        return

    # Run cross-match
    logger.info("Querying eROSITA DR1...")
    matches = crossmatch_erosita_bulk(sources, args.radius)

    # Match back to sources
    logger.info("Matching results to sources...")
    result = match_back_to_sources(matches, sources, args.radius)

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        result.to_parquet(output_path, index=False)
    else:
        result.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")

    # Summary
    n_xray = result["has_erosita"].sum()
    n_quiet = (~result["has_erosita"]).sum()

    logger.info("\n" + "=" * 60)
    logger.info("eROSITA CROSS-MATCH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total sources: {len(result)}")
    logger.info(f"X-ray detected: {n_xray}")
    logger.info(f"X-ray quiet: {n_quiet}")
    logger.info(f"X-ray quiet fraction: {n_quiet/len(result)*100:.1f}%")

    if n_xray > 0:
        xray_sources = result[result["has_erosita"]]
        logger.info("\nX-ray detections:")
        for _, row in xray_sources.head(10).iterrows():
            logger.info(
                f"  {row['designation']}: "
                f"flux={row['erosita_flux']:.2e}, "
                f"sep={row['erosita_separation']:.1f}\""
            )

    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
