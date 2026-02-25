"""
TASNI: Cross-match Tier 4 Candidates with LAMOST Spectroscopy
=============================================================

Cross-matches TASNI anomaly candidates with LAMOST DR12 spectroscopy
to identify known stellar types and find spectral anomalies.

Key Functions:
1. Identify known stellar types (M/L/T dwarfs, carbon stars) -> veto
2. Detect spectral anomalies (temp mismatch between IR and spectral Teff)
3. Flag "Unknown" classifications as potentially interesting

Scoring Logic (from config.py):
- LAMOST_KNOWN_TYPE_PENALTY = -20.0 (known M/L/T dwarf explains IR)
- LAMOST_UNKNOWN_BONUS = 5.0 (has spectrum but unknown = interesting)
- LAMOST_TEMP_MISMATCH_BONUS = 10.0 (spectral Teff disagrees with IR Teff)

Usage:
    python crossmatch_lamost.py --input output/tier4_final.parquet
    python crossmatch_lamost.py --test-region orion
    python crossmatch_lamost.py --use-local-catalog
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from astropy import units as u
    from astropy.coordinates import SkyCoord, match_coordinates_sky

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    import pyvo as vo

    PYVO_AVAILABLE = True
except ImportError:
    PYVO_AVAILABLE = False

try:
    from astroquery.vizier import Vizier

    VIZIER_AVAILABLE = True
except ImportError:
    VIZIER_AVAILABLE = False

# VizieR catalog for LAMOST - V/164 is the main DR7 catalog
VIZIER_LAMOST_CATALOG = "V/164"
from tasni.core.config import (
    LAMOST_DIR,
    LAMOST_KNOWN_TYPE_PENALTY,
    LAMOST_TAP_URL,
    LAMOST_TEMP_MISMATCH_BONUS,
    LAMOST_UNKNOWN_BONUS,
    LOG_DIR,
    OUTPUT_DIR,
    RADIUS_WISE_LAMOST,
    ensure_dirs,
)

ensure_dirs()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [LAMOST-XMATCH] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "crossmatch_lamost.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_ir_temperature(w1, w2, w3=None):
    """
    Estimate effective temperature from WISE colors.

    Uses empirical relation for brown dwarfs/cool stars:
    W1-W2 color correlates with Teff for cool objects.

    Args:
        w1: W1 magnitude
        w2: W2 magnitude
        w3: W3 magnitude (optional)

    Returns:
        Estimated Teff in Kelvin, or None if cannot estimate
    """
    if pd.isna(w1) or pd.isna(w2) or w1 > 90 or w2 > 90:
        return None

    w1_w2 = w1 - w2

    # Empirical relations for different color ranges
    # Based on Kirkpatrick et al. (2021) and other brown dwarf studies

    if w1_w2 < 0.5:
        # Hot stars (probably not real orphans)
        return None
    elif w1_w2 < 1.5:
        # M/L transition (~2000-3000K)
        teff = 3500 - 1000 * w1_w2
    elif w1_w2 < 2.5:
        # L dwarfs (~1300-2000K)
        teff = 2500 - 500 * (w1_w2 - 1.5)
    elif w1_w2 < 4.0:
        # T dwarfs (~600-1300K)
        teff = 1500 - 400 * (w1_w2 - 2.5)
    else:
        # Y dwarfs (<600K)
        teff = 700 - 100 * (w1_w2 - 4.0)
        teff = max(teff, 250)  # Floor at 250K

    return teff


def classify_lamost_subclass(subclass):
    """
    Classify LAMOST subclass into categories relevant for IR emission.

    Args:
        subclass: LAMOST subclass string (e.g., "M5", "K2V", "CV", "FGK", "A")

    Returns:
        Tuple of (category, is_known_ir_type)
    """
    if pd.isna(subclass) or subclass == "" or subclass == "Unknown":
        return "UNKNOWN", False

    subclass = str(subclass).upper().strip()

    # Known IR-bright types (should explain WISE emission)
    if any(subclass.startswith(t) for t in ["M", "L", "T", "Y"]):
        return "COOL_DWARF", True

    if subclass.startswith("C") or "CARBON" in subclass:
        return "CARBON_STAR", True

    if subclass.startswith("S") or "S-TYPE" in subclass:
        return "S_TYPE", True

    if "WC" in subclass or "WN" in subclass or "WOLF" in subclass:
        return "WOLF_RAYET", True

    if "BE" in subclass or "AE" in subclass:
        return "EMISSION_STAR", True

    # VizieR table-derived types (from V/164)
    if subclass == "FGK":
        return "NORMAL_STAR", False  # FGK stars don't explain orphan IR
    if subclass == "A":
        return "NORMAL_STAR", False  # A-type stars don't explain orphan IR

    # Other stellar types (less likely to explain anomalous IR)
    if any(subclass.startswith(t) for t in ["O", "B", "A", "F", "G", "K"]):
        return "NORMAL_STAR", False

    if "CV" in subclass or "DN" in subclass or "NOVA" in subclass:
        return "CATACLYSMIC", False

    if "WD" in subclass or "DA" in subclass or "DB" in subclass:
        return "WHITE_DWARF", False

    return "OTHER", False


def query_lamost_for_position(ra, dec, radius_arcsec=3.0, tap_service=None):
    """
    Query LAMOST TAP for sources near a position.

    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Search radius in arcseconds
        tap_service: Optional pre-initialized TAP service

    Returns:
        DataFrame with LAMOST matches, or empty DataFrame if no matches
    """
    if not PYVO_AVAILABLE:
        logger.warning("pyvo not available")
        return pd.DataFrame()

    radius_deg = radius_arcsec / 3600.0

    query = f"""
    SELECT obsid, ra, dec, snr_g, snr_r, snr_i,
           teff, teff_err, logg, logg_err, feh, feh_err,
           rv, rv_err, class, subclass
    FROM dr12_v1_1_lr_stellar
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    """

    try:
        if tap_service is None:
            tap_service = vo.dal.TAPService(LAMOST_TAP_URL)

        result = tap_service.search(query, maxrec=100)

        if len(result) == 0:
            return pd.DataFrame()

        return result.to_table().to_pandas()

    except Exception as e:
        logger.debug(f"LAMOST query failed for ({ra}, {dec}): {e}")
        return pd.DataFrame()


def crossmatch_with_local_catalog(candidates_df, lamost_dir=None):
    """
    Cross-match candidates against locally downloaded LAMOST catalog.

    This is much faster than individual TAP queries if you've already
    downloaded the LAMOST catalog via download_lamost.py.

    Args:
        candidates_df: DataFrame with 'ra', 'dec' columns
        lamost_dir: Directory containing LAMOST parquet files

    Returns:
        DataFrame with LAMOST match columns added
    """
    if lamost_dir is None:
        lamost_dir = LAMOST_DIR

    # Check if local catalog exists
    lamost_files = list(Path(lamost_dir).glob("lamost_hp*.parquet"))

    if not lamost_files:
        logger.warning(f"No local LAMOST catalog found in {lamost_dir}")
        logger.info("Run 'python download_lamost.py --method bulk' first")
        return None

    logger.info(f"Found {len(lamost_files)} LAMOST tile files")

    # Load all LAMOST data
    logger.info("Loading local LAMOST catalog...")
    lamost_dfs = []
    for f in lamost_files:
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                lamost_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")

    if not lamost_dfs:
        logger.warning("No LAMOST data loaded from local files")
        return None

    lamost_df = pd.concat(lamost_dfs, ignore_index=True)
    logger.info(f"Loaded {len(lamost_df):,} LAMOST sources")

    # Coordinate matching
    if not ASTROPY_AVAILABLE:
        logger.error("astropy not available for coordinate matching")
        return None

    # Find RA/Dec columns in LAMOST
    lamost_ra_col = next((c for c in lamost_df.columns if c.lower() == "ra"), None)
    lamost_dec_col = next((c for c in lamost_df.columns if c.lower() in ["dec", "decl"]), None)

    if lamost_ra_col is None or lamost_dec_col is None:
        logger.error(f"Cannot find RA/Dec in LAMOST columns: {lamost_df.columns.tolist()}")
        return None

    # Create SkyCoord objects
    cand_coords = SkyCoord(
        ra=candidates_df["ra"].values * u.deg, dec=candidates_df["dec"].values * u.deg, frame="icrs"
    )

    lamost_coords = SkyCoord(
        ra=lamost_df[lamost_ra_col].values * u.deg,
        dec=lamost_df[lamost_dec_col].values * u.deg,
        frame="icrs",
    )

    logger.info("Performing spatial cross-match...")

    # Cross-match
    idx, sep2d, _ = match_coordinates_sky(cand_coords, lamost_coords)

    # Create result DataFrame
    result = candidates_df.copy()

    # Initialize LAMOST columns
    result["lamost_match"] = False
    result["lamost_sep_arcsec"] = np.inf
    result["lamost_obsid"] = None
    result["lamost_class"] = None
    result["lamost_subclass"] = None
    result["lamost_teff"] = np.nan
    result["lamost_logg"] = np.nan
    result["lamost_feh"] = np.nan
    result["lamost_rv"] = np.nan
    result["lamost_snr"] = np.nan

    # Apply matches within radius
    match_mask = sep2d.arcsec < RADIUS_WISE_LAMOST

    for i, (is_match, match_idx, sep) in enumerate(
        zip(match_mask, idx, sep2d.arcsec, strict=False)
    ):
        if is_match:
            lamost_row = lamost_df.iloc[match_idx]
            result.loc[result.index[i], "lamost_match"] = True
            result.loc[result.index[i], "lamost_sep_arcsec"] = sep
            result.loc[result.index[i], "lamost_obsid"] = lamost_row.get("obsid")
            result.loc[result.index[i], "lamost_class"] = lamost_row.get("class")
            result.loc[result.index[i], "lamost_subclass"] = lamost_row.get("subclass")
            result.loc[result.index[i], "lamost_teff"] = lamost_row.get("teff")
            result.loc[result.index[i], "lamost_logg"] = lamost_row.get("logg")
            result.loc[result.index[i], "lamost_feh"] = lamost_row.get("feh")
            result.loc[result.index[i], "lamost_rv"] = lamost_row.get("rv")
            result.loc[result.index[i], "lamost_snr"] = lamost_row.get(
                "snr_g", lamost_row.get("snrg")
            )

    n_matches = match_mask.sum()
    logger.info(f"Found {n_matches:,} LAMOST matches ({100*n_matches/len(result):.1f}%)")

    return result


def compute_lamost_scores(df):
    """
    Compute LAMOST-based anomaly scores for each candidate.

    Args:
        df: DataFrame with LAMOST match columns

    Returns:
        DataFrame with additional score columns
    """
    result = df.copy()

    # Initialize score columns
    result["lamost_score"] = 0.0
    result["lamost_category"] = "NO_SPECTRUM"
    result["lamost_is_known_ir"] = False
    result["lamost_temp_mismatch"] = False
    result["ir_teff"] = np.nan

    for idx, row in result.iterrows():
        score = 0.0

        # Compute IR temperature estimate
        w1 = row.get("w1mpro", row.get("w1", np.nan))
        w2 = row.get("w2mpro", row.get("w2", np.nan))
        ir_teff = compute_ir_temperature(w1, w2)
        result.loc[idx, "ir_teff"] = ir_teff

        if not row.get("lamost_match", False):
            # No LAMOST spectrum - neither penalty nor bonus
            result.loc[idx, "lamost_category"] = "NO_SPECTRUM"
            result.loc[idx, "lamost_score"] = 0.0
            continue

        # Classify the subclass
        subclass = row.get("lamost_subclass")
        category, is_known_ir = classify_lamost_subclass(subclass)
        result.loc[idx, "lamost_category"] = category
        result.loc[idx, "lamost_is_known_ir"] = is_known_ir

        # Apply scoring
        if is_known_ir:
            # Known IR-bright type explains the emission
            score = LAMOST_KNOWN_TYPE_PENALTY  # -20.0
            logger.debug(f"{row.get('designation')}: Known IR type {subclass} -> penalty {score}")

        elif category == "UNKNOWN":
            # Has spectrum but unknown classification - interesting!
            score = LAMOST_UNKNOWN_BONUS  # +5.0
            logger.debug(f"{row.get('designation')}: Unknown type -> bonus {score}")

        else:
            # Other stellar type - might explain IR, might not
            score = -5.0  # Mild penalty

        # Check for temperature mismatch
        lamost_teff = row.get("lamost_teff")
        if pd.notna(lamost_teff) and ir_teff is not None:
            teff_diff = abs(lamost_teff - ir_teff)

            # If spectral and IR temps disagree by >500K, that's interesting
            if teff_diff > 500:
                result.loc[idx, "lamost_temp_mismatch"] = True
                score += LAMOST_TEMP_MISMATCH_BONUS  # +10.0
                logger.debug(
                    f"{row.get('designation')}: Temp mismatch "
                    f"(spec={lamost_teff}, IR={ir_teff}) -> bonus {LAMOST_TEMP_MISMATCH_BONUS}"
                )

        result.loc[idx, "lamost_score"] = score

    return result


def crossmatch_with_vizier(candidates_df, batch_size=500):
    """
    Cross-match candidates against LAMOST using VizieR catalog V/164.

    This queries VizieR directly for batches of candidates, which is more
    efficient than tile-by-tile downloads and more reliable than TAP.

    V/164 has multiple tables:
    - Table 0: Basic observations (lr)
    - Table 1: FGK stars with Teff, [Fe/H] (lr-fgk7)
    - Table 2: A-type stars with H-line info (lr-a7)
    - Table 3: M stars (lr-m7)

    We combine all tables to get maximum coverage.

    Args:
        candidates_df: DataFrame with 'ra', 'dec' columns
        batch_size: Number of candidates per query batch

    Returns:
        DataFrame with LAMOST match columns added
    """
    if not VIZIER_AVAILABLE or not ASTROPY_AVAILABLE:
        logger.error("astroquery.vizier and astropy required for VizieR crossmatch")
        return None

    logger.info(f"Cross-matching {len(candidates_df):,} candidates with LAMOST via VizieR")
    logger.info(f"Using catalog: {VIZIER_LAMOST_CATALOG} (all sub-tables)")

    # Initialize result DataFrame
    result = candidates_df.copy()
    result["lamost_match"] = False
    result["lamost_sep_arcsec"] = np.inf
    result["lamost_obsid"] = None
    result["lamost_class"] = None
    result["lamost_subclass"] = None
    result["lamost_teff"] = np.nan
    result["lamost_logg"] = np.nan
    result["lamost_feh"] = np.nan
    result["lamost_rv"] = np.nan
    result["lamost_snr"] = np.nan

    # Configure Vizier
    v = Vizier(columns=["*"], row_limit=-1)  # Get all columns  # No limit

    # Process in batches grouped by sky region
    n_batches = (len(candidates_df) + batch_size - 1) // batch_size
    total_matches = 0

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(candidates_df))
        batch = candidates_df.iloc[start_idx:end_idx]

        if len(batch) == 0:
            continue

        # Get RA/Dec bounds for this batch
        ra_min, ra_max = batch["ra"].min(), batch["ra"].max()
        dec_min, dec_max = batch["dec"].min(), batch["dec"].max()

        # Add margin
        margin = RADIUS_WISE_LAMOST / 3600.0  # Convert arcsec to degrees
        ra_min -= margin
        ra_max += margin
        dec_min -= margin
        dec_max += margin

        # Query center and radius
        ra_center = (ra_min + ra_max) / 2
        dec_center = (dec_min + dec_max) / 2

        # Calculate search radius (half-diagonal of the box)
        ra_span = (ra_max - ra_min) * np.cos(np.radians(dec_center))
        dec_span = dec_max - dec_min
        search_radius = np.sqrt(ra_span**2 + dec_span**2) / 2

        # Limit to reasonable radius (VizieR has limits)
        search_radius = min(search_radius, 5.0)  # Max 5 degrees

        try:
            coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.deg, u.deg))

            # Query VizieR - returns multiple tables
            vizier_result = v.query_region(
                coord, radius=search_radius * u.deg, catalog=VIZIER_LAMOST_CATALOG
            )

            if vizier_result and len(vizier_result) > 0:
                # Combine all tables from V/164
                all_dfs = []
                for tbl_idx, tbl in enumerate(vizier_result):
                    try:
                        tbl_df = tbl.to_pandas()
                        # Identify star type based on columns
                        if "Teff" in tbl_df.columns:
                            tbl_df["lamost_star_type"] = "FGK"
                        elif "HaD0.2" in tbl_df.columns:
                            tbl_df["lamost_star_type"] = "A"
                        else:
                            tbl_df["lamost_star_type"] = "UNKNOWN"
                        all_dfs.append(tbl_df)
                    except Exception as e:
                        logger.debug(f"Error converting table {tbl_idx}: {e}")

                if not all_dfs:
                    continue

                # Merge all tables
                lamost_df = pd.concat(all_dfs, ignore_index=True)

                # Find RA/Dec columns (VizieR uses RAJ2000/DEJ2000)
                ra_col = next(
                    (c for c in lamost_df.columns if c.upper() in ["RAJ2000", "RA"]), None
                )
                dec_col = next(
                    (c for c in lamost_df.columns if c.upper() in ["DEJ2000", "DEC"]), None
                )

                if ra_col and dec_col and len(lamost_df) > 0:
                    # Filter out rows with NaN coordinates
                    valid_mask = ~(lamost_df[ra_col].isna() | lamost_df[dec_col].isna())
                    lamost_df = lamost_df[valid_mask].copy()

                    if len(lamost_df) == 0:
                        continue

                    # Cross-match this batch
                    batch_coords = SkyCoord(
                        ra=batch["ra"].values * u.deg, dec=batch["dec"].values * u.deg, frame="icrs"
                    )

                    lamost_coords = SkyCoord(
                        ra=lamost_df[ra_col].values * u.deg,
                        dec=lamost_df[dec_col].values * u.deg,
                        frame="icrs",
                    )

                    # Match
                    idx, sep2d, _ = match_coordinates_sky(batch_coords, lamost_coords)
                    match_mask = sep2d.arcsec < RADIUS_WISE_LAMOST

                    # Apply matches
                    for i, (orig_idx, is_match, match_idx, sep) in enumerate(
                        zip(batch.index, match_mask, idx, sep2d.arcsec, strict=False)
                    ):
                        if is_match:
                            lamost_row = lamost_df.iloc[match_idx]
                            result.loc[orig_idx, "lamost_match"] = True
                            result.loc[orig_idx, "lamost_sep_arcsec"] = sep
                            result.loc[orig_idx, "lamost_obsid"] = lamost_row.get(
                                "ObsID", lamost_row.get("obsid")
                            )
                            # Use star type from table identification
                            star_type = lamost_row.get("lamost_star_type", "UNKNOWN")
                            result.loc[orig_idx, "lamost_class"] = "STAR"
                            result.loc[orig_idx, "lamost_subclass"] = star_type
                            result.loc[orig_idx, "lamost_teff"] = lamost_row.get(
                                "Teff", lamost_row.get("teff", np.nan)
                            )
                            result.loc[orig_idx, "lamost_logg"] = lamost_row.get("logg", np.nan)
                            result.loc[orig_idx, "lamost_feh"] = lamost_row.get(
                                "[Fe/H]", lamost_row.get("feh", np.nan)
                            )
                            result.loc[orig_idx, "lamost_rv"] = lamost_row.get(
                                "HRV", lamost_row.get("RV", lamost_row.get("rv", np.nan))
                            )
                            result.loc[orig_idx, "lamost_snr"] = lamost_row.get(
                                "snrg", lamost_row.get("snr_g", np.nan)
                            )
                            total_matches += 1

        except Exception as e:
            logger.warning(f"Batch {batch_idx+1}: VizieR query failed: {e}")

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"[{batch_idx+1}/{n_batches}] {total_matches} matches so far")

        time.sleep(0.2)  # Rate limit

    logger.info(
        f"VizieR crossmatch complete: {total_matches:,} matches "
        f"({100*total_matches/len(result):.1f}%)"
    )

    return result


def crossmatch_tier4_with_lamost(input_file=None, output_file=None, use_local=True):
    """
    Main function to cross-match Tier 4 candidates with LAMOST.

    Args:
        input_file: Path to Tier 4 candidate parquet
        output_file: Path to save results
        use_local: Use locally downloaded LAMOST catalog (faster)

    Returns:
        DataFrame with LAMOST match data and scores
    """
    if input_file is None:
        input_file = OUTPUT_DIR / "tier4_final.parquet"
    if output_file is None:
        output_file = OUTPUT_DIR / "tier4_with_lamost.parquet"

    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return None

    # Load candidates
    logger.info(f"Loading candidates from {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} Tier 4 candidates")

    # Cross-match
    if use_local:
        result = crossmatch_with_local_catalog(df)

        if result is None:
            logger.info("Falling back to VizieR crossmatch...")
            use_local = False

    if not use_local:
        result = None

        # Try VizieR first (more reliable than TAP)
        if VIZIER_AVAILABLE:
            logger.info("Using VizieR for cross-matching...")
            result = crossmatch_with_vizier(df)
            if result is None:
                logger.warning("VizieR crossmatch failed, trying TAP...")

        if result is None and PYVO_AVAILABLE:
            # TAP-based cross-match (slower but doesn't require local catalog)
            logger.info("Using TAP service for cross-matching (this may take a while)...")

            try:
                tap_service = vo.dal.TAPService(LAMOST_TAP_URL)
            except:
                tap_service = None

            result = df.copy()
            result["lamost_match"] = False
            result["lamost_sep_arcsec"] = np.inf
            result["lamost_obsid"] = None
            result["lamost_class"] = None
            result["lamost_subclass"] = None
            result["lamost_teff"] = np.nan
            result["lamost_logg"] = np.nan
            result["lamost_feh"] = np.nan
            result["lamost_rv"] = np.nan

            for i, (idx, row) in enumerate(df.iterrows()):
                ra = row.get("ra", row.get("ra_deg"))
                dec = row.get("dec", row.get("dec_deg"))

                lamost_df = query_lamost_for_position(ra, dec, RADIUS_WISE_LAMOST, tap_service)

                if len(lamost_df) > 0:
                    # Take closest match
                    lamost_row = lamost_df.iloc[0]
                    result.loc[idx, "lamost_match"] = True
                    result.loc[idx, "lamost_obsid"] = lamost_row.get("obsid")
                    result.loc[idx, "lamost_class"] = lamost_row.get("class")
                    result.loc[idx, "lamost_subclass"] = lamost_row.get("subclass")
                    result.loc[idx, "lamost_teff"] = lamost_row.get("teff")
                    result.loc[idx, "lamost_logg"] = lamost_row.get("logg")
                    result.loc[idx, "lamost_feh"] = lamost_row.get("feh")
                    result.loc[idx, "lamost_rv"] = lamost_row.get("rv")

                if (i + 1) % 100 == 0:
                    n_matches = result["lamost_match"].sum()
                    logger.info(f"[{i+1}/{len(df)}] {n_matches} matches so far")

                time.sleep(0.1)  # Rate limit

        if result is None:
            logger.error("No crossmatch method available - returning None")
            return None

    # Compute scores
    logger.info("Computing LAMOST-based scores...")
    result = compute_lamost_scores(result)

    # Summary statistics
    n_total = len(result)
    n_matches = result["lamost_match"].sum()
    n_known_ir = result["lamost_is_known_ir"].sum()
    n_unknown = (result["lamost_category"] == "UNKNOWN").sum()
    n_temp_mismatch = result["lamost_temp_mismatch"].sum()

    logger.info("=" * 60)
    logger.info("LAMOST Cross-match Summary:")
    logger.info(f"  Total candidates: {n_total:,}")
    logger.info(f"  LAMOST matches: {n_matches:,} ({100*n_matches/n_total:.1f}%)")
    logger.info(f"  Known IR types (veto): {n_known_ir:,}")
    logger.info(f"  Unknown types (bonus): {n_unknown:,}")
    logger.info(f"  Temp mismatches: {n_temp_mismatch:,}")
    logger.info("  Score distribution:")
    logger.info(f"    Mean: {result['lamost_score'].mean():.2f}")
    logger.info(f"    Max: {result['lamost_score'].max():.2f}")
    logger.info(f"    Min: {result['lamost_score'].min():.2f}")
    logger.info("=" * 60)

    # Category breakdown
    if n_matches > 0:
        logger.info("Category breakdown:")
        for cat, count in result["lamost_category"].value_counts().items():
            logger.info(f"  {cat}: {count:,}")

    # Save results
    result.to_parquet(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

    # Save CSV of high-interest candidates
    high_interest = result[
        (result["lamost_score"] > 0)
        | (result["lamost_temp_mismatch"])  # Bonus candidates
        | (result["lamost_category"] == "UNKNOWN")  # Temp mismatch  # Unknown
    ].sort_values("lamost_score", ascending=False)

    if len(high_interest) > 0:
        hi_file = output_file.parent / "lamost_high_interest.csv"
        high_interest.to_csv(hi_file, index=False)
        logger.info(f"Saved {len(high_interest)} high-interest candidates to {hi_file}")

    # Save summary JSON
    summary = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "n_candidates": n_total,
        "n_lamost_matches": int(n_matches),
        "n_known_ir_types": int(n_known_ir),
        "n_unknown_types": int(n_unknown),
        "n_temp_mismatches": int(n_temp_mismatch),
        "n_high_interest": len(high_interest),
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = output_file.parent / "lamost_crossmatch_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-match TASNI candidates with LAMOST")
    parser.add_argument("--input", type=str, help="Input candidate parquet file")
    parser.add_argument("--output", type=str, help="Output file with LAMOST matches")
    parser.add_argument(
        "--use-local-catalog",
        action="store_true",
        help="Use locally downloaded LAMOST catalog (faster)",
    )
    parser.add_argument(
        "--use-tap",
        action="store_true",
        help="Use TAP service for queries (slower but doesn't need local data)",
    )
    parser.add_argument("--test-region", type=str, choices=["orion"], help="Test with known region")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TASNI: LAMOST Cross-match")
    logger.info("=" * 60)

    if args.test_region == "orion":
        # Create test data
        test_data = pd.DataFrame(
            {
                "designation": ["TEST_ORION_1", "TEST_ORION_2", "TEST_ORION_3"],
                "ra": [83.63, 83.80, 84.05],
                "dec": [-5.39, -5.50, -5.20],
                "w1mpro": [14.0, 15.0, 13.5],
                "w2mpro": [13.0, 13.5, 12.0],
            }
        )

        test_file = OUTPUT_DIR / "test_orion_candidates.parquet"
        test_data.to_parquet(test_file, index=False)

        result = crossmatch_tier4_with_lamost(
            input_file=test_file,
            output_file=OUTPUT_DIR / "test_orion_lamost.parquet",
            use_local=not args.use_tap,
        )
        return

    use_local = not args.use_tap

    crossmatch_tier4_with_lamost(
        input_file=args.input, output_file=args.output, use_local=use_local
    )


if __name__ == "__main__":
    main()
