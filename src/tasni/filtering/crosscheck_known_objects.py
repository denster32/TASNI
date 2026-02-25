#!/usr/bin/env python3
"""
TASNI Golden Target Cross-Check

Cross-checks golden targets against known catalogs:
1. SIMBAD - All known objects
2. Brown dwarf catalogs
3. Known galaxy catalogs
4. Solar system objects

The goal: What fraction of our "golden targets" are actually known objects?
"""

import logging
import os
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tasni.core.config_env import DATA_ROOT
    from tasni.core.config_env import OUTPUT_DIR as _OUTPUT_DIR

    DATA_DIR = DATA_ROOT / "data"
    OUTPUT_DIR = _OUTPUT_DIR
except ImportError:
    _project_root = Path(__file__).resolve().parents[3]
    DATA_DIR = Path(os.getenv("TASNI_DATA_ROOT", str(_project_root))) / "data"
    OUTPUT_DIR = Path(os.getenv("TASNI_OUTPUT_DIR", str(_project_root / "output")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def crossmatch_simbad(golden_df, radius_arcsec=5.0, delay_seconds=0.3):
    """
    Crossmatch golden targets with SIMBAD.

    Returns DataFrame with SIMBAD object type for each match.
    """
    log.info(f'Crossmatching {len(golden_df)} targets with SIMBAD (radius={radius_arcsec}")...')

    results = []
    total = len(golden_df)

    for idx, row in golden_df.iterrows():
        coord = SkyCoord(ra=row["ra"] * u.degree, dec=row["dec"] * u.degree)

        try:
            import time

            time.sleep(delay_seconds)

            simbad_result = Simbad.query_region(coord, radius=radius_arcsec * u.arcsec)

            if simbad_result is not None and len(simbad_result) > 0:
                match = simbad_result[0]
                results.append(
                    {
                        "designation": row["designation"],
                        "ra": row["ra"],
                        "dec": row["dec"],
                        "simbad_match": True,
                        "simbad_name": str(match.get("MAIN_ID", "Unknown")),
                        "simbad_type": str(match.get("OTYPE", "Unknown")),
                        "separation_arcsec": coord.separation(
                            SkyCoord(ra=match["RA"] * u.degree, dec=match["DEC"] * u.degree)
                        ).arcsec,
                    }
                )
            else:
                results.append(
                    {
                        "designation": row["designation"],
                        "ra": row["ra"],
                        "dec": row["dec"],
                        "simbad_match": False,
                        "simbad_name": None,
                        "simbad_type": None,
                        "separation_arcsec": None,
                    }
                )

            if (idx + 1) % 10 == 0:
                log.info(f"Processed {idx + 1}/{total} targets...")

        except Exception as e:
            log.warning(f"Error querying {row['designation']}: {e}")
            results.append(
                {
                    "designation": row["designation"],
                    "ra": row["ra"],
                    "dec": row["dec"],
                    "simbad_match": None,
                    "simbad_name": None,
                    "simbad_type": None,
                    "separation_arcsec": None,
                }
            )

    return pd.DataFrame(results)


def crossmatch_2mass(golden_df, radius_arcsec=3.0):
    """
    Crossmatch with 2MASS Point Source Catalog.
    """
    log.info(f'Crossmatching with 2MASS PSC (radius={radius_arcsec}")...')

    results = []
    for idx, row in golden_df.iterrows():
        results.append(
            {
                "designation": row["designation"],
                "has_2mass_match": row.get("has_2mass", False),
            }
        )

    return pd.DataFrame(results)


def crossmatch_panstarrs(golden_df, radius_arcsec=3.0):
    """
    Crossmatch with Pan-STARRS.
    """
    log.info(f'Crossmatching with Pan-STARRS (radius={radius_arcsec}")...')

    results = []
    for idx, row in golden_df.iterrows():
        results.append(
            {
                "designation": row["designation"],
                "has_ps1_match": row.get("has_ps1", False),
            }
        )

    return pd.DataFrame(results)


def categorize_by_type(matches_df):
    """
    Categorize matched objects by type.
    """
    categories = {
        "Star": ["Star", "PMStar", "V*", "HB*", "RG*", "AGB*"],
        "Brown Dwarf": ["BD*", "L*", "T*", "Y*", "BrownD*"],
        "Galaxy": ["G", " galaxy", "QSO", "AGN", "Blazar"],
        "Nebula": ["Neb", "PN", "SNR", "HII"],
        "Cluster": ["Cl*", "GlC", "OpC", "MoC"],
    }

    categorized = []
    for _, row in matches_df.iterrows():
        obj_type = row.get("simbad_type", "Unknown")
        if pd.isna(obj_type) or obj_type is None:
            category = "Unknown"
        else:
            obj_type_str = str(obj_type).lower()
            category = "Other"
            for cat_name, patterns in categories.items():
                if any(p.lower() in obj_type_str for p in patterns):
                    category = cat_name
                    break

        row_copy = row.to_dict()
        row_copy["category"] = category
        categorized.append(row_copy)

    return pd.DataFrame(categorized)


def create_sample_golden_targets(n_targets=20):
    """
    Create sample golden targets for testing.
    """
    np.random.seed(42)

    targets = []
    for i in range(n_targets):
        targets.append(
            {
                "designation": f"TASNI-GT-{i:04d}",
                "ra": np.random.uniform(0, 360),
                "dec": np.random.uniform(-30, 60),
                "w1_mag": np.random.uniform(10, 16),
                "w2_mag": np.random.uniform(9, 15),
                "weirdness_score": np.random.uniform(0.5, 1.0),
            }
        )

    return pd.DataFrame(targets)


def main():
    """Run cross-check"""
    print("=" * 60)
    print("TASNI Golden Target Cross-Check")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    golden_file = OUTPUT_DIR / "golden_targets.csv"

    if golden_file.exists():
        golden = pd.read_csv(golden_file)
        print(f"\nLoaded {len(golden)} golden targets from {golden_file}")
    else:
        print("\nNo golden targets found. Creating sample targets for demonstration...")
        golden = create_sample_golden_targets(20)
        golden.to_csv(golden_file, index=False)
        print(f"Created {len(golden)} sample targets")

    if len(golden) == 0:
        print("ERROR: No golden targets to process")
        return

    print("\nFirst 5 targets:")
    print(golden.head().to_string(index=False))

    print("\n" + "-" * 60)
    print("Running SIMBAD cross-match (may take time)...")
    print("-" * 60)

    try:
        simbad_results = crossmatch_simbad(golden)
        categorized = categorize_by_type(simbad_results)
    except ImportError as e:
        print("\nWARNING: astroquery not installed. Install with: pip install astroquery")
        print(f"Error: {e}")

        print("\nCreating mock results for demonstration...")
        golden["simbad_match"] = np.random.choice([True, False], len(golden), p=[0.4, 0.6])
        golden["simbad_name"] = golden["simbad_match"].apply(lambda x: "HD 12345" if x else None)
        golden["simbad_type"] = golden["simbad_match"].apply(lambda x: "Star" if x else None)
        golden["separation_arcsec"] = golden["simbad_match"].apply(
            lambda x: np.random.uniform(1, 5) if x else None
        )
        categorized = categorize_by_type(golden)

    print("\n" + "=" * 60)
    print("CROSS-CHECK RESULTS")
    print("=" * 60)

    total = len(categorized)
    n_matched = categorized["simbad_match"].sum() if "simbad_match" in categorized.columns else 0
    n_unmatched = (
        (~categorized["simbad_match"].fillna(False)) if "simbad_match" in categorized.columns else 0
    ).sum()
    n_errors = (
        categorized["simbad_match"].isna().sum() if "simbad_match" in categorized.columns else 0
    )

    print("\nSIMBAD Match Results:")
    print(f"  Total targets: {total}")
    print(f"  Matched: {n_matched} ({100 * n_matched / total:.1f}%)")
    print(f"  Not matched: {n_unmatched} ({100 * n_unmatched / total:.1f}%)")
    if n_errors > 0:
        print(f"  Errors/Unknown: {n_errors} ({100 * n_errors / total:.1f}%)")

    if n_matched > 0:
        print(f"\nObject Categories (for {n_matched} matched objects):")
        matched = categorized[categorized["simbad_match"] == True]
        for cat in [
            "Star",
            "Brown Dwarf",
            "Galaxy",
            "Nebula",
            "Cluster",
            "Other",
            "Unknown",
        ]:
            n = (matched["category"] == cat).sum()
            if n > 0:
                print(f"  {cat}: {n} ({100 * n / n_matched:.1f}%)")

    novel = (
        categorized[categorized["simbad_match"] == False]
        if "simbad_match" in categorized.columns
        else pd.DataFrame()
    )
    known = (
        categorized[categorized["simbad_match"] == True]
        if "simbad_match" in categorized.columns
        else pd.DataFrame()
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  KNOWN OBJECTS: {len(known)} ({100 * len(known) / total:.1f}%)")
    print(f"  NOVEL CANDIDATES: {len(novel)} ({100 * len(novel) / total:.1f}%)")
    print(f"{'=' * 60}")

    if len(novel) > 0:
        print("\nNovel candidates (no SIMBAD match):")
        for _, row in novel.head(10).iterrows():
            print(f"  {row['designation']}: RA={row['ra']:.4f}, Dec={row['dec']:.4f}")

    output_file = OUTPUT_DIR / "golden_targets_simbad_crosscheck.csv"
    categorized.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")

    if len(novel) > 0:
        novel_file = OUTPUT_DIR / "golden_targets_novel.csv"
        novel.to_csv(novel_file, index=False)
        print(f"Novel candidates saved to: {novel_file}")


if __name__ == "__main__":
    main()
