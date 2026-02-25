import os
import sqlite3

import numpy as np
import pandas as pd

# Files
TIER5_FILE = "./data/processed/tier5_radio_silent.parquet"
PHYSICS_FILE = "./data/processed/tier4_physics.parquet"
CLUSTERS_FILE = "./data/processed/image_clusters.csv"
VOTES_DB = "./data/processed/votes.db"
LAMOST_FILE = "./data/processed/tier4_with_lamost.parquet"
LEGACY_FILE = "./data/processed/tier4_with_legacy.parquet"
OUTPUT_CSV = "./data/processed/golden_targets.csv"

# LAMOST Scoring Constants (from config.py)
LAMOST_KNOWN_TYPE_PENALTY = -20.0
LAMOST_UNKNOWN_BONUS = 5.0
LAMOST_TEMP_MISMATCH_BONUS = 10.0

# Legacy Survey Scoring Constants
LEGACY_DEEP_OPTICAL_PENALTY = -15.0
LEGACY_FAINT_PENALTY = -5.0

# Kinematic Scoring Constants
HIGH_PM_THRESHOLD = 500.0  # mas/yr - very high PM (definitely nearby)
MEDIUM_PM_THRESHOLD = 200.0  # mas/yr - moderate PM (likely nearby)
LOW_PM_THRESHOLD = 50.0  # mas/yr - low PM (consistent with distant)

HIGH_PM_PENALTY = -30.0  # Very high PM = definitely nearby object
MEDIUM_PM_PENALTY = -15.0  # Moderate PM = likely nearby object
LOW_PM_BONUS = 10.0  # Low PM = consistent with distant object

HALO_LOCATION_BONUS = 5.0  # High galactic latitude (less confused)
DISK_LOCATION_PENALTY = -5.0  # Low galactic latitude (more confusion)


def main():
    print("Loading datasets...")

    # 1. survivors (Radio Silent)
    df = pd.read_parquet(TIER5_FILE)
    print(f"Radio Silent Base: {len(df)}")

    # 2. Physics (Temperature)
    if os.path.exists(PHYSICS_FILE):
        phys = pd.read_parquet(PHYSICS_FILE)
        # Merge on designation or position?
        # Assuming index alignment or designation match.
        # Both came from tier4_prime, so designation should be unique key.
        df = df.merge(
            phys[["designation", "T_eff_K", "SolidAngle_scale"]], on="designation", how="left"
        )
    else:
        print("Warning: Physics file missing.")
        df["T_eff_K"] = -999

    # 3. Clusters (Visual)
    if os.path.exists(CLUSTERS_FILE):
        clusters = pd.read_csv(CLUSTERS_FILE)
        # Cluster file has 'filename' -> 'dr9_{designation}.jpg' usually
        # Let's extract designation from filename
        # filename format from earlier script: dr9_J000017.78p132338.0.jpg
        # designation format: J000017.78+132338.0
        # Need to clean up mapping.

        # Actually generate_manifest.py logic:
        # obj_id = basename.replace('dr9_', '').replace('.jpg', '')
        # Matches designation? "p" vs "+" in filename?
        # Let's try to normalize.

        clusters["designation_clean"] = (
            clusters["filename"].str.replace("dr9_", "").str.replace(".jpg", "")
        )
        # Fix p/m to + / -?
        # The cutouts script likely sanitized the name.
        # Let's assume precise match or try fuzzy if needed.
        # Actually fast_cutouts used: fname = f"dr9_{row['designation']}.jpg"
        # So it should match perfectly.

        df = df.merge(
            clusters[["filename", "cluster", "umap_x", "umap_y"]],
            left_on="designation",
            right_on=clusters["designation_clean"],
            how="left",
        )
    else:
        print("Warning: Clusters file missing.")
        df["cluster"] = -1

    # 4. Human Votes
    try:
        conn = sqlite3.connect(VOTES_DB)
        votes = pd.read_sql_query("SELECT filename, vote FROM votes", conn)
        conn.close()

        # Merge
        df = df.merge(votes, on="filename", how="left")
    except Exception as e:
        print(f"Warning: Could not load votes: {e}")
        df["vote"] = "none"

    # 5. LAMOST Spectroscopy (NADC Integration)
    if os.path.exists(LAMOST_FILE):
        print("Loading LAMOST cross-match data...")
        lamost = pd.read_parquet(LAMOST_FILE)

        # Select LAMOST columns to merge
        lamost_cols = [
            "designation",
            "lamost_match",
            "lamost_class",
            "lamost_subclass",
            "lamost_teff",
            "lamost_score",
            "lamost_category",
            "lamost_is_known_ir",
            "lamost_temp_mismatch",
            "ir_teff",
        ]
        lamost_cols = [c for c in lamost_cols if c in lamost.columns]

        df = df.merge(lamost[lamost_cols], on="designation", how="left")

        n_lamost = df["lamost_match"].fillna(False).sum()
        print(f"LAMOST matches: {n_lamost}")

        n_known_ir = df["lamost_is_known_ir"].fillna(False).sum()
        print(f"Known IR types (will be penalized): {n_known_ir}")
    else:
        print("Warning: LAMOST cross-match file missing.")
        print("Run: python crossmatch_lamost.py --input output/tier4_final.parquet")
        df["lamost_match"] = False
        df["lamost_score"] = 0
        df["lamost_is_known_ir"] = False
        df["lamost_temp_mismatch"] = False

    # 6. Legacy Survey DR10 Deep Optical (NADC Integration)
    if os.path.exists(LEGACY_FILE):
        print("Loading Legacy Survey cross-match data...")
        legacy = pd.read_parquet(LEGACY_FILE)

        # Select Legacy columns to merge
        legacy_cols = [
            "designation",
            "legacy_match",
            "legacy_mag_g",
            "legacy_mag_r",
            "legacy_mag_z",
            "legacy_type",
            "legacy_score",
            "legacy_is_bright",
            "legacy_is_faint",
        ]
        legacy_cols = [c for c in legacy_cols if c in legacy.columns]

        df = df.merge(legacy[legacy_cols], on="designation", how="left")

        n_legacy = df["legacy_match"].fillna(False).sum()
        print(f"Legacy Survey matches (optical detections): {n_legacy}")

        n_bright = df["legacy_is_bright"].fillna(False).sum()
        n_faint = df["legacy_is_faint"].fillna(False).sum()
        print(f"  Bright detections (strong veto): {n_bright}")
        print(f"  Faint detections (mild veto): {n_faint}")
    else:
        print("Warning: Legacy Survey cross-match file missing.")
        print("Run: python crossmatch_legacy.py --input output/tier4_final.parquet")
        df["legacy_match"] = False
        df["legacy_score"] = 0
        df["legacy_is_bright"] = False
        df["legacy_is_faint"] = False

    # 7. Kinematic Analysis (Proper Motion)
    print("Computing kinematic properties...")

    # Compute total proper motion if not already present
    if "pm_total" not in df.columns and "pmra" in df.columns and "pmdec" in df.columns:
        df["pm_total"] = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)

    # Compute Galactic coordinates
    if "ra" in df.columns and "dec" in df.columns:
        ra_rad = np.radians(df["ra"].values)
        dec_rad = np.radians(df["dec"].values)

        # North Galactic Pole (J2000)
        ra_ngp = np.radians(192.85948)
        dec_ngp = np.radians(27.12825)
        l_ncp = np.radians(122.93192)

        # Compute Galactic latitude b
        sin_b = np.sin(dec_ngp) * np.sin(dec_rad) + np.cos(dec_ngp) * np.cos(dec_rad) * np.cos(
            ra_rad - ra_ngp
        )
        df["gal_b"] = np.degrees(np.arcsin(np.clip(sin_b, -1, 1)))

        # Compute Galactic longitude l
        cos_b = np.cos(np.radians(df["gal_b"]))
        cos_b = np.where(np.abs(cos_b) < 1e-10, 1e-10, cos_b)
        sin_l_minus_lncp = np.cos(dec_rad) * np.sin(ra_rad - ra_ngp) / cos_b
        cos_l_minus_lncp = (
            np.cos(dec_ngp) * np.sin(dec_rad)
            - np.sin(dec_ngp) * np.cos(dec_rad) * np.cos(ra_rad - ra_ngp)
        ) / cos_b
        df["gal_l"] = np.degrees(
            np.mod(l_ncp - np.arctan2(sin_l_minus_lncp, cos_l_minus_lncp), 2 * np.pi)
        )

    # Classify proper motion
    if "pm_total" in df.columns:
        df["pm_category"] = "moderate"
        df.loc[df["pm_total"] >= HIGH_PM_THRESHOLD, "pm_category"] = "very_high"
        df.loc[
            (df["pm_total"] >= MEDIUM_PM_THRESHOLD) & (df["pm_total"] < HIGH_PM_THRESHOLD),
            "pm_category",
        ] = "high"
        df.loc[df["pm_total"] < LOW_PM_THRESHOLD, "pm_category"] = "low"

        n_high_pm = len(df[df["pm_category"].isin(["high", "very_high"])])
        n_low_pm = len(df[df["pm_category"] == "low"])
        print(f"High-PM sources (likely nearby): {n_high_pm}")
        print(f"Low-PM sources (consistent with distant): {n_low_pm}")

    # Classify galactic region
    if "gal_b" in df.columns:
        df["galactic_region"] = "thick_disk"
        df.loc[np.abs(df["gal_b"]) >= 30.0, "galactic_region"] = "halo"
        df.loc[np.abs(df["gal_b"]) < 10.0, "galactic_region"] = "thin_disk"

        n_halo = len(df[df["galactic_region"] == "halo"])
        print(f"Halo-direction sources (|b|>30): {n_halo}")

    print("Filtering for GOLD...")

    # Criteria:
    # 1. Radio Silent (Implicit in TIER5 file)
    # 2. Temperature in [200, 450] K (Extended Room Temp)
    # 3. NOT explicitly voted "artifact" or "fake"

    candidates = df.copy()

    # Filter Temp
    room_temp = candidates[(candidates["T_eff_K"] >= 200) & (candidates["T_eff_K"] <= 450)]
    print(f"Room Temp (200-450K): {len(room_temp)}")

    # If we have valid votes, prioritize "REAL"
    voted_real = candidates[candidates["vote"] == "real"]
    print(f"Voted REAL: {len(voted_real)}")

    # Combine lists
    # Gold Tier 1: Room Temp AND (Voted Real OR Cluster 3 OR Cluster 2)
    # Actually, let's just score them.

    candidates["score"] = 0

    # Points for Temp
    candidates.loc[(candidates["T_eff_K"] >= 250) & (candidates["T_eff_K"] <= 350), "score"] += 50
    candidates.loc[(candidates["T_eff_K"] >= 200) & (candidates["T_eff_K"] <= 500), "score"] += 10

    # Points for Cluster (Outliers)
    candidates.loc[candidates["cluster"] == 3, "score"] += 30
    candidates.loc[candidates["cluster"] == 2, "score"] += 20

    # Points for Vote
    candidates.loc[candidates["vote"] == "real", "score"] += 100
    candidates.loc[candidates["vote"] == "weird", "score"] += 50
    candidates.loc[candidates["vote"] == "artifact", "score"] -= 1000  # Kill it

    # Points for brightness (Signal to Noise)
    # w1snr?
    if "w1snr" in candidates.columns:
        candidates["score"] += candidates["w1snr"]

    # === LAMOST Spectroscopy Scoring (NADC Integration) ===
    if "lamost_score" in candidates.columns:
        candidates["score"] += candidates["lamost_score"].fillna(0)

    # Additional LAMOST penalties/bonuses
    if "lamost_is_known_ir" in candidates.columns:
        # Penalize known IR types (M/L/T dwarfs, carbon stars)
        # Already included in lamost_score, but double-check
        n_known = candidates["lamost_is_known_ir"].fillna(False).sum()
        if n_known > 0:
            print(f"LAMOST: Penalizing {n_known} known IR stellar types")

    if "lamost_temp_mismatch" in candidates.columns:
        # Extra bonus for temperature mismatches (spectral vs IR)
        n_mismatch = candidates["lamost_temp_mismatch"].fillna(False).sum()
        if n_mismatch > 0:
            print(f"LAMOST: {n_mismatch} sources with temp mismatch (bonus applied)")

    # === Legacy Survey DR10 Deep Optical Scoring (NADC Integration) ===
    if "legacy_score" in candidates.columns:
        candidates["score"] += candidates["legacy_score"].fillna(0)

    if "legacy_match" in candidates.columns:
        n_legacy = candidates["legacy_match"].fillna(False).sum()
        if n_legacy > 0:
            print(f"Legacy Survey: Penalizing {n_legacy} sources with optical detections")

    # Bonus for truly dark sources (no Legacy detection)
    if "legacy_match" in candidates.columns:
        truly_dark = ~candidates["legacy_match"].fillna(False)
        candidates.loc[truly_dark, "score"] += 5  # Small bonus for being truly dark
        n_dark = truly_dark.sum()
        print(f"Legacy Survey: {n_dark} truly dark sources (bonus applied)")

    # === Kinematic Scoring ===
    if "pm_category" in candidates.columns:
        # Proper motion penalties/bonuses
        candidates.loc[candidates["pm_category"] == "very_high", "score"] += HIGH_PM_PENALTY
        candidates.loc[candidates["pm_category"] == "high", "score"] += MEDIUM_PM_PENALTY
        candidates.loc[candidates["pm_category"] == "low", "score"] += LOW_PM_BONUS

        n_penalized = len(candidates[candidates["pm_category"].isin(["high", "very_high"])])
        n_bonus = len(candidates[candidates["pm_category"] == "low"])
        print(
            f"Kinematics: {n_penalized} high-PM sources penalized, {n_bonus} low-PM sources bonus"
        )

    if "galactic_region" in candidates.columns:
        # Galactic location modifiers
        candidates.loc[candidates["galactic_region"] == "halo", "score"] += HALO_LOCATION_BONUS
        candidates.loc[
            candidates["galactic_region"] == "thin_disk", "score"
        ] += DISK_LOCATION_PENALTY

        n_halo = len(candidates[candidates["galactic_region"] == "halo"])
        print(f"Kinematics: {n_halo} halo-direction sources (bonus applied)")

    # Sort
    golden = candidates.sort_values(by="score", ascending=False).head(100)

    # Select columns - include LAMOST, Legacy, and Kinematic data
    cols = [
        "designation",
        "ra",
        "dec",
        "w1mpro",
        "w2mpro",
        "T_eff_K",
        "cluster",
        "vote",
        "score",
        "filename",
        "lamost_class",
        "lamost_subclass",
        "lamost_teff",
        "ir_teff",
        "lamost_temp_mismatch",
        "lamost_score",
        "legacy_match",
        "legacy_mag_g",
        "legacy_mag_r",
        "legacy_score",
        "pm_total",
        "pm_category",
        "gal_l",
        "gal_b",
        "galactic_region",
    ]
    # Filter cols that exist
    cols = [c for c in cols if c in golden.columns]

    golden.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved Top 100 Golden Targets to {OUTPUT_CSV}")
    print(golden[cols].head())


if __name__ == "__main__":
    main()
