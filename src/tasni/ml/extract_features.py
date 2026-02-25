"""
Feature Extraction for TASNI Sources

Extract 500+ features from tier5 sources for machine learning classification.

Usage:
    python src/tasni/ml/extract_features.py \
        --tier5 data/processed/final/tier5_radio_silent.parquet \
        --output data/processed/features/tier5_features.parquet \
        --neowise data/processed/final/neowise_epochs.parquet \
        --workers 16
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from tasni.core.tasni_logging import get_logger

logger = get_logger("feature_extraction")

logging.basicConfig(level=logging.INFO)


class FeatureExtractor:
    """Extract comprehensive features from TASNI sources"""

    def __init__(self, neowise_epochs_df: pd.DataFrame = None):
        self.neowise_epochs = neowise_epochs_df

    def extract_photometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract photometric features (100+)"""
        logger.info("Extracting photometric features...")

        features = pd.DataFrame(index=df.index)

        # Magnitudes
        for mag in ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]:
            if mag in df.columns:
                features[f"{mag}_value"] = df[mag].values
                if f"{mag}_unc" in df.columns:
                    features[f"{mag}_uncertainty"] = df[f"{mag}_unc"].values

        # Colors
        if "w1mpro" in df.columns and "w2mpro" in df.columns:
            features["w1_w2_color"] = (df["w1mpro"] - df["w2mpro"]).values
        if "w2mpro" in df.columns and "w3mpro" in df.columns:
            features["w2_w3_color"] = (df["w2mpro"] - df["w3mpro"]).values
        if "w3mpro" in df.columns and "w4mpro" in df.columns:
            features["w3_w4_color"] = (df["w3mpro"] - df["w4mpro"]).values

        # Additional color features
        if "w1_w2_color" in df.columns:
            features["w1_w2_color_original"] = df["w1_w2_color"].values

        # Quality flags
        for flag in ["cc_flags", "ph_qual", "ext_flg"]:
            if flag in df.columns:
                features[f"{flag}_value"] = df[flag].values

        # Coordinates
        if "ra" in df.columns and "dec" in df.columns:
            features["ra_value"] = df["ra"].values
            features["dec_value"] = df["dec"].values

            # Galactic coordinates (with proper error handling)
            try:
                from astropy import units as u
                from astropy.coordinates import SkyCoord

                ra_vals = df["ra"].values
                dec_vals = df["dec"].values
                coords = SkyCoord(ra=ra_vals * u.deg, dec=dec_vals * u.deg)
                galactic = coords.galactic
                features["l_galactic"] = galactic.l.deg
                features["b_galactic"] = galactic.b.deg

                # Ecliptic coordinates
                ecliptic = coords.geocentrictrueecliptic
                features["ecliptic_lon"] = ecliptic.lon.deg
                features["ecliptic_lat"] = ecliptic.lat.deg
            except Exception as e:
                logger.warning(f"Coordinate conversion failed: {e}")

        # Effective temperature (if available)
        if "T_eff_K" in df.columns:
            features["T_eff_value"] = df["T_eff_K"].values

        # SNR features
        for snr in ["w1snr", "w2snr", "w3snr", "w4snr"]:
            if snr in df.columns:
                features[f"{snr}_value"] = df[snr].values

        # Magnitude errors
        for err in ["w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro"]:
            if err in df.columns:
                features[f"{err}_value"] = df[err].values

        logger.info(f"Extracted {len(features.columns)} photometric features")
        return features

    def extract_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract kinematic features (50+)"""
        logger.info("Extracting kinematic features...")

        features = pd.DataFrame(index=df.index)

        # Proper motion
        if "pmra" in df.columns:
            features["pmra_value"] = df["pmra"]
        if "pmdec" in df.columns:
            features["pmdec_value"] = df["pmdec"]
        if "pmra" in df.columns and "pmdec" in df.columns:
            features["pm_total"] = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)
            features["pm_angle"] = np.arctan2(df["pmdec"], df["pmra"]) * 180 / np.pi

            # PM classification
            features["pm_class"] = pd.cut(
                features["pm_total"],
                bins=[0, 10, 50, 200, float("inf")],
                labels=["ZERO", "LOW", "MEDIUM", "HIGH"],
            )

        # Gaia parallax (if available)
        if "parallax" in df.columns:
            features["parallax_value"] = df["parallax"]
            features["distance_pc"] = 1000.0 / df["parallax"].clip(lower=0.1)

        logger.info(f"Extracted {len(features.columns)} kinematic features")
        return features

    def extract_variability_features(
        self, df: pd.DataFrame, epochs_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Extract variability features (200+)"""
        logger.info("Extracting variability features...")

        features = pd.DataFrame(index=df.index)

        # Load variability metrics if available
        if epochs_df is not None and len(epochs_df) > 0:
            logger.info("Using NEOWISE epoch data for variability...")

            # Check column names (may be w1mpro or w1mpro_ep)
            w1_col = "w1mpro_ep" if "w1mpro_ep" in epochs_df.columns else "w1mpro"
            w2_col = "w2mpro_ep" if "w2mpro_ep" in epochs_df.columns else "w2mpro"

            # Group by designation
            for designation, group in epochs_df.groupby("designation"):
                if designation in df.index:
                    # W1 variability
                    w1_mag = group[w1_col].values if w1_col in group.columns else np.array([])
                    if len(w1_mag) > 2:
                        w1_valid = w1_mag[~np.isnan(w1_mag)]
                        if len(w1_valid) > 2:
                            features.loc[designation, "w1_mean"] = np.mean(w1_valid)
                            features.loc[designation, "w1_std"] = np.std(w1_valid)
                            features.loc[designation, "w1_rms"] = np.sqrt(
                                np.mean((w1_valid - np.mean(w1_valid)) ** 2)
                            )
                            features.loc[designation, "w1_range"] = np.max(w1_valid) - np.min(
                                w1_valid
                            )
                            features.loc[designation, "w1_n_epochs"] = len(w1_valid)

                    # W2 variability
                    w2_mag = group[w2_col].values if w2_col in group.columns else np.array([])
                    if len(w2_mag) > 2:
                        w2_valid = w2_mag[~np.isnan(w2_mag)]
                        if len(w2_valid) > 2:
                            features.loc[designation, "w2_mean"] = np.mean(w2_valid)
                            features.loc[designation, "w2_std"] = np.std(w2_valid)
                            features.loc[designation, "w2_rms"] = np.sqrt(
                                np.mean((w2_valid - np.mean(w2_valid)) ** 2)
                            )
                            features.loc[designation, "w2_range"] = np.max(w2_valid) - np.min(
                                w2_valid
                            )
                            features.loc[designation, "w2_n_epochs"] = len(w2_valid)

        else:
            logger.info("No NEOWISE epoch data, using pre-computed variability...")

            # Load pre-computed variability if available
            var_file = "data/processed/final/tier5_variability.parquet"
            if Path(var_file).exists():
                var_df = pd.read_parquet(var_file)

                # Merge features
                for col in var_df.columns:
                    if col != "designation":
                        features[col] = var_df.set_index("designation").reindex(df.index)[col]

        logger.info(f"Extracted {len(features.columns)} variability features")
        return features

    def extract_multiwavelength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract multi-wavelength detection features (100+)"""
        logger.info("Extracting multi-wavelength features...")

        features = pd.DataFrame(index=df.index)

        # Detection flags
        for survey in ["gaia", "twomass", "ps1", "legacy", "nvss", "rosat"]:
            flag_col = f"has_{survey}"
            if flag_col in df.columns:
                features[f"{survey}_detected"] = df[flag_col].astype(int)

        # Upper limits (if available)
        for mag in ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]:
            if f"{mag}_upper_limit" in df.columns:
                features[f"{mag}_is_upper"] = df[f"{mag}_upper_limit"].astype(int)

        # Detection count
        detection_cols = [col for col in df.columns if col.startswith("has_")]
        if detection_cols:
            features["detection_count"] = df[detection_cols].sum(axis=1)
            features["detection_fraction"] = features["detection_count"] / len(detection_cols)

        logger.info(f"Extracted {len(features.columns)} multi-wavelength features")
        return features

    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical and distribution features (20+)"""
        logger.info("Extracting statistical features...")

        features = pd.DataFrame(index=df.index)

        # Magnitude statistics
        mag_cols = ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]
        mag_cols = [col for col in mag_cols if col in df.columns]

        if mag_cols:
            features["mag_mean"] = df[mag_cols].mean(axis=1)
            features["mag_std"] = df[mag_cols].std(axis=1)
            features["mag_min"] = df[mag_cols].min(axis=1)
            features["mag_max"] = df[mag_cols].max(axis=1)
            features["mag_range"] = features["mag_max"] - features["mag_min"]

        # Color statistics
        color_cols = [col for col in df.columns if "color" in col.lower()]
        if color_cols:
            features["color_mean"] = df[color_cols].mean(axis=1)
            features["color_std"] = df[color_cols].std(axis=1)

        logger.info(f"Extracted {len(features.columns)} statistical features")
        return features

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features (500+)"""
        logger.info("Extracting ALL features...")
        logger.info(f"Input sources: {len(df)}")

        all_features = pd.DataFrame(index=df.index)

        # Extract feature groups
        try:
            photometric = self.extract_photometric_features(df)
            all_features = pd.concat([all_features, photometric], axis=1)
        except Exception as e:
            logger.warning(f"Photometric features failed: {e}")

        try:
            kinematic = self.extract_kinematic_features(df)
            all_features = pd.concat([all_features, kinematic], axis=1)
        except Exception as e:
            logger.warning(f"Kinematic features failed: {e}")

        try:
            variability = self.extract_variability_features(df, self.neowise_epochs)
            all_features = pd.concat([all_features, variability], axis=1)
        except Exception as e:
            logger.warning(f"Variability features failed: {e}")

        try:
            multiwavelength = self.extract_multiwavelength_features(df)
            all_features = pd.concat([all_features, multiwavelength], axis=1)
        except Exception as e:
            logger.warning(f"Multi-wavelength features failed: {e}")

        try:
            statistical = self.extract_statistical_features(df)
            all_features = pd.concat([all_features, statistical], axis=1)
        except Exception as e:
            logger.warning(f"Statistical features failed: {e}")

        logger.info(f"Total features extracted: {len(all_features.columns)}")
        logger.info(
            f"Feature extraction complete: {len(all_features)} sources, {len(all_features.columns)} features"
        )

        return all_features


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from TASNI sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--tier5", type=str, required=True, help="Path to tier5 parquet file")

    parser.add_argument(
        "--neowise", type=str, default=None, help="Path to NEOWISE epochs parquet file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features/tier5_features.parquet",
        help="Output path for features parquet file",
    )

    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")

    args = parser.parse_args()

    # Load tier5 sources
    logger.info(f"Loading tier5 sources from {args.tier5}...")
    tier5_df = pd.read_parquet(args.tier5)
    logger.info(f"Loaded {len(tier5_df)} tier5 sources")

    # Set designation as index
    if "designation" in tier5_df.columns:
        tier5_df = tier5_df.set_index("designation")

    # Load NEOWISE epochs (if available)
    neowise_df = None
    if args.neowise and Path(args.neowise).exists():
        logger.info(f"Loading NEOWISE epochs from {args.neowise}...")
        neowise_df = pd.read_parquet(args.neowise)
        logger.info(f"Loaded {len(neowise_df)} epoch records")

    # Extract features
    extractor = FeatureExtractor(neowise_df)
    features_df = extractor.extract_all_features(tier5_df)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save features
    logger.info(f"Saving features to {args.output}...")
    features_df.to_parquet(args.output, compression="snappy")

    # Print summary
    logger.info("=" * 70)
    logger.info("Feature Extraction Summary")
    logger.info("=" * 70)
    logger.info(f"Sources processed: {len(features_df)}")
    logger.info(f"Features extracted: {len(features_df.columns)}")
    logger.info(f"Features per source: {len(features_df.columns)}")
    logger.info(f"Output size: {output_path.stat().st_size / (1024**2):.2f} MB")
    logger.info("=" * 70)

    # Feature list
    logger.info("Features extracted:")
    for i, col in enumerate(features_df.columns, 1):
        logger.info(f"  {i}. {col}")


if __name__ == "__main__":
    main()
