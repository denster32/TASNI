"""
Light Curve Visualization Tool

Visualize NEOWISE light curves for TASNI sources.

Usage:
    python src/tasni/analysis/light_curve_visualizer.py \
        --designation J143046.35-025927.8 \
        --neowise data/processed/final/neowise_epochs.parquet \
        --output reports/figures/light_curves/
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from tasni.core.tasni_logging import setup_logging

logger = setup_logging("light_curve_visualizer", level="INFO")


class LightCurveVisualizer:
    """Visualize NEOWISE light curves"""

    def __init__(self, epochs_df=None):
        self.epochs_df = epochs_df

    def load_epochs(self, epochs_path: str) -> None:
        """Load NEOWISE epoch data"""
        if not HAS_PANDAS:
            logger.error("pandas not available, cannot load epochs")
            return

        logger.info(f"Loading epochs from {epochs_path}...")
        self.epochs_df = pd.read_parquet(epochs_path)
        logger.info(f"Loaded {len(self.epochs_df)} epoch records")

    def extract_light_curve(self, designation: str) -> dict[str, Any]:
        """Extract light curve data for a single source"""
        if self.epochs_df is None:
            logger.error("No epochs data loaded")
            return None

        # Filter by designation
        source_epochs = self.epochs_df[self.epochs_df["designation"] == designation]

        if len(source_epochs) == 0:
            logger.warning(f"No epochs found for {designation}")
            return None

        # Extract W1 and W2 data
        w1_mags = source_epochs["w1mpro"].values
        w2_mags = source_epochs["w2mpro"].values
        mjd = source_epochs["mjd"].values

        # Calculate Julian years
        year = (mjd - 51544.5) / 365.25 + 2000.0

        return {
            "designation": designation,
            "mjd": mjd,
            "year": year,
            "w1_mag": w1_mags,
            "w2_mag": w2_mags,
            "w1_w2": w1_mags - w2_mags,
            "n_epochs": len(mjd),
        }

    def fit_linear_trend(self, mags: np.ndarray, times: np.ndarray) -> dict[str, float]:
        """Fit linear trend to light curve"""
        # Remove NaN values
        mask = ~np.isnan(mags) & ~np.isnan(times)
        mags_clean = mags[mask]
        times_clean = times[mask]

        if len(mags_clean) < 2:
            return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

        # Linear regression
        coeffs = np.polyfit(times_clean, mags_clean, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Calculate R-squared
        predicted = slope * times_clean + intercept
        residuals = mags_clean - predicted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mags_clean - np.mean(mags_clean)) ** 2)

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }

    def calculate_statistics(self, lc: dict[str, Any]) -> dict[str, float]:
        """Calculate light curve statistics"""
        w1_mag = lc["w1_mag"]
        w2_mag = lc["w2_mag"]

        # Remove NaN values
        w1_clean = w1_mag[~np.isnan(w1_mag)]
        w2_clean = w2_mag[~np.isnan(w2_mag)]

        stats = {
            "designation": lc["designation"],
            "n_epochs": lc["n_epochs"],
            "w1_mean": np.mean(w1_clean),
            "w1_std": np.std(w1_clean),
            "w1_range": np.max(w1_clean) - np.min(w1_clean),
            "w2_mean": np.mean(w2_clean),
            "w2_std": np.std(w2_clean),
            "w2_range": np.max(w2_clean) - np.min(w2_clean),
        }

        # Linear trends
        w1_trend = self.fit_linear_trend(w1_mag, lc["year"])
        w2_trend = self.fit_linear_trend(w2_mag, lc["year"])

        stats.update(
            {
                "w1_slope": w1_trend["slope"],
                "w1_r2": w1_trend["r_squared"],
                "w2_slope": w2_trend["slope"],
                "w2_r2": w2_trend["r_squared"],
            }
        )

        return stats

    def create_plot_data(self, lc: dict[str, Any]) -> dict[str, Any]:
        """Create data structure for plotting"""
        plot_data = {
            "designation": lc["designation"],
            "times": lc["year"].tolist(),
            "w1_mag": lc["w1_mag"].tolist(),
            "w2_mag": lc["w2_mag"].tolist(),
            "w1_w2": lc["w1_w2"].tolist(),
            "n_epochs": lc["n_epochs"],
        }

        # Linear trend lines
        w1_trend = self.fit_linear_trend(lc["w1_mag"], lc["year"])
        w2_trend = self.fit_linear_trend(lc["w2_mag"], lc["year"])

        plot_data["w1_trend"] = (w1_trend["slope"] * lc["year"] + w1_trend["intercept"]).tolist()
        plot_data["w2_trend"] = (w2_trend["slope"] * lc["year"] + w2_trend["intercept"]).tolist()

        return plot_data

    def save_plot_data(self, lc: dict[str, Any], output_path: str) -> None:
        """Save plot data as JSON"""
        import json

        plot_data = self.create_plot_data(lc)

        with open(output_path, "w") as f:
            json.dump(plot_data, f, indent=2)

        logger.info(f"Saved plot data to {output_path}")

    def save_statistics(self, stats: dict[str, float], output_path: str) -> None:
        """Save light curve statistics"""
        import json

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize light curves for TASNI sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--designation",
        type=str,
        required=True,
        help="Source designation (e.g., J143046.35-025927.8)",
    )

    parser.add_argument(
        "--neowise",
        type=str,
        default="data/processed/final/neowise_epochs.parquet",
        help="Path to NEOWISE epochs parquet file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/figures/light_curves/",
        help="Output directory for light curve data",
    )

    args = parser.parse_args()

    # Check for pandas
    if not HAS_PANDAS:
        logger.error("pandas not available. Install with: conda install pandas")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load epochs
    visualizer = LightCurveVisualizer()
    visualizer.load_epochs(args.neowise)

    # Extract light curve
    logger.info(f"Extracting light curve for {args.designation}...")
    lc = visualizer.extract_light_curve(args.designation)

    if lc is None:
        logger.error(f"Failed to extract light curve for {args.designation}")
        return

    # Calculate statistics
    stats = visualizer.calculate_statistics(lc)

    # Save results
    output_base = output_dir / args.designation
    visualizer.save_plot_data(lc, f"{output_base}_plot.json")
    visualizer.save_statistics(stats, f"{output_base}_stats.json")

    # Print summary
    logger.info("=" * 70)
    logger.info("Light Curve Summary")
    logger.info("=" * 70)
    logger.info(f"Designation: {stats['designation']}")
    logger.info(f"Epochs: {stats['n_epochs']}")
    logger.info(
        f"W1: {stats['w1_mean']:.2f} ± {stats['w1_std']:.2f} (range: {stats['w1_range']:.2f})"
    )
    logger.info(
        f"W2: {stats['w2_mean']:.2f} ± {stats['w2_std']:.2f} (range: {stats['w2_range']:.2f})"
    )
    logger.info(f"W1 Slope: {stats['w1_slope']:.4f} mag/yr (R² = {stats['w1_r2']:.3f})")
    logger.info(f"W2 Slope: {stats['w2_slope']:.4f} mag/yr (R² = {stats['w2_r2']:.3f})")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
