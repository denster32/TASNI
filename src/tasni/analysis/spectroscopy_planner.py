"""
Spectroscopy Target Planning Tool

Plan spectroscopic observations for TASNI fading thermal orphans.

Usage:
    python src/tasni/analysis/spectroscopy_planner.py \
        --targets data/processed/final/golden_targets.csv \
        --output data/processed/spectroscopy/observation_plan.csv
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tasni.core.tasni_logging import setup_logging

logger = setup_logging("spectroscopy_planner", level="INFO")


class SpectroscopyPlanner:
    """Plan spectroscopic observations for fading orphans"""

    def __init__(self):
        self.telescopes = {
            "keck_nires": {
                "name": "Keck II / NIRES",
                "wavelength_range": [0.95, 2.45],  # μm
                "spectral_resolution": 2700,
                "min_temp": 200,  # K
                "max_temp": 400,  # K
                "sensitivity_mag": 19.0,  # J band
                "exposure_time": 3600,  # seconds
                "overhead": 300,  # seconds
            },
            "vlt_kmos": {
                "name": "VLT / KMOS",
                "wavelength_range": [2.0, 2.45],  # μm
                "spectral_resolution": 4000,
                "min_temp": 200,  # K
                "max_temp": 350,  # K
                "sensitivity_mag": 18.0,  # H band
                "exposure_time": 2400,  # seconds
                "overhead": 300,  # seconds
            },
            "jwst_nirspec": {
                "name": "JWST / NIRSpec",
                "wavelength_range": [0.6, 5.3],  # μm
                "spectral_resolution": 2700,
                "min_temp": 100,  # K
                "max_temp": 500,  # K
                "sensitivity_mag": 22.0,  # F444W band
                "exposure_time": 600,  # seconds
                "overhead": 300,  # seconds
            },
            "irtf_speX": {
                "name": "IRTF / SpeX",
                "wavelength_range": [0.8, 5.5],  # μm
                "spectral_resolution": 2000,
                "min_temp": 200,  # K
                "max_temp": 400,  # K
                "sensitivity_mag": 17.0,  # J band
                "exposure_time": 1800,  # seconds
                "overhead": 300,  # seconds
            },
        }

    def calculate_visibility(self, ra: float, dec: float, obs_date: datetime) -> dict[str, float]:
        """Calculate visibility information for a source"""
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time

        coord = SkyCoord(ra * u.deg, dec * u.deg)
        time = Time(obs_date)

        # Calculate hour angle at transit
        lst = time.sidereal_time("apparent")
        ha = (lst - coord.ra).wrap_at(12 * u.hour)

        # Calculate airmass at transit (minimum airmass)
        zenith_dist = np.abs(coord.dec.deg - 19.82)  # Maunakea latitude
        min_airmass = 1.0 / np.cos(np.radians(zenith_dist))

        # Calculate hours of visibility (airmass < 2.0)
        max_airmass = 2.0
        max_zenith = np.degrees(np.arccos(1.0 / max_airmass))
        hours_visible = 2 * (max_zenith - zenith_dist) / 15.0  # deg to hours

        return {
            "lst_transit_deg": lst.deg,
            "ha_transit_hours": ha.hour,
            "min_airmass": min_airmass,
            "hours_visible": max(0, hours_visible),
        }

    def estimate_exposure_time(
        self, source_temp: float, magnitude: float, telescope: str, snr_target: float = 10.0
    ) -> float:
        """Estimate required exposure time for desired SNR"""
        if telescope not in self.telescopes:
            return float("inf")

        tele = self.telescopes[telescope]

        # Base exposure time from telescope specs
        base_exposure = tele["exposure_time"]

        # Temperature adjustment (colder = fainter = longer)
        # Reference: 300K, scaling factor
        temp_factor = (300.0 / source_temp) ** 4.0  # Stefan-Boltzmann

        # Magnitude adjustment (fainter = longer)
        # Reference: J=16, scaling factor
        mag_factor = 10 ** (0.4 * (magnitude - 16.0))

        # SNR adjustment (higher SNR = longer)
        snr_factor = (snr_target / 10.0) ** 2.0

        # Calculate required exposure
        required_exposure = base_exposure * temp_factor * mag_factor * snr_factor

        return required_exposure

    def select_best_telescope(self, source_temp: float, magnitude: float) -> str:
        """Select best telescope for a given source"""
        best_telescope = None
        best_score = float("inf")

        for tele_name, tele_specs in self.telescopes.items():
            # Check temperature range
            if source_temp < tele_specs["min_temp"] or source_temp > tele_specs["max_temp"]:
                continue

            # Check magnitude sensitivity
            if magnitude > tele_specs["sensitivity_mag"]:
                continue

            # Estimate exposure time
            exposure = self.estimate_exposure_time(source_temp, magnitude, tele_name)

            # Score: exposure time (lower is better)
            if exposure < best_score:
                best_score = exposure
                best_telescope = tele_name

        return best_telescope if best_telescope else "None"

    def plan_observation(self, source: pd.Series, obs_date: datetime) -> dict[str, Any]:
        """Plan complete observation for a source"""
        designation = source["designation"]
        ra = source["ra"]
        dec = source["dec"]
        temp = source.get("T_eff_K", 300)
        w1_mag = source.get("w1mpro", 16.0)

        # Calculate visibility
        visibility = self.calculate_visibility(ra, dec, obs_date)

        # Select best telescope
        best_tele = self.select_best_telescope(temp, w1_mag)

        # Estimate exposure time
        if best_tele != "None":
            exposure = self.estimate_exposure_time(temp, w1_mag, best_tele)
            overhead = self.telescopes[best_tele]["overhead"]
            total_time = exposure + overhead
        else:
            exposure = float("inf")
            overhead = 0
            total_time = float("inf")

        return {
            "designation": designation,
            "ra_deg": ra,
            "dec_deg": dec,
            "T_eff_K": temp,
            "w1_mag": w1_mag,
            "best_telescope": best_tele,
            "telescope_name": (
                self.telescopes[best_tele]["name"] if best_tele != "None" else "N/A"
            ),
            "exposure_seconds": exposure,
            "overhead_seconds": overhead,
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60.0,
            "lst_transit_deg": visibility["lst_transit_deg"],
            "ha_transit_hours": visibility["ha_transit_hours"],
            "min_airmass": visibility["min_airmass"],
            "hours_visible": visibility["hours_visible"],
            "observation_date": obs_date.strftime("%Y-%m-%d"),
        }

    def plan_all_observations(
        self, targets_df: pd.DataFrame, obs_date: datetime = None
    ) -> pd.DataFrame:
        """Plan observations for all targets"""
        if obs_date is None:
            obs_date = datetime.now()

        logger.info(f"Planning observations for {len(targets_df)} targets...")
        logger.info(f"Observation date: {obs_date.strftime('%Y-%m-%d')}")

        plans = []
        for idx, source in targets_df.iterrows():
            try:
                plan = self.plan_observation(source, obs_date)
                plans.append(plan)
            except Exception as e:
                logger.warning(f"Failed to plan observation for {idx}: {e}")

        plans_df = pd.DataFrame(plans)
        logger.info(f"Planned observations for {len(plans_df)} targets")

        return plans_df

    def create_schedule(
        self, plans_df: pd.DataFrame, telescope: str = "keck_nires"
    ) -> pd.DataFrame:
        """Create observation schedule for a specific telescope"""
        # Filter by telescope
        tele_plans = plans_df[plans_df["best_telescope"] == telescope].copy()
        tele_plans = tele_plans.sort_values("lst_transit_deg")

        if len(tele_plans) == 0:
            logger.warning(f"No targets planned for {telescope}")
            return pd.DataFrame()

        # Calculate cumulative time
        tele_plans["cumulative_time_hours"] = tele_plans["total_time_minutes"].cumsum() / 60.0

        # Calculate start times (assume observations start at sunset)
        # This is simplified; real scheduling would need more logic
        tele_plans["start_hour"] = tele_plans["lst_transit_deg"] / 15.0
        tele_plans["end_hour"] = tele_plans["start_hour"] + tele_plans["total_time_minutes"] / 60.0

        logger.info(f"Created schedule for {telescope}: {len(tele_plans)} targets")
        logger.info(f"Total time: {tele_plans['total_time_minutes'].sum() / 60.0:.2f} hours")

        return tele_plans


def main():
    parser = argparse.ArgumentParser(
        description="Plan spectroscopic observations for TASNI targets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--targets",
        type=str,
        default="data/processed/final/golden_targets.csv",
        help="Path to targets CSV file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/spectroscopy/observation_plan.csv",
        help="Output path for observation plan",
    )

    parser.add_argument(
        "--obs-date", type=str, default=None, help="Observation date (YYYY-MM-DD), default: today"
    )

    parser.add_argument(
        "--schedule", type=str, default="keck_nires", help="Create schedule for specific telescope"
    )

    args = parser.parse_args()

    # Load targets
    logger.info(f"Loading targets from {args.targets}...")
    targets_df = pd.read_csv(args.targets)
    logger.info(f"Loaded {len(targets_df)} targets")

    # Parse observation date
    obs_date = None
    if args.obs_date:
        obs_date = datetime.strptime(args.obs_date, "%Y-%m-%d")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Plan observations
    planner = SpectroscopyPlanner()
    plans_df = planner.plan_all_observations(targets_df, obs_date)

    # Save observation plan
    logger.info(f"Saving observation plan to {args.output}...")
    plans_df.to_csv(args.output, index=False)

    # Create schedule
    if args.schedule:
        schedule_path = str(args.output).replace(".csv", f"_{args.schedule}_schedule.csv")
        schedule_df = planner.create_schedule(plans_df, args.schedule)

        if len(schedule_df) > 0:
            schedule_df.to_csv(schedule_path, index=False)
            logger.info(f"Saved schedule to {schedule_path}")

    # Print summary
    logger.info("=" * 70)
    logger.info("Spectroscopy Observation Planning Summary")
    logger.info("=" * 70)
    logger.info(f"Targets planned: {len(plans_df)}")
    logger.info("Telescope recommendations:")
    for tele in plans_df["best_telescope"].unique():
        if tele != "None":
            count = (plans_df["best_telescope"] == tele).sum()
            logger.info(f"  {tele}: {count} targets")
    logger.info(f"Total observation time: {plans_df['total_time_minutes'].sum() / 60.0:.2f} hours")
    logger.info(f"Output saved: {args.output}")
    logger.info("=" * 70)

    # Print top 5 targets
    logger.info("\nTop 5 Targets by Exposure Time:")
    top_5 = plans_df.nsmallest(5, "total_time_minutes")
    for i, (idx, row) in enumerate(top_5.iterrows(), 1):
        logger.info(
            f"  {i}. {row['designation']} - {row['telescope_name']} - {row['total_time_minutes']:.1f} min - T_eff: {row['T_eff_K']:.1f} K"
        )


if __name__ == "__main__":
    main()
