#!/usr/bin/env python3
"""
TASNI: Full AllWISE Processing Pipeline

Processes all 747 million AllWISE sources to identify thermal anomalies.

Architecture:
- HEALPix tiling (12,288 tiles at NSIDE=32)
- Parallel tile processing
- GPU-accelerated crossmatch (optional)
- Multi-wavelength veto cascade
- ML-based candidate ranking

Compute requirements:
- CPU: 10,000 core-hours
- GPU: 1,000 hours (10-20x speedup)
- Storage: 500GB intermediate

Usage:
    python full_allwise_pipeline.py --start-tile 0 --end-tile 1000 --workers 16
    python full_allwise_pipeline.py --resume --checkpoint-dir /path/to/checkpoints
"""

import argparse
import json
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import healpy as hp
import pandas as pd

warnings.filterwarnings("ignore")

from tasni.core.config import (
    CROSSMATCH_DIR,
    GAIA_DIR,
    GOOD_PH_QUAL,
    HEALPIX_NSIDE,
    LOG_DIR,
    MATCH_RADIUS_ARCSEC,
    MIN_SNR_W1,
    MIN_SNR_W2,
    N_WORKERS,
    OUTPUT_DIR,
    WISE_DIR,
    ensure_dirs,
)

# Setup
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [FULL_PIPELINE] - %(levelname)s - %(message)s",
    handlers=[
        (
            logging.FileHandler(LOG_DIR / "full_allwise_pipeline.log")
            if LOG_DIR
            else logging.NullHandler()
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Constants
N_TILES = hp.nside2npix(HEALPIX_NSIDE)  # 12,288 tiles


class PipelineCheckpoint:
    """Manage pipeline checkpoints for resume capability."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = checkpoint_dir / "pipeline_state.json"

    def load(self) -> dict:
        """Load checkpoint state."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {
            "completed_tiles": [],
            "failed_tiles": [],
            "start_time": None,
            "total_processed": 0,
            "total_orphans": 0,
        }

    def save(self, state: dict):
        """Save checkpoint state."""
        with open(self.checkpoint_file, "w") as f:
            json.dump(state, f, indent=2)

    def mark_tile_complete(self, tile_idx: int, n_processed: int, n_orphans: int):
        """Mark a tile as complete."""
        state = self.load()
        state["completed_tiles"].append(tile_idx)
        state["total_processed"] += n_processed
        state["total_orphans"] += n_orphans
        self.save(state)

    def mark_tile_failed(self, tile_idx: int, error: str):
        """Mark a tile as failed."""
        state = self.load()
        state["failed_tiles"].append({"tile": tile_idx, "error": str(error)})
        self.save(state)


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to WISE sources."""
    initial_count = len(df)

    # SNR filters
    if "w1snr" in df.columns:
        df = df[df["w1snr"] >= MIN_SNR_W1]
    if "w2snr" in df.columns:
        df = df[df["w2snr"] >= MIN_SNR_W2]

    # Photometric quality
    if "ph_qual" in df.columns:
        # Require at least B quality in W1 and W2
        df = df[df["ph_qual"].str[0].isin(GOOD_PH_QUAL)]
        df = df[df["ph_qual"].str[1].isin(GOOD_PH_QUAL)]

    # Contamination flags
    if "cc_flags" in df.columns:
        # Require clean flags
        df = df[df["cc_flags"].str[:2] == "00"]

    logger.debug(f"Quality filter: {initial_count} -> {len(df)} sources")
    return df


def thermal_color_filter(df: pd.DataFrame, min_color: float = 0.5) -> pd.DataFrame:
    """Filter for sources with red W1-W2 colors (thermal anomalies)."""
    initial_count = len(df)

    if "w1mpro" in df.columns and "w2mpro" in df.columns:
        df["w1_w2_color"] = df["w1mpro"] - df["w2mpro"]
        df = df[df["w1_w2_color"] >= min_color]

    logger.debug(f"Thermal color filter (>{min_color}): {initial_count} -> {len(df)} sources")
    return df


def crossmatch_gaia(wise_df: pd.DataFrame, gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-match WISE sources with Gaia to find orphans."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord, match_coordinates_sky

    if len(wise_df) == 0:
        return wise_df

    if len(gaia_df) == 0:
        # All are orphans
        wise_df["has_gaia_match"] = False
        return wise_df

    wise_coords = SkyCoord(ra=wise_df["ra"].values * u.degree, dec=wise_df["dec"].values * u.degree)
    gaia_coords = SkyCoord(ra=gaia_df["ra"].values * u.degree, dec=gaia_df["dec"].values * u.degree)

    idx, sep2d, _ = match_coordinates_sky(wise_coords, gaia_coords)

    # Mark matches within radius
    wise_df["has_gaia_match"] = sep2d.arcsec <= MATCH_RADIUS_ARCSEC
    wise_df["gaia_separation_arcsec"] = sep2d.arcsec

    return wise_df


def process_single_tile(tile_idx: int) -> dict[str, Any]:
    """
    Process a single HEALPix tile.

    Steps:
    1. Load WISE tile
    2. Apply quality filters
    3. Apply thermal color filter
    4. Load Gaia tile
    5. Cross-match to find orphans
    6. Save orphans

    Returns statistics for this tile.
    """
    stats = {
        "tile_idx": tile_idx,
        "n_wise_raw": 0,
        "n_wise_quality": 0,
        "n_wise_thermal": 0,
        "n_orphans": 0,
        "status": "unknown",
        "error": None,
    }

    try:
        # Load WISE data
        wise_file = WISE_DIR / f"wise_hp{tile_idx:05d}.parquet"
        if not wise_file.exists():
            stats["status"] = "no_wise_file"
            return stats

        wise_df = pd.read_parquet(wise_file)
        stats["n_wise_raw"] = len(wise_df)

        # Quality filter
        wise_df = quality_filter(wise_df)
        stats["n_wise_quality"] = len(wise_df)

        # Thermal color filter
        wise_df = thermal_color_filter(wise_df, min_color=0.5)
        stats["n_wise_thermal"] = len(wise_df)

        if len(wise_df) == 0:
            stats["status"] = "no_thermal_sources"
            return stats

        # Load Gaia data
        gaia_file = GAIA_DIR / f"gaia_hp{tile_idx:05d}.parquet"
        gaia_df = pd.DataFrame()
        if gaia_file.exists():
            gaia_df = pd.read_parquet(gaia_file)

        # Cross-match
        wise_df = crossmatch_gaia(wise_df, gaia_df)

        # Filter to orphans only
        orphans = wise_df[~wise_df["has_gaia_match"]].copy()
        stats["n_orphans"] = len(orphans)

        # Save orphans
        if len(orphans) > 0:
            output_file = CROSSMATCH_DIR / f"orphans_hp{tile_idx:05d}.parquet"
            orphans.to_parquet(output_file, index=False)

        stats["status"] = "success"

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        logger.exception("Tile %s failed with unhandled exception", tile_idx)

    return stats


def run_pipeline(
    start_tile: int = 0,
    end_tile: int = N_TILES,
    n_workers: int = N_WORKERS,
    resume: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run the full AllWISE processing pipeline.

    Args:
        start_tile: First tile to process
        end_tile: Last tile to process (exclusive)
        n_workers: Number of parallel workers
        resume: Whether to resume from checkpoint
        checkpoint_dir: Directory for checkpoints

    Returns:
        Summary statistics
    """
    if checkpoint_dir is None:
        checkpoint_dir = OUTPUT_DIR / "checkpoints"

    checkpoint = PipelineCheckpoint(checkpoint_dir)

    # Load state if resuming
    state = checkpoint.load() if resume else {"completed_tiles": [], "failed_tiles": []}

    # Determine tiles to process
    completed = set(state.get("completed_tiles", []))
    tiles_to_process = [t for t in range(start_tile, end_tile) if t not in completed]

    logger.info("=" * 60)
    logger.info("TASNI Full AllWISE Processing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Tiles: {start_tile} to {end_tile} ({end_tile - start_tile} total)")
    logger.info(f"Already completed: {len(completed)}")
    logger.info(f"To process: {len(tiles_to_process)}")
    logger.info(f"Workers: {n_workers}")
    logger.info("=" * 60)

    # Process tiles in parallel
    total_stats = {"n_processed": 0, "n_thermal": 0, "n_orphans": 0, "n_success": 0, "n_failed": 0}

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_tile, t): t for t in tiles_to_process}

        for i, future in enumerate(as_completed(futures)):
            tile_idx = futures[future]

            try:
                stats = future.result()

                if stats["status"] == "success":
                    total_stats["n_success"] += 1
                    total_stats["n_processed"] += stats["n_wise_raw"]
                    total_stats["n_thermal"] += stats["n_wise_thermal"]
                    total_stats["n_orphans"] += stats["n_orphans"]
                    checkpoint.mark_tile_complete(tile_idx, stats["n_wise_raw"], stats["n_orphans"])
                else:
                    total_stats["n_failed"] += 1
                    if stats["error"]:
                        checkpoint.mark_tile_failed(tile_idx, stats["error"])

            except Exception as e:
                logger.exception("Tile %s crashed in worker future", tile_idx)
                total_stats["n_failed"] += 1
                checkpoint.mark_tile_failed(tile_idx, str(e))

            # Progress update
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(tiles_to_process) - i - 1) / rate / 60
                logger.info(
                    f"Progress: {i+1}/{len(tiles_to_process)} tiles | "
                    f"Rate: {rate:.2f} tiles/s | "
                    f"ETA: {eta:.1f} min"
                )

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Successful tiles: {total_stats['n_success']}")
    logger.info(f"Failed tiles: {total_stats['n_failed']}")
    logger.info(f"Total WISE sources processed: {total_stats['n_processed']:,}")
    logger.info(f"Thermal sources: {total_stats['n_thermal']:,}")
    logger.info(f"Orphans found: {total_stats['n_orphans']:,}")
    logger.info("=" * 60)

    return total_stats


def estimate_resource_requirements(n_tiles: int = N_TILES) -> dict[str, float]:
    """Estimate compute requirements for processing."""
    # Based on ~60,000 sources per tile average
    sources_per_tile = 60000
    total_sources = n_tiles * sources_per_tile

    # Processing rates (sources/second)
    cpu_rate = 1000  # Single core
    gpu_rate = 20000  # With cuDF acceleration

    # Time estimates
    cpu_hours = total_sources / cpu_rate / 3600
    gpu_hours = total_sources / gpu_rate / 3600

    return {
        "total_sources": total_sources,
        "n_tiles": n_tiles,
        "cpu_hours_single": cpu_hours,
        "cpu_hours_16core": cpu_hours / 16,
        "gpu_hours": gpu_hours,
        "storage_gb": total_sources * 0.0005,  # ~0.5KB per source
        "ram_gb": 32,  # Minimum recommended
    }


def main():
    parser = argparse.ArgumentParser(description="TASNI Full AllWISE Processing Pipeline")
    parser.add_argument("--start-tile", type=int, default=0)
    parser.add_argument("--end-tile", type=int, default=N_TILES)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--estimate", action="store_true", help="Show resource estimates")
    args = parser.parse_args()

    if args.estimate:
        estimates = estimate_resource_requirements(args.end_tile - args.start_tile)
        print("\nResource Estimates:")
        print(f"  Total sources: {estimates['total_sources']:,}")
        print(f"  CPU (1 core): {estimates['cpu_hours_single']:.0f} hours")
        print(f"  CPU (16 cores): {estimates['cpu_hours_16core']:.0f} hours")
        print(f"  GPU (cuDF): {estimates['gpu_hours']:.0f} hours")
        print(f"  Storage: {estimates['storage_gb']:.0f} GB")
        print(f"  RAM: {estimates['ram_gb']} GB minimum")
        return

    run_pipeline(
        start_tile=args.start_tile,
        end_tile=args.end_tile,
        n_workers=args.workers,
        resume=not args.no_resume,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
