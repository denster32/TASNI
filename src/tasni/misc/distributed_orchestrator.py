"""
TASNI: Heterogeneous Computing Orchestrator
============================================

Orchestrates work across:
- CPU (20 threads)
- NVIDIA RTX 3060 (12GB VRAM, cuDF)
- Intel Arc A770 (16GB VRAM, PyTorch XPU)

Strategy:
- Large tiles (>1M sources) → GPU (RTX 3060)
- Medium tiles → CPU (16 workers)
- Classification → Intel Arc (larger VRAM)
- I/O coordination → CPU main process

Usage:
    python distributed_orchestrator.py [--mode full] [--workers N]
"""

import argparse
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import healpy as hp

from tasni.core.config import (
    COMPUTE_ENV,
    CROSSMATCH_DIR,
    GAIA_DIR,
    HEALPIX_NSIDE,
    LOG_DIR,
    N_WORKERS,
    USE_CUDA,
    USE_XPU,
    WISE_DIR,
    XPU_ENV,
    ensure_dirs,
)

ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "orchestrator.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

N_TILES = hp.nside2npix(HEALPIX_NSIDE)


def get_tile_size(tile_idx):
    """Estimate tile size by checking file size"""
    wise_file = WISE_DIR / f"wise_hp{tile_idx:05d}.parquet"

    if not wise_file.exists():
        return 0

    # Get file size as proxy for source count
    file_size = wise_file.stat().st_size
    # Approximate: 1KB ~ 100 sources (rough estimate)
    return file_size // 100  # Estimated source count


def assign_tile_to_device(tile_idx):
    """
    Decide which device should process this tile

    Rules:
    - Large tiles (>500K sources) → RTX 3060 (cuDF)
    - Medium tiles (50K-500K) → CPU
    - Small tiles (<50K) → CPU (batch)
    """
    size = get_tile_size(tile_idx)

    if size > 500_000:
        return "cuda"  # RTX 3060
    elif size > 50_000:
        return "cpu"
    else:
        return "cpu_batch"


def run_gpu_crossmatch(tile_idx):
    """Run cuDF-accelerated cross-match on RTX 3060"""
    try:
        result = subprocess.run(
            [f"{COMPUTE_ENV}/bin/python", "gpu_crossmatch.py", "--tile", str(tile_idx)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        return tile_idx, 0, "ok" if result.returncode == 0 else f"error: {result.stderr[-100:]}"
    except subprocess.TimeoutExpired:
        return tile_idx, 0, "timeout"
    except Exception as e:
        return tile_idx, 0, f"error: {e}"


def run_cpu_crossmatch(tile_idx):
    """Run CPU cross-match (original implementation)"""
    try:
        result = subprocess.run(
            [f"{COMPUTE_ENV}/bin/python", "crossmatch_full.py", "--tile", str(tile_idx)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return tile_idx, 0, "ok" if result.returncode == 0 else f"error: {result.stderr[-100:]}"
    except Exception as e:
        return tile_idx, 0, f"error: {e}"


def run_xpu_classification():
    """Run Intel Arc classification"""
    try:
        result = subprocess.run(
            [f"{XPU_ENV}/bin/python", "xpu_classify.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"XPU classification failed: {e}")
        return False


def orchestrate_crossmatch(start_tile=0, end_tile=N_TILES, workers=4):
    """
    Orchestrate cross-match across heterogeneous devices

    Strategy:
    1. Assess tile sizes
    2. Assign to appropriate device
    3. Execute in parallel where possible
    """
    logger.info("=" * 60)
    logger.info("Heterogeneous Cross-Match Orchestration")
    logger.info("=" * 60)
    logger.info(f"CUDA available: {USE_CUDA}")
    logger.info(f"XPU available: {USE_XPU}")
    logger.info(f"Workers: {workers}")

    # Get ready tiles
    wise_tiles = {int(f.stem.split("hp")[1]) for f in WISE_DIR.glob("wise_hp*.parquet")}
    gaia_tiles = {int(f.stem.split("hp")[1]) for f in GAIA_DIR.glob("gaia_hp*.parquet")}
    ready = wise_tiles & gaia_tiles

    completed = {int(f.stem.split("hp")[1]) for f in CROSSMATCH_DIR.glob("orphans_hp*.parquet")}
    to_process = sorted(ready - completed)
    to_process = [t for t in to_process if start_tile <= t < end_tile]

    logger.info(f"Tiles ready: {len(ready)}")
    logger.info(f"Tiles completed: {len(completed)}")
    logger.info(f"Tiles to process: {len(to_process)}")

    if not to_process:
        logger.info("Nothing to process")
        return

    # Categorize tiles by device
    cuda_tiles = []
    cpu_tiles = []

    for tile in to_process:
        device = assign_tile_to_device(tile)
        if device == "cuda" and USE_CUDA:
            cuda_tiles.append(tile)
        else:
            cpu_tiles.append(tile)

    logger.info(f"GPU tiles (RTX 3060): {len(cuda_tiles)}")
    logger.info(f"CPU tiles: {len(cpu_tiles)}")

    # Process
    total_orphans = 0
    failed = []

    start_time = time.time()

    # Process GPU tiles (limited parallelism due to VRAM)
    if cuda_tiles and USE_CUDA:
        logger.info("Processing GPU tiles...")

        for i, tile in enumerate(cuda_tiles):
            tile_idx, n_orphans, status = run_gpu_crossmatch(tile)

            if "error" in status:
                failed.append(tile_idx)
                logger.error(f"GPU Tile {tile_idx}: {status}")
            else:
                logger.info(f"GPU [{i+1}/{len(cuda_tiles)}] Tile {tile_idx}: {status}")

    # Process CPU tiles in parallel
    if cpu_tiles:
        logger.info(f"Processing CPU tiles with {workers} workers...")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_tile_cpu, t): t for t in cpu_tiles}

            for i, future in enumerate(as_completed(futures)):
                tile_idx, n_orphans, status = future.result()
                total_orphans += n_orphans

                if "error" in status:
                    failed.append(tile_idx)
                    logger.error(f"CPU Tile {tile_idx}: {status}")
                elif (i + 1) % 100 == 0:
                    logger.info(
                        f"CPU [{i+1}/{len(cpu_tiles)}] Tile {tile_idx}: {n_orphans} orphans"
                    )

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"Cross-match complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Total orphans: {total_orphans:,}")
    if failed:
        logger.warning(f"Failed tiles: {len(failed)}")
    logger.info("=" * 60)


def process_tile_cpu(tile_idx):
    """CPU tile processing (imports from crossmatch_full)"""
    # Import here to avoid issues
    from crossmatch_full import process_tile

    return process_tile(tile_idx)


def run_full_pipeline(use_gpu=True, use_secondary=False):
    """
    Run complete pipeline with heterogeneous computing

    Steps:
    1. GPU-accelerated cross-match (RTX 3060)
    2. CPU parallel processing
    3. Multi-wavelength scoring
    4. Intel Arc classification
    """
    logger.info("=" * 60)
    logger.info("TASNI: Full Heterogeneous Pipeline")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now().isoformat()}")

    pipeline_start = time.time()

    # Step 1: Cross-match
    logger.info("Step 1: Heterogeneous cross-match...")
    orchestrate_crossmatch()

    # Step 2: Merge orphans
    logger.info("Step 2: Merging orphans...")
    subprocess.run(
        [f"{COMPUTE_ENV}/bin/python", "gpu_crossmatch.py", "--merge-only"],
        cwd=Path(__file__).parent,
    )

    # Step 3: Multi-wavelength scoring
    if use_secondary:
        logger.info("Step 3: Multi-wavelength scoring...")
        subprocess.run(
            [f"{COMPUTE_ENV}/bin/python", "multi_wavelength_scoring.py"], cwd=Path(__file__).parent
        )

    # Step 4: Classification
    if USE_XPU:
        logger.info("Step 4: Intel Arc classification...")
        run_xpu_classification()

    elapsed = time.time() - pipeline_start

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous computing orchestrator")
    parser.add_argument(
        "--mode", choices=["crossmatch", "full", "classify"], default="full", help="Pipeline mode"
    )
    parser.add_argument("--start-tile", type=int, default=0)
    parser.add_argument("--end-tile", type=int, default=N_TILES)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--no-secondary", action="store_true", help="Skip secondary catalogs")
    args = parser.parse_args()

    if args.mode == "crossmatch":
        orchestrate_crossmatch(args.start_tile, args.end_tile, args.workers)
    elif args.mode == "full":
        run_full_pipeline(use_gpu=not args.no_gpu, use_secondary=not args.no_secondary)
    elif args.mode == "classify":
        run_xpu_classification()


if __name__ == "__main__":
    main()
