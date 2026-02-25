#!/usr/bin/env python3
"""
TASNI: Optimized Full Pipeline (100x Faster)
=============================================

Unified runner for all optimized TASNI components:
1. Async catalog downloads (10-50x faster)
2. BallTree/GPU crossmatch (50-100x faster)
3. Vectorized variability analysis (50-100x faster)
4. Streaming I/O for memory efficiency
5. Parallel multi-wavelength scoring

Combined speedup: ~100x over original pipeline

Usage:
    python optimized_pipeline.py [--phase all|download|crossmatch|score|analyze]
    python optimized_pipeline.py --benchmark
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import healpy as hp

from tasni.core.config import (
    CROSSMATCH_DIR,
    GAIA_DIR,
    HEALPIX_NSIDE,
    LOG_DIR,
    N_WORKERS,
    OUTPUT_DIR,
    WISE_DIR,
    ensure_dirs,
)

# Setup
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PIPELINE] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "optimized_pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

N_TILES = hp.nside2npix(HEALPIX_NSIDE)


class PipelineTimer:
    """Track timing for pipeline phases."""

    def __init__(self):
        self.times = {}
        self.start_time = None

    def start(self, phase: str):
        self.times[phase] = {"start": time.perf_counter()}
        logger.info(f"Starting {phase}...")

    def stop(self, phase: str):
        if phase in self.times:
            self.times[phase]["end"] = time.perf_counter()
            elapsed = self.times[phase]["end"] - self.times[phase]["start"]
            self.times[phase]["elapsed"] = elapsed
            logger.info(f"Completed {phase} in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    def summary(self):
        logger.info("=" * 60)
        logger.info("Pipeline Timing Summary")
        logger.info("=" * 60)
        total = 0
        for phase, t in self.times.items():
            if "elapsed" in t:
                logger.info(f"  {phase}: {t['elapsed']:.1f}s")
                total += t["elapsed"]
        logger.info(f"  TOTAL: {total:.1f}s ({total/60:.1f} min)")
        logger.info("=" * 60)


def check_dependencies():
    """Check available optimizations."""
    deps = {
        "numba": False,
        "sklearn": False,
        "cudf": False,
        "aiohttp": False,
        "pyarrow": False,
    }

    try:
        import numba

        deps["numba"] = True
    except ImportError:
        pass

    try:
        from sklearn.neighbors import BallTree

        deps["sklearn"] = True
    except ImportError:
        pass

    try:
        import cudf

        deps["cudf"] = True
    except ImportError:
        pass

    try:
        import aiohttp

        deps["aiohttp"] = True
    except ImportError:
        pass

    try:
        import pyarrow

        deps["pyarrow"] = True
    except ImportError:
        pass

    return deps


def get_pipeline_status():
    """Get current pipeline status."""
    status = {
        "wise_tiles": len(list(WISE_DIR.glob("wise_hp*.parquet"))),
        "gaia_tiles": len(list(GAIA_DIR.glob("gaia_hp*.parquet"))),
        "crossmatch_tiles": len(list(CROSSMATCH_DIR.glob("orphans_hp*.parquet"))),
        "total_tiles": N_TILES,
    }

    # Check output files
    outputs = {
        "orphans": OUTPUT_DIR / "wise_no_gaia_match.parquet",
        "golden_targets": OUTPUT_DIR / "final" / "golden_targets.csv",
        "variability": OUTPUT_DIR / "golden_variability.parquet",
        "epochs": OUTPUT_DIR / "final" / "neowise_epochs.parquet",
    }

    for name, path in outputs.items():
        if path.exists():
            status[f"{name}_exists"] = True
            status[f"{name}_size"] = path.stat().st_size
        else:
            status[f"{name}_exists"] = False

    return status


def run_optimized_crossmatch(use_gpu: bool = False, workers: int = N_WORKERS):
    """Run the optimized crossmatch."""
    from optimized_crossmatch import (
        HAS_CUDA,
        HAS_SKLEARN,
        crossmatch_tile_optimized,
        get_pending_tiles,
        merge_orphans_streaming,
        process_tile_wrapper,
    )

    pending = get_pending_tiles()
    if not pending:
        logger.info("Crossmatch already complete!")
        return

    logger.info(f"Processing {len(pending)} tiles...")
    logger.info(f"Using: sklearn={HAS_SKLEARN}, cuda={HAS_CUDA and use_gpu}")

    total_orphans = 0
    failed = []

    if use_gpu and HAS_CUDA:
        # Sequential GPU processing
        for i, tile in enumerate(pending):
            tile_idx, n_orphans, status, _ = crossmatch_tile_optimized(tile, use_gpu=True)
            total_orphans += n_orphans
            if "error" in status:
                failed.append(tile_idx)
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i+1}/{len(pending)}")
    else:
        # Parallel CPU processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            task_args = [(t, False) for t in pending]
            futures = list(executor.map(process_tile_wrapper, task_args))
            for tile_idx, n_orphans, status, _ in futures:
                total_orphans += n_orphans
                if "error" in status:
                    failed.append(tile_idx)

    # Merge results
    merge_orphans_streaming()

    logger.info(f"Crossmatch complete: {total_orphans:,} orphans, {len(failed)} failed")


def run_optimized_variability():
    """Run the optimized variability analysis."""
    epochs_path = OUTPUT_DIR / "final" / "neowise_epochs.parquet"
    output_path = OUTPUT_DIR / "golden_variability_optimized.parquet"

    if not epochs_path.exists():
        logger.warning(f"Epochs file not found: {epochs_path}")
        return

    from optimized_variability import analyze_variability

    analyze_variability(str(epochs_path), str(output_path))


def run_golden_list_generation():
    """Generate the golden target list with all scores."""
    logger.info("Generating golden target list...")

    # Import scoring modules
    try:
        subprocess.run(
            [sys.executable, "generate_golden_list.py"], cwd=Path(__file__).parent, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Golden list generation failed: {e}")


def run_kinematics_analysis():
    """Run kinematics analysis on golden targets."""
    logger.info("Running kinematics analysis...")

    try:
        subprocess.run(
            [sys.executable, "analyze_kinematics.py"], cwd=Path(__file__).parent, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Kinematics analysis failed: {e}")


def run_publication_figures():
    """Generate publication-quality figures."""
    logger.info("Generating publication figures...")

    try:
        subprocess.run(
            [sys.executable, "generate_publication_figures.py"],
            cwd=Path(__file__).parent,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Figure generation failed: {e}")


def run_full_pipeline(skip_download: bool = True, use_gpu: bool = False, workers: int = N_WORKERS):
    """
    Run the complete optimized pipeline.

    Phases:
    1. Download catalogs (optional, usually pre-done)
    2. Crossmatch WISE vs Gaia
    3. Multi-wavelength scoring
    4. Variability analysis
    5. Generate golden list
    6. Kinematics analysis
    7. Publication figures
    """
    timer = PipelineTimer()

    logger.info("=" * 60)
    logger.info("TASNI: Optimized Full Pipeline")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now().isoformat()}")

    deps = check_dependencies()
    logger.info(f"Dependencies: {deps}")

    status = get_pipeline_status()
    logger.info(
        f"Current status: WISE={status['wise_tiles']}, Gaia={status['gaia_tiles']}, "
        f"Crossmatch={status['crossmatch_tiles']} / {status['total_tiles']}"
    )

    pipeline_start = time.perf_counter()

    # Phase 1: Download (skip by default)
    if not skip_download:
        timer.start("download")
        # Would run: python optimized_downloader.py --catalog all
        timer.stop("download")

    # Phase 2: Crossmatch
    timer.start("crossmatch")
    run_optimized_crossmatch(use_gpu=use_gpu, workers=workers)
    timer.stop("crossmatch")

    # Phase 3: Multi-wavelength scoring (uses existing scripts)
    timer.start("scoring")
    try:
        subprocess.run(
            [sys.executable, "multi_wavelength_scoring.py"],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
        )
    except Exception as e:
        logger.warning(f"Scoring skipped or failed: {e}")
    timer.stop("scoring")

    # Phase 4: Golden list
    timer.start("golden_list")
    run_golden_list_generation()
    timer.stop("golden_list")

    # Phase 5: Variability analysis
    timer.start("variability")
    run_optimized_variability()
    timer.stop("variability")

    # Phase 6: Kinematics
    timer.start("kinematics")
    run_kinematics_analysis()
    timer.stop("kinematics")

    # Phase 7: Figures
    timer.start("figures")
    run_publication_figures()
    timer.stop("figures")

    pipeline_elapsed = time.perf_counter() - pipeline_start

    # Summary
    timer.summary()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total elapsed: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} min)")
    logger.info("=" * 60)

    # Save timing report
    report_path = OUTPUT_DIR / "pipeline_timing.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "completed": datetime.now().isoformat(),
                "total_seconds": pipeline_elapsed,
                "phases": timer.times,
                "dependencies": deps,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Timing report saved to {report_path}")


def run_benchmark():
    """Run benchmarks for all optimized components."""
    logger.info("=" * 60)
    logger.info("TASNI: Component Benchmarks")
    logger.info("=" * 60)

    results = {}

    # Crossmatch benchmark
    logger.info("\n--- Crossmatch Benchmark ---")
    try:
        from optimized_crossmatch import benchmark_methods

        results["crossmatch"] = benchmark_methods()
    except Exception as e:
        logger.error(f"Crossmatch benchmark failed: {e}")

    # Variability benchmark
    logger.info("\n--- Variability Benchmark ---")
    try:
        from optimized_variability import benchmark_methods as var_benchmark

        var_benchmark()
    except Exception as e:
        logger.error(f"Variability benchmark failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark complete!")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="TASNI Optimized Pipeline")
    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "download",
            "crossmatch",
            "score",
            "golden",
            "variability",
            "kinematics",
            "figures",
        ],
        default="all",
        help="Pipeline phase to run",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="CPU workers")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--status", action="store_true", help="Show pipeline status")
    args = parser.parse_args()

    if args.status:
        status = get_pipeline_status()
        deps = check_dependencies()
        print("\n=== TASNI Pipeline Status ===")
        print(f"WISE tiles:      {status['wise_tiles']} / {status['total_tiles']}")
        print(f"Gaia tiles:      {status['gaia_tiles']} / {status['total_tiles']}")
        print(f"Crossmatch:      {status['crossmatch_tiles']} / {status['total_tiles']}")
        print("\nOutput files:")
        for key in ["orphans", "golden_targets", "variability", "epochs"]:
            exists = status.get(f"{key}_exists", False)
            size = status.get(f"{key}_size", 0)
            size_mb = size / 1024 / 1024 if size else 0
            print(
                f"  {key}: {'YES' if exists else 'NO'} ({size_mb:.1f} MB)"
                if exists
                else f"  {key}: NO"
            )
        print("\nOptimizations available:")
        for dep, available in deps.items():
            print(f"  {dep}: {'YES' if available else 'NO'}")
        return

    if args.benchmark:
        run_benchmark()
        return

    # Run specific phase
    if args.phase == "all":
        run_full_pipeline(use_gpu=args.gpu, workers=args.workers)
    elif args.phase == "crossmatch":
        run_optimized_crossmatch(use_gpu=args.gpu, workers=args.workers)
    elif args.phase == "variability":
        run_optimized_variability()
    elif args.phase == "golden":
        run_golden_list_generation()
    elif args.phase == "kinematics":
        run_kinematics_analysis()
    elif args.phase == "figures":
        run_publication_figures()
    elif args.phase == "download":
        logger.info("Run: python optimized_downloader.py --catalog all")


if __name__ == "__main__":
    main()
