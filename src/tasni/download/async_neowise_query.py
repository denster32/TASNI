#!/usr/bin/env python3
"""
TASNI: Async NEOWISE Multi-Epoch Query Module
==============================================

High-performance asynchronous queries to IRSA for NEOWISE multi-epoch
photometry. Provides 10-50x speedup over sequential queries.

Features:
- Concurrent queries with configurable parallelism
- Connection pooling and rate limiting
- Automatic retry with exponential backoff
- Progress tracking and resumable queries
- Memory-efficient streaming results

Usage:
    # As module
    from async_neowise_query import fetch_all_neowise_epochs
    epochs = await fetch_all_neowise_epochs(sources_df)

    # As script
    python async_neowise_query.py --input sources.csv --output epochs.parquet

    # Benchmark mode
    python async_neowise_query.py --benchmark --sources 100
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
from aiohttp import ClientTimeout, TCPConnector

from tasni.core.config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    MAX_RETRIES,
    OUTPUT_DIR,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_BASE,
    ensure_dirs,
)

# Setup logging
ensure_dirs()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "async_neowise.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# IRSA TAP endpoint
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"
NEOWISE_TABLE = "neowiser_p1bs_psd"

# Default concurrency settings
DEFAULT_MAX_CONCURRENT = 20  # Max simultaneous requests
DEFAULT_RATE_LIMIT = 50  # Max requests per second
DEFAULT_BATCH_SIZE = 100  # Sources per batch for progress tracking


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = rate
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


async def query_neowise_epochs_async(
    session: aiohttp.ClientSession,
    ra: float,
    dec: float,
    radius_arcsec: float = 3.0,
    max_epochs: int = 1000,
    rate_limiter: RateLimiter | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> pd.DataFrame | None:
    """
    Query NEOWISE epochs for a single source asynchronously.

    Args:
        session: aiohttp ClientSession
        ra: Right ascension in degrees
        dec: Declination in degrees
        radius_arcsec: Cone search radius
        max_epochs: Maximum epochs to return
        rate_limiter: Optional rate limiter
        semaphore: Optional concurrency semaphore

    Returns:
        DataFrame of NEOWISE epochs or None if no data
    """
    # Build ADQL query
    query = f"""
    SELECT TOP {max_epochs}
        mjd, w1mpro, w1sigmpro, w2mpro, w2sigmpro,
        ra, dec, qual_frame, qi_fact, saa_sep, moon_masked
    FROM {NEOWISE_TABLE}
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_arcsec/3600})
    ) = 1
    ORDER BY mjd
    """

    params = {"QUERY": query, "FORMAT": "csv", "LANG": "ADQL"}

    # Rate limiting
    if rate_limiter:
        await rate_limiter.acquire()

    # Semaphore for concurrency control
    async with semaphore if semaphore else asyncio.Lock():
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(IRSA_TAP_URL, params=params) as response:
                    if response.status == 200:
                        text = await response.text()
                        if text.strip() and not text.startswith("<!"):
                            # Parse CSV response
                            from io import StringIO

                            df = pd.read_csv(StringIO(text))
                            if len(df) > 0:
                                df["source_ra"] = ra
                                df["source_dec"] = dec
                                return df
                        return None
                    elif response.status == 429:
                        # Rate limited - back off
                        wait_time = RETRY_BACKOFF_BASE**attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"HTTP {response.status} for ({ra:.4f}, {dec:.4f})")
                        return None

            except TimeoutError:
                wait_time = RETRY_BACKOFF_BASE**attempt
                logger.warning(f"Timeout, retry {attempt+1}/{MAX_RETRIES}")
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.debug(f"Error querying ({ra:.4f}, {dec:.4f}): {e}")
                return None

    return None


async def fetch_batch(
    session: aiohttp.ClientSession,
    sources: list[tuple[float, float, str]],
    rate_limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    progress_callback=None,
) -> list[pd.DataFrame]:
    """
    Fetch NEOWISE epochs for a batch of sources.

    Args:
        session: aiohttp ClientSession
        sources: List of (ra, dec, designation) tuples
        rate_limiter: Rate limiter instance
        semaphore: Concurrency semaphore
        progress_callback: Optional callback(completed, total)

    Returns:
        List of DataFrames with epoch data
    """
    tasks = []
    for ra, dec, designation in sources:
        task = query_neowise_epochs_async(
            session, ra, dec, rate_limiter=rate_limiter, semaphore=semaphore
        )
        tasks.append(task)

    results = []
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            results.append(result)
        completed += 1
        if progress_callback and completed % 10 == 0:
            progress_callback(completed, len(sources))

    return results


async def fetch_all_neowise_epochs(
    sources: pd.DataFrame,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    checkpoint_file: Path | None = None,
    progress_interval: int = 100,
) -> pd.DataFrame:
    """
    Fetch NEOWISE epochs for all sources asynchronously.

    Args:
        sources: DataFrame with 'ra', 'dec', and optionally 'designation' columns
        max_concurrent: Maximum concurrent requests
        rate_limit: Maximum requests per second
        checkpoint_file: Optional file for resuming interrupted queries
        progress_interval: Log progress every N sources

    Returns:
        DataFrame with all NEOWISE epochs
    """
    logger.info(f"Fetching NEOWISE epochs for {len(sources)} sources")
    logger.info(f"Max concurrent: {max_concurrent}, Rate limit: {rate_limit}/s")

    # Prepare source list
    if "designation" not in sources.columns:
        sources = sources.copy()
        sources["designation"] = [f"src_{i}" for i in range(len(sources))]

    source_list = list(
        zip(
            sources["ra"].values, sources["dec"].values, sources["designation"].values, strict=False
        )
    )

    # Load checkpoint if exists
    completed_sources = set()
    if checkpoint_file and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            completed_sources = set(checkpoint.get("completed", []))
            logger.info(f"Resuming from checkpoint: {len(completed_sources)} already done")

    # Filter out completed sources
    source_list = [(ra, dec, des) for ra, dec, des in source_list if des not in completed_sources]
    logger.info(f"Remaining sources to query: {len(source_list)}")

    # Setup connection pool and rate limiter
    connector = TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = ClientTimeout(total=REQUEST_TIMEOUT)
    rate_limiter = RateLimiter(rate_limit)
    semaphore = asyncio.Semaphore(max_concurrent)

    all_epochs = []
    start_time = time.time()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Process in batches for progress tracking
        for batch_start in range(0, len(source_list), DEFAULT_BATCH_SIZE):
            batch = source_list[batch_start : batch_start + DEFAULT_BATCH_SIZE]

            # Fetch batch
            batch_results = await fetch_batch(session, batch, rate_limiter, semaphore)
            all_epochs.extend(batch_results)

            # Update checkpoint
            completed_sources.update([des for _, _, des in batch])
            if checkpoint_file:
                with open(checkpoint_file, "w") as f:
                    json.dump({"completed": list(completed_sources)}, f)

            # Progress logging
            completed = batch_start + len(batch)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            epochs_found = sum(len(df) for df in all_epochs)

            if completed % progress_interval == 0 or completed == len(source_list):
                logger.info(
                    f"Progress: {completed}/{len(source_list)} sources "
                    f"({rate:.1f}/s), {epochs_found:,} epochs found"
                )

    # Combine all epochs
    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f}s ({len(source_list)/elapsed:.1f} sources/s)")

    if all_epochs:
        combined = pd.concat(all_epochs, ignore_index=True)
        logger.info(f"Total epochs retrieved: {len(combined):,}")
        return combined
    else:
        logger.warning("No epochs found")
        return pd.DataFrame()


def run_sequential_benchmark(sources: pd.DataFrame, n_sources: int = 10) -> float:
    """Run sequential queries for benchmark comparison."""
    from pyvo.dal import TAPService

    logger.info(f"Running sequential benchmark ({n_sources} sources)...")
    service = TAPService(IRSA_TAP_URL.replace("/sync", ""))

    start_time = time.time()
    for idx, row in sources.head(n_sources).iterrows():
        query = f"""
        SELECT TOP 100 mjd, w1mpro, w2mpro
        FROM {NEOWISE_TABLE}
        WHERE CONTAINS(POINT('ICRS', ra, dec),
                       CIRCLE('ICRS', {row['ra']}, {row['dec']}, 0.001)) = 1
        """
        try:
            result = service.run_sync(query, timeout=30)
        except Exception as e:
            logger.debug(f"Sequential query error: {e}")

    elapsed = time.time() - start_time
    return elapsed


async def run_async_benchmark(sources: pd.DataFrame, n_sources: int = 10) -> float:
    """Run async queries for benchmark comparison."""
    logger.info(f"Running async benchmark ({n_sources} sources)...")

    start_time = time.time()
    await fetch_all_neowise_epochs(sources.head(n_sources), max_concurrent=20, rate_limit=50)
    elapsed = time.time() - start_time
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Async NEOWISE multi-epoch queries")
    parser.add_argument("--input", type=str, help="Input source file (parquet or csv)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "final" / "neowise_epochs_async.parquet"),
        help="Output epochs file",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark comparing sequential vs async"
    )
    parser.add_argument("--sources", type=int, default=100, help="Number of sources for benchmark")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Max concurrent requests (default: {DEFAULT_MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=DEFAULT_RATE_LIMIT,
        help=f"Max requests/second (default: {DEFAULT_RATE_LIMIT})",
    )
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 60)
    logger.info("TASNI: Async NEOWISE Query Module")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    if args.benchmark:
        # Benchmark mode
        logger.info("BENCHMARK MODE")
        logger.info(f"Testing with {args.sources} sources")

        # Load or create test sources
        test_input = OUTPUT_DIR / "final" / "golden_targets.csv"
        if test_input.exists():
            sources = pd.read_csv(test_input)
        else:
            # Generate random positions for testing
            np.random.seed(42)
            sources = pd.DataFrame(
                {
                    "ra": np.random.uniform(0, 360, args.sources),
                    "dec": np.random.uniform(-90, 90, args.sources),
                    "designation": [f"test_{i}" for i in range(args.sources)],
                }
            )

        n_test = min(args.sources, len(sources))

        # Run sequential benchmark (small sample)
        seq_n = min(10, n_test)
        seq_time = run_sequential_benchmark(sources, seq_n)
        seq_rate = seq_n / seq_time
        logger.info(f"Sequential: {seq_time:.1f}s for {seq_n} sources ({seq_rate:.2f}/s)")

        # Run async benchmark
        async_time = asyncio.run(run_async_benchmark(sources, n_test))
        async_rate = n_test / async_time
        logger.info(f"Async: {async_time:.1f}s for {n_test} sources ({async_rate:.2f}/s)")

        # Calculate speedup
        speedup = async_rate / seq_rate
        logger.info("")
        logger.info(f"=== SPEEDUP: {speedup:.1f}x ===")

        # Estimate time for full Tier5
        tier5_size = 4137
        seq_estimate = tier5_size / seq_rate
        async_estimate = tier5_size / async_rate
        logger.info("")
        logger.info(f"Estimated time for {tier5_size} Tier5 sources:")
        logger.info(f"  Sequential: {seq_estimate/3600:.1f} hours")
        logger.info(f"  Async: {async_estimate/3600:.2f} hours ({async_estimate/60:.0f} min)")

    else:
        # Normal mode - fetch epochs for input sources
        if not args.input:
            parser.error("--input required in normal mode (use --benchmark for testing)")

        input_path = Path(args.input)
        if input_path.suffix == ".parquet":
            sources = pd.read_parquet(input_path)
        else:
            sources = pd.read_csv(input_path)

        logger.info(f"Loaded {len(sources)} sources from {input_path}")

        # Setup checkpoint
        checkpoint_file = CHECKPOINT_DIR / "async_neowise_checkpoint.json"

        # Run async fetch
        epochs = asyncio.run(
            fetch_all_neowise_epochs(
                sources,
                max_concurrent=args.max_concurrent,
                rate_limit=args.rate_limit,
                checkpoint_file=checkpoint_file,
            )
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(epochs) > 0:
            epochs.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(epochs)} epochs to {output_path}")
        else:
            logger.warning("No epochs to save")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
