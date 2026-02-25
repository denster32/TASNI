#!/usr/bin/env python3
"""
TASNI Pipeline: Tier Vetoes (UKIDSS/VHS/CatWISE) - Modularized & Typed
======================================================================
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from pydantic import BaseModel, Field, field_validator

# Logging (structlog integration pending)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/tier_vetoes.log"), logging.StreamHandler()],
)
logger: logging.Logger = logging.getLogger(__name__)


class VetoConfig(BaseModel):
    radius_arcsec: float = Field(3.0, ge=0.1, description="Match radius in arcsec")
    batch_size: int = Field(50, ge=1, description="Batch size for processing")
    max_workers: int = Field(8, ge=1, le=32, description="Parallel workers per batch")
    pause: float = Field(1.0, ge=0.1, description="Pause between batches (sec)")

    @field_validator("batch_size")
    @classmethod
    def batch_size_multiple_of_10(cls, v: int) -> int:
        if v % 10 != 0:
            raise ValueError("batch_size must be multiple of 10")
        return v


def query_catalog(
    coord: SkyCoord, radius: u.Quantity, catalog_list: list[str]
) -> pd.DataFrame | None:
    """Query Vizier catalogs for matches."""
    v: Vizier = Vizier(columns=["*"], row_limit=5)
    for cat in catalog_list:
        try:
            result = v.query_region(coord, radius=radius, catalog=cat)
            if result and len(result) > 0:
                return result[0].to_pandas()
        except Exception as e:
            logger.debug("Query %s failed: %s", cat, e)
    return None


def query_ukidss(coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame | None:
    """Query UKIDSS (DR10)."""
    catalogs: list[str] = [
        "II/319/las10",  # LAS
        "II/319/gps10",  # GPS
        "II/319/gcs10",  # GCS
        "II/319/dxs10",  # DXS
    ]
    return query_catalog(coord, radius, catalogs)


def query_vhs(coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame | None:
    """Query VHS (DR5)."""
    catalogs: list[str] = ["J/A+A/618/A92/vhsdr3s"]
    return query_catalog(coord, radius, catalogs)


def query_catwise(coord: SkyCoord, radius: u.Quantity) -> pd.DataFrame | None:
    """Query CatWISE2020."""
    catalogs: list[str] = ["II/365/catwise"]
    return query_catalog(coord, radius, catalogs)


def apply_veto(df: pd.DataFrame, config: VetoConfig) -> pd.DataFrame:
    """Apply all vetoes to DataFrame with RA/Dec columns."""
    missing = [col for col in ("ra", "dec") if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for Tier1 vetoes: {missing}")

    df = df.copy()
    df[["has_ukidss", "has_vhs", "has_catwise"]] = False
    df["veto_sep_arcsec"] = np.nan

    n_ukidss: int = 0
    n_vhs: int = 0
    n_catwise: int = 0
    start_time: float = time.time()

    radius: u.Quantity = config.radius_arcsec * u.arcsec

    def _process_source(idx: int, row: pd.Series) -> tuple[int, dict]:
        coord: SkyCoord = SkyCoord(ra=row["ra"] * u.deg, dec=row["dec"] * u.deg)
        out = {
            "has_ukidss": False,
            "has_vhs": False,
            "has_catwise": False,
            "veto_sep_arcsec": np.nan,
        }

        ukidss = query_ukidss(coord, radius)
        if ukidss is not None and len(ukidss) > 0:
            out["has_ukidss"] = True
            out["veto_sep_arcsec"] = ukidss.iloc[0].get("_r", np.nan)

        vhs = query_vhs(coord, radius)
        if vhs is not None and len(vhs) > 0:
            out["has_vhs"] = True

        catwise = query_catwise(coord, radius)
        if catwise is not None and len(catwise) > 0:
            out["has_catwise"] = True

        return idx, out

    for i in range(0, len(df), config.batch_size):
        batch: pd.DataFrame = df.iloc[i : i + config.batch_size]
        logger.info("Processing batch %d (%d sources)", i // config.batch_size + 1, len(batch))

        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(_process_source, idx, row): idx for idx, row in batch.iterrows()}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, out = future.result()
                    df.at[idx, "has_ukidss"] = out["has_ukidss"]
                    df.at[idx, "has_vhs"] = out["has_vhs"]
                    df.at[idx, "has_catwise"] = out["has_catwise"]
                    df.at[idx, "veto_sep_arcsec"] = out["veto_sep_arcsec"]

                    n_ukidss += int(out["has_ukidss"])
                    n_vhs += int(out["has_vhs"])
                    n_catwise += int(out["has_catwise"])
                except Exception:
                    logger.exception("Failed Tier1 veto query for row index %s", idx)

        time.sleep(config.pause)

        elapsed: float = time.time() - start_time
        logger.info(
            "Progress: %d/%d | UKIDSS:%d VHS:%d CatWISE:%d | %.1fs",
            i + len(batch),
            len(df),
            n_ukidss,
            n_vhs,
            n_catwise,
            elapsed,
        )

    df["tier1_improved"] = ~(df["has_ukidss"] | df["has_vhs"] | df["has_catwise"])
    return df


def run_tier_vetoes(
    input_path: Path, output_path: Path, config: VetoConfig, test_mode: bool = False
) -> None:
    """Main runner function (CLI entrypoint)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    df: pd.DataFrame = pd.read_parquet(input_path)
    if test_mode:
        df = df.head(100)
        logger.info("TEST MODE: Limited to 100 sources")
        output_path = output_path.with_suffix(".test.parquet")

    logger.info("Loaded %d Tier1 candidates", len(df))

    df = apply_veto(df, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved %d improved Tier1 candidates to %s",
        len(df[df["tier1_improved"]]),
        output_path,
    )

    # Summary
    logger.info("Veto Summary:")
    logger.info("  UKIDSS veto: %d", df["has_ukidss"].sum())
    logger.info("  VHS veto: %d", df["has_vhs"].sum())
    logger.info("  CatWISE veto: %d", df["has_catwise"].sum())
    logger.info(
        "  Survived all: %d (%.1f%%)",
        df["tier1_improved"].sum(),
        100 * df["tier1_improved"].mean(),
    )


if __name__ == "__main__":

    from typer import Argument, Option, Typer

    app = Typer()

    @app.command()
    def main(
        input: Path = Argument(
            "data/interim/checkpoints/tier1/orphans.parquet", help="Tier1 input"
        ),
        output: Path = Argument(
            "data/interim/checkpoints/tier1_improved/tier1_vetoes.parquet", help="Output"
        ),
        radius: float = Option(3.0, help="Match radius (arcsec)"),
        workers: int = Option(8, "--workers", "-w", help="Parallel query workers per batch"),
        test: bool = Option(False, "--test", help="Test on 100 sources"),
    ):
        config = VetoConfig(radius_arcsec=radius, max_workers=workers)
        run_tier_vetoes(input, output, config, test)

    app()
