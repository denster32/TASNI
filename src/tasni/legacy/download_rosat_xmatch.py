#!/usr/bin/env python3
"""
Download pre-computed AllWISE x ROSAT (2RXS) cross-match from CDS.

This is the NWAY cross-match: Salvato et al. 2018, MNRAS 473, 4937
~135,000 ROSAT sources matched to AllWISE counterparts.

File: 2rxswg.dat.gz (~30-50 MB)
"""

import gzip
from pathlib import Path

import requests

# Configuration
CDS_URL = "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/473/4937/2rxswg.dat.gz"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = _PROJECT_ROOT / "data" / "rosat"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "rosat" / "processed"


def download_file():
    """Download the AllWISE x ROSAT cross-match file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "2rxswg.dat.gz"

    if output_path.exists():
        print(f"File already exists: {output_path}")
        return output_path

    print(f"Downloading from {CDS_URL}")
    print(f"Saving to {output_path}")

    response = requests.get(CDS_URL, stream=True, timeout=600)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192 * 8):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\rProgress: {pct:.1f}% ({downloaded/1024/1024:.1f} MB)", end="")

    print(f"\nDownload complete: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def extract_wise_designations(gz_path: Path) -> set[str]:
    """
    Extract AllWISE designations from the cross-match file.

    The 2rxswg.dat format is described in the ReadMe file.
    We need the AllWISE designation column to match against our orphans.

    Returns a set of AllWISE designations that have ROSAT counterparts.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "rosat_wise_ids.txt"

    if output_path.exists():
        print(f"Loading existing designations from {output_path}")
        with open(output_path) as f:
            return set(line.strip() for line in f if line.strip())

    wise_designations = set()

    print(f"Extracting AllWISE designations from {gz_path}")

    with gzip.open(gz_path, "rt") as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith("#"):
                continue

            # Parse the pipe-delimited format
            # Column 2 (index 1) is the AllWISE designation
            parts = line.strip().split("|")
            if len(parts) > 1:
                wise_design = parts[1].strip()
                if wise_design and wise_design != "\\N":
                    wise_designations.add(wise_design)

            if line_num % 10000 == 0:
                print(
                    f"\rProcessed {line_num:,} lines, found {len(wise_designations):,} matches",
                    end="",
                )

    print(f"\nExtracted {len(wise_designations):,} unique AllWISE designations")

    # Save to file for fast reloading
    with open(output_path, "w") as f:
        for design in sorted(wise_designations):
            f.write(f"{design}\n")

    print(f"Saved to {output_path}")
    return wise_designations


def main():
    print("=" * 60)
    print("AllWISE x ROSAT Cross-Match Download")
    print("=" * 60)

    # Download
    gz_path = download_file()

    # Extract WISE designations
    wise_designations = extract_wise_designations(gz_path)

    print("\n" + "=" * 60)
    print(f"Summary: {len(wise_designations):,} AllWISE sources have ROSAT counterparts")
    print("=" * 60)


if __name__ == "__main__":
    main()
