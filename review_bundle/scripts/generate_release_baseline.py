#!/usr/bin/env python3
"""
Generate a reproducibility baseline snapshot for TASNI release hardening.

The snapshot captures:
- environment metadata
- git metadata (if available)
- existence + SHA-256 digests for key release artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Curated release-critical targets. Globs are expanded relative to project root.
RELEASE_TARGETS = [
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "pyproject.toml",
    "requirements.txt",
    "docs/DATA_AVAILABILITY.md",
    "docs/reproducibility.md",
    "docs/REPRODUCIBILITY_QUICKSTART.md",
    "docs/publication_checklist.md",
    "docs/PUBLICATION_STATUS.md",
    "data/processed/final/*.csv",
    "data/processed/final/*.parquet",
    "data/processed/final/checksums.txt",
    "reports/figures/*",
    "tasni_paper_final/figures/*",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def collect_files() -> dict[str, Any]:
    expanded: list[Path] = []
    for pattern in RELEASE_TARGETS:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        if matches:
            expanded.extend(m for m in matches if m.is_file())
            continue
        candidate = PROJECT_ROOT / pattern
        if candidate.is_file():
            expanded.append(candidate)

    # Stable ordering + dedup
    unique_files = sorted(set(expanded))

    artifacts: list[dict[str, Any]] = []
    for path in unique_files:
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        stat = path.stat()
        artifacts.append(
            {
                "path": rel_path,
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                "sha256": _sha256(path),
            }
        )

    missing_patterns = []
    for pattern in RELEASE_TARGETS:
        if not list(PROJECT_ROOT.glob(pattern)) and not (PROJECT_ROOT / pattern).is_file():
            missing_patterns.append(pattern)

    return {"artifacts": artifacts, "missing_patterns": missing_patterns}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TASNI release baseline snapshot")
    parser.add_argument(
        "--output",
        default="output/release/baseline_snapshot.json",
        help="Output JSON path relative to project root",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any target pattern is missing",
    )
    args = parser.parse_args()

    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_payload = collect_files()
    snapshot: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "git": {
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _run_git(["rev-parse", "HEAD"]),
            "dirty": (_run_git(["status", "--porcelain"]) or "") != "",
        },
        "artifacts": file_payload["artifacts"],
        "missing_patterns": file_payload["missing_patterns"],
    }

    output_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote baseline snapshot: {output_path}")
    print(f"Artifacts captured: {len(snapshot['artifacts'])}")
    if snapshot["missing_patterns"]:
        print("Missing target patterns:")
        for pattern in snapshot["missing_patterns"]:
            print(f"  - {pattern}")
        if args.strict:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
