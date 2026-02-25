#!/usr/bin/env python3
"""Verify release checksums from checksums.txt and/or release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_checksums_file(
    checksums_path: Path, base_dir: Path, project_root: Path
) -> tuple[int, int]:
    checked = 0
    failed = 0

    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        expected, rel = parts[0], parts[-1]
        file_path = (base_dir / rel).resolve()
        if not file_path.exists():
            file_path = (project_root / rel).resolve()
        checked += 1

        if not file_path.exists():
            print(f"[FAIL] Missing file: {file_path}")
            failed += 1
            continue

        actual = sha256_file(file_path)
        if actual != expected:
            print(f"[FAIL] Hash mismatch: {file_path}")
            print(f"       expected={expected}")
            print(f"       actual  ={actual}")
            failed += 1
        else:
            print(f"[ OK ] {file_path}")

    return checked, failed


def verify_manifest(manifest_path: Path, project_root: Path) -> tuple[int, int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    failed = 0
    for item in payload.get("files", []):
        rel = item.get("path")
        expected = item.get("sha256")
        if not rel or not expected:
            continue

        checked += 1
        file_path = (project_root / rel).resolve()
        if not file_path.exists():
            print(f"[FAIL] Missing manifest file: {file_path}")
            failed += 1
            continue

        actual = sha256_file(file_path)
        if actual != expected:
            print(f"[FAIL] Manifest hash mismatch: {file_path}")
            failed += 1

    return checked, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify TASNI release checksums")
    parser.add_argument(
        "--checksums",
        default="data/processed/final/checksums.txt",
        help="Checksums file relative to project root",
    )
    parser.add_argument(
        "--base-dir",
        default="data/processed/final",
        help="Base directory for checksum entries (relative to project root)",
    )
    parser.add_argument(
        "--manifest",
        default="output/release/RELEASE_MANIFEST.json",
        help="Optional release manifest path (relative to project root)",
    )
    args = parser.parse_args()

    checksums_path = (PROJECT_ROOT / args.checksums).resolve()
    base_dir = (PROJECT_ROOT / args.base_dir).resolve()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()

    total_checked = 0
    total_failed = 0

    if checksums_path.exists():
        checked, failed = verify_checksums_file(checksums_path, base_dir, PROJECT_ROOT)
        total_checked += checked
        total_failed += failed
    else:
        print(f"[WARN] checksums file not found: {checksums_path}")

    if manifest_path.exists():
        checked, failed = verify_manifest(manifest_path, PROJECT_ROOT)
        print(f"[INFO] manifest entries checked: {checked}")
        total_checked += checked
        total_failed += failed
    else:
        print(f"[WARN] manifest file not found: {manifest_path}")

    print(f"[SUMMARY] checked={total_checked} failed={total_failed}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
