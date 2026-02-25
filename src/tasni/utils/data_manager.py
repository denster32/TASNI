"""
Data lifecycle management for TASNI

Handles automatic cleanup, archival, and retention of intermediate data files.

Usage:
    python src/tasni/utils/data_manager.py                    # Run cleanup
    python src/tasni/utils/data_manager.py --manifest           # Generate manifest
    python src/tasni/utils/data_manager.py --dry-run           # Preview cleanup
"""

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataManager:
    """Manage TASNI data lifecycle"""

    def __init__(self, tasni_root: Path = None):
        if tasni_root is None:
            # Default to current directory
            tasni_root = Path.cwd()
            # Navigate to tasni root
            while tasni_root.name != "tasni" and tasni_root.parent != tasni_root:
                tasni_root = tasni_root.parent

        self.tasni_root = Path(tasni_root)
        self.archive_dir = self.tasni_root / "archive"
        self.log_dir = self.tasni_root / "logs"
        self.checkpoint_dir = self.tasni_root / "checkpoints"
        self.output_dir = self.tasni_root / "output"

        # Configuration
        self.log_retention_days = 30
        self.intermediate_retention_days = 90
        self.max_archive_size_gb = 100

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def _git_metadata(self) -> dict[str, Any]:
        def _run_git(args: list[str]) -> str:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=self.tasni_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return result.stdout.strip()
            except Exception:
                return ""

        dirty = bool(_run_git(["status", "--porcelain"]))
        return {
            "commit": _run_git(["rev-parse", "HEAD"]) or None,
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or None,
            "dirty": dirty,
        }

    def cleanup_old_logs(self, dry_run: bool = False) -> dict[str, int]:
        """Compress or remove old log files"""
        results = {"compressed": 0, "removed": 0, "freed_bytes": 0}

        cutoff_date = datetime.now() - timedelta(days=self.log_retention_days)

        for log_file in self.log_dir.glob("*.log"):
            if not log_file.is_file():
                continue

            # Check file age
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_mtime < cutoff_date:
                # Compress if not already gzipped
                gz_file = log_file.with_suffix(log_file.suffix + ".gz")
                if not gz_file.exists():
                    if dry_run:
                        logger.info(f"Would compress: {log_file}")
                        results["compressed"] += 1
                    else:
                        try:
                            with open(log_file, "rb") as f_in:
                                with gzip.open(gz_file, "wb") as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            size_diff = log_file.stat().st_size - gz_file.stat().st_size
                            log_file.unlink()
                            results["freed_bytes"] += size_diff
                            results["compressed"] += 1
                            logger.info(f"Compressed: {log_file} -> {gz_file}")
                        except Exception as e:
                            logger.error(f"Error compressing {log_file}: {e}")

        return results

    def archive_intermediate_files(self, dry_run: bool = False) -> dict[str, int]:
        """Move intermediate files to archive"""
        results = {"archived": 0, "skipped": 0, "freed_bytes": 0}

        # Define intermediate file patterns in output directory
        intermediate_patterns = [
            "tier2_*.parquet",
            "tier3_*.parquet",
            "anomalies_*.parquet",
            "orphan_*.parquet",
            "*_progress_batch*.parquet",
            "*_temp.parquet",
            "*_draft.parquet",
        ]

        for pattern in intermediate_patterns:
            for file in self.output_dir.glob(pattern):
                if not file.is_file():
                    continue

                # Skip if already in archive
                if file.parent == self.archive_dir:
                    continue

                # Skip if in final/ directory (golden targets)
                if "final" in file.parts:
                    results["skipped"] += 1
                    continue

                archive_path = self.archive_dir / file.name
                if dry_run:
                    logger.info(f"Would archive: {file} -> {archive_path}")
                    results["archived"] += 1
                else:
                    try:
                        file_size = file.stat().st_size
                        shutil.move(str(file), str(archive_path))
                        results["freed_bytes"] += file_size
                        results["archived"] += 1
                        logger.info(f"Archived: {file} -> {archive_path}")
                    except Exception as e:
                        logger.error(f"Error archiving {file}: {e}")

        return results

    def cleanup_archive(self, dry_run: bool = False) -> dict[str, int]:
        """Remove old/duplicate files from archive"""
        results = {"removed": 0, "freed_bytes": 0}

        cutoff_date = datetime.now() - timedelta(days=self.intermediate_retention_days)

        for file in self.archive_dir.glob("*.parquet"):
            if not file.is_file():
                continue

            remove = False

            # Check for old files
            file_mtime = datetime.fromtimestamp(file.stat().st_mtime)
            if file_mtime < cutoff_date:
                # Check for duplicate or temporary files
                if any(x in file.name for x in ["_old", "progress", "_temp", "_draft", "backup"]):
                    remove = True

            # Remove duplicates
            if "_old" in file.name.lower():
                # Check if newer version exists
                base_name = file.name.replace("_old", "").replace("_OLD", "")
                if (self.archive_dir / base_name).exists():
                    remove = True

            if remove:
                if dry_run:
                    logger.info(f"Would remove: {file}")
                    results["removed"] += 1
                else:
                    try:
                        size = file.stat().st_size
                        file.unlink()
                        results["freed_bytes"] += size
                        results["removed"] += 1
                        logger.info(f"Removed: {file}")
                    except Exception as e:
                        logger.error(f"Error removing {file}: {e}")

        return results

    def check_archive_size(self) -> dict[str, float]:
        """Check current archive size"""
        total_size = sum(f.stat().st_size for f in self.archive_dir.rglob("*") if f.is_file())

        size_gb = total_size / (1024**3)

        return {
            "total_bytes": total_size,
            "total_gb": size_gb,
            "limit_gb": self.max_archive_size_gb,
            "usage_percent": (size_gb / self.max_archive_size_gb) * 100,
            "under_limit": size_gb < self.max_archive_size_gb,
        }

    def generate_manifest(self) -> str:
        """Generate release-grade manifest with SHA-256 and provenance metadata."""
        logger.info("Generating release manifest...")

        manifest: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "tasni_root": str(self.tasni_root),
            "provenance": {
                "generator": "src/tasni/utils/data_manager.py::DataManager.generate_manifest",
                "git": self._git_metadata(),
                "python_version": sys.version,
            },
            "files": [],
        }

        include_roots = [
            self.tasni_root / "data" / "processed" / "final",
            self.tasni_root / "docs",
            self.tasni_root / "reports" / "figures",
            self.tasni_root / "tasni_paper_final" / "figures",
            self.tasni_root / "src" / "tasni",
            self.tasni_root / ".github" / "workflows",
        ]
        include_ext = {
            ".csv",
            ".parquet",
            ".json",
            ".txt",
            ".md",
            ".yml",
            ".yaml",
            ".py",
            ".pdf",
            ".png",
        }

        exclude_substrings = {
            "/archive/",
            "/logs/",
            "/checkpoints/",
            "/data/interim/",
            "/__pycache__/",
        }

        for root in include_roots:
            if not root.exists():
                continue
            for file in root.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in include_ext:
                    continue
                rel_path = file.relative_to(self.tasni_root).as_posix()
                if any(token in f"/{rel_path}" for token in exclude_substrings):
                    continue

                try:
                    stat = file.stat()
                    manifest["files"].append(
                        {
                            "path": rel_path,
                            "size_bytes": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "sha256": self._sha256_file(file),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing {file}: {e}")

        manifest["files"] = sorted(manifest["files"], key=lambda x: x["path"])
        total_size = sum(item["size_bytes"] for item in manifest["files"])
        manifest["summary"] = {
            "total_files": len(manifest["files"]),
            "total_bytes": total_size,
            "total_gb": total_size / (1024**3),
        }

        release_manifest_dir = self.tasni_root / "output" / "release"
        release_manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = release_manifest_dir / "RELEASE_MANIFEST.json"

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Keep backward-compatible pointer in archive.
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        archive_manifest_path = self.archive_dir / "manifest.json"
        with archive_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Release manifest generated: {manifest_path}")
        logger.info(f"  Total files: {manifest['summary']['total_files']}")
        logger.info(f"  Total size: {manifest['summary']['total_gb']:.2f} GB")

        return str(manifest_path)

    def run_cleanup_cycle(self, dry_run: bool = False) -> dict:
        """Run complete cleanup cycle"""
        logger.info("=" * 70)
        logger.info("Starting TASNI data cleanup cycle...")
        logger.info("=" * 70)

        if dry_run:
            logger.info("DRY RUN MODE - No changes will be made")

        results = {
            "log_cleanup": self.cleanup_old_logs(dry_run),
            "archive_files": self.archive_intermediate_files(dry_run),
            "archive_cleanup": self.cleanup_archive(dry_run),
            "archive_status": self.check_archive_size(),
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
        }

        # Summary
        logger.info("=" * 70)
        logger.info("Cleanup Summary")
        logger.info("=" * 70)

        # Logs
        log_results = results["log_cleanup"]
        logger.info("Logs:")
        logger.info(f"  Compressed: {log_results['compressed']}")
        logger.info(f"  Freed: {log_results['freed_bytes'] / (1024**2):.2f} MB")

        # Archive files
        archive_results = results["archive_files"]
        logger.info("Archive (move from output):")
        logger.info(f"  Archived: {archive_results['archived']}")
        logger.info(f"  Skipped: {archive_results['skipped']}")
        logger.info(f"  Freed: {archive_results['freed_bytes'] / (1024**3):.2f} GB")

        # Archive cleanup
        cleanup_results = results["archive_cleanup"]
        logger.info("Archive cleanup:")
        logger.info(f"  Removed: {cleanup_results['removed']}")
        logger.info(f"  Freed: {cleanup_results['freed_bytes'] / (1024**3):.2f} GB")

        # Archive status
        status = results["archive_status"]
        logger.info("Archive status:")
        logger.info(f"  Total size: {status['total_gb']:.2f} GB")
        logger.info(f"  Limit: {status['limit_gb']:.2f} GB")
        logger.info(f"  Usage: {status['usage_percent']:.1f}%")
        logger.info(f"  Under limit: {status['under_limit']}")

        # Total freed
        total_freed = (
            log_results["freed_bytes"]
            + archive_results["freed_bytes"]
            + cleanup_results["freed_bytes"]
        )
        logger.info(f"Total space freed: {total_freed / (1024**3):.2f} GB")

        logger.info("=" * 70)

        return results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TASNI Data Lifecycle Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/tasni/utils/data_manager.py                    # Run cleanup
  python src/tasni/utils/data_manager.py --manifest           # Generate manifest
  python src/tasni/utils/data_manager.py --dry-run           # Preview cleanup
  python src/tasni/utils/data_manager.py --status             # Check status
        """,
    )

    parser.add_argument("--manifest", action="store_true", help="Generate data manifest")

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview cleanup without making changes"
    )

    parser.add_argument("--status", action="store_true", help="Show data status only")

    parser.add_argument(
        "--root", type=str, default=None, help="TASNI root directory (default: auto-detect)"
    )

    args = parser.parse_args()

    # Initialize data manager
    tasni_root = Path(args.root) if args.root else None
    manager = DataManager(tasni_root)

    try:
        if args.status:
            # Just show status
            status = manager.check_archive_size()
            print(
                f"Archive size: {status['total_gb']:.2f} GB / {status['limit_gb']:.2f} GB ({status['usage_percent']:.1f}%)"
            )
            print(f"Under limit: {status['under_limit']}")

        elif args.manifest:
            # Generate manifest
            manifest_path = manager.generate_manifest()
            print(f"✓ Manifest generated: {manifest_path}")

        else:
            # Run cleanup cycle
            results = manager.run_cleanup_cycle(dry_run=args.dry_run)

            # Save results
            results_dir = manager.tasni_root / "output"
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / "cleanup_results.json"

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"✓ Cleanup results saved: {results_path}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
