"""
TASNI Health Check Script

Checks system health, data integrity, and pipeline status.

Usage:
    python src/tasni/utils/health_check.py
    python src/tasni/utils/health_check.py --check data
    python src/tasni/utils/health_check.py --check all
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class HealthChecker:
    """TASNI health checker"""

    def __init__(self, tasni_root: Path = None):
        if tasni_root is None:
            tasni_root = Path.cwd()
            while tasni_root.name != "tasni" and tasni_root.parent != tasni_root:
                tasni_root = tasni_root.parent

        self.tasni_root = Path(tasni_root)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tasni_root": str(self.tasni_root),
            "checks": {},
        }

    def check_disk_space(self) -> dict[str, Any]:
        """Check disk space"""
        check_result = {"name": "Disk Space", "status": "OK", "details": {}}

        try:
            stat = os.statvfs(self.tasni_root)
            total = stat.f_frsize * stat.f_blocks
            free = stat.f_frsize * stat.f_bavail
            used = total - free
            percent_used = (used / total) * 100

            check_result["details"] = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "percent_used": percent_used,
            }

            # Warning threshold: 80%
            if percent_used > 80:
                check_result["status"] = "WARNING"
                check_result["message"] = f"Disk usage at {percent_used:.1f}%"

            # Critical threshold: 90%
            if percent_used > 90:
                check_result["status"] = "CRITICAL"
                check_result["message"] = f"Disk usage at {percent_used:.1f}%"

        except Exception as e:
            check_result["status"] = "ERROR"
            check_result["message"] = str(e)

        self.results["checks"]["disk_space"] = check_result
        return check_result

    def check_directories(self) -> dict[str, Any]:
        """Check directory structure"""
        check_result = {"name": "Directories", "status": "OK", "details": {}, "missing": []}

        required_dirs = [
            "src/tasni/core",
            "src/tasni/download",
            "src/tasni/crossmatch",
            "src/tasni/analysis",
            "src/tasni/filtering",
            "src/tasni/generation",
            "data",
            "output",
            "logs",
            "archive",
            "docs",
            "tests",
        ]

        for dir_path in required_dirs:
            full_path = self.tasni_root / dir_path
            exists = full_path.exists()

            check_result["details"][dir_path] = exists

            if not exists:
                check_result["status"] = "ERROR"
                check_result["missing"].append(dir_path)

        if check_result["status"] == "ERROR":
            check_result["message"] = f"Missing {len(check_result['missing'])} directories"

        self.results["checks"]["directories"] = check_result
        return check_result

    def check_data_integrity(self) -> dict[str, Any]:
        """Check data integrity"""
        check_result = {"name": "Data Integrity", "status": "OK", "details": {}}

        # Check WISE data
        wise_dir = self.tasni_root / "data" / "wise"
        wise_files = list(wise_dir.glob("wise_hp*.parquet")) if wise_dir.exists() else []
        check_result["details"]["wise_tiles"] = len(wise_files)

        # Check Gaia data
        gaia_dir = self.tasni_root / "data" / "gaia"
        gaia_files = list(gaia_dir.glob("gaia_hp*.parquet")) if gaia_dir.exists() else []
        check_result["details"]["gaia_tiles"] = len(gaia_files)

        # Check expected tile count
        expected_tiles = 12288  # NSIDE=32
        wise_percent = (len(wise_files) / expected_tiles) * 100 if expected_tiles > 0 else 0
        gaia_percent = (len(gaia_files) / expected_tiles) * 100 if expected_tiles > 0 else 0

        check_result["details"]["wise_percent"] = wise_percent
        check_result["details"]["gaia_percent"] = gaia_percent

        # Check if data is missing
        if len(wise_files) < expected_tiles * 0.5:  # Less than 50%
            check_result["status"] = "WARNING"
            check_result["message"] = f"WISE data incomplete ({wise_percent:.1f}%)"

        if len(gaia_files) < expected_tiles * 0.5:  # Less than 50%
            check_result["status"] = "WARNING"
            check_result["message"] = f"Gaia data incomplete ({gaia_percent:.1f}%)"

        self.results["checks"]["data_integrity"] = check_result
        return check_result

    def check_outputs(self) -> dict[str, Any]:
        """Check output files"""
        check_result = {"name": "Outputs", "status": "OK", "details": {}}

        output_dir = self.tasni_root / "output"
        final_dir = output_dir / "final"

        # Check golden targets
        golden_csv = final_dir / "golden_targets.csv"
        golden_parquet = final_dir / "golden_targets.parquet"
        check_result["details"]["golden_targets"] = golden_csv.exists() or golden_parquet.exists()

        # Check variability
        var_csv = final_dir / "golden_variability.csv"
        check_result["details"]["variability"] = var_csv.exists()

        # Check figures
        figures_dir = output_dir / "figures"
        figures = list(figures_dir.glob("*")) if figures_dir.exists() else []
        check_result["details"]["figures_count"] = len(figures)

        # Check for critical outputs
        if not (golden_csv.exists() or golden_parquet.exists()):
            check_result["status"] = "WARNING"
            check_result["message"] = "Golden targets not generated"

        self.results["checks"]["outputs"] = check_result
        return check_result

    def check_dependencies(self) -> dict[str, Any]:
        """Check Python dependencies"""
        check_result = {"name": "Dependencies", "status": "OK", "details": {}}

        critical_packages = ["numpy", "pandas", "pyarrow", "astropy", "scipy", "tqdm", "healpy"]

        optional_packages = ["cupy", "cudf", "torch", "sklearn", "umap"]

        check_result["details"]["critical"] = {}
        for package in critical_packages:
            try:
                __import__(package)
                check_result["details"]["critical"][package] = "OK"
            except ImportError:
                check_result["details"]["critical"][package] = "MISSING"
                check_result["status"] = "ERROR"

        check_result["details"]["optional"] = {}
        for package in optional_packages:
            try:
                __import__(package)
                check_result["details"]["optional"][package] = "OK"
            except ImportError:
                check_result["details"]["optional"][package] = "NOT INSTALLED"
                check_result["status"] = "WARNING" if check_result["status"] != "ERROR" else "ERROR"

        missing_critical = sum(
            1 for v in check_result["details"]["critical"].values() if v == "MISSING"
        )
        if missing_critical > 0:
            check_result["message"] = f"{missing_critical} critical packages missing"

        self.results["checks"]["dependencies"] = check_result
        return check_result

    def check_git(self) -> dict[str, Any]:
        """Check git status"""
        check_result = {"name": "Git", "status": "OK", "details": {}}

        try:
            # Check if git repository
            result = subprocess.run(
                ["git", "status"], cwd=self.tasni_root, capture_output=True, text=True
            )

            if result.returncode == 0:
                check_result["details"]["repo"] = True

                # Check for uncommitted changes
                output = result.stdout
                if "modified:" in output or "Untracked files:" in output:
                    check_result["status"] = "WARNING"
                    check_result["message"] = "Uncommitted changes present"
                    check_result["details"]["clean"] = False
                else:
                    check_result["details"]["clean"] = True
            else:
                check_result["status"] = "WARNING"
                check_result["message"] = "Not a git repository"
                check_result["details"]["repo"] = False

        except FileNotFoundError:
            check_result["status"] = "WARNING"
            check_result["message"] = "Git not installed"
            check_result["details"]["git_installed"] = False

        self.results["checks"]["git"] = check_result
        return check_result

    def check_logs(self) -> dict[str, Any]:
        """Check log files"""
        check_result = {"name": "Logs", "status": "OK", "details": {}}

        log_dir = self.tasni_root / "logs"

        if not log_dir.exists():
            check_result["status"] = "WARNING"
            check_result["message"] = "Logs directory not found"
            return check_result

        # Count log files
        log_files = list(log_dir.glob("*.log"))
        check_result["details"]["log_files"] = len(log_files)

        # Count compressed logs
        gz_files = list(log_dir.glob("*.log.gz"))
        check_result["details"]["compressed_logs"] = len(gz_files)

        # Check log sizes
        total_size = sum(f.stat().st_size for f in log_files)
        check_result["details"]["total_size_mb"] = total_size / (1024**2)

        # Warning if logs are very large (>100MB)
        if total_size > 100 * 1024**2:
            check_result["status"] = "WARNING"
            check_result["message"] = f"Logs exceed 100MB ({total_size/(1024**3):.2f}GB)"

        self.results["checks"]["logs"] = check_result
        return check_result

    def check_archive_size(self) -> dict[str, Any]:
        """Check archive size"""
        check_result = {"name": "Archive Size", "status": "OK", "details": {}}

        archive_dir = self.tasni_root / "archive"

        if not archive_dir.exists():
            check_result["status"] = "OK"
            check_result["message"] = "Archive directory not found"
            return check_result

        # Calculate total size
        total_size = sum(f.stat().st_size for f in archive_dir.rglob("*") if f.is_file())

        size_gb = total_size / (1024**3)
        check_result["details"]["size_gb"] = size_gb

        # Limit: 100GB
        limit_gb = 100
        percent = (size_gb / limit_gb) * 100

        check_result["details"]["limit_gb"] = limit_gb
        check_result["details"]["usage_percent"] = percent

        if size_gb > limit_gb:
            check_result["status"] = "CRITICAL"
            check_result["message"] = f"Archive exceeds limit ({size_gb:.1f}GB > {limit_gb}GB)"
        elif size_gb > limit_gb * 0.9:
            check_result["status"] = "WARNING"
            check_result["message"] = f"Archive near limit ({percent:.1f}%)"

        self.results["checks"]["archive_size"] = check_result
        return check_result

    def run_all_checks(self) -> dict:
        """Run all health checks"""
        print("=" * 70)
        print("TASNI Health Check")
        print("=" * 70)
        print()

        # Run checks
        self.check_disk_space()
        self.check_directories()
        self.check_data_integrity()
        self.check_outputs()
        self.check_dependencies()
        self.check_git()
        self.check_logs()
        self.check_archive_size()

        # Print results
        for check_name, check_result in self.results["checks"].items():
            status = check_result["status"]
            status_symbol = "✓" if status == "OK" else "⚠️" if status == "WARNING" else "✗"
            print(f"{status_symbol} {check_result['name']}: {status}")

            if "message" in check_result:
                print(f"  └─ {check_result['message']}")

            # Print details
            if check_result["details"]:
                for key, value in check_result["details"].items():
                    if key not in ["message"]:
                        if isinstance(value, dict):
                            for k, v in value.items():
                                print(f"    - {k}: {v}")
                        else:
                            print(f"    - {key}: {value}")
            print()

        # Summary
        print("=" * 70)
        print("Summary")
        print("=" * 70)

        status_counts = {"OK": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
        for check_result in self.results["checks"].values():
            status_counts[check_result["status"]] += 1

        print(f"OK: {status_counts['OK']}")
        print(f"WARNING: {status_counts['WARNING']}")
        print(f"ERROR: {status_counts['ERROR']}")
        print(f"CRITICAL: {status_counts['CRITICAL']}")
        print()

        if status_counts["CRITICAL"] > 0:
            overall_status = "CRITICAL"
            print("✗ CRITICAL: Immediate action required")
        elif status_counts["ERROR"] > 0:
            overall_status = "ERROR"
            print("✗ ERROR: Issues detected")
        elif status_counts["WARNING"] > 0:
            overall_status = "WARNING"
            print("⚠️ WARNING: Non-critical issues detected")
        else:
            overall_status = "OK"
            print("✓ OK: All systems healthy")

        self.results["overall_status"] = overall_status
        self.results["summary"] = status_counts

        print("=" * 70)

        return self.results

    def save_report(self, output_path: str = None) -> str:
        """Save health check report to JSON"""
        if output_path is None:
            output_path = self.tasni_root / "output" / "health_check_report.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        return str(output_path)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TASNI Health Check", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--check",
        choices=["disk", "directories", "data", "outputs", "deps", "git", "logs", "archive", "all"],
        default="all",
        help="Check type (default: all)",
    )

    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")

    parser.add_argument(
        "--root", type=str, default=None, help="TASNI root directory (default: auto-detect)"
    )

    args = parser.parse_args()

    # Initialize health checker
    tasni_root = Path(args.root) if args.root else None
    checker = HealthChecker(tasni_root)

    try:
        # Run checks
        if args.check == "all":
            checker.run_all_checks()
        elif args.check == "disk":
            checker.check_disk_space()
        elif args.check == "directories":
            checker.check_directories()
        elif args.check == "data":
            checker.check_data_integrity()
        elif args.check == "outputs":
            checker.check_outputs()
        elif args.check == "deps":
            checker.check_dependencies()
        elif args.check == "git":
            checker.check_git()
        elif args.check == "logs":
            checker.check_logs()
        elif args.check == "archive":
            checker.check_archive_size()

        # Save report
        if args.check == "all":
            report_path = checker.save_report(args.output)
            print(f"Report saved: {report_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
