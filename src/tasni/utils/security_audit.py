"""
Security audit for TASNI project

Checks for hardcoded credentials, sensitive data, and security issues.

Usage:
    python src/tasni/utils/security_audit.py
"""

import argparse
import re
import sys
from pathlib import Path


def scan_for_secrets(file_path: Path) -> list[tuple[str, int, str]]:
    """Scan a file for potential secrets"""
    secrets = []

    if not file_path.is_file():
        return secrets

    # Patterns that might indicate secrets
    patterns = [
        (r"password\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded password"),
        (r"passwd\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded password (passwd)"),
        (r"pwd\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded password (pwd)"),
        (r"api_key\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded API key"),
        (r"apikey\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded API key (apikey)"),
        (r"api-key\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded API key (api-key)"),
        (r"token\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded token"),
        (r"access_token\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded access token"),
        (r"secret\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded secret"),
        (r"private_key\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded private key"),
        (r"secret_key\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded secret key"),
        (r"auth\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded auth"),
        (r"authorization\s*=\s*[\"\']([^\"\']+)[\"\']", "Hardcoded authorization"),
        (r"://[^:]+:([^@]+)@", "URL with credentials"),
    ]

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue

                for pattern, description in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        secrets.append((str(file_path), line_num, description))

    except Exception as e:
        import logging

        logging.getLogger(__name__).debug("Could not scan %s for secrets: %s", file_path, e)

    return secrets


def scan_for_hardcoded_paths(file_path: Path) -> list[tuple[str, int, str]]:
    """Scan for hardcoded absolute paths"""
    issues = []

    if not file_path.is_file():
        return issues

    # Pattern for absolute paths
    pattern = r"[\"\'](/[a-zA-Z0-9_/\-\.]+)[\"\']"

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                matches = re.finditer(pattern, line)
                for match in matches:
                    path = match.group(1)

                    # Skip common non-sensitive paths
                    if any(
                        x in path
                        for x in [
                            "/dev/",
                            "/usr/",
                            "/opt/",
                            "/proc/",
                            "/bin/",
                            "/sbin/",
                            "/lib/",
                            "/home/",
                            "/tmp/",
                            "/var/log/",
                        ]
                    ):
                        continue

                    # Skip example/template paths
                    if "example" in line.lower() or "template" in line.lower():
                        continue

                    # Skip documentation
                    if any(
                        x in str(file_path) for x in ["README", "docs/", "examples/", "templates/"]
                    ):
                        continue

                    issues.append((str(file_path), line_num, f"Hardcoded path: {path}"))

    except Exception as e:
        import logging

        logging.getLogger(__name__).debug("Could not scan %s for paths: %s", file_path, e)

    return issues


def scan_for_sql_injection(file_path: Path) -> list[tuple[str, int, str]]:
    """Scan for potential SQL injection vectors"""
    issues = []

    if not file_path.is_file():
        return issues

    # Pattern for SQL injection
    patterns = [
        (r'execute\s*\(\s*["\'].*%s', "SQL query with %s formatting"),
        (r'execute\s*\(\s*["\'].*\+', "SQL query with string concatenation"),
        (r'cursor\.execute\s*\(\s*["\'].*\{', "SQL query with f-string format"),
    ]

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                for pattern, description in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        # Check if it's using proper parameterization
                        if "param" in line.lower() or "params" in line.lower():
                            continue
                        issues.append((str(file_path), line_num, description))

    except Exception as e:
        import logging

        logging.getLogger(__name__).debug("Could not scan %s for SQL: %s", file_path, e)

    return issues


def check_file_permissions(tasni_root: Path) -> list[tuple[str, str]]:
    """Check for files with overly permissive permissions"""
    issues = []

    for file in tasni_root.rglob("*"):
        if not file.is_file():
            continue

        try:
            mode = file.stat().st_mode

            # Check if world-writable
            if mode & 0o002:
                issues.append((str(file), "World-writable"))

            # Check if group-writable for scripts
            if file.suffix == ".py" and mode & 0o020:
                issues.append((str(file), "Group-writable Python file"))

        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("Could not check permissions for %s: %s", file, e)

    return issues


def check_for_sensitive_data(tasni_root: Path) -> list[tuple[str, str]]:
    """Check for potential sensitive data files"""
    issues = []

    sensitive_patterns = [
        "*password*",
        "*secret*",
        "*key*",
        "*credential*",
        "*token*",
        "*auth*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "id_rsa*",
        "id_dsa*",
        "*_rsa*",
        "*_dsa*",
    ]

    for pattern in sensitive_patterns:
        for file in tasni_root.rglob(pattern):
            if not file.is_file():
                continue

            # Skip examples/templates
            if any(x in str(file).lower() for x in ["example", "template", "test"]):
                continue

            # Skip .git directory
            if ".git" in str(file):
                continue

            # Skip venv
            if "venv" in str(file) or ".venv" in str(file):
                continue

            issues.append((str(file), "Potential sensitive file"))

    return issues


def check_for_debug_code(tasni_root: Path) -> list[tuple[str, int, str]]:
    """Scan for debug code that should not be in production"""
    issues = []

    # Debug patterns
    patterns = [
        (r"pprint\s*\(", "pprint (debug output)"),
        (r"pdb\s*\.set_trace", "pdb breakpoint"),
        (r"ipdb\s*\.set_trace", "ipdb breakpoint"),
        (r"breakpoint\s*\(\)", "Python 3.7+ breakpoint"),
        (r"print\s*\(\s*[\"\']\*{20,}", "Long debug print statement"),
        (r"print\s*\(\s*[\"\']DEBUG", "Debug print statement"),
    ]

    for py_file in tasni_root.rglob("*.py"):
        if not py_file.is_file():
            continue

        # Skip test files
        if "test" in py_file.name:
            continue

        # Skip examples
        if "example" in str(py_file).lower():
            continue

        try:
            with open(py_file, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    # Skip comments
                    if line.strip().startswith("#"):
                        continue

                    for pattern, description in patterns:
                        match = re.search(pattern, line)
                        if match:
                            issues.append((str(py_file), line_num, description))

        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("Could not scan %s for debug code: %s", py_file, e)

    return issues


def main():
    """Run security audit"""
    parser = argparse.ArgumentParser(
        description="TASNI Security Audit", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--root", type=str, default=None, help="TASNI root directory (default: auto-detect)"
    )

    parser.add_argument("--secrets", action="store_true", help="Scan for hardcoded secrets")

    parser.add_argument("--paths", action="store_true", help="Scan for hardcoded paths")

    parser.add_argument("--sql", action="store_true", help="Scan for SQL injection vectors")

    parser.add_argument("--permissions", action="store_true", help="Check file permissions")

    parser.add_argument("--sensitive", action="store_true", help="Check for sensitive data files")

    parser.add_argument("--debug", action="store_true", help="Scan for debug code")

    args = parser.parse_args()

    # Auto-detect root
    if args.root:
        tasni_root = Path(args.root)
    else:
        tasni_root = Path.cwd()
        while tasni_root.name != "tasni" and tasni_root.parent != tasni_root:
            tasni_root = tasni_root.parent

    print("=" * 80)
    print("TASNI Security Audit")
    print("=" * 80)
    print(f"Scanning: {tasni_root}")
    print()

    # Run all scans if none specified
    run_all = not any(
        [args.secrets, args.paths, args.sql, args.permissions, args.sensitive, args.debug]
    )

    total_issues = 0

    # Scan for secrets
    if run_all or args.secrets:
        print("Scanning for hardcoded secrets...")
        secrets_found = 0
        for py_file in tasni_root.rglob("*.py"):
            if "venv" in str(py_file) or ".venv" in str(py_file):
                continue
            if ".git" in str(py_file):
                continue
            secrets = scan_for_secrets(py_file)
            if secrets:
                for file_path, line_num, description in secrets:
                    print(f"  ⚠️  {file_path}:{line_num} - {description}")
                    secrets_found += 1
                    total_issues += 1
        if secrets_found == 0:
            print("  ✓ No hardcoded secrets found")
        print()

    # Scan for hardcoded paths
    if run_all or args.paths:
        print("Scanning for hardcoded paths...")
        paths_found = 0
        for py_file in tasni_root.rglob("*.py"):
            if "venv" in str(py_file) or ".venv" in str(py_file):
                continue
            if ".git" in str(py_file):
                continue
            paths = scan_for_hardcoded_paths(py_file)
            if paths:
                # Show first 20 to avoid spam
                for file_path, line_num, description in paths[:20]:
                    print(f"  ⚠️  {file_path}:{line_num} - {description}")
                    paths_found += 1
                    total_issues += 1
                if len(paths) > 20:
                    print(f"  ... and {len(paths) - 20} more")
                    paths_found += len(paths) - 20
                    total_issues += len(paths) - 20
        if paths_found == 0:
            print("  ✓ No hardcoded paths found")
        elif paths_found >= 20:
            print(f"  ⚠️  Found {paths_found} hardcoded paths (showing first 20)")
        print()

    # Scan for SQL injection
    if run_all or args.sql:
        print("Scanning for SQL injection vectors...")
        sql_found = 0
        for py_file in tasni_root.rglob("*.py"):
            if "venv" in str(py_file) or ".venv" in str(py_file):
                continue
            sql_issues = scan_for_sql_injection(py_file)
            if sql_issues:
                for file_path, line_num, description in sql_issues:
                    print(f"  ⚠️  {file_path}:{line_num} - {description}")
                    sql_found += 1
                    total_issues += 1
        if sql_found == 0:
            print("  ✓ No SQL injection vectors found")
        print()

    # Check file permissions
    if run_all or args.permissions:
        print("Checking file permissions...")
        permission_issues = check_file_permissions(tasni_root)
        if permission_issues:
            for file_path, issue in permission_issues[:20]:
                print(f"  ⚠️  {file_path} - {issue}")
                total_issues += 1
            if len(permission_issues) > 20:
                print(f"  ... and {len(permission_issues) - 20} more")
                total_issues += len(permission_issues) - 20
        else:
            print("  ✓ No permission issues found")
        print()

    # Check for sensitive data files
    if run_all or args.sensitive:
        print("Checking for sensitive data files...")
        sensitive_issues = check_for_sensitive_data(tasni_root)
        if sensitive_issues:
            for file_path, issue in sensitive_issues[:20]:
                print(f"  ⚠️  {file_path} - {issue}")
                total_issues += 1
            if len(sensitive_issues) > 20:
                print(f"  ... and {len(sensitive_issues) - 20} more")
                total_issues += len(sensitive_issues) - 20
        else:
            print("  ✓ No sensitive files found")
        print()

    # Check for debug code
    if run_all or args.debug:
        print("Checking for debug code...")
        debug_issues = check_for_debug_code(tasni_root)
        if debug_issues:
            for file_path, line_num, description in debug_issues[:20]:
                print(f"  ⚠️  {file_path}:{line_num} - {description}")
                total_issues += 1
            if len(debug_issues) > 20:
                print(f"  ... and {len(debug_issues) - 20} more")
                total_issues += len(debug_issues) - 20
        else:
            print("  ✓ No debug code found")
        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total security issues found: {total_issues}")
    print()

    if total_issues == 0:
        print("✓ No security issues detected")
        print("  Great! Keep it up!")
        return 0
    else:
        print(f"⚠️  Found {total_issues} potential security issues")
        print("  Please review and address the issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
