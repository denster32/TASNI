# Security Policy

## Supported Versions

The TASNI project is currently in active development. Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of TASNI seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **Email**: Send a detailed report to paluckide@yahoo.com with the subject line "SECURITY: Vulnerability Report"

2. **GitHub Security Advisory**: Use the [Security Advisories](https://github.com/dpalucki/tasni/security/advisories) feature

### What to Include

Please include the following information in your report:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths of source file(s) related to the manifestation of the issue**
- **The location of the affected source code** (tag/branch/commit or direct URL)
- **Any special configuration required to reproduce the issue**
- **Step-by-step instructions to reproduce the issue**
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity (Critical: 7 days, High: 14 days, Medium: 30 days, Low: Next release)

### Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any potential similar problems
3. Prepare fixes for all supported versions
4. Release patched versions
5. Announce the vulnerability through our security advisory

## Security Best Practices

When using TASNI:

1. **Do not commit sensitive data**: Never commit API keys, passwords, or personal data to the repository
2. **Use environment variables**: Store credentials in environment variables or `.env` files (already in `.gitignore`)
3. **Keep dependencies updated**: Regularly run `pip audit` or `poetry update` to update dependencies
4. **Review code changes**: Carefully review any code from external contributors before merging

## Known Security Considerations

### Data Handling

TASNI processes astronomical catalog data which may include:
- Large datasets (500GB+)
- External API queries (IRSA, VizieR, etc.)

Please ensure:
- Adequate disk space before processing
- Rate limiting when querying external services
- Secure storage of any downloaded data

### Dependencies

TASNI depends on numerous scientific Python packages. We recommend:
- Using a virtual environment
- Regularly auditing dependencies with `pip-audit`
- Reviewing the `pyproject.toml` for all dependencies

## Contact

For any security-related questions or concerns:
- Email: paluckide@yahoo.com
- GitHub: [@dpalucki](https://github.com/dpalucki)

---

**Last Updated**: 2026-02-23
