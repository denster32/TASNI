# Contributing to TASNI

Thank you for your interest in contributing to TASNI (Thermal Anomaly Search for Non-communicating Intelligence)!

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Making Changes](#making-changes)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Pull Request Guidelines](#pull-request-guidelines)

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- 500GB+ free storage (for catalogs)
- (Optional) NVIDIA GPU with CUDA 12.x
- (Optional) Intel Arc GPU

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
```bash
git clone https://github.com/your-username/tasni.git
cd tasni
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/original-username/tasni.git
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Production dependencies
make install
# or
pip install -r requirements.txt

# Development dependencies (recommended)
make install-dev
# or
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

Pre-commit hooks will automatically:
- Format code with black
- Sort imports with isort
- Check code style with flake8
- Run security checks with bandit

### 4. Verify Installation

```bash
# Run tests
make test

# Check imports
python -m pytest tests/unit/test_imports.py -v

# Test configuration
python src/tasni/core/config.py
```

## Making Changes

### 1. Create a Branch

Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
# or
git checkout -b docs/your-docs-name
```

### 2. Make Your Changes

- Edit code in `src/tasni/` subdirectories
- Follow the directory structure:
  - `core/` - Configuration, logging
  - `download/` - Data acquisition
  - `crossmatch/` - Spatial matching
  - `analysis/` - Data analysis
  - `filtering/` - Anomaly detection
  - `generation/` - Output generation
  - `optimized/` - Performance versions
  - `ml/` - Machine learning
  - `utils/` - Utilities
  - `checks/` - Validation
  - `misc/` - Miscellaneous

### 3. Write Tests

- Add unit tests in `tests/unit/`
- Add integration tests in `tests/integration/`
- Use fixtures from `tests/fixtures/sample_data.py`

Example test:
```python
import pytest
import pandas as pd
from analysis.analyze_kinematics import calculate_pm

def test_calculate_pm():
    """Test proper motion calculation"""
    pmra = 100.0
    pmdec = 50.0
    result = calculate_pm(pmra, pmdec)

    assert result == pytest.approx(111.8, rel=0.01)
```

### 4. Update Documentation

- Update relevant documentation in `docs/`
- Add docstrings to functions
- Update README if needed
- Update CHANGELOG if applicable

## Code Style

### Formatting

TASNI uses:
- **Black** for code formatting
- **isort** for import sorting

Format your code:
```bash
make format
# or
black src/tasni/ tests/
isort src/tasni/ tests/
```

Configuration (in `pyproject.toml`):
- Line length: 100 characters
- String quotes: Double quotes
- Import sorting: `isort` profile

### Linting

TASNI uses **flake8** for linting.

Check your code:
```bash
make lint
# or
flake8 src/tasni/ tests/ --max-line-length=100
```

Common issues to fix:
- Unused imports
- Line too long (>100)
- Missing docstrings
- Undefined variables

### Naming Conventions

- **Variables**: `snake_case` - `my_variable`
- **Functions**: `snake_case` - `my_function()`
- **Classes**: `PascalCase` - `MyClass`
- **Constants**: `UPPER_SNAKE_CASE` - `MY_CONSTANT`
- **Private methods**: `_leading_underscore` - `_private_method()`

### Docstrings

Use Google-style docstrings:

```python
def calculate_temperature(w1mag: float, w2mag: float) -> float:
    """Calculate effective temperature from WISE magnitudes.

    Uses the relation from Kirkpatrick et al. (2012).

    Args:
        w1mag: W1 magnitude (mag)
        w2mag: W2 magnitude (mag)

    Returns:
        Effective temperature (K)

    Raises:
        ValueError: If magnitude values are invalid

    Examples:
        >>> calculate_temperature(12.5, 11.5)
        450.0
    """
    # Implementation
```

## Testing

### Run All Tests

```bash
make test
# or
python -m pytest tests/ -v
```

### Run Specific Tests

```bash
# Unit tests only
make test-unit
# or
python -m pytest tests/unit/ -v

# Integration tests only
make test-integration
# or
python -m pytest tests/integration/ -v

# Tests matching a pattern
python -m pytest tests/ -k "config" -v

# Specific test file
python -m pytest tests/unit/test_config.py -v
```

### Test Coverage

Generate coverage report:
```bash
make test-coverage
# or
python -m pytest tests/ --cov=scripts --cov-report=html
```

View coverage:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

Aim for >80% coverage for new code.

### Test Markers

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests (slower)
- `@pytest.mark.slow`: Slow tests (require large data)
- `@pytest.mark.gpuskip`: Skip if GPU not available

Example:
```python
@pytest.mark.unit
def test_fast_function():
    pass

@pytest.mark.slow
@pytest.mark.gpuskip
def test_slow_gpu_function():
    pass
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings in source code
2. **User Documentation**: Guides in `docs/`
3. **API Documentation**: Auto-generated from docstrings
4. **Architecture Documentation**: System design in `docs/ARCHITECTURE.md`

### Documentation Style

- Use clear, concise language
- Provide examples where helpful
- Use code blocks for code
- Use tables for comparisons
- Use diagrams for architecture (if needed)

### Building Documentation

```bash
# Sphinx documentation (if configured)
make docs

# View documentation
# Open docs/QUICKSTART.md, docs/ARCHITECTURE.md, etc.
```

## Submitting Changes

### 1. Sync with Upstream

```bash
git fetch upstream
git rebase upstream/master
```

### 2. Run Tests and Linters

```bash
# Format code
make format

# Run linters
make lint

# Run tests
make test
```

Fix any issues before proceeding.

### 3. Commit Your Changes

```bash
git add .
git commit -m "type: brief description

More detailed explanation (optional).

- Change 1
- Change 2

Refs: #123
"
```

**Commit message format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance tasks

**Examples:**
```
feat: Add NEOWISE variability detection
fix: Correct proper motion calculation error
docs: Update installation guide
test: Add unit tests for scoring module
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill in PR template

## Pull Request Guidelines

### PR Title

Use clear, descriptive titles:
- **Good**: `feat: Add NEOWISE light curve analysis`
- **Bad**: `update stuff` or `fix code`

### PR Description

Include:
- What this PR does
- Why it's needed
- How it was tested
- Screenshots (for UI changes)
- Breaking changes (if any)
- Related issues

### PR Checklist

- [ ] Code follows project style (`make format`)
- [ ] Tests pass locally (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow guidelines
- [ ] Self-review completed

### Review Process

1. Automated checks (CI/CD) must pass
2. At least one maintainer approval
3. Address review comments
4. Squash and merge when approved

## Additional Guidelines

### Performance

- Profile code before optimizing
- Use `src/tasni/optimized/` for performance-critical code
- Benchmark with `make benchmark`
- Consider GPU acceleration for heavy computations

### Error Handling

- Use specific exceptions (not bare `except`)
- Provide helpful error messages
- Log errors appropriately
- Handle edge cases

### Logging

- Use `tasni_logging` module
- Appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Structured logging with context
- Avoid logging sensitive data

### Security

- Never commit credentials
- Use environment variables for secrets
- Run security audit: `make security-audit`
- Follow security best practices

### Data Handling

- Use streaming for large files
- Clean up temporary files
- Use efficient data structures
- Respect data retention policies

## Getting Help

### Resources
- **Documentation**: `docs/` directory
- **Examples**: `notebooks/` directory
- **Issues**: GitHub Issues

### Questions
- Open an issue with the `question` label
- Be specific and provide context
- Include code snippets if applicable
- Mention your environment (OS, Python version)

## Recognition

Contributors will be acknowledged in:
- AUTHORS file
- CHANGELOG
- Publication acknowledgments (if appropriate)

## License

By contributing, you agree that your contributions will be licensed under the project's license.

---

Thank you for contributing to TASNI! Your contributions help advance the search for non-communicating intelligence signatures.
