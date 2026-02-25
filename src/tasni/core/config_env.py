"""
Compatibility layer for environment-driven TASNI config access.

`src/tasni/core/config.py` is the single source of truth. This module provides:
1) helper env parsing functions used in tests/integration code, and
2) re-exported config symbols for legacy imports.

It intentionally has no filesystem side effects at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

# Re-export canonical configuration to preserve backward compatibility.
from tasni.core.config import *  # noqa: F401,F403


def get_path(env_var: str, default: Path) -> Path:
    """Return a Path from environment variable, else default."""
    value = os.getenv(env_var)
    return Path(value) if value else default


def get_int(env_var: str, default: int) -> int:
    """Return an int from environment variable, else default."""
    value = os.getenv(env_var)
    if value is None or value == "":
        return default
    return int(value)


def get_float(env_var: str, default: float) -> float:
    """Return a float from environment variable, else default."""
    value = os.getenv(env_var)
    if value is None or value == "":
        return default
    return float(value)


def get_bool(env_var: str, default: bool) -> bool:
    """Return a bool from environment variable, else default."""
    value = os.getenv(env_var)
    if value is None:
        return default
    norm = value.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    return default
