"""Centralized deterministic seed policy for TASNI."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_RANDOM_SEED = int(os.getenv("TASNI_RANDOM_SEED", "42"))


def seed_numpy_and_python(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Set deterministic seeds for Python's random and NumPy RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


def make_rng(seed: int = DEFAULT_RANDOM_SEED) -> np.random.Generator:
    """Return a deterministic NumPy Generator."""
    return np.random.default_rng(seed)
