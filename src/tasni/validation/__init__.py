"""
TASNI Validation Module

Provides rigorous validation of the TASNI pipeline including:
- Expanded brown dwarf test catalog (100+ objects)
- K-fold cross-validation
- Precision/recall metrics
- Bootstrap confidence intervals
"""

from .expanded_bd_catalog import KNOWN_BROWN_DWARFS, load_expanded_brown_dwarf_catalog
from .ml_validation import MLValidator, cross_validate_model
from .rigorous_validation import (
    RigorousValidator,
    calculate_precision_recall,
    calculate_roc_metrics,
    validate_with_kfold,
)

__all__ = [
    "load_expanded_brown_dwarf_catalog",
    "KNOWN_BROWN_DWARFS",
    "RigorousValidator",
    "validate_with_kfold",
    "calculate_precision_recall",
    "calculate_roc_metrics",
    "MLValidator",
    "cross_validate_model",
]
