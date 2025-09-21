"""Configuration boundary for task schemas and validation.

Centralizes config enums, schema builders, and validation helpers.
"""

from .schema import (
    THRESHOLD_MODES,
    COMP_REJ_METHODS,
    ICA_METHODS,
    IC_FLAGS,
    validate_task_module_config,
)

__all__ = [
    "THRESHOLD_MODES",
    "COMP_REJ_METHODS",
    "ICA_METHODS",
    "IC_FLAGS",
    "validate_task_module_config",
]
