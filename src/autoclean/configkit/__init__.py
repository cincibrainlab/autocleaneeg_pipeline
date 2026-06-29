"""Configuration boundary for task schemas and validation.

Centralizes config enums, schema builders, and validation helpers.
"""

from .schema import (
    COMP_REJ_METHODS,
    IC_FLAGS,
    ICA_METHODS,
    THRESHOLD_MODES,
    format_task_config_error,
    validate_task_module_config,
)

__all__ = [
    "THRESHOLD_MODES",
    "COMP_REJ_METHODS",
    "ICA_METHODS",
    "IC_FLAGS",
    "format_task_config_error",
    "validate_task_module_config",
]
