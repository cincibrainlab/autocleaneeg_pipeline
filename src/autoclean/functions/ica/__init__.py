"""ICA functions for AutoClean.

This module provides standalone functions for Independent Component Analysis (ICA)
including component fitting, classification, and artifact rejection.
"""

from .ica_processing import (
    apply_ica_component_rejection,
    apply_ica_rejection,
    classify_ica_components,
    fit_ica,
    process_ica_control_sheet,
)

__all__ = [
    "fit_ica",
    "classify_ica_components",
    "apply_ica_rejection",
    "apply_ica_component_rejection",
    "process_ica_control_sheet",
]
