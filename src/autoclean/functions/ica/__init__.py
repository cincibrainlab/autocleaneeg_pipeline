"""ICA functions for AutoClean.

This module provides standalone functions for Independent Component Analysis (ICA)
including component fitting, classification, and artifact rejection.
"""

from .ica_processing import fit_ica, classify_ica_components, apply_ica_rejection, apply_iclabel_rejection

__all__ = [
    "fit_ica",
    "classify_ica_components",
    "apply_ica_rejection",
    "apply_iclabel_rejection",
]