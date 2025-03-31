"""Automated EEG preprocessing pipeline.

This package provides tools for automated EEG data preprocessing,
supporting multiple experimental paradigms and processing workflows.
"""

from .core.pipeline import Pipeline

__version__ = "1.2.0"

__all__ = [
    "Pipeline",
]
