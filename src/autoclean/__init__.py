"""Automated EEG preprocessing pipeline.

This package provides tools for automated EEG data preprocessing,
supporting multiple experimental paradigms and processing workflows.
"""

__version__ = "1.4.1"


def __getattr__(name):
    """Lazy import for Pipeline to avoid loading heavy dependencies on simple imports."""
    if name == "Pipeline":
        from .core.pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Pipeline",
]
