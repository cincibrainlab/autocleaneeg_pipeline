"""GUI and visualization tools.

These tools require GUI dependencies (PyQt6) which may not be available
in headless/server environments. Imports are lazy to avoid loading
GUI dependencies when not needed.
"""


def __getattr__(name):
    """Lazy import for GUI tools to avoid loading PyQt6 at import time."""
    if name == "run_autoclean_exclusion_tool":
        from .autoclean_exclude import run_autoclean_exclusion_tool

        return run_autoclean_exclusion_tool
    elif name == "run_autoclean_review":
        from .autoclean_review import run_autoclean_review

        return run_autoclean_review

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["run_autoclean_review", "run_autoclean_exclusion_tool"]
