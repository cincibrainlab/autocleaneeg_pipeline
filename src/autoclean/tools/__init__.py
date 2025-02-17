"""Tools for the autoclean package."""

def run_autoclean_review(*args, **kwargs):
    """Lazy-load and run the autoclean review GUI.
    
    This function is a wrapper that only imports PyQt5 when the GUI is actually needed,
    allowing the rest of the package to function without GUI dependencies.
    """
    from .autoclean_review import run_autoclean_review as _run_review
    return _run_review(*args, **kwargs)

__all__ = ["run_autoclean_review"]
