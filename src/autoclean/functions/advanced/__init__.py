"""Advanced Processing Functions.

This module contains standalone functions for advanced EEG processing techniques
including autoreject-based epoch cleaning.

Functions
---------
autoreject_epochs : Apply AutoReject for automatic epoch cleaning
"""

# Import implemented functions
from .autoreject import autoreject_epochs

__all__ = [
    "autoreject_epochs",
]