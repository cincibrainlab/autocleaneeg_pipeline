"""Advanced Processing Functions.

This module contains standalone functions for advanced EEG processing techniques
including autoreject-based epoch cleaning and segment-based rejection methods.

Functions
---------
autoreject_epochs : Apply AutoReject for automatic epoch cleaning
annotate_noisy_segments : Identify and annotate noisy data segments
annotate_uncorrelated_segments : Identify and annotate uncorrelated segments
"""

# Import implemented functions
from .autoreject import autoreject_epochs
from .segment_rejection import annotate_noisy_segments, annotate_uncorrelated_segments

__all__ = [
    "autoreject_epochs",
    "annotate_noisy_segments", 
    "annotate_uncorrelated_segments",
]