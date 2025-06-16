"""Artifact Detection and Channel Operations Functions.

This module contains standalone functions for detecting and handling artifacts
in EEG data, including bad channel detection, interpolation, and channel operations.

Functions
---------
detect_bad_channels : Detect bad channels using multiple methods
interpolate_bad_channels : Interpolate bad channels using spherical splines
drop_channels : Remove specified channels from data
"""

# Import implemented functions
from .channels import detect_bad_channels, interpolate_bad_channels

__all__ = [
    "detect_bad_channels",
    "interpolate_bad_channels",
    # "drop_channels",  # Already in preprocessing
]