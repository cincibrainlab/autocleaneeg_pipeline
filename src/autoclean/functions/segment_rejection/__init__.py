"""Segment rejection functions for AutoClean.

This module provides standalone functions for identifying and rejecting noisy segments
in EEG data, including oscillatory artifacts and other temporal anomalies.
"""

from .dense_oscillatory import detect_dense_oscillatory_artifacts
from .segment_rejection import annotate_noisy_segments, annotate_uncorrelated_segments

__all__ = [
    "detect_dense_oscillatory_artifacts",
    "annotate_noisy_segments",
    "annotate_uncorrelated_segments",
]