"""Mixin classes for autoclean tasks.

This package contains mixin classes that provide common functionality
that can be shared across different task types.
"""

from autoclean.mixins.signal_processing.main import SignalProcessingMixin
from autoclean.mixins.signal_processing import (
    ResamplingMixin,
    ArtifactsMixin,
    ChannelsMixin,
    SegmentationMixin,
    ReferenceMixin,
    EpochsMixin,
)

__all__ = [
    "SignalProcessingMixin",
    "ResamplingMixin",
    "ArtifactsMixin",
    "ChannelsMixin",
    "SegmentationMixin",
    "ReferenceMixin",
    "EpochsMixin",
]
