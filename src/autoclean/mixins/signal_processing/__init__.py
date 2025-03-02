"""Signal processing mixin classes for autoclean tasks.

This package contains mixin classes that provide signal processing functionality
that can be shared across different task types.
"""

from autoclean.mixins.signal_processing.base import SignalProcessingMixin
from autoclean.mixins.signal_processing.resampling import ResamplingMixin
from autoclean.mixins.signal_processing.artifacts import ArtifactsMixin
from autoclean.mixins.signal_processing.channels import ChannelsMixin
from autoclean.mixins.signal_processing.segmentation import SegmentationMixin
from autoclean.mixins.signal_processing.reference import ReferenceMixin
from autoclean.mixins.signal_processing.epochs import EpochsMixin

__all__ = [
    "SignalProcessingMixin",
    "ResamplingMixin",
    "ArtifactsMixin",
    "ChannelsMixin",
    "SegmentationMixin",
    "ReferenceMixin",
    "EpochsMixin",
]
