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
from autoclean.mixins.signal_processing.regular_epochs import RegularEpochsMixin
from autoclean.mixins.signal_processing.eventid_epochs import EventIDEpochsMixin
from autoclean.mixins.signal_processing.prepare_epochs_ica import PrepareEpochsICAMixin
from autoclean.mixins.signal_processing.gfp_clean_epochs import GFPCleanEpochsMixin
from autoclean.mixins.signal_processing.autoreject_epochs import AutoRejectEpochsMixin

__all__ = [
    "SignalProcessingMixin",
    "ResamplingMixin",
    "ArtifactsMixin",
    "ChannelsMixin",
    "SegmentationMixin",
    "ReferenceMixin",
    "EpochsMixin",
    "RegularEpochsMixin",
    "EventIDEpochsMixin",
    "PrepareEpochsICAMixin",
    "GFPCleanEpochsMixin",
    "AutoRejectEpochsMixin",
]
