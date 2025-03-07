"""Main signal processing mixin that combines all specialized mixins."""

from autoclean.mixins.signal_processing.base import SignalProcessingMixin as BaseSignalProcessingMixin
from autoclean.mixins.signal_processing.resampling import ResamplingMixin
from autoclean.mixins.signal_processing.artifacts import ArtifactsMixin
from autoclean.mixins.signal_processing.channels import ChannelsMixin
from autoclean.mixins.signal_processing.segmentation import SegmentationMixin
from autoclean.mixins.signal_processing.reference import ReferenceMixin
from autoclean.mixins.signal_processing.epochs import EpochsMixin

class SignalProcessingMixin(
    BaseSignalProcessingMixin,
    ResamplingMixin,
    ArtifactsMixin,
    ChannelsMixin,
    SegmentationMixin,
    ReferenceMixin,
    EpochsMixin
):
    """Main mixin class that combines all signal processing functionality.
    
    This class inherits from all specialized signal processing mixins to provide
    a comprehensive set of signal processing methods for EEG data.
    """
    pass
