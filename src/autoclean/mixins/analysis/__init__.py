"""Analysis mixins for autoclean tasks.

This module provides mixins for various analysis functions that can be
used within the autoclean pipeline. These mixins handle configuration,
metadata tracking, and result saving while wrapping standalone analysis functions.

Classes
-------
InterTrialCoherenceMixin : Mixin for computing inter-trial coherence analysis
"""

from .intertrial_coherence import InterTrialCoherenceMixin

__all__ = [
    "InterTrialCoherenceMixin",
]