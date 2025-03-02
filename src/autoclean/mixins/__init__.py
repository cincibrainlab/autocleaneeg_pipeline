"""Mixin classes for autoclean tasks.

This package contains mixin classes that provide common functionality
that can be shared across different task types.
"""

from autoclean.mixins.data_processing import SignalProcessingMixin

__all__ = ["SignalProcessingMixin"]
