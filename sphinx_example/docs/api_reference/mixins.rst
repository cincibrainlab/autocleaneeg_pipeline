.. _api_mixins:

=============
Mixins
=============

This section covers the mixin classes that provide reusable functionality for EEG data processing in AutoClean.

.. currentmodule:: autoclean.mixins.signal_processing

SignalProcessingMixin
-------------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst
   :nosignatures:
   
   resampling.ResamplingMixin
   artifacts.ArtifactsMixin
   channels.ChannelsMixin
   segmentation.SegmentationMixin
   reference.ReferenceMixin
   eventid_epochs.EventIDEpochsMixin
   regular_epochs.RegularEpochsMixin
   pylossless.PyLosslessMixin

.. currentmodule:: autoclean.mixins.reporting

ReportingMixin
------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst
   :nosignatures:
   
   visualization.VisualizationMixin
   ica.ICAReportingMixin