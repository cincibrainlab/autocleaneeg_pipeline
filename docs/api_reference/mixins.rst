.. _api_mixins:

===============================
Mixins *(autoclean.mixins)*
===============================

This section covers the mixin classes that provide reusable functionality for EEG data processing in AutoClean.
Mixins are the preferred way to add functions to your tasks in autoclean and act as a replacement for step functions. 
They are designed as classes that are added to the base task class so that functions may be natively accessible from any task implementation.
This simplifies the process of creating new tasks as you do not need to worry about manually importing each processing function.

Mixins should be designed to be used in task implementations as such: 

.. code-block:: python

   from autoclean.core.task import Task
   # Mixins are imported inside the Task base class

   class MyTask(Task):
       def run(self):
           # Calling a mixin function
           self.create_regular_epochs()  # Modifies self.epochs

*Note:* Most mixins may have a return value or a data parameter but are designed to use and update the task object and its data attributes in place. 
If you decide to use both mixin functions and non-mixin functions, be careful to update your task's data attributes accordingly.

*Example:*

.. code-block:: python

   # Since the self.raw attribute has been updated, we can use the mixin function
   self.create_regular_epochs()  # Modifies self.epochs

Available Mixins
-----------------


.. currentmodule:: autoclean.mixins.signal_processing

SignalProcessingMixin
---------------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst
   :nosignatures:
   
   

   basic_steps.BasicStepsMixin
   channels.ChannelsMixin
   artifacts.ArtifactsMixin
   autoreject_epochs.AutoRejectEpochsMixin
   channels.ChannelsMixin
   eventid_epochs.EventIDEpochsMixin
   regular_epochs.RegularEpochsMixin
   prepare_epochs_ica.PrepareEpochsICAMixin
   gfp_clean_epochs.GFPCleanEpochsMixin
   ica.IcaMixin
   segment_rejection.SegmentRejectionMixin

.. currentmodule:: autoclean.mixins.viz

ReportingMixin
--------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst
   :nosignatures:
   
   visualization.VisualizationMixin
   ica.ICAReportingMixin