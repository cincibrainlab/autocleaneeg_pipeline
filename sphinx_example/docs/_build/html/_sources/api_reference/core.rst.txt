.. _api_core:

==========
Core Modules
==========

This section covers the core classes for EEG data processing in AutoClean. These classes form the 
foundation of the processing pipeline and define the main abstractions for tasks and data flow.

Pipeline
-------

The :class:`~autoclean.core.pipeline.Pipeline` class is the main entry point for using AutoClean. 
It manages the processing of EEG data files through various tasks.

.. toctree::
   :maxdepth: 2
   
   core/pipeline

Task
----

The :class:`~autoclean.core.task.Task` class is the base class for all processing tasks in AutoClean.
Task implementations define specific processing pipelines for different EEG paradigms.

.. toctree::
   :maxdepth: 2
   
   core/task 
   