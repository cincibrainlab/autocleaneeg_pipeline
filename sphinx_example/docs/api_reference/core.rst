.. _api_core:

==========
Core API
==========

.. currentmodule:: autoclean

This section covers the core classes for EEG data processing in AutoClean.

.. toctree::
   :maxdepth: 1
   
   core/pipeline
   core/task

Pipeline
========

The Pipeline class is the main entry point for AutoClean, providing the interface for EEG data processing.

.. autoclass:: Pipeline
   :members:
   :inherited-members:
   :exclude-members: __init__, __weakref__
   
   .. rubric:: Methods
   
   .. autosummary::
      :toctree: ../generated/
      
      Pipeline.__init__
      Pipeline.process_file
      Pipeline.process_directory
      Pipeline.process_directory_async
      Pipeline._entrypoint
      Pipeline.list_tasks
      Pipeline.list_stage_files
      Pipeline.start_autoclean_review
      Pipeline._validate_task
      Pipeline._validate_file

Task
====

The Task abstract base class defines the interface for all EEG processing tasks.

.. currentmodule:: autoclean.core.task

.. autoclass:: Task
   :members:
   :inherited-members:
   :exclude-members: __init__, __weakref__
   
   .. rubric:: Methods
   
   .. autosummary::
      :toctree: ../generated/
      
      Task.__init__
      Task.import_raw
      Task.run
      Task.validate_config
      Task._validate_task_config
      Task.get_flagged_status
      Task.get_raw
      Task.get_epochs
   