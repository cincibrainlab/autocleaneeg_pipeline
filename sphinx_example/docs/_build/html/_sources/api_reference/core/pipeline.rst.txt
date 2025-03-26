.. _api_pipeline:

========
Pipeline
========

.. currentmodule:: autoclean

Main pipeline class for EEG processing.

This class serves as the primary interface for the autoclean package.
It manages the complete processing workflow including:

- Configuration loading and validation
- Directory structure setup
- Task instantiation and execution
- Progress tracking and error handling
- Results saving and report generation

The pipeline supports multiple EEG processing paradigms through its task registry,
allowing researchers to process different types of EEG recordings with appropriate
analysis pipelines.

.. autoclass:: Pipeline
   :members:
   :member-order: groupwise
   :exclude-members: __init__, __weakref__, task_registry

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Core Methods

   .. automethod:: process_file
   .. automethod:: process_directory
   .. automethod:: process_directory_async

   .. rubric:: Utility Methods

   .. automethod:: list_tasks
   .. automethod:: list_stage_files
   .. automethod:: start_autoclean_review

Usage Examples
-------------

.. code-block:: python

   from autoclean import Pipeline

   # Initialize the pipeline
   pipeline = Pipeline(
       autoclean_dir="results/",
       autoclean_config="configs/default.yaml"
   )

   # Process a single file
   pipeline.process_file(
       file_path="data/sub-01_task-rest_eeg.set",
       task="rest_eyesopen"
   ) 