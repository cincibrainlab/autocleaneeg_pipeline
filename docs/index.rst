Autoclean
====================================

**AutoClean** is an automated EEG processing pipeline that leverages MNE and PyLossless to provide 
a modular framework for EEG data analysis. Designed as an importable Python package, it allows 
researchers to create custom pipelines tailored to their specific experimental paradigms.

Features
--------

* **Modular Architecture**: Task-based system for custom processing pipelines
* **Automated Preprocessing**: Streamlined workflows for EEG data cleaning
* **Artifact Detection**: Advanced algorithms for removing common EEG artifacts
* **Flexible Configuration**: YAML-based configuration for reproducible analysis
* **Comprehensive Reporting**: Detailed visual reports for quality control
* **Database Integration**: Tracking of processing runs and results

Quick Example
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

.. toctree::
   :maxdepth: 2
   :hidden:
   
   self

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api_reference/index

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   getting_started
   tutorial



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 