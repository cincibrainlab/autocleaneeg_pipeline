Autoclean
====================================

**Autoclean** is an automated EEG processing pipeline leveraging MNE and PyLossless. It provides a modular, task-based framework for EEG data analysis, designed as an importable Python package. Create custom pipelines tailored to your specific experimental needs.

Features
--------
* **Modular Architecture**: Task-based system for custom processing pipelines
* **Automated Preprocessing**: Streamlined workflows for EEG data cleaning
* **Artifact Detection**: Advanced algorithms for removing common EEG artifacts
* **Flexible Configuration**: YAML-based configuration for reproducible analysis
* **Comprehensive Reporting**: Detailed visual reports for quality control
* **Database Integration**: Tracking of processing runs and results

Who is this for?
----------------

This pipeline is designed for:

**EEG Researchers:** Streamline your preprocessing and analysis workflows with reproducible and configurable pipelines.

**Why use this pipeline?** 
Preprocessing EEG data often suffers from variability, even among highly trained researchers, due to differences in methods and quality standards. Additionally, existing pipelines may lack flexibility or fail to include critical processing steps required by specific experiments. To address these challenges, Autoclean provides a fully customizable, modular pipeline using a task-based architecture. While initial setup involves creating and configuring tasks to precisely fit your analysis needs, the result is a robust, reproducible, and automated workflow that can include both preprocessing and analysis. This ensures consistent, high-quality outputs tailored specifically to your dataset, significantly improving both the reliability and efficiency of your EEG data analysis.

Core Concepts
-------------

The pipeline is built around a few key ideas:

*   **Pipeline:** The central orchestrator that manages configuration, data flow, and task execution.
*   **Task:** A defined sequence of processing steps for a specific experimental paradigm (e.g., resting-state, auditory steady-state response). You can use pre-built tasks or create your own.
*   **Config:** A YAML file that defines the parameters for the pipeline and tasks.
*   **Mixins:** Individual, reusable processing units (e.g., loading data, filtering, artifact rejection, reporting) that make up a Task.
*   **Step Functions:** Legacy functions that may still be used to process data.
*   **Plugins:** Plugins handle importing different formats and montages as well as different event types

Quick Example
-------------

.. code-block:: python

   from autoclean import Pipeline

   # Initialize the pipeline with configuration and output directory
   pipeline = Pipeline(
       autoclean_dir="path/to/your/results",
       autoclean_config="path/to/your/autoclean_config.yaml",
       # lossless_config="path/to/your/lossless_config.yaml" # Optional
   )

   # Process a single EEG file using a specific task
   pipeline.process_file(
       file_path="path/to/your/data/sub-01_task-rest_eeg.set", # Example file
       task="resting_eyes_open"  # Example task
   )

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Table of Contents

   self

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development
   CONTRIBUTING
   CHANGELOG
