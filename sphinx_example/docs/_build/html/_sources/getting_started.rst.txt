Getting Started
===============

This guide will walk you through the basics of using AutoClean for EEG data preprocessing.

Initializing a Pipeline
----------------------

The ``Pipeline`` class is the main entry point for the AutoClean package:

.. code-block:: python

    from autoclean import Pipeline
    
    # Initialize the pipeline with configuration
    pipeline = Pipeline(
        autoclean_dir="results/",
        autoclean_config="configs/autoclean_config.yaml",
        verbose="info"
    )

Processing a Single File
----------------------

Once your pipeline is initialized, you can process a single EEG file:

.. code-block:: python

    # Process a single file with a specific task
    pipeline.process_file(
        file_path="data/sub-01_task-rest_eeg.set",
        task="rest_eyesopen"
    )
    
    # The results will be saved in the specified autoclean_dir
    # with subdirectories for each processing stage

Batch Processing
--------------

Process multiple files in a directory:

.. code-block:: python

    # Process all .set files in a directory
    pipeline.process_directory(
        directory="data/",
        task="rest_eyesopen",
        pattern="*.set"
    )
    
    # Process asynchronously with concurrency control
    pipeline.process_directory_async(
        directory="data/",
        task="rest_eyesopen",
        pattern="*.set",
        max_concurrent=3
    )

Understanding Configuration Files
------------------------------

AutoClean uses YAML configuration files to control processing parameters:

.. code-block:: yaml

    # Example configuration snippet
    task: rest_eyesopen
    eeg_system: EGI_HydroCel_129
    
    tasks:
      rest_eyesopen:
        pylossless:
          lo_freq: 0.1
          hi_freq: 80
          ref_method: average
        
        rejection_policy:
          reject_criteria:
            max_bad_channels: 10
            ica_threshold: 0.8
    
    stage_files:
      post_import:
        enabled: true
        suffix: "_import"
      post_prepipeline:
        enabled: true
        suffix: "_preproc"

Creating Custom Tasks
------------------

To create a custom processing pipeline, you can extend the ``Task`` class:

.. code-block:: python

    from autoclean.core.task import Task
    import mne
    
    class MyCustomTask(Task):
        def __init__(self, config):
            super().__init__(config)
        
        def _validate_task_config(self, config):
            # Add task-specific validation here
            return config
        
        def run(self):
            # Import the raw data
            self.import_raw()
            
            # Apply your custom preprocessing steps
            # Example: Filtering
            self.raw.filter(l_freq=1.0, h_freq=40.0)
            
            # Create epochs if needed
            events = mne.find_events(self.raw)
            self.epochs = mne.Epochs(self.raw, events)
            
            # Generate reports
            self._generate_reports()

Using the AutoClean CLI
--------------------

AutoClean also provides a command-line interface:

.. code-block:: bash

    # Process a file using the CLI
    autoclean --config configs/autoclean_config.yaml --file data/sub-01.set --task rest_eyesopen
    
    # Process a directory
    autoclean --config configs/autoclean_config.yaml --dir data/ --task rest_eyesopen --pattern "*.set" 