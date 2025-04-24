Getting Started
===============

Installation
------------

AutoClean EEG can be installed via pip:

.. code-block:: bash

   pip install autoclean-eeg

For development installation:

.. code-block:: bash

   git clone https://github.com/cincibrainlab/autoclean_pipeline
   cd autoclean_pipeline
   pip install -e .

Development Installations are recommended in order to fully utilize the features of Autoclean. 

Basic Usage
-----------

Single File
^^^^^^^^^^^

In order to run a task on a dataset you will use the :class:`~autoclean.core.pipeline.Pipeline` class.


.. code-block:: python

   from autoclean import Pipeline

   # Initialize the pipeline with configuration
   pipeline = Pipeline(
       autoclean_dir="/path/to/output",
       autoclean_config="configs/autoclean_config.yaml"
   )


   # Process a single file
   pipeline.process_file(
       file_path="/path/to/data.set",
       task="RestingEyesOpen"
   )


Processing Multiple Files
^^^^^^^^^^^^^^^^^^^^^^^^^

AutoClean supports batch processing of files, with both synchronous and asynchronous options:

.. code-block:: python

   # Synchronous processing (one file at a time)
   pipeline.process_directory(
       directory="/path/to/data",
       task="RestingEyesOpen",
       pattern="*.set"
   )

   # Asynchronous processing (multiple files concurrently)
   import asyncio
   
   asyncio.run(pipeline.process_directory_async(
       directory="/path/to/data",
       task="RestingEyesOpen",
       pattern="*.raw",
       max_concurrent=3  # Process up to 3 files simultaneously
   ))

**Note:** The inputted task name must match the name in the configuration file and the task class.


*For example:*

.. code-block:: python

    #src/autoclean/tasks/resting_eyes_open.py
    class RestingEyesOpen(Task):


.. code-block:: yaml

   #configs/autoclean_config.yaml
   tasks:
     RestingEyesOpen:



Configuration
-------------

AutoClean uses YAML files for configuration. The main configuration file specifies processing parameters for different tasks:

.. code-block:: yaml

   tasks:
     RestingEyesOpen:
       mne_task: "rest"
       description: "Resting state with eyes open"
       lossless_config: configs/pylossless/lossless_config.yaml
       settings:
         resample_step:
           enabled: true
           value: 250
         # Additional settings...
       rejection_policy:
         # Artifact rejection settings...

Provided and Example configuration files can be found in the `configs` directory.

Tasks
---------------

AutoClean comes with several pre-configured :class:`~autoclean.core.task.Task` classes:

- **RestingEyesOpen**: Processing for resting state EEG with eyes open
- **ChirpDefault**: Processing for chirp auditory stimulus paradigms
- **AssrDefault**: Processing for auditory steady state response paradigms
- **HBCD_MMN**: Processing for mismatch negativity paradigms
- **TEMPLATE**: Template for creating custom tasks


However the real power of Autoclean comes from the ability to create custom dataflows backed by structure and modular framework we have developed.

See :doc:`tutorials/creating_custom_task` for more information.



Output Structure
----------------

AutoClean organizes processing outputs in a structured directory hierarchy:

- **bids/**: Data and derivatives saved in BIDS format
- **logs/**: Logs of the processing steps
- **metadata/**: Full metadata in json format and a generic run report pdf
- **post_comps/**: Post completion files
- **stage/**: Where the stage files are saved

Docker Usage
------------

AutoClean can be run in a containerized environment using Docker. This ensures consistent execution across different systems.

Windows PowerShell
^^^^^^^^^^^^^^^^^^

.. code-block:: powershell

   # Add the autoclean command to your PowerShell profile
   Copy-Item profile.ps1 $PROFILE
   # or add to existing profile
   . "C:\path\to\autoclean.ps1"

   # Run the pipeline
   autoclean -DataPath "C:\Data\raw" -Task "RestingEyesOpen" -ConfigPath "C:\configs\autoclean_config.yaml"

Linux/WSL/Mac
^^^^^^^^^^^^^

.. code-block:: bash

   # Add the autoclean command to your system
   mkdir -p ~/.local/bin
   cp autoclean.sh ~/.local/bin/autoclean
   chmod +x ~/.local/bin/autoclean

   # Run the pipeline
   autoclean -DataPath "/path/to/data" -Task "RestingEyesOpen" -ConfigPath "/path/to/config.yaml"

Next Steps
----------

- See the :doc:`tutorials/index` for a step-by-step walkthrough
- Explore the :doc:`api_reference/index` for detailed API documentation
