Getting Started
===============

AutoClean is a framework for automated EEG data processing built on MNE-Python. This guide covers installation, basic usage, and the most common processing workflows.

Installation
------------

Install AutoClean from PyPI:

.. code-block:: bash

   pip install autocleaneeg

First-time Setup
----------------

Initialize your workspace directory:

.. code-block:: bash

   autoclean setup

This creates a workspace directory structure for organizing your processing tasks, configuration, and results. The default location is platform-specific (e.g., `~/Documents/AutoClean-EEG` on most systems).

Basic Usage
-----------

AutoClean provides built-in tasks for common EEG paradigms. Process your data using the command line:

.. code-block:: bash

   autoclean process RestingEyesOpen my_eeg_data.raw

Or use the Python API:

.. code-block:: python

   from autoclean import Pipeline
   
   pipeline = Pipeline(output_dir="results")
   pipeline.process_file("my_eeg_data.raw", task="RestingEyesOpen")

Processing includes artifact removal, ICA decomposition, epoching, and quality control reporting. Results are saved in BIDS-compatible format with comprehensive metadata.

Built-in Tasks
--------------

AutoClean includes tasks for common EEG paradigms:

**Resting State Tasks**
- ``RestingEyesOpen`` - Resting state with eyes open
- ``RestingEyesClosed`` - Resting state with eyes closed

**Auditory Tasks**
- ``ASSR`` - Auditory steady-state response
- ``ChirpDefault`` - Chirp stimulus paradigm
- ``HBCD_MMN`` - Mismatch negativity for HBCD protocol

**Analysis Tasks**
- ``StatisticalLearning`` - Statistical learning paradigm

List all available tasks:

.. code-block:: bash

   autoclean list-tasks

Command Line Interface
----------------------

The CLI provides efficient processing for single files or batch operations:

.. code-block:: bash

   # Process a single file
   autoclean process RestingEyesOpen data_file.raw
   
   # Process multiple files in a directory
   autoclean process RestingEyesOpen data_folder/
   
   # Custom output location
   autoclean process RestingEyesOpen data.raw --output results/
   
   # View processing options
   autoclean process --help

Python API
----------

For programmatic control and integration into research workflows:

.. code-block:: python

   from autoclean import Pipeline
   
   # Initialize pipeline
   pipeline = Pipeline(output_dir="results")
   
   # Process single files
   pipeline.process_file("subject01.raw", "RestingEyesOpen")
   
   # Batch process multiple files
   pipeline.process_directory("data/", "RestingEyesOpen")

**Advanced Command Line Usage**

.. code-block:: bash

   # Process with custom output location
   autoclean process RestingEyesOpen data.raw --output results/
   
   # Dry run to preview what will be processed
   autoclean process RestingEyesOpen data.raw --dry-run
   
   # Use a custom task file
   autoclean process --task-file my_custom_task.py data.raw

**Workspace Management**

.. code-block:: bash

   # Add custom tasks to your workspace
   autoclean task add my_task.py
   
   # List all available tasks (built-in + custom)
   autoclean list-tasks --include-custom
   
   # Manage your workspace
   autoclean config show          # See workspace location
   autoclean setup               # Reconfigure workspace
   
   # Manage custom tasks
   autoclean task list           # List custom tasks
   autoclean task remove MyTask  # Remove a custom task

**Jupyter Notebook Integration**

.. code-block:: python

   # Perfect for interactive data analysis
   from autoclean import Pipeline
   
   pipeline = Pipeline()  # Uses your workspace automatically
   
   # Process data
   pipeline.process_file("subject01.raw", "RestingEyesOpen")
   
   # Results are automatically saved to workspace/output/
   # Quality control reports are generated automatically

Custom Tasks
------------

In addition to built-in tasks, you can create custom processing workflows:

**Adding Custom Tasks**

.. code-block:: bash

   # Add a task file to your workspace
   autoclean task add my_custom_task.py
   
   # List all available tasks
   autoclean list-tasks

**Workspace Structure**

Your workspace directory contains:

.. code-block::

   ~/Documents/AutoClean-EEG/
   ├── tasks/                    # Custom task files
   ├── output/                   # Processing results
   └── example_basic_usage.py    # Example script

Custom task files are automatically discovered when placed in the tasks directory. Results are organized in timestamped folders within the output directory.

.. code-block:: bash

   autoclean process RestingEyesOpen my_data.raw

📈 Output and Results
--------------------

AutoClean creates comprehensive outputs for every processing run:

**Processed Data**
- Clean EEG data in standard formats (.set, .fif)
- Epoch data ready for analysis
- Artifact-corrected continuous data

**Quality Control Reports**
- Visual summaries of processing steps
- Before/after comparison plots
- Statistical summaries of data quality

**Metadata and Logs**
- Complete processing parameters
- Detailed logs of all processing steps
- Database tracking of all runs

All results are organized in timestamped folders so you never lose previous analyses.

🆘 Getting Help
---------------

**Documentation**
- :doc:`tutorials/index` - Step-by-step guides for common tasks
- :doc:`api_reference/index` - Complete technical reference

**Support**
- Check our FAQ for common questions
- Visit our GitHub issues page for bug reports
- Join our community forums for discussions

**Quick Troubleshooting**

.. code-block:: bash

   # Check if AutoClean is installed correctly
   autoclean version
   
   # Verify your workspace setup
   autoclean config show
   
   # List available tasks
   autoclean list-tasks

🚀 Next Steps
-------------

Now that you have AutoClean installed:

1. **Try the quick start example** above with your own data
2. **Explore the tutorials** to learn specific workflows
3. **Create custom tasks** using our task builder or Python templates
4. **Integrate with your analysis pipeline** using Python or command-line automation

Happy analyzing! 🧠