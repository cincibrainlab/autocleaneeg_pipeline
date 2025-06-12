Getting Started
===============

AutoClean is a framework for automated EEG data processing built on MNE-Python. This guide covers installation, workspace setup, and the Python-based workflow for EEG processing.

Installation
------------

Install AutoClean using Python's package manager:

.. code-block:: bash

   pip install autocleaneeg

Workspace Setup
---------------

AutoClean uses a simple "drop-and-go" workflow centered around a workspace directory. Set this up once:

.. code-block:: python

   import subprocess
   subprocess.run(["autoclean", "setup"])

Or run this from your file manager by opening a terminal/command prompt and typing ``autoclean setup``.

This creates a workspace directory (typically ``~/Documents/AutoClean-EEG``) with a simple structure:

.. code-block::

   AutoClean-EEG/
   ├── tasks/                    # Drop custom task files here
   ├── output/                   # All processing results
   └── example_basic_usage.py    # Example Python script

Python API Workflow
--------------------

AutoClean is designed for Python-based workflows. Here's the basic pattern:

.. code-block:: python

   from autoclean import Pipeline
   
   # Initialize the pipeline
   pipeline = Pipeline(output_dir="results")
   
   # Process your EEG data using built-in tasks
   pipeline.process_file("your_data.raw", task="RestingEyesOpen")

Navigation with File Manager
----------------------------

Use your system's file manager (Finder on Mac, File Explorer on Windows, Files on Linux) to:

- **Navigate to your data files** - Browse to where your EEG files are stored
- **View results** - Open the ``output`` folder in your workspace to see processed data
- **Manage tasks** - Drop new task files into the ``tasks`` folder
- **Backup** - Copy your entire workspace folder to backup your setup

This approach eliminates the need for command-line navigation and integrates with your familiar file management workflow.

Built-in Tasks
--------------

AutoClean includes several ready-to-use tasks for common EEG paradigms:

- ``RestingEyesOpen`` - Resting state with eyes open
- ``RestingEyesClosed`` - Resting state with eyes closed  
- ``ASSR`` - Auditory steady-state response
- ``ChirpDefault`` - Chirp stimulus paradigm
- ``HBCD_MMN`` - Mismatch negativity for HBCD protocol
- ``StatisticalLearning`` - Statistical learning paradigm

Use these directly in your Python scripts without any setup:

.. code-block:: python

   from autoclean import Pipeline
   
   pipeline = Pipeline(output_dir="results")
   
   # For resting state data
   pipeline.process_file("rest_data.raw", task="RestingEyesOpen")
   
   # For auditory experiments  
   pipeline.process_file("assr_data.raw", task="ASSR")

Creating Custom Tasks
---------------------

The real power of AutoClean comes from easily creating custom processing workflows. Use the **AutoClean Config Wizard** at https://cincibrainlab.github.io/Autoclean-ConfigWizard/ to create task files tailored to your experiment.

**Step 1: Create Your Task**

1. Open https://cincibrainlab.github.io/Autoclean-ConfigWizard/ in your browser
2. Select your EEG system, experimental paradigm, and processing options
3. Download the generated Python task file

**Step 2: Drop Into Workspace**

Simply save the downloaded file to your ``tasks`` folder:

.. code-block::

   AutoClean-EEG/
   ├── tasks/
   │   └── MyCustomTask.py    # <-- Drop your task file here
   ├── output/
   └── example_basic_usage.py

**Step 3: Use Immediately**

.. code-block:: python

   from autoclean import Pipeline
   
   pipeline = Pipeline(output_dir="results")
   
   # AutoClean automatically finds your custom task
   pipeline.process_file("my_data.raw", task="MyCustomTask")

The workspace automatically discovers new task files - no installation, registration, or configuration required.

Working with Multiple Files
----------------------------

Process multiple files using Python's file handling. Use your file manager to locate your data directory, then:

.. code-block:: python

   from pathlib import Path
   from autoclean import Pipeline
   
   pipeline = Pipeline(output_dir="results")
   
   # Process all .raw files in a directory
   data_dir = Path("/path/to/your/data")  # Use file manager to find this path
   for eeg_file in data_dir.glob("*.raw"):
       pipeline.process_file(str(eeg_file), task="RestingEyesOpen")

**Finding File Paths with File Manager:**

1. Navigate to your data folder using Finder/File Explorer/Files
2. Right-click on the folder and select "Copy Path" or "Properties" to get the full path
3. Use this path in your Python script

Example Workflow
----------------

Here's a complete example of the typical AutoClean workflow:

.. code-block:: python

   from pathlib import Path
   from autoclean import Pipeline
   
   # Set up paths using your file manager
   data_path = Path("/Users/researcher/EEG_Data/subject01_rest.raw")
   output_path = Path("/Users/researcher/AutoClean_Results")
   
   # Initialize pipeline
   pipeline = Pipeline(output_dir=str(output_path))
   
   # Process the file
   result = pipeline.process_file(str(data_path), task="RestingEyesOpen")
   
   # Results are automatically saved to output_path
   print(f"Processing complete! Results in: {output_path}")

Results are organized in timestamped folders that you can browse with your file manager. Each processing run creates a complete record including cleaned data, quality reports, and processing logs.

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