Getting Started
===============

Welcome to AutoClean! This guide will help you get up and running quickly, whether you're new to programming or an experienced researcher.

🎯 Quick Start for Everyone
---------------------------

**Step 1: Install AutoClean**

.. code-block:: bash

   pip install autocleaneeg

**Step 2: Set up your workspace**

.. code-block:: bash

   autoclean setup

This creates a folder in your Documents called "Autoclean-EEG" where all your tasks and results will be stored.

**Step 3: Get a task file**

You can either:
- Use the web-based task builder (coming soon) to create custom tasks
- Download example tasks from our gallery
- Use the built-in tasks that come with AutoClean

**Step 4: Process your data**

.. code-block:: bash

   autoclean process RestingEyesOpen my_eeg_data.raw

That's it! AutoClean will automatically:
- Find your task configuration
- Process your EEG data with advanced artifact removal
- Save results to your workspace output folder
- Generate quality control reports

📊 For Non-Technical Users
--------------------------

If you're not comfortable with command lines or programming, AutoClean is designed with you in mind:

**Installation Options**

*Option A: Ask your IT department*
   Share this guide with your IT support - the installation is straightforward and requires only one command.

*Option B: Use a research computing specialist*
   Many universities have research computing groups that can help install and set up AutoClean.

*Option C: Learn the basics*
   The command line sounds scary but it's just typing simple commands. See our :doc:`tutorials/command_line_basics` guide.

**Using AutoClean Without Programming**

Once installed, you only need to remember a few simple commands:

.. code-block:: bash

   # See what tasks are available
   autoclean list-tasks
   
   # Process a single file  
   autoclean process TaskName data_file.raw
   
   # Process multiple files in a folder
   autoclean process TaskName data_folder/
   
   # Check where your results are saved
   autoclean config show

**Getting Task Files**

1. **Web Task Builder** (recommended): Visit our online task builder to create processing workflows through a point-and-click interface
2. **Task Gallery**: Download pre-made tasks for common experimental paradigms
3. **Ask a colleague**: If someone in your lab has created tasks, they can share the files with you

🔧 For Technical Users  
----------------------

If you're comfortable with Python or command-line tools, AutoClean offers powerful automation and customization options:

**Python Integration**

.. code-block:: python

   from autoclean import Pipeline
   
   # Simple usage - automatically uses your workspace
   pipeline = Pipeline()
   
   # Process single files
   pipeline.process_file("subject01.raw", "RestingEyesOpen")
   
   # Batch process multiple files
   pipeline.process_directory("data/", "RestingEyesOpen") 
   
   # Custom output location
   pipeline = Pipeline(output_dir="custom_results/")

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

**Jupyter Notebook Integration**

.. code-block:: python

   # Perfect for interactive data analysis
   from autoclean import Pipeline
   
   pipeline = Pipeline()  # Uses your workspace automatically
   
   # Process data
   pipeline.process_file("subject01.raw", "RestingEyesOpen")
   
   # Results are automatically saved to workspace/output/
   # Quality control reports are generated automatically

📁 Understanding Your Workspace
-------------------------------

After running setup, you'll find this structure in your Documents folder:

.. code-block::

   Documents/Autoclean-EEG/
   ├── tasks/           # Your custom task files go here
   ├── output/          # Processing results saved here  
   └── user_config.json # AutoClean settings

**Key Points:**

- **Drop-and-Go**: Drop .py task files into the tasks/ folder and AutoClean automatically finds them
- **Organized Results**: All processing outputs go to the output/ folder with clear naming
- **Backup Friendly**: The entire Autoclean-EEG folder can be copied to backup or share your setup
- **Cross-Platform**: Same folder structure works on Windows, Mac, and Linux

🎯 Built-in Tasks
-----------------

AutoClean comes with several ready-to-use tasks:

- **RestingEyesOpen**: For resting-state EEG recordings
- **ChirpDefault**: For chirp auditory stimulus experiments  
- **AssrDefault**: For auditory steady-state response paradigms
- **HBCD_MMN**: For mismatch negativity experiments

You can use these immediately without any configuration:

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