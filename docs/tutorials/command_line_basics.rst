AutoClean EEG CLI Usage Guide
==============================

Don't worry - you don't need to be a programmer to use AutoClean! This guide will teach you the simple commands you need to know, starting from the very basics.

🚀 Getting Started (No Arguments)
----------------------------------

When you first install AutoClean EEG, try running it without any arguments to see what happens:

.. code-block:: bash

   autocleaneeg-pipeline

This shows you a helpful overview with:

* System information (Python version, OS, current time)
* Your workspace directory location
* Quick links to get help and tutorials
* First-time setup reminder if needed

This is your starting point - bookmark this command!

🏠 Setting Up Your Workspace
-----------------------------

Before processing any data, you need to configure your workspace:

.. code-block:: bash

   # First time setup - interactive wizard
   autocleaneeg-pipeline workspace

This runs an interactive setup that helps you:

* Choose your workspace directory location
* Create the necessary folder structure
* Set up basic configuration

**Workspace management commands:**

.. code-block:: bash

   # See current workspace location
   autocleaneeg-pipeline workspace
   
   # Open workspace folder in your file manager
   autocleaneeg-pipeline workspace explore
   
   # Change workspace location
   autocleaneeg-pipeline workspace set /path/to/new/location
   
   # Reset to default location
   autocleaneeg-pipeline workspace default
   
   # Navigate to workspace in terminal
   autocleaneeg-pipeline workspace cd

📋 Finding Available Tasks
---------------------------

Before processing data, see what analysis workflows are available:

.. code-block:: bash

   # List all available tasks (built-in + custom)
   autocleaneeg-pipeline list-tasks
   
   # Or use the task command
   autocleaneeg-pipeline task list
   
   # Show detailed information about tasks
   autocleaneeg-pipeline list-tasks --verbose

Common built-in tasks you'll see:
* ``RestingEyesOpen`` - Resting state with eyes open
* ``RestingEyesClosed`` - Resting state with eyes closed  
* ``MMN`` - Mismatch negativity paradigm
* ``ASSR`` - Auditory steady-state response
* ``Chirp`` - Chirp stimulus paradigm

🔄 Processing Your Data (Main Commands)
----------------------------------------

The most important command is ``process`` - this is where the magic happens!

**Simple processing (recommended):**

.. code-block:: bash

   # Process a single file
   autocleaneeg-pipeline process RestingEyesOpen my_data.raw
   
   # Process all .raw files in current directory  
   autocleaneeg-pipeline process RestingEyesOpen --dir . --format "*.raw"
   
   # Process with custom output directory
   autocleaneeg-pipeline process RestingEyesOpen data.raw --output ~/EEG_Results

**Advanced processing options:**

.. code-block:: bash

   # Process multiple formats
   autocleaneeg-pipeline process RestingEyesOpen --dir /data --format "*.set"
   
   # Recursive directory search
   autocleaneeg-pipeline process RestingEyesOpen --dir /data --recursive
   
   # Parallel processing (faster for multiple files)
   autocleaneeg-pipeline process RestingEyesOpen --dir /data --parallel 4
   
   # See what would be processed (don't actually run)
   autocleaneeg-pipeline process RestingEyesOpen --dir /data --dry-run
   
   # Verbose output for troubleshooting
   autocleaneeg-pipeline process RestingEyesOpen data.raw --verbose

🛠️ Managing Custom Tasks
-------------------------

Create and manage your own analysis workflows:

**Adding custom tasks:**

.. code-block:: bash

   # Add a task file to your workspace
   autocleaneeg-pipeline task add my_custom_analysis.py
   
   # Add with a specific name
   autocleaneeg-pipeline task add analysis.py --name MySpecialTask
   
   # Force overwrite if task exists
   autocleaneeg-pipeline task add analysis.py --force

**Editing tasks:**

.. code-block:: bash

   # Edit a workspace task in your default editor
   autocleaneeg-pipeline task edit MyCustomTask
   
   # Copy a built-in task to edit it
   autocleaneeg-pipeline task edit RestingEyesOpen --name MyRestingTask

**Managing task files:**

.. code-block:: bash

   # Copy any task to a new workspace file
   autocleaneeg-pipeline task copy RestingEyesOpen --name MyModifiedResting
   
   # Import a task file from anywhere
   autocleaneeg-pipeline task import /path/to/task.py --name ImportedTask
   
   # Remove a custom task
   autocleaneeg-pipeline task remove MyCustomTask
   
   # Delete task file completely
   autocleaneeg-pipeline task delete MyCustomTask
   
   # Open tasks folder in file manager
   autocleaneeg-pipeline task explore

📊 Reviewing Results
--------------------

After processing, review your results with the GUI:

.. code-block:: bash

   # Start the review GUI (uses workspace output directory)
   autocleaneeg-pipeline review
   
   # Review a specific output directory
   autocleaneeg-pipeline review --output /path/to/results

The review GUI lets you:
* Browse processed files visually
* Compare before/after signal quality
* Export results and reports
* Generate publication-ready figures

⚙️ Configuration Management
----------------------------

Manage your AutoClean configuration:

.. code-block:: bash

   # Show configuration directory location
   autocleaneeg-pipeline config show
   
   # Reconfigure workspace (same as workspace setup)
   autocleaneeg-pipeline config setup
   
   # Reset all configuration to defaults
   autocleaneeg-pipeline config reset --confirm
   
   # Export your configuration  
   autocleaneeg-pipeline config export ~/my-autoclean-backup
   
   # Import configuration from backup
   autocleaneeg-pipeline config import ~/my-autoclean-backup

🔍 Viewing EEG Files
--------------------

Quickly view EEG files without processing:

.. code-block:: bash

   # View an EEG file in MNE's browser
   autocleaneeg-pipeline view data.raw
   
   # Just validate the file (don't open viewer)
   autocleaneeg-pipeline view data.raw --no-view

🧹 Maintenance Commands
-----------------------

Keep your workspace clean and organized:

.. code-block:: bash

   # Remove all outputs for a specific task
   autocleaneeg-pipeline clean-task RestingEyesOpen
   
   # See what would be deleted (don't actually delete)
   autocleaneeg-pipeline clean-task RestingEyesOpen --dry-run
   
   # Force deletion without confirmation
   autocleaneeg-pipeline clean-task RestingEyesOpen --force
   
   # Check workspace size
   autocleaneeg-pipeline workspace size

📋 Audit and Compliance
------------------------

For research documentation and compliance:

.. code-block:: bash

   # Export complete audit trail
   autocleaneeg-pipeline export-access-log --output audit-trail.jsonl
   
   # Export specific date range
   autocleaneeg-pipeline export-access-log \
     --start-date 2025-01-01 \
     --end-date 2025-01-31 \
     --output monthly-audit.jsonl
   
   # Export as CSV for spreadsheet analysis
   autocleaneeg-pipeline export-access-log --format csv --output audit.csv
   
   # Just verify database integrity (no export)
   autocleaneeg-pipeline export-access-log --verify-only

🔐 Authentication (Advanced)
-----------------------------

For compliance environments requiring user authentication:

.. code-block:: bash

   # Login to authentication system
   autocleaneeg-pipeline login
   
   # Check who is currently logged in
   autocleaneeg-pipeline whoami
   
   # Logout and clear tokens
   autocleaneeg-pipeline logout
   
   # Diagnose authentication issues
   autocleaneeg-pipeline auth0-diagnostics --verbose

ℹ️ Getting Help
----------------

When you need more information:

.. code-block:: bash

   # Show version information
   autocleaneeg-pipeline version
   
   # Show general help
   autocleaneeg-pipeline help
   
   # Show help for specific commands
   autocleaneeg-pipeline process --help
   autocleaneeg-pipeline task --help
   
   # Run the interactive tutorial
   autocleaneeg-pipeline tutorial

🎯 Your First Analysis (Step by Step)
--------------------------------------

Here's exactly what to type for your first EEG analysis:

**1. Check if AutoClean is working:**

.. code-block:: bash

   autocleaneeg-pipeline

You should see the welcome screen.

**2. Set up your workspace:**

.. code-block:: bash

   autocleaneeg-pipeline workspace

Follow the interactive prompts.

**3. See what tasks are available:**

.. code-block:: bash

   autocleaneeg-pipeline list-tasks

**4. Navigate to your data:**

.. code-block:: bash

   # Change to your data directory
   cd /path/to/your/eeg/data

**5. Process your first file:**

.. code-block:: bash

   autocleaneeg-pipeline process RestingEyesOpen your_file.raw

**6. Review results:**

.. code-block:: bash

   autocleaneeg-pipeline review

**7. Check where everything was saved:**

.. code-block:: bash

   autocleaneeg-pipeline config show

🚨 Troubleshooting Common Issues
---------------------------------

**"Command not found":**

.. code-block:: bash

   # Install AutoClean EEG
   pip install autocleaneeg-pipeline

**"Workspace not configured":**

.. code-block:: bash

   # Run workspace setup
   autocleaneeg-pipeline workspace

**"Task not found":**

.. code-block:: bash

   # List available tasks
   autocleaneeg-pipeline list-tasks

**"File not found":**

.. code-block:: bash

   # Check you're in the right directory
   pwd           # Mac/Linux
   cd            # Windows
   
   # List files
   ls            # Mac/Linux  
   dir           # Windows

**Processing seems stuck:**
* Wait - EEG processing takes time (especially for large files)
* Press Ctrl+C to cancel if needed
* Use ``--verbose`` flag to see detailed progress

💡 Pro Tips
-----------

**Tab completion:**
Start typing filenames and press Tab for auto-completion.

**Command history:**
Use up/down arrow keys to repeat previous commands.

**Batch processing:**
Use ``--dir`` and ``--format`` to process multiple files at once.

**Parallel processing:**
Use ``--parallel N`` for faster processing of multiple files.

**Dry run:**
Always use ``--dry-run`` first when processing directories to see what will happen.

**Keep organized:**
Use meaningful task names and organize your workspace folders.

🎉 You're Ready to Go!
-----------------------

With these commands, you can:

✅ Set up AutoClean EEG  
✅ Process single files or entire directories  
✅ Manage custom analysis workflows  
✅ Review and export results  
✅ Maintain audit trails for research  

Start with the basic workflow above, then gradually explore more advanced features as your needs grow.

**Next Steps:**
* Try :doc:`first_time_processing` for a detailed walkthrough
* Learn about :doc:`understanding_results` to interpret your data
* Explore :doc:`creating_custom_task` for specialized analyses
