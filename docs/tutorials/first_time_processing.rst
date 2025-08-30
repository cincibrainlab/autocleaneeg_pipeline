Python-Based EEG Processing
===========================

This tutorial shows how to process EEG data using AutoClean's Python API with file manager navigation and the Config Wizard for creating custom tasks.

What You Need
-------------

- Python environment with AutoClean installed
- EEG data files in supported formats (.raw, .set, .eeg, .bdf, .fif)
- Access to your system's file manager (Finder, File Explorer, Files)
- Web browser for the Config Wizard

Setup Your Environment
----------------------

Install and set up AutoClean:

.. code-block:: python

   # Install if needed
   import subprocess
   subprocess.run(["pip", "install", "autocleaneeg-pipeline"])
   
   # Set up workspace
   subprocess.run(["autoclean", "setup"])

This creates your workspace folder (typically ``~/Documents/AutoClean-EEG``) with the drop-and-go structure.

Basic Processing with Built-in Tasks
-------------------------------------

Start by processing data with AutoClean's built-in tasks:

.. code-block:: python

   from autoclean import Pipeline
   from pathlib import Path
   
   # Initialize pipeline
   pipeline = Pipeline(output_dir="my_results")
   
   # Use file manager to find your data file path
   data_file = "/path/to/your/data.raw"  # Copy path from file manager
   
   # Process with a built-in task
   pipeline.process_file(data_file, task="RestingEyesOpen")

**Available Built-in Tasks:**
- ``RestingEyesOpen`` - Resting state with eyes open
- ``ASSR`` - Auditory steady-state response  
- ``ChirpDefault`` - Chirp auditory stimuli
- ``HBCD_MMN`` - Mismatch negativity
- ``StatisticalLearning`` - Statistical learning paradigms

Creating Custom Tasks with Config Wizard
-----------------------------------------

For your specific experimental needs, create custom tasks using the web-based Config Wizard:

**Step 1: Open the Config Wizard**

Navigate to https://cincibrainlab.github.io/Autoclean-ConfigWizard/ in your web browser.

**Step 2: Configure Your Task**

1. **Select EEG System** - Choose your hardware (e.g., EGI, BrainVision, BioSemi)
2. **Choose Paradigm** - Select your experimental type (resting, auditory, visual, etc.)
3. **Set Processing Options** - Configure filtering, ICA, epoching parameters
4. **Download Task File** - Get your custom Python task file

**Step 3: Drop Into Workspace**

Use your file manager to save the downloaded task file:

1. Open your file manager (Finder/File Explorer/Files)
2. Navigate to your AutoClean workspace (typically ``~/Documents/AutoClean-EEG``)
3. Drop the downloaded Python file into the ``tasks`` folder
4. Done! AutoClean automatically discovers the new task

**Step 4: Use Your Custom Task**

.. code-block:: python

   from autoclean import Pipeline
   
   pipeline = Pipeline(output_dir="results")
   
   # Your custom task is now available
   pipeline.process_file("data.raw", task="MyCustomTask")

File Management with GUI
-------------------------

Use your system's file manager throughout the process:

**Finding Data Files:**
1. Open Finder/File Explorer/Files
2. Navigate to your EEG data folder
3. Right-click on files/folders to copy paths for Python scripts

**Viewing Results:**
1. Navigate to your output directory
2. Browse timestamped result folders
3. Open quality reports (HTML/PDF files) by double-clicking

**Managing Tasks:**
1. Open workspace ``tasks`` folder in file manager
2. Drag and drop new task files from downloads
3. Remove tasks by moving files to trash

Processing Multiple Files
-------------------------

Use Python loops with file manager navigation for batch processing:

.. code-block:: python

   from pathlib import Path
   from autoclean import Pipeline
   
   # Use file manager to find your data directory
   data_dir = Path("/path/to/data/folder")  # Copy from file manager
   pipeline = Pipeline(output_dir="batch_results")
   
   # Process all .raw files
   for eeg_file in data_dir.glob("*.raw"):
       print(f"Processing: {eeg_file.name}")
       pipeline.process_file(str(eeg_file), task="RestingEyesOpen")
   
   print("Batch processing complete!")

**GUI Workflow for Batch Processing:**
1. Use file manager to navigate to your data folder
2. Note the folder path for your Python script
3. Run the script to process all files
4. Use file manager to browse individual result folders

Complete Example
----------------

Here's a complete workflow from setup to results:

.. code-block:: python

   from pathlib import Path
   from autoclean import Pipeline
   
   # Step 1: Setup (run once)
   import subprocess
   subprocess.run(["autoclean", "setup"])
   
   # Step 2: Create custom task at Config Wizard
   # Visit: https://cincibrainlab.github.io/Autoclean-ConfigWizard/
   # Download task file and drop into workspace/tasks/ folder
   
   # Step 3: Process your data
   data_path = Path("/Users/researcher/EEG_Study/subject01.raw")
   output_path = Path("/Users/researcher/Results")
   
   pipeline = Pipeline(output_dir=str(output_path))
   result = pipeline.process_file(str(data_path), task="MyCustomTask")
   
   # Step 4: View results in file manager
   print(f"Results saved to: {output_path}")
   print("Open the folder in your file manager to view reports!")

This workflow integrates Python processing with familiar file management, making EEG analysis accessible while maintaining the power of programmatic control.

**Real example:**

.. code-block:: bash

   autoclean process RestingEyesOpen subject001_rest.raw

**What you'll see:**
- Welcome message and setup information
- Progress messages as AutoClean works
- "Processing completed successfully!" when done

**How long does it take?**
- Small files (< 10 minutes): 2-5 minutes
- Medium files (10-60 minutes): 5-15 minutes  
- Large files (> 1 hour): 15-30 minutes

**While it's running:**
- Don't close the command window
- You can minimize it and do other work
- Watch for any error messages

📊 Step 6: Find Your Results
----------------------------

**Check where results are saved:**

.. code-block:: bash

   autoclean config show

This shows your workspace location. Your results are in the "output" folder.

**Navigate to your results:**

.. code-block:: bash

   # Go to your workspace output folder
   cd Documents/Autoclean-EEG/output
   
   # See what's there
   ls    # Mac/Linux  
   dir   # Windows

**What you'll find:**

.. code-block::

   output/
   ├── subject001_rest_TIMESTAMP/
   │   ├── bids/                 # Processed data files
   │   ├── logs/                 # Processing logs
   │   ├── metadata/             # Reports and summaries
   │   └── stage/                # Intermediate files

🔍 Step 7: View Your Results
----------------------------

**Open your results folder in file explorer:**

.. code-block:: bash

   # Windows
   explorer Documents\Autoclean-EEG\output
   
   # Mac
   open ~/Documents/Autoclean-EEG/output
   
   # Linux
   xdg-open ~/Documents/Autoclean-EEG/output

**Key files to look at:**

**metadata/run_report.pdf**
   Visual summary of processing results - open this first!

**bids/derivatives/**
   Your cleaned EEG data ready for analysis

**logs/**
   Detailed logs if you need to troubleshoot

📈 Step 8: Understanding Your Results
-------------------------------------

**Quality Control Report (run_report.pdf):**
- Shows before/after data comparison
- Highlights removed artifacts
- Provides data quality metrics
- Red flags any potential issues

**Look for:**
- ✅ Green indicators = good data quality
- ⚠️ Yellow warnings = check these issues  
- ❌ Red errors = data may need attention

**Processed Data Files:**
- Clean continuous EEG data
- Artifact-free epochs (if applicable)
- ICA components and artifact classifications

🆘 Troubleshooting Common Issues
-------------------------------

**"Task not found" error:**

.. code-block:: bash

   # Check available tasks
   autoclean list-tasks
   
   # Make sure you typed the task name exactly

**"File not found" error:**

.. code-block:: bash

   # Check you're in the right folder
   pwd    # Mac/Linux
   cd     # Windows
   
   # List files to see exact names
   ls     # Mac/Linux
   dir    # Windows

**Processing fails with errors:**
- Check the logs folder for detailed error messages
- Ensure your EEG file isn't corrupted
- Try a different task if the current one doesn't fit your data

**No results appear:**
- Check that processing completed successfully
- Look for error messages in the command window
- Verify the output folder location with `autoclean config show`

🎉 Success! What's Next?
------------------------

Congratulations! You've successfully processed your first EEG file with AutoClean.

**Next steps:**

1. **Analyze your results:** Import the cleaned data into your analysis software
2. **Process more files:** Use the same command with different filenames
3. **Learn batch processing:** Process multiple files automatically
4. **Explore custom tasks:** Create workflows specific to your experiments

**Useful follow-up tutorials:**
- :doc:`understanding_results` - Deep dive into what AutoClean produces
- :doc:`creating_custom_task` - Create workflows specific to your experiments
- :doc:`command_line_basics` - Learn more command line skills

💡 Tips for Success
-------------------

**Keep good records:**
- Note which task you used for each experiment type
- Save the processing logs for your records
- Document any custom settings you use

**Start simple:**
- Use built-in tasks when possible
- Process one file first before doing batches
- Review quality control reports carefully

**Get help when needed:**
- Check our troubleshooting guide
- Ask on the community forums
- Contact your lab's technical support

Happy analyzing! 🧠
