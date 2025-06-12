EEG Processing Tutorial
=======================

This tutorial demonstrates how to process EEG data using AutoClean's built-in tasks and Python API.

Prerequisites
-------------

- EEG data file (.raw, .set, .eeg, or .bdf format)
- Python 3.10+ with pip installed
- Basic understanding of your experimental paradigm

Supported paradigms include resting state, auditory steady-state response (ASSR), chirp stimuli, and mismatch negativity (MMN).

Installation
------------

Install AutoClean from PyPI:

.. code-block:: bash

   pip install autocleaneeg

Verify the installation:

.. code-block:: bash

   autoclean version

This should display the current version (e.g., "AutoClean EEG Pipeline v2.0.0").

Workspace Setup
---------------

Initialize your workspace directory:

.. code-block:: bash

   autoclean setup

This creates a workspace directory structure with folders for custom tasks, output results, and configuration. The default location is platform-specific (typically in your Documents folder).

Selecting a Processing Task
---------------------------

AutoClean includes built-in tasks for common EEG paradigms:

.. code-block:: bash

   autoclean list-tasks

Available built-in tasks:

**RestingEyesOpen**
   Resting state recordings with eyes open
   
**ChirpDefault**  
   Auditory experiments using chirp stimuli
   
**ASSR**
   Auditory steady-state response experiments
   
**HBCD_MMN**
   Mismatch negativity experiments

**StatisticalLearning**
   Statistical learning paradigms

Select the task that matches your experimental paradigm. If none match exactly, ``RestingEyesOpen`` provides a good general-purpose processing pipeline.

Processing EEG Data
-------------------

AutoClean supports common EEG file formats including .raw, .set, .eeg, .bdf, and .fif files. 

**Command Line Processing**

Process a single file:

.. code-block:: bash

   autoclean process RestingEyesOpen subject01_rest.raw

Process multiple files in a directory:

.. code-block:: bash

   autoclean process RestingEyesOpen data_directory/

**Python API Processing**

.. code-block:: bash

   autoclean process RestingEyesOpen your_file_name.raw

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