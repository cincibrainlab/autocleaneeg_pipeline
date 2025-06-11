Your First EEG Analysis with AutoClean
=======================================

This tutorial walks you through processing your first EEG file with AutoClean, from installation to viewing results. Perfect for beginners!

🎯 What You'll Learn
--------------------

By the end of this tutorial, you'll know how to:
- Install and set up AutoClean
- Choose the right task for your experiment
- Process an EEG file
- Find and understand your results
- Troubleshoot common issues

📋 What You'll Need
-------------------

**Before starting:**
- An EEG data file (.raw, .set, .eeg, or .bdf format)
- Basic knowledge of what experiment the data is from
- About 15 minutes

**Common experiment types we'll cover:**
- Resting state (eyes open/closed)
- Auditory experiments (tones, speech, music)
- Visual experiments  
- Cognitive tasks

🚀 Step 1: Install AutoClean
----------------------------

If you haven't installed AutoClean yet:

.. code-block:: bash

   pip install autocleaneeg

**Troubleshooting installation:**
- If you get "pip not found", you may need to install Python first
- Ask your IT department for help if needed
- Some systems use `pip3` instead of `pip`

✅ **Test your installation:**

.. code-block:: bash

   autoclean version

You should see something like "AutoClean EEG Pipeline v1.4.1"

⚙️ Step 2: Set Up Your Workspace
--------------------------------

This creates your personal AutoClean folder where everything will be organized:

.. code-block:: bash

   autoclean setup

**What happens:**
- Creates a folder called "Autoclean-EEG" in your Documents
- Sets up the folder structure for tasks and results
- Configures AutoClean for your system

**If asked where to put your workspace:**
- Press Enter to use the default (Documents/Autoclean-EEG)
- Or type a custom path if you prefer

🎯 Step 3: Choose Your Task
---------------------------

Different experiments need different processing approaches. AutoClean includes several built-in tasks:

**See what's available:**

.. code-block:: bash

   autoclean list-tasks

**Built-in tasks you'll see:**

**RestingEyesOpen**
   For resting state recordings where participants keep their eyes open
   
**ChirpDefault**  
   For auditory experiments using chirp stimuli
   
**AssrDefault**
   For auditory steady-state response experiments
   
**HBCD_MMN**
   For mismatch negativity experiments

**How to choose:**

1. **Resting state data?** → Use "RestingEyesOpen"
2. **Auditory experiment?** → Try "ChirpDefault" or "AssrDefault"  
3. **Not sure?** → Start with "RestingEyesOpen" (works for most data)
4. **Need something custom?** → See our task creation guides later

📁 Step 4: Locate Your Data File
--------------------------------

**Find your EEG file:**

.. code-block:: bash

   # Navigate to where your data is stored
   cd Documents/My_EEG_Data
   
   # See what files are there
   ls    # Mac/Linux
   dir   # Windows

**Common file extensions:**
- `.raw` - Continuous EEG data
- `.set` - EEGLAB format  
- `.eeg` - BrainVision format
- `.bdf` - BioSemi format
- `.fif` - MNE format

**Example filenames you might see:**
- subject001_rest.raw
- participant_01.set
- sub-01_task-rest_eeg.raw

🎬 Step 5: Process Your Data
----------------------------

Now for the magic! This single command processes your entire EEG file:

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

   # Go to your workspace
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
- :doc:`batch_processing_datasets` - Process multiple files efficiently  
- :doc:`quality_control_best_practices` - Ensure reliable results

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