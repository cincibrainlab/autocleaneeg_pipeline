Troubleshooting Guide
====================

This guide helps you solve common issues with AutoClean. Most problems can be resolved quickly by following these steps.

🔧 Installation Issues
----------------------

**"autoclean command not found"**

.. code-block:: bash

   # Check if AutoClean is installed
   pip list | grep autocleaneeg-pipeline
   
   # If not found, install it
   pip install autocleaneeg-pipeline
   
   # Some systems need pip3
   pip3 install autocleaneeg-pipeline
   
   # Verify installation
   autoclean version

**"Permission denied" errors**

.. code-block:: bash

   # On Windows: Run as administrator
   # Right-click Command Prompt → "Run as administrator"
   
   # On Mac/Linux: Use sudo (be careful!)
   sudo pip install autocleaneeg-pipeline

**Python/pip not found**

- Install Python from python.org
- Make sure Python is added to your system PATH
- Restart your command prompt after installing Python

⚙️ Setup and Workspace Issues
-----------------------------

**Setup wizard not starting**

.. code-block:: bash

   # Force reconfiguration
   autoclean setup
   
   # Check current configuration
   autoclean config show

**Workspace deleted or moved**

AutoClean detects workspace deletion automatically. When you run any command, it will offer to recreate your workspace:

.. code-block:: text

   ⚠ Previous workspace no longer exists
   
   🔧 Workspace Setup
   
   Workspace location: /Users/yourname/Documents/Autoclean-EEG
   • Custom tasks  • Configuration  • Results  • Easy backup

**Can't find workspace location**

.. code-block:: bash

   # Show current workspace location
   autoclean config show
   
   # Reset to default location
   autoclean config reset --confirm

📁 File and Data Issues
-----------------------

**"File not found" errors**

.. code-block:: bash

   # Check you're in the right directory
   pwd     # Mac/Linux
   cd      # Windows (shows current directory)
   
   # List files to see exact names
   ls      # Mac/Linux
   dir     # Windows
   
   # Use full file path if needed
   autoclean process RestingEyesOpen "/full/path/to/your/file.raw"

**Unsupported file format**

AutoClean supports: .raw, .set, .eeg, .bdf, .fif, .cnt, .vhdr

If your file isn't supported:
- Convert to a supported format using your acquisition software
- Export as .set from EEGLAB
- Export as .fif from MNE-Python

**File corruption issues**

.. code-block:: bash

   # Check file integrity
   # Try opening the file in your original software first
   
   # If file opens elsewhere but fails in AutoClean:
   # Check the logs for detailed error messages
   autoclean config show
   # Look in output/*/logs/ folder

🎯 Task and Processing Issues
----------------------------

**"Task not found" errors**

.. code-block:: bash

   # List available tasks
   autoclean list-tasks
   
   # Include custom tasks
   autoclean list-tasks --include-custom
   
   # Check exact spelling (case-sensitive!)
   # Use: RestingEyesOpen
   # Not: restingeyesopen or Resting_Eyes_Open

**Custom task not discovered**

.. code-block:: bash

   # Check task file is in correct location
   autoclean config show
   ls ~/Documents/Autoclean-EEG/tasks/
   
   # Verify Python syntax
   python -m py_compile your_task_file.py
   
   # Check class inherits from Task
   grep "class.*Task" ~/Documents/Autoclean-EEG/tasks/your_task.py

**Processing fails or hangs**

.. code-block:: bash

   # Check available memory (EEG files can be large)
   # Close other programs if needed
   
   # Try with a smaller file first
   autoclean process RestingEyesOpen small_test_file.raw
   
   # Check logs for error details
   # Look in workspace/output/*/logs/

**Poor processing results**

1. **Review quality control reports**: Check metadata/run_report.pdf
2. **Try different task**: Some tasks work better for specific data types
3. **Check data quality**: Ensure original data isn't too noisy
4. **Verify experimental paradigm**: Make sure you're using the right task type

⚡ Performance Issues
--------------------

**Processing takes too long**

- **Large files**: Normal for files >1GB - can take 30+ minutes
- **Insufficient RAM**: Close other programs, ensure 8GB+ available
- **Slow storage**: Move data to faster drive (SSD vs HDD)

**Out of memory errors**

.. code-block:: bash

   # Check available memory
   # Windows: Task Manager → Performance → Memory
   # Mac: Activity Monitor → Memory
   # Linux: free -h
   
   # Solutions:
   # 1. Close other programs
   # 2. Process smaller segments
   # 3. Use a machine with more RAM

**Disk space issues**

AutoClean creates several copies of your data during processing:

.. code-block:: bash

   # Check available space
   df -h    # Mac/Linux
   dir      # Windows (shows drive space)
   
   # Clean up old results if needed
   # Delete old output folders you don't need

🔍 Output and Results Issues
---------------------------

**No output files generated**

.. code-block:: bash

   # Check processing completed successfully
   # Look for "Processing completed successfully!" message
   
   # Check output directory
   autoclean config show
   ls ~/Documents/Autoclean-EEG/output/
   
   # Check logs for errors
   # Look in latest output/*/logs/ folder

**Can't open result files**

- **PDF reports**: Need PDF viewer (Adobe, Chrome, etc.)
- **.set files**: Open with EEGLAB in MATLAB
- **.fif files**: Use MNE-Python
- **Missing software**: Install required analysis software

**Results look wrong**

1. **Check data quality**: Review run_report.pdf first
2. **Verify task choice**: Ensure task matches your experiment type
3. **Check parameters**: May need custom task with different settings
4. **Compare with original**: Is original data good quality?

💻 Command Line Issues
---------------------

**Commands not working**

.. code-block:: bash

   # Make sure you're typing commands exactly
   # AutoClean is case-sensitive
   
   # Copy/paste commands to avoid typos
   
   # Check you're in the right directory
   pwd     # Shows current location

**Permission errors**

.. code-block:: bash

   # Don't run as administrator unless necessary
   # AutoClean should work with normal user permissions
   
   # If needed on Mac/Linux:
   sudo autoclean setup

**Terminal/Command prompt closes**

- Don't close the window while processing is running
- If it closes unexpectedly, restart and run autoclean config show to find partial results

🌐 Network and Environment Issues
---------------------------------

**Conda environment issues**

.. code-block:: bash

   # If using conda/miniconda
   conda activate your_environment
   pip install autocleaneeg-pipeline
   
   # Or install in conda directly
   conda install -c conda-forge autocleaneeg-pipeline

**Corporate firewall blocking installation**

- Contact IT department for assistance
- May need to use proxy settings or internal package repository
- Alternative: download offline installer from IT

🆘 Getting More Help
--------------------

**Still having problems?**

1. **Check error messages carefully**: Often they tell you exactly what's wrong
2. **Look at log files**: Detailed information in output/*/logs/
3. **Try with test data**: Confirm AutoClean works with known-good files
4. **Update AutoClean**: New versions fix common issues

.. code-block:: bash

   # Update to latest version
   pip install --upgrade autocleaneeg-pipeline

**Report bugs or ask for help:**

- GitHub Issues: Report specific bugs with error messages
- Community Forums: Ask questions and share solutions
- Documentation: Check other tutorial sections

**What to include when asking for help:**

1. **Error message**: Copy/paste exact error text
2. **Command used**: What exactly did you type?
3. **File type**: What format is your EEG data?
4. **System info**: Windows/Mac/Linux, Python version
5. **AutoClean version**: Output of `autoclean version`

🧹 Clean Installation
---------------------

**Start fresh if nothing works:**

.. code-block:: bash

   # Uninstall AutoClean
   pip uninstall autocleaneeg-pipeline
   
   # Clear pip cache
   pip cache purge
   
   # Reinstall
   pip install autocleaneeg-pipeline
   
   # Reset configuration
   autoclean config reset --confirm

**Complete reset:**

1. Uninstall AutoClean (above)
2. Delete workspace folder: Documents/Autoclean-EEG
3. Delete config folder:
   - Windows: %APPDATA%\autoclean\autoclean
   - Mac: ~/Library/Application Support/autoclean/autoclean  
   - Linux: ~/.config/autoclean/autoclean
4. Reinstall and reconfigure

💡 Prevention Tips
------------------

**Avoid common problems:**

- **Keep backups**: Copy important results before processing new data
- **Test first**: Try new tasks on small test files
- **Document settings**: Keep notes on what tasks work for your data
- **Regular updates**: Update AutoClean occasionally for bug fixes
- **Stable environment**: Don't change Python/conda environments mid-project

**Best practices:**

- Use descriptive filenames for your EEG data
- Keep data organized in clear folder structures
- Process one file before doing batch operations
- Review quality control reports after each processing run

Remember: Most issues are simple fixes! Check the error message, verify your file paths, and make sure you're using the right task for your data type.
