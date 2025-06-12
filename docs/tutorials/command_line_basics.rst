Command Line Basics for AutoClean
==================================

Don't worry - you don't need to be a programmer to use AutoClean! This guide will teach you the few simple commands you need to know.

🖥️ What is the Command Line?
----------------------------

The command line (also called "terminal", "command prompt", or "shell") is a way to give instructions to your computer by typing text commands instead of clicking buttons.

**Why use it?**
- It's actually faster once you learn a few commands
- AutoClean can process your data automatically 
- You can repeat the same analysis easily
- It works the same way on Windows, Mac, and Linux

🚀 Opening the Command Line
---------------------------

**Windows:**
1. Press `Windows key + R`
2. Type `cmd` and press Enter
3. A black window opens - this is your command prompt

**Mac:**
1. Press `Command + Space` to open Spotlight
2. Type `terminal` and press Enter
3. A window opens - this is your terminal

**Linux:**
1. Press `Ctrl + Alt + T`
2. Or search for "Terminal" in your applications

📁 Basic Navigation
-------------------

These commands help you move around your computer:

**See where you are:**

.. code-block:: bash

   # Windows
   cd
   
   # Mac/Linux  
   pwd

**See what files are in your current folder:**

.. code-block:: bash

   # Windows
   dir
   
   # Mac/Linux
   ls

**Change to a different folder:**

.. code-block:: bash

   # Go to your Documents folder
   cd Documents
   
   # Go to a specific folder
   cd "path/to/your/folder"
   
   # Go back to the previous folder
   cd ..

**Pro tip:** Use quotes around folder names that have spaces!

🧠 AutoClean Commands You Need to Know
--------------------------------------

Once AutoClean is installed, you only need to remember these commands:

**1. Check if AutoClean is working:**

.. code-block:: bash

   autoclean version

This should show you the AutoClean version number.

**2. Set up your workspace (first time only):**

.. code-block:: bash

   autoclean setup

This runs a clean setup wizard that creates your personal AutoClean workspace in Documents/Autoclean-EEG.

**3. See what tasks are available:**

.. code-block:: bash

   # Built-in tasks only
   autoclean list-tasks
   
   # Include your custom tasks too
   autoclean list-tasks --include-custom

This shows all the processing workflows you can use.

**4. Process your data:**

.. code-block:: bash

   autoclean process RestingEyesOpen my_data_file.raw

Replace "RestingEyesOpen" with your task name and "my_data_file.raw" with your actual file.

**5. Manage your custom tasks:**

.. code-block:: bash

   # Add a custom task
   autoclean task add my_custom_task.py
   
   # List your custom tasks
   autoclean task list
   
   # Remove a custom task
   autoclean task remove MyTaskName

**6. Check your results:**

.. code-block:: bash

   autoclean config show

This shows where your results are saved.

🎯 Step-by-Step: Your First Analysis
------------------------------------

Let's walk through processing your first EEG file:

**Step 1: Open the command line** (see instructions above)

**Step 2: Navigate to your data folder**

If your EEG files are in Documents/EEG_Data:

.. code-block:: bash

   cd Documents/EEG_Data

**Step 3: Check what files are there**

.. code-block:: bash

   # Windows
   dir
   
   # Mac/Linux
   ls

You should see your .raw, .set, or other EEG files listed.

**Step 4: Process your data**

.. code-block:: bash

   autoclean process RestingEyesOpen subject001.raw

Replace "subject001.raw" with your actual filename.

**Step 5: Wait for processing to complete**

You'll see messages showing the progress. When it's done, you'll see "Processing completed successfully!"

**Step 6: Check your results**

.. code-block:: bash

   autoclean config show

This tells you where to find your processed data and reports.

📋 Common File and Folder Names
-------------------------------

**Your data files might be named like:**
- subject001.raw
- participant_01_rest.set  
- data_session1.eeg
- sub-01_task-rest_eeg.raw

**Folder paths you might use:**
- Documents/Research/EEG_Data
- Desktop/Experiment_Data
- C:\Research\Subject_Data (Windows)
- /Users/yourname/Research (Mac)

🆘 What If Something Goes Wrong?
-------------------------------

**"Command not found" error:**
This means AutoClean isn't installed properly. Try:

.. code-block:: bash

   pip install autocleaneeg

**"File not found" error:**
Check that you're in the right folder and the filename is correct:

.. code-block:: bash

   # See what files are available
   dir    # Windows
   ls     # Mac/Linux

**"Permission denied" error:**
Try running the command as administrator (Windows) or with sudo (Mac/Linux).

**AutoClean seems stuck:**
- Wait a few minutes - EEG processing takes time
- Press Ctrl+C to cancel if needed
- Check the error messages for clues

💡 Helpful Tips
---------------

**Use Tab completion:**
Start typing a filename and press Tab - the computer will try to complete it for you!

**Use the up arrow:**
Press the up arrow key to repeat your last command.

**Copy and paste:**
- Windows: Right-click to paste
- Mac: Cmd+V to paste  
- Linux: Ctrl+Shift+V to paste

**Keep a cheat sheet:**
Write down the commands you use most often until you memorize them.

🎉 You're Ready!
----------------

That's all you need to know! With these few commands, you can:
- Navigate to your data
- Process EEG files with AutoClean
- Find your results

The command line becomes easier with practice. Start with these basics and gradually learn more as needed.

**Next steps:**
- Try the :doc:`first_time_processing` tutorial
- Learn about :doc:`understanding_results`
- Explore :doc:`creating_custom_task` for your research