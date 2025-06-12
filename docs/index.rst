AutoClean EEG Pipeline
======================

**AutoClean** makes EEG data processing simple and accessible for everyone - from researchers with no programming experience to advanced developers building custom analysis workflows.

🎯 **For Non-Technical Users**
------------------------------

If you're new to EEG processing or programming, AutoClean provides:

* **One-click setup**: Install and configure your workspace in minutes
* **Web-based task builder**: Create processing pipelines through an intuitive interface  
* **Drop-and-go workflow**: Simply drop your task files and run
* **No coding required**: Process your data with simple commands

🔧 **For Technical Users**  
--------------------------

If you're a programmer or advanced researcher, AutoClean offers:

* **Python integration**: Full API access for custom scripts and Jupyter notebooks
* **Modular architecture**: Build custom processing pipelines with reusable components
* **Advanced customization**: Create sophisticated workflows with mixins and plugins
* **Developer tools**: CLI commands and configuration management

Key Features
------------
* **Automated Preprocessing**: Intelligent artifact detection and removal
* **Quality Control Reports**: Visual summaries of processing results  
* **User-Friendly Workspace**: Organized file structure in your Documents folder
* **Cross-Platform**: Works on Windows, Mac, and Linux
* **Reproducible**: Consistent results across different users and systems
* **Extensible**: Easy to add new processing methods and experimental paradigms

Why Choose AutoClean?
---------------------

**Consistent Results**: Eliminates variability between researchers and labs by providing standardized, validated processing workflows.

**Easy to Use**: Whether you're clicking through a web interface or writing Python code, AutoClean adapts to your preferred way of working.

**Research-Focused**: Built by neuroscientists for neuroscientists, with features that address real research needs and workflows.

Core Concepts
-------------

AutoClean is built around simple, intuitive concepts:

**Workspace**
   Your personal folder (in Documents/Autoclean-EEG) containing all your custom tasks, configuration, and processing results.

**Tasks** 
   Pre-configured processing workflows for specific experiments (e.g., resting-state, auditory experiments). Each task contains all the settings and steps needed to process your data.

**Pipeline**
   The processing engine that takes your data and task, then automatically handles all the complex EEG preprocessing steps.

**Auto-Discovery**
   Simply drop task files into your workspace - AutoClean automatically finds and makes them available for use.

Quick Start Examples
-------------------

**🎯 For Non-Technical Users (Command Line)**

.. code-block:: bash

   # Install AutoClean
   pip install autocleaneeg
   
   # Run first-time setup
   autoclean setup
   
   # Process your data (that's it!)
   autoclean process RestingEyesOpen my_eeg_data.raw

**🔧 For Technical Users (Python)**

.. code-block:: python

   from autoclean import Pipeline

   # Simple usage - uses your workspace automatically  
   pipeline = Pipeline()
   pipeline.process_file("my_data.raw", "RestingEyesOpen")
   
   # Custom output location
   pipeline = Pipeline(output_dir="my_results/")
   pipeline.process_file("my_data.raw", "CustomTask")

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Table of Contents

   self

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
