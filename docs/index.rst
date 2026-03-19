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
* **Serve docs portal**: Start with `Serve Docs Portal <serve-docs-portal.html>`_ if you want a browsable local menu instead of a single standalone guide page
* **Serve UI tutorial**: Read :doc:`serve_ui_workflow` for the CLI-first setup and daily-use workflow
* **Serve tutorial**: Open `First Route Tutorial <serve-first-route-tutorial.html>`_ for the step-by-step workflow from workspace creation through Draft validation and Production promotion
* **Edit tutorial**: Open `Edit Existing Route Tutorial <serve-edit-route-tutorial.html>`_ when a route already exists and you need to change it without casually changing live behavior
* **Recovery tutorial**: Open `Failures and Recovery Tutorial <serve-failures-recovery-tutorial.html>`_ when validation fails, a route is missing, the wrong files are matched, or Production no longer feels safe
* **Handoff checklist**: Open `Operator Handoff Checklist <serve-operator-handoff-checklist.html>`_ for readiness, daily checks, weekly checks, and escalation guidance
* **Serve operator guide**: Start with `Route-First Serve Guide <serve-route-first-operator-guide.html>`_ for the visual operator workflow: Draft vs Production, route-first setup, edit-vs-create decisions, safe reruns, and the current TUI route actions

🔧 **For Technical Users**  
--------------------------

If you're a programmer or advanced researcher, AutoClean offers:

* **Python integration**: Full API access for custom scripts and Jupyter notebooks
* **Modular architecture**: Build custom processing pipelines with reusable components
* **Advanced customization**: Create sophisticated workflows with mixins and plugins
* **Developer tools**: CLI commands and configuration management
* **Serve devlog**: Open `Route-First Serve Devlog <serve-route-first-devlog.md>`_ for the route registry contract, exact serve route commands, and the current boundaries between backend, TUI, and future setup flows

Key Features
------------
* **Automated Preprocessing**: Intelligent artifact detection and removal
* **Quality Control Reports**: Visual summaries of processing results  
* **User-Friendly Workspace**: Organized file structure in your Documents folder
* **Cross-Platform**: Works on Windows, Mac, and Linux
* **Reproducible**: Consistent results across different users and systems
* **Extensible**: Easy to add new processing methods and experimental paradigms
* **Compliance Ready**: Tamper-proof audit trails and integrity verification for regulated environments

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
   pip install autocleaneeg-pipeline
   
   # Run first-time setup
   autoclean setup
   
   # Process your data (that's it!)
   autoclean process RestingEyesOpen my_eeg_data.raw
   
   # Export audit trail for compliance
   autoclean export-access-log --output audit.jsonl

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
   serve_ui_workflow

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
