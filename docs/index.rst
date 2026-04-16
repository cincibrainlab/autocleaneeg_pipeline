AutoClean EEG Pipeline
======================

**AutoClean** provides task-driven EEG preprocessing, review, and operational
workflows for research teams using MNE-Python.

🎯 **For Non-Technical Users**
------------------------------

If you're new to EEG processing or programming, AutoClean provides:

* **Workspace setup**: Configure a local workspace for tasks and outputs
* **Task-driven processing**: Run built-in or workspace-installed tasks from the CLI
* **Serve workflow**: Use the web/API and TUI surfaces for route-based operation
* **Published guides**: Follow the getting-started and tutorial pages for supported workflows
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
* **Audit tooling**: Export and inspect access logs for research and operational review

Why Choose AutoClean?
---------------------

**Consistent Results**: Eliminates variability between researchers and labs by providing standardized, validated processing workflows.

**Multiple entrypoints**: Use the CLI, Python API, TUI, or Serve workflow depending on your deployment style.

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
--------------------

**🎯 For Non-Technical Users (Command Line)**

``uv tool install autocleaneeg-pipeline``
   Install the CLI.

.. code-block:: bash

   uv tool install autocleaneeg-pipeline

``autocleaneeg-pipeline config setup``
   Set up or reconfigure your workspace.

.. code-block:: bash

   autocleaneeg-pipeline config setup

``autocleaneeg-pipeline process RestingEyesOpen my_eeg_data.raw``
   Process one EEG file with a built-in task.

.. code-block:: bash

   autocleaneeg-pipeline process RestingEyesOpen my_eeg_data.raw

``autocleaneeg-pipeline export-access-log --output audit.jsonl``
   Export the audit trail for records or review.

.. code-block:: bash

   autocleaneeg-pipeline export-access-log --output audit.jsonl

Important boundaries
--------------------

- GitHub issues are the public path for bug reports and documentation gaps.
- Optional integrations such as MATLAB, Redis/RQ, and GUI tooling are not part
  of the minimal base workflow.
- Audit logging and export tools can support internal controls, but the
  repository does not claim certification for any regulatory framework by
  itself.
- GitHub Pages publishing comes from the repository workflow on `main`, not a
  manually curated docs branch.
- For Serve, use ``autocleaneeg-serve`` as the normal launcher and
  ``autocleaneeg-pipeline serve ...`` for lower-level operator control.

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
   command_reference
   serve_command_reference
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
