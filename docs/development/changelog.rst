Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[2.0.0] - 06/12/2025
--------------------

**🚨 BREAKING CHANGES**

* Pipeline API: Changed ``autoclean_dir`` parameter to ``output_dir`` in Pipeline constructor
* Configuration: Removed ``autoclean_config`` parameter - YAML configuration no longer required
* Task System: Introduced new Python task file system replacing YAML-based task configuration
* Workspace Management: Complete overhaul of user workspace setup and management
* Task Validation: Simplified requirements - only ``run_id``, ``unprocessed_file``, and ``task`` now required

Added
^^^^^
* Python Task Files: Create custom tasks as Python files with embedded configuration
* Workspace Setup Wizard: Interactive setup for first-time users with automatic workspace creation
* Dynamic Task Discovery: Automatic discovery and registration of custom Python task files
* Export Counter System: Streamlined data export tracking replacing complex stage file management
* Production Deployment: Complete dependency locking with requirements.txt generation
* Enhanced Error Handling: Improved error messages and validation throughout pipeline

Changed
^^^^^^^
* Simplified Architecture: Removed YAML configuration dependencies for built-in tasks
* Modern API Design: Consistent parameter naming across all components
* User Experience: Streamlined workflow for both basic and advanced users
* Test Coverage: Achieved 85.8% test pass rate with comprehensive integration testing
* Code Quality: 100% compliance with Black, isort, and Ruff formatting standards

Fixed
^^^^^
* Test Infrastructure: Resolved 48+ test failures improving reliability from 48% to 85.8% pass rate
* MNE Compatibility: Fixed file naming conventions throughout test suite
* Pipeline Stability: Eliminated initialization errors and API inconsistencies
* Memory Management: Optimized processing workflows for better resource utilization

`[2.0.0] <https://github.com/cincibrainlab/autoclean_pipeline/releases/tag/v2.0.0>`_

[1.2.0] - 03/19/2025
--------------------

* Added robust system for flagging concerning behavior in processing
* Added customized pylossless pipeline function
* Added task for converting .raw to .set files
* Further optimizations and testing for ideal cleaning parameters

`[1.2.0] <https://github.com/cincibrainlab/autoclean_pipeline/releases/tag/v1.2.0>`_

[1.1.0] - 03/3/2025
---------------------

Added
^^^^^
* Modularized import system further using mixins
* Mixins are imported in task base class
* Plugins added for custom import behavior
* Refresh files button to autoclean_review
* Complete documentation site

Deprecated
^^^^^^^^^^
* Most basic step functions

`[1.1.0] <https://github.com/cincibrainlab/autoclean_pipeline/releases/tag/v1.1.0>`_

[1.0.0] - 02/28/2025
---------------------

Added
^^^^^
* Initial release of AutoClean EEG
* Core pipeline functionality
* Support for multiple EEG paradigms
* BIDS-compatible data organization
* Quality control and reporting system
* Database-backed processing tracking
* Task-based modular architecture

`[1.0.0] <https://github.com/cincibrainlab/autoclean_pipeline/releases/tag/v1.0.0>`_ 