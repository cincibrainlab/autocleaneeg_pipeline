Development
============

This guide provides information for developers who want to contribute to or
extend AutoClean EEG.

Setting Up Development Environment
----------------------------------

Prerequisites:

- Python 3.11 to 3.13
- Git

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/cincibrainlab/autoclean_pipeline
      cd autoclean_pipeline

2. Install the package as an editable `uv` tool:

   .. code-block:: bash

      uv tool install -e --upgrade . --force

3. Install contributor tooling:

   .. code-block:: bash

      make install-dev
      python3 scripts/uv_tools.py run pre-commit install

Project Structure
-----------------

The project is organized as follows:

- ``configs/``: Configuration templates
- ``src/autoclean/configkit/``: Config schema helpers and schema exports

- ``src/autoclean/core/``: Core classes and functionality
   - ``pipeline.py``: Main entry point for the API
   - ``task.py``: Base class for all task implementations

- ``src/autoclean/api/``: Serve API endpoints, auth hooks, notifications, and
  shipped runtime web assets

- ``src/autoclean/blocks/``: Optional processing and analysis blocks

- ``src/autoclean/io/``: Import/export helpers
   - ``export.py``: Exporting functions
   - ``import_.py``: Importing functions

- ``src/autoclean/mixins/signal_processing/``: Signal processing related functions

- ``src/autoclean/mixins/viz/``: Visualization related functions
   
- ``src/autoclean/step_functions/``: Modular processing functions
   - ``continuous.py``: Core preprocessing steps
   - ``reports.py``: Post-task reports such as processing log

- ``src/autoclean/plugins/``: Import and event handling plugins
   
- ``src/autoclean/tasks/``: Task implementations and curated built-in tasks

- ``src/autoclean/tui/``: Terminal UI surfaces for Serve and operator workflows
   
- ``src/autoclean/utils/``: Utility functions
   - ``config.py``: Configuration handling
   - ``database.py``: Database operations
   - ``logging.py``: Logging functionality

- ``src/autoclean/tools/``: Additional features for the pipeline
   - ``autoclean_review.py``: Review GUI

Architecture
------------

AutoClean follows a modular architecture with several key components:

1. **Pipeline Class**: Central coordinator that manages configuration, processing, and output.

2. **Task Classes**: Implementations for specific EEG paradigms (resting state, ASSR, etc.).

3. **Step Functions**: Modular processing operations that can be combined into workflows.

4. **Database Tracking**: Database-backed tracking of processing runs.

The architecture uses a combination of:

- **Abstract Base Classes**: For extensibility and consistent interfaces
- **Mixins**: For shared functionality across tasks
- **Asynchronous Processing**: For parallel file processing
- **Python Task Modules**: For reproducible task configuration and workflow logic

Canonical contributor docs
--------------------------

Use these as the main sources of truth:

- ``CONTRIBUTING.md`` for setup, validation, and issue reporting
- ``README.md`` for user-facing install and supported surface guidance
- ``docs/INDEX.md`` for documentation layout and canonical locations

.. toctree::
   :maxdepth: 2
   
   development/contributing
   development/changelog
