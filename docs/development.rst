Development
============

This guide provides information for developers who want to contribute to or extend AutoClean EEG.

Setting Up Development Environment
---------------------------------

Prerequisites:

- Python 3.10 or higher
- Git

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/cincibrainlab/autoclean_pipeline
      cd autoclean_pipeline

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

Project Structure
----------------

The project is organized as follows:

- ``configs/``: Configuration templates

- ``src/autoclean/core/``: Core classes and functionality
   - ``pipeline.py``: Main entry point for the API
   - ``task.py``: Base class for all task implementations

- ``src/autoclean/io/``: Modular processing functions
   - ``export.py``: Exporting functions
   - ``import.py``: Importing functions

- ``src/autoclean/mixins/signal_processing/``: Signal processing related functions

- ``src/autoclean/mixins/viz/``: Visualization related functions
   
- ``src/autoclean/step_functions/``: Modular processing functions
   - ``continuous.py``: Core preprocessing steps
   - ``reports.py``: Post-task reports such as processing log

- ``src/autoclean/plugins/``: Import and event handling plugins
   
- ``src/autoclean/tasks/``: Task implementations
   - ``resting_eyes_open.py``: Resting state task
   - ``assr_default.py``: ASSR task
   - And others...
   
- ``src/autoclean/utils/``: Utility functions
   - ``config.py``: Configuration handling
   - ``database.py``: Database operations
   - ``logging.py``: Logging functionality

- ``src/autoclean/tools/``: Additional features for the pipeline
   - ``autoclean_review.py``: Review GUI

Architecture
-----------

AutoClean follows a modular architecture with several key components:

1. **Pipeline Class**: Central coordinator that manages configuration, processing, and output.

2. **Task Classes**: Implementations for specific EEG paradigms (resting state, ASSR, etc.).

3. **Step Functions**: Modular processing operations that can be combined into workflows.

4. **Database Tracking**: Database-backed tracking of processing runs.

The architecture uses a combination of:

- **Abstract Base Classes**: For extensibility and consistent interfaces
- **Mixins**: For shared functionality across tasks
- **Asynchronous Processing**: For parallel file processing
- **YAML Configuration**: For reproducible processing parameters

.. toctree::
   :maxdepth: 2
   
   development/contributing
   development/changelog


