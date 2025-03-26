Installation
============

This guide covers multiple ways to install AutoClean, from basic to advanced installation methods.

Requirements
-----------

AutoClean is compatible with:

- Python 3.10 or higher (Python 3.11 recommended)
- Operating systems: Windows, macOS, and Linux

Dependencies
-----------

AutoClean requires the following core libraries:

- numpy >= 1.20.0
- mne >= 1.0.0
- rich >= 10.0.0
- python-dotenv >= 0.19.0
- pyyaml >= 5.1
- schema >= 0.7.0
- mne-bids >= 0.10
- pandas >= 1.3.0
- and several other dependencies listed in pyproject.toml

Install from GitHub
------------------

Since AutoClean is currently in development, the recommended installation method is directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/cincibrainlab/autoclean_complete.git

Install with optional GUI dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AutoClean has optional dependencies for GUI features:

.. code-block:: bash

    # For GUI features
    pip install git+https://github.com/cincibrainlab/autoclean_complete.git#egg=autoclean[gui]

Development installation
---------------------

For contributors or users who want to install for development:

.. code-block:: bash

    git clone https://github.com/cincibrainlab/autoclean_complete.git
    cd autoclean_complete
    pip install -e .

For development with GUI dependencies:

.. code-block:: bash

    pip install -e ".[gui]"

Docker installation
----------------

AutoClean can also be installed using Docker:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/cincibrainlab/autoclean_complete.git
    cd autoclean_complete
    
    # Build and run the Docker container
    docker-compose up 