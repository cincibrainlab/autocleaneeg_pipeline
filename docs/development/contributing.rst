Contributing
============

We love your input! We want to make contributing to AutoClean EEG as easy and transparent as possible, whether it's:

* Reporting a bug
* Discussing the current state of the code
* Submitting a fix
* Proposing new features
* Becoming a maintainer

Development Process
-------------------

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from ``main``.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

Development Setup
-----------------

1. Clone your fork and install development dependencies:

   ``git clone https://github.com/cincibrainlab/autoclean_pipeline.git``
      Clone the repository locally.

   .. code-block:: bash

      git clone https://github.com/cincibrainlab/autoclean_pipeline.git

   ``cd autoclean_pipeline``
      Enter the repository root.

   .. code-block:: bash

      cd autoclean_pipeline

   ``uv tool install -e --upgrade . --force``
      Install the package as an editable uv tool.

   .. code-block:: bash

      uv tool install -e --upgrade . --force

   ``make install-dev``
      Install contributor tooling used by the repo.

   .. code-block:: bash

      make install-dev

2. Set up pre-commit hooks:

   ``python3 scripts/uv_tools.py run pre-commit install``
      Install the repository pre-commit hooks.

   .. code-block:: bash

      python3 scripts/uv_tools.py run pre-commit install

Code Style
----------

We use several tools to maintain code quality:

* `Black <https://black.readthedocs.io/>`_ for code formatting
* `isort <https://pycqa.github.io/isort/>`_ for import sorting
* `mypy <http://mypy-lang.org/>`_ for static type checking
* `Ruff <https://docs.astral.sh/ruff/>`_ for linting

Run the following before committing:

``black src tests scripts``
   Run Black directly against the main source and test trees.

.. code-block:: bash

   black src tests scripts

``isort src tests scripts``
   Sort imports directly with isort.

.. code-block:: bash

   isort src tests scripts

``mypy src/autoclean``
   Run mypy directly against the package source.

.. code-block:: bash

   mypy src/autoclean

``ruff check src tests scripts``
   Run Ruff directly against the main source and test trees.

.. code-block:: bash

   ruff check src tests scripts

Testing
-------

We use pytest for testing. Run the test suite:

``make test``
   Run the main test target.

.. code-block:: bash

   make test

For coverage report:

``make test-cov``
   Run tests with coverage reporting.

.. code-block:: bash

   make test-cov

Documentation
-------------

We use Sphinx for documentation. Build the docs:

``make docs-setup``
   Install documentation dependencies.

.. code-block:: bash

   make docs-setup

``make docs-build``
   Build the Sphinx docs locally.

.. code-block:: bash

   make docs-build

Pull Request Process
--------------------

1. Update the README.md with details of changes to the interface
2. Update the docs/ with any new documentation
3. Update CHANGELOG.md with a note describing your changes
4. The PR will be merged once you have the sign-off of at least one maintainer

Licensing
---------

Any contributions you make will be under the MIT Software License.

In short, when you submit code changes, your submissions are understood to be under the same MIT License that covers the project. Feel free to contact the maintainers if that's a concern.

Bug Reports
-----------

We use GitHub issues to track public bugs. Report a bug by `opening a new issue <https://github.com/cincibrainlab/autoclean_pipeline/issues/new>`_.

Great Bug Reports tend to have:

* A quick summary and/or background
* Steps to reproduce
   * Be specific!
   * Give sample code if you can.
* What you expected would happen
* What actually happens
* Notes (possibly including why you think this might be happening, or stuff you tried that didn't work) 
