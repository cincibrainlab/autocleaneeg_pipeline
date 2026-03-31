MATLAB Integration
==================

AutoClean supports MATLAB-backed tasks and bundled blocks without making MATLAB a core dependency for every install.

When to Use This
----------------

Use this guide if you need any of the following:

- ``autocleaneeg-pipeline matlab doctor``
- MATLAB-backed route execution in serve mode
- thin Python wrappers that call MATLAB functions
- the bundled MATLAB FOOOF block

Install Model
-------------

Base AutoClean install remains separate from MATLAB enablement.

Base install:

.. code-block:: bash

   pip install autocleaneeg-pipeline

MATLAB-enabled install:

1. Install a MATLAB release that matches the Python architecture you plan to use.
2. Create and activate the AutoClean environment that will run both the CLI and serve workers.
3. Install MATLAB Engine into that same environment from the MATLAB engine source tree or a compatible ``matlabengine`` release.

Recommended local workflow on this repo:

.. code-block:: bash

   arch -x86_64 python3 -m venv .venv
   arch -x86_64 .venv/bin/python /path/to/MATLAB/extern/engines/python/setup.py install

Important rules:

- Do not assume ``pip install autocleaneeg-pipeline`` alone enables MATLAB.
- Do not assume ``uv tool install`` is enough for MATLAB-backed routes.
- Use one MATLAB-capable interpreter for the CLI, worker, and engine.

Compatibility Caveats
---------------------

MATLAB Engine is sensitive to all of the following:

- Python version
- Python architecture
- MATLAB release
- MATLAB architecture
- local license configuration

On this project, the validated working path is:

- Intel MATLAB ``R2025b`` under Rosetta
- x86_64 Python in ``.venv``
- MATLAB Engine installed into that same environment

If the architectures do not match, ``matlab doctor`` commonly fails with missing ``maca64`` or ``maci64`` engine directories.

Readiness Checks
----------------

Inspect the current interpreter:

.. code-block:: bash

   autocleaneeg-pipeline matlab doctor

Smoke-test the engine:

.. code-block:: bash

   autocleaneeg-pipeline matlab test-engine

The doctor output now reports:

- install mode
- engine package presence and version
- MATLAB root and binary
- whether startup was verified
- whether the current interpreter is suitable for MATLAB-backed routes
- remediation guidance for common failure modes

Task Config Examples
--------------------

Thin wrapper call from Python:

.. code-block:: python

   from autoclean.functions.matlab import call_matlab

   sqrt_16 = call_matlab("sqrt", 16.0, startup_options="")

Config-driven function call:

.. code-block:: python

   config = {
       "apply_matlab": {
           "enabled": True,
           "value": {
               "kind": "function",
               "entrypoint": "write_probe",
               "args": ["/tmp/output.json"],
               "paths": ["./matlab"],
               "startup_options": "",
               "startup_timeout_seconds": 60.0,
               "nargout": 0,
           },
       }
   }

Config-driven script execution:

.. code-block:: python

   config = {
       "run_matlab": {
           "enabled": True,
           "value": {
               "kind": "script",
               "entrypoint": "./matlab/my_script.m",
               "paths": ["./matlab"],
               "startup_options": "",
               "startup_timeout_seconds": 60.0,
           },
       }
   }

Route Example
-------------

MATLAB-backed Python task file:

.. code-block:: python

   from autoclean.mixins.utils.matlab import MatlabExecutionMixin


   class MatlabTask(MatlabExecutionMixin):
       def run(self):
           self.execute_matlab_step("apply_matlab")

Serve behavior:

- route parsing detects that the task requires MATLAB
- the worker runs ``matlab doctor`` in the target runtime before full processing
- route execution fails fast if the runtime is not MATLAB-capable

Bundled MATLAB FOOOF Block
--------------------------

The bundled FOOOF block lives under:

- ``src/autoclean/blocks/analysis/matlab_fooof``

It writes outputs under:

- ``derivatives/matlab/fooof/<subject>/``

Expected artifacts include:

- manifest JSON
- summary CSV
- aperiodic CSV
- native MATLAB output folder

Output and Provenance
---------------------

MATLAB-backed steps follow the same broad provenance rules as other AutoClean steps:

- route outputs go into predictable route/task derivative folders
- block metadata is written back through the task metadata update path
- preflight failures surface in route worker results and logs

Troubleshooting
---------------

Common failures and actions:

``MATLAB Engine API unavailable``
   Install the engine into the same interpreter that runs AutoClean.

``Could not find directory .../maca64`` or ``.../maci64``
   Python and MATLAB architectures do not match.

``Remote MVMs are disabled for this session``
   Avoid ``mwpython`` for this project workflow. Use the validated x86_64 AutoClean ``.venv`` interpreter with MATLAB runtime paths available.

License errors
   Verify the active MATLAB license or pass ``--license-file`` to the doctor/test commands.

Serve worker can process non-MATLAB routes but fails MATLAB routes
   The worker runtime is not the same interpreter that has MATLAB Engine installed.

Migration Notes
---------------

If you previously ran ad hoc MATLAB scripts from ``temp/``:

- move reusable code into repo-local parameterized ``.m`` assets
- prefer function entrypoints for bundled blocks
- keep task wrappers thin and route-safe
- use ``apply_matlab`` / ``run_matlab`` config blocks or the bundled MATLAB FOOOF block instead of hard-coded one-off paths
