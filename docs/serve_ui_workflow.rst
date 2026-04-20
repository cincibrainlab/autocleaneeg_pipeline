Serve UI Workflow
=================

This tutorial shows the normal way to use AutoClean Serve:

1. create or select the workspace Serve should use from the CLI
2. start Serve with one command
3. finish route setup in the web UI if needed
4. confirm that processing is actually operational

This guide is intentionally CLI-first. The web UI is a first-class operator surface, but the CLI remains the source of truth for setup, status, and operational control.

Command model
-------------

Use the Serve commands with this split:

* ``autocleaneeg-serve`` is the normal launcher for daemon lifecycle commands:
  foreground start, ``up``, ``down``, ``restart``, ``status``, and ``share``
* ``autocleaneeg-pipeline serve ...`` is the lower-level control surface for
  workspace setup, route management, validation, deployment, dispatcher
  control, queue inspection, API/TUI/worker commands, and mode switching

Before you start
----------------

You need:

* a working ``autocleaneeg-pipeline`` install
* a workspace directory you want Serve to use
* at least one EEG input folder
* a task file and montage for your route

In examples below, replace these paths with your own:

* workspace: ``/path/to/serve-workspace``
* input folder: ``/path/to/input-folder``
* task file: ``/path/to/task.py``
* route id: ``resting-test``

Step 1: Create or link the workspace Serve should use
-----------------------------------------------------

``autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace``
   Create a new workspace for Serve.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace

Use an empty directory for ``--mode new``. If the target directory already contains files, use ``--mode existing`` instead.

``autocleaneeg-pipeline serve workspace --mode existing --path /path/to/serve-workspace``
   Link an existing directory as the Serve workspace.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace --mode existing --path /path/to/serve-workspace

``autocleaneeg-pipeline serve workspace status``
   Show which workspace Serve is currently using.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace status

``autocleaneeg-pipeline serve workspace doctor``
   Check the selected workspace for structural problems.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace doctor

``autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace``
   Switch Serve to a different workspace.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace

What to look for:

* ``serve workspace status`` should show the selected workspace root that Serve is using
* ``serve workspace doctor`` should tell you whether the workspace is broken or just not fully configured yet

Step 2: Start Serve the normal way
----------------------------------

``autocleaneeg-serve up``
   Start the UI and normal processing path.

.. code-block:: bash

   autocleaneeg-serve up

That is the recommended operator command.

If you want to start Serve through the older alias under the main CLI, persist
the workspace first and then start it:

.. code-block:: bash

   autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace
   autocleaneeg-pipeline serve up

``autocleaneeg-pipeline serve up`` uses the workspace already selected for
Serve and does not accept ``--path``. If you need an explicit-path start in one
command, use ``autocleaneeg-serve up --path /path/to/serve-workspace``.

For most users, you should not need to start ``serve api`` and ``serve run``
separately.

``autocleaneeg-serve status``
   Check whether Serve is actually operational.

.. code-block:: bash

   autocleaneeg-serve status

What to look for:

* whether the UI server is running
* whether the dispatcher is running
* whether routes exist
* whether the queue is active, idle, or blocked

Step 3: Create a route
----------------------

You can create the route in the web UI, or directly in the CLI.

``autocleaneeg-pipeline serve route upsert resting-test --taskfile /path/to/task.py --montage GSN-HydroCel-129 --ingestion-folder /path/to/input-folder --file-glob "*.set" --recursive --enabled``
   Create or update a route from the CLI.

.. code-block:: bash

   autocleaneeg-pipeline serve route upsert resting-test \
     --taskfile /path/to/task.py \
     --montage GSN-HydroCel-129 \
     --ingestion-folder /path/to/input-folder \
     --file-glob "*.set" \
     --recursive \
     --enabled

``autocleaneeg-pipeline serve route list``
   List the routes currently configured in the workspace.

.. code-block:: bash

   autocleaneeg-pipeline serve route list

``autocleaneeg-pipeline serve route promote resting-test``
   Promote one route inside the route set.

.. code-block:: bash

   autocleaneeg-pipeline serve route promote resting-test

``autocleaneeg-pipeline serve route archive resting-test``
   Archive a route without deleting it.

.. code-block:: bash

   autocleaneeg-pipeline serve route archive resting-test

``autocleaneeg-pipeline serve route unarchive resting-test``
   Restore an archived route.

.. code-block:: bash

   autocleaneeg-pipeline serve route unarchive resting-test

``autocleaneeg-pipeline serve route sync``
   Sync route state from workspace files.

.. code-block:: bash

   autocleaneeg-pipeline serve route sync

Notes:

* ``route upsert`` is the main create/edit CLI command
* use the web UI if route creation is easier there
* the same operator capability should still exist in the CLI

Step 4: Validate configuration
------------------------------

If you want explicit control, you can validate from the CLI:

``autocleaneeg-pipeline serve validate --mode test``
   Validate the current draft configuration.

.. code-block:: bash

   autocleaneeg-pipeline serve validate --mode test

Validation checks whether the current draft config is usable. It does not publish the draft for processing.

To make processing use the current config, you must apply or deploy it explicitly:

* in the web UI, use the Apply action in Settings
* in the CLI, use ``autocleaneeg-pipeline serve deploy --mode <test|live>`` to
  publish the current draft into ``deploy/``
* route changes are not live for processing until that apply/deploy step happens

Step 5: Confirm dispatcher and queue state
------------------------------------------

``autocleaneeg-pipeline serve service status``
   Show dispatcher status through the service API.

.. code-block:: bash

   autocleaneeg-pipeline serve service status

``autocleaneeg-pipeline serve service start``
   Start dispatcher processing through the service API.

.. code-block:: bash

   autocleaneeg-pipeline serve service start

``autocleaneeg-pipeline serve service stop``
   Stop dispatcher processing through the service API.

.. code-block:: bash

   autocleaneeg-pipeline serve service stop

``autocleaneeg-pipeline serve queue status``
   Show queue health and activity.

.. code-block:: bash

   autocleaneeg-pipeline serve queue status

``autocleaneeg-pipeline serve queue list``
   List queued files and queue history items.

.. code-block:: bash

   autocleaneeg-pipeline serve queue list

``autocleaneeg-pipeline serve queue retry-failed``
   Retry failed queue items.

.. code-block:: bash

   autocleaneeg-pipeline serve queue retry-failed

``autocleaneeg-pipeline serve queue clear-processed``
   Remove processed items from queue history.

.. code-block:: bash

   autocleaneeg-pipeline serve queue clear-processed

``autocleaneeg-pipeline serve queue remove /full/path/to/file.set``
   Remove one file from the queue by path.

.. code-block:: bash

   autocleaneeg-pipeline serve queue remove /full/path/to/file.set

Step 6: Use the web UI for normal operations
--------------------------------------------

Once Serve is up, open the UI in your browser.

Normal operator flow:

1. confirm the workspace is correct
2. create or inspect routes
3. apply config if needed
4. confirm the dispatcher is running
5. watch Queue, Results, and Exclude from the route-aware UI

Daily-use commands
------------------

``autocleaneeg-serve up``
   Start the normal Serve daemon workflow.

.. code-block:: bash

   autocleaneeg-serve up

``autocleaneeg-serve status``
   Check whether Serve is operational.

.. code-block:: bash

   autocleaneeg-serve status

``autocleaneeg-pipeline serve service status``
   Check dispatcher status.

.. code-block:: bash

   autocleaneeg-pipeline serve service status

``autocleaneeg-pipeline serve queue status``
   Check queue health.

.. code-block:: bash

   autocleaneeg-pipeline serve queue status

Advanced commands
-----------------

``autocleaneeg-pipeline serve api``
   Start only the Serve API process.

.. code-block:: bash

   autocleaneeg-pipeline serve api

``autocleaneeg-pipeline serve run``
   Start only the dispatcher loop.

.. code-block:: bash

   autocleaneeg-pipeline serve run

``autocleaneeg-pipeline serve worker``
   Start only the worker process.

.. code-block:: bash

   autocleaneeg-pipeline serve worker

``autocleaneeg-pipeline serve mode test``
   Switch the running Serve session to test mode.

.. code-block:: bash

   autocleaneeg-pipeline serve mode test

``autocleaneeg-pipeline serve mode live``
   Switch the running Serve session to live mode.

.. code-block:: bash

   autocleaneeg-pipeline serve mode live

``autocleaneeg-pipeline serve share status``
   Show sharing status through the Serve CLI family.

.. code-block:: bash

   autocleaneeg-pipeline serve share status

Use these when you are debugging, developing, or deliberately controlling Serve internals.

Troubleshooting
---------------

Serve is up, but nothing is processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run:

``autocleaneeg-serve status``
   Check high-level Serve status first.

.. code-block:: bash

   autocleaneeg-serve status

``autocleaneeg-pipeline serve service status``
   Confirm whether the dispatcher is running.

.. code-block:: bash

   autocleaneeg-pipeline serve service status

``autocleaneeg-pipeline serve queue status``
   Inspect whether files are queued, blocked, or idle.

.. code-block:: bash

   autocleaneeg-pipeline serve queue status

If routes are missing, create one.

If config is not applied, validate it and then apply or deploy it:

``autocleaneeg-pipeline serve validate --mode test``
   Validate the current draft without deploying it.

.. code-block:: bash

   autocleaneeg-pipeline serve validate --mode test

Validation alone does not make processing operational.

For the normal web UI and API service path, the dispatcher only starts from the deployed config.

The TUI service screen can still run against the operator config for a deliberate Draft-only test path when no deployed config exists yet, but that is an advanced exception rather than the normal operator workflow.

Serve workspace setup fails with ``UnknownIssuer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace``
fails while ``uv pip install`` is running and the error mentions
``invalid peer certificate: UnknownIssuer``, rerun the setup with uv's system
TLS trust enabled:

.. code-block:: bash

   export UV_NATIVE_TLS=1
   autocleaneeg-pipeline serve workspace --mode existing --path /path/to/serve-workspace

Use ``--mode existing`` on the retry because the first attempt already created
the workspace directory structure. This is common on managed networks or
machines that require a custom root CA in the system trust store.

I am not sure which workspace Serve is using
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run:

``autocleaneeg-pipeline serve workspace status``
   Show the workspace currently selected for Serve.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace status

``autocleaneeg-pipeline serve workspace doctor``
   Check whether the selected workspace is healthy.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace doctor

I want to switch to a different Serve workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run:

``autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace``
   Switch Serve to a different workspace.

.. code-block:: bash

   autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace

Task Manager parity notes
-------------------------

For the Serve UI Task Manager, the clean CLI equivalents today are:

``autocleaneeg-pipeline task create MyCustomTask``
   Create a new task in the active workspace.

.. code-block:: bash

   autocleaneeg-pipeline task create MyCustomTask

``autocleaneeg-pipeline task install RestingState_Basic``
   Install a task into the active workspace.

.. code-block:: bash

   autocleaneeg-pipeline task install RestingState_Basic

``autocleaneeg-pipeline task delete MyCustomTask``
   Delete a task from the active workspace.

.. code-block:: bash

   autocleaneeg-pipeline task delete MyCustomTask

``autocleaneeg-pipeline task update``
   Refresh task metadata used by the task browser and manager.

.. code-block:: bash

   autocleaneeg-pipeline task update

Notes:

* when Serve is pointed at a workspace, ``task create`` writes into that workspace's ``tasks/`` directory
* ``task install`` and ``task delete`` also operate on that same workspace task directory
* ``task update`` refreshes the shared task registry metadata used by the task browser and manager
* ``task use``, ``task sync``, ``task diff``, ``task diagnose``, and ``task search`` now follow the active operator task workspace too
* legacy active-task behavior only matters when you are working outside the Serve operator flow

Recommended mental model
------------------------

Use this simple model:

* ``serve workspace`` sets up or selects the normal workspace root that Serve uses
* ``autocleaneeg-serve up`` starts the normal operator experience
* ``autocleaneeg-serve status`` tells you whether Serve is really operational
* ``serve route ...`` manages routes directly from the CLI
* ``serve service ...`` manages the dispatcher
