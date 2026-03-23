Serve UI Workflow
=================

This tutorial shows the normal way to use AutoClean Serve:

1. create or select the workspace Serve should use from the CLI
2. start Serve with one command
3. finish route setup in the web UI if needed
4. confirm that processing is actually operational

This guide is intentionally CLI-first. The web UI is a first-class operator surface, but the CLI remains the source of truth for setup, status, and operational control.

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

Create a new workspace for Serve:

.. code-block:: bash

   autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace

Use an empty directory for ``--mode new``. If the target directory already contains files, use ``--mode existing`` instead.

Link an existing workspace for Serve:

.. code-block:: bash

   autocleaneeg-pipeline serve workspace --mode existing --path /path/to/serve-workspace

Useful workspace commands:

.. code-block:: bash

   autocleaneeg-pipeline serve workspace status
   autocleaneeg-pipeline serve workspace doctor
   autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace

What to look for:

* ``serve workspace status`` should show the selected workspace root that Serve is using
* ``serve workspace doctor`` should tell you whether the workspace is broken or just not fully configured yet

Step 2: Start Serve the normal way
----------------------------------

Start the UI and normal processing path:

.. code-block:: bash

   autocleaneeg-pipeline serve up

That is the recommended operator command.

For most users, you should not need to start ``serve api`` and ``serve run`` separately.

Check whether Serve is actually operational:

.. code-block:: bash

   autocleaneeg-pipeline serve status

What to look for:

* whether the UI server is running
* whether the dispatcher is running
* whether routes exist
* whether the queue is active, idle, or blocked

Step 3: Create a route
----------------------

You can create the route in the web UI, or directly in the CLI.

CLI route creation example:

.. code-block:: bash

   autocleaneeg-pipeline serve route upsert resting-test \
     --taskfile /path/to/task.py \
     --montage GSN-HydroCel-129 \
     --ingestion-folder /path/to/input-folder \
     --file-glob "*.set" \
     --recursive \
     --enabled

Useful route commands:

.. code-block:: bash

   autocleaneeg-pipeline serve route list
   autocleaneeg-pipeline serve route promote resting-test
   autocleaneeg-pipeline serve route archive resting-test
   autocleaneeg-pipeline serve route unarchive resting-test
   autocleaneeg-pipeline serve route sync

Notes:

* ``route upsert`` is the main create/edit CLI command
* use the web UI if route creation is easier there
* the same operator capability should still exist in the CLI

Step 4: Validate configuration
------------------------------

If you want explicit control, you can validate from the CLI:

.. code-block:: bash

   autocleaneeg-pipeline serve validate --mode test

Validation checks whether the current draft config is usable. It does not publish the draft for processing.

To make processing use the current config, you must apply or deploy it explicitly:

* in the web UI, use the Apply action in Settings
* in the CLI, use the config deployment path that publishes the current draft into ``deploy/``

Step 5: Confirm dispatcher and queue state
------------------------------------------

Dispatcher controls:

.. code-block:: bash

   autocleaneeg-pipeline serve service status
   autocleaneeg-pipeline serve service start
   autocleaneeg-pipeline serve service stop

Queue controls:

.. code-block:: bash

   autocleaneeg-pipeline serve queue status
   autocleaneeg-pipeline serve queue list
   autocleaneeg-pipeline serve queue retry-failed
   autocleaneeg-pipeline serve queue clear-processed
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

.. code-block:: bash

   autocleaneeg-pipeline serve up
   autocleaneeg-pipeline serve status
   autocleaneeg-pipeline serve service status
   autocleaneeg-pipeline serve queue status

Advanced commands
-----------------

.. code-block:: bash

   autocleaneeg-pipeline serve api
   autocleaneeg-pipeline serve run
   autocleaneeg-pipeline serve worker
   autocleaneeg-pipeline serve mode test
   autocleaneeg-pipeline serve mode live
   autocleaneeg-pipeline serve share status

Use these when you are debugging, developing, or deliberately controlling Serve internals.

Troubleshooting
---------------

Serve is up, but nothing is processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run:

.. code-block:: bash

   autocleaneeg-pipeline serve status
   autocleaneeg-pipeline serve service status
   autocleaneeg-pipeline serve queue status

If routes are missing, create one.

If config is not applied, validate it and then apply or deploy it:

.. code-block:: bash

   autocleaneeg-pipeline serve validate --mode test

Validation alone does not make processing operational. The dispatcher should only run from the deployed config.

I am not sure which workspace Serve is using
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run:

.. code-block:: bash

   autocleaneeg-pipeline serve workspace status
   autocleaneeg-pipeline serve workspace doctor

I want to switch to a different Serve workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run:

.. code-block:: bash

   autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace

Task Manager parity notes
-------------------------

For the Serve UI Task Manager, the clean CLI equivalents today are:

.. code-block:: bash

   autocleaneeg-pipeline task create MyCustomTask
   autocleaneeg-pipeline task install RestingState_Basic
   autocleaneeg-pipeline task delete MyCustomTask
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
* ``serve up`` starts the normal operator experience
* ``serve status`` tells you whether Serve is really operational
* ``serve route ...`` manages routes directly from the CLI
* ``serve service ...`` manages the dispatcher
