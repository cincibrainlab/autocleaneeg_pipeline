Serve Command Reference
=======================

This page documents the current Serve command surface.

Serve command split
-------------------

Use the commands with this split:

- ``autocleaneeg-serve`` for the normal launcher lifecycle
- ``autocleaneeg-pipeline serve ...`` for workspace, routes, validation,
  deployment, queue inspection, dispatcher control, API/TUI/worker commands,
  and advanced operator actions

Normal launcher commands
------------------------

Foreground start:

.. code-block:: bash

   autocleaneeg-serve
   autocleaneeg-serve --path /path/to/serve-workspace
   autocleaneeg-serve --port 8000 --host 127.0.0.1

Daemon lifecycle:

.. code-block:: bash

   autocleaneeg-serve up
   autocleaneeg-serve up --path /path/to/serve-workspace --mode test
   autocleaneeg-serve status
   autocleaneeg-serve restart
   autocleaneeg-serve down

Tunnel sharing:

.. code-block:: bash

   autocleaneeg-serve share
   autocleaneeg-serve share status
   autocleaneeg-serve share setup
   autocleaneeg-serve share stop
   autocleaneeg-serve share clear

Serve workspace commands
------------------------

.. code-block:: bash

   autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace
   autocleaneeg-pipeline serve workspace --mode existing --path /path/to/serve-workspace
   autocleaneeg-pipeline serve workspace status
   autocleaneeg-pipeline serve workspace doctor
   autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace

Serve overview and docs commands
--------------------------------

.. code-block:: bash

   autocleaneeg-pipeline serve list --path /path/to/serve-workspace
   autocleaneeg-pipeline serve docs --port 7933 --host 127.0.0.1

Route registry commands
-----------------------

List routes:

.. code-block:: bash

   autocleaneeg-pipeline serve route list
   autocleaneeg-pipeline serve route list --path /path/to/serve-workspace
   autocleaneeg-pipeline serve route list --mode test
   autocleaneeg-pipeline serve route list --include-archived

Create or update a route:

.. code-block:: bash

   autocleaneeg-pipeline serve route upsert resting-test \
     --path /path/to/serve-workspace \
     --taskfile /path/to/task.py \
     --montage GSN-HydroCel-129 \
     --ingestion-folder /path/to/input-folder \
     --file-glob "*.set" \
     --recursive \
     --enabled

More route actions:

.. code-block:: bash

   autocleaneeg-pipeline serve route promote resting-test --path /path/to/serve-workspace
   autocleaneeg-pipeline serve route archive resting-test --path /path/to/serve-workspace
   autocleaneeg-pipeline serve route unarchive resting-test --path /path/to/serve-workspace
   autocleaneeg-pipeline serve route delete resting-test --workspace /path/to/serve-workspace --force
   autocleaneeg-pipeline serve route sync --path /path/to/serve-workspace

Validation and deployment commands
----------------------------------

.. code-block:: bash

   autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode test
   autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode live
   autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode test
   autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode live

Service loop and runtime commands
---------------------------------

Dispatcher loop:

.. code-block:: bash

   autocleaneeg-pipeline serve run --path /path/to/serve-workspace --mode test
   autocleaneeg-pipeline serve run --path /path/to/serve-workspace --mode live --max-cycles 0 --idle-limit 0
   autocleaneeg-pipeline serve run --path /path/to/serve-workspace --dry-run
   autocleaneeg-pipeline serve run --path /path/to/serve-workspace --no-watch --no-sentinel

TUI, API, and worker:

.. code-block:: bash

   autocleaneeg-pipeline serve tui --path /path/to/serve-workspace --mode test
   autocleaneeg-pipeline serve api --path /path/to/serve-workspace --mode test --api-port 8000
   autocleaneeg-pipeline serve worker --redis-url redis://localhost:6379

Dispatcher control through the running API
------------------------------------------

.. code-block:: bash

   autocleaneeg-pipeline serve service status
   autocleaneeg-pipeline serve service start
   autocleaneeg-pipeline serve service start --max-cycles 0 --idle-limit 0
   autocleaneeg-pipeline serve service stop

Mode switching on a running Serve session
-----------------------------------------

.. code-block:: bash

   autocleaneeg-pipeline serve mode
   autocleaneeg-pipeline serve mode status
   autocleaneeg-pipeline serve mode test
   autocleaneeg-pipeline serve mode live

Queue commands
--------------

.. code-block:: bash

   autocleaneeg-pipeline serve queue status
   autocleaneeg-pipeline serve queue list
   autocleaneeg-pipeline serve queue list --status failed --route-id resting-test --limit 20
   autocleaneeg-pipeline serve queue retry-failed
   autocleaneeg-pipeline serve queue retry-failed /full/path/to/file.set
   autocleaneeg-pipeline serve queue clear-processed
   autocleaneeg-pipeline serve queue remove /full/path/to/file.set

Legacy daemon aliases under ``autocleaneeg-pipeline serve``
------------------------------------------------------------

These mirror the launcher lifecycle, but the normal operator path should prefer
``autocleaneeg-serve``:

.. code-block:: bash

   autocleaneeg-pipeline serve up
   autocleaneeg-pipeline serve down
   autocleaneeg-pipeline serve restart
   autocleaneeg-pipeline serve status
   autocleaneeg-pipeline serve share status

Recommended operator flow
-------------------------

For most operators, the usual sequence is:

.. code-block:: bash

   autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace
   autocleaneeg-serve up
   autocleaneeg-pipeline serve route upsert resting-test \
     --path /path/to/serve-workspace \
     --taskfile /path/to/task.py \
     --montage GSN-HydroCel-129 \
     --ingestion-folder /path/to/input-folder \
     --file-glob "*.set" \
     --recursive \
     --enabled
   autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode test
   autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode test
   autocleaneeg-serve status
   autocleaneeg-pipeline serve service status
   autocleaneeg-pipeline serve queue status

