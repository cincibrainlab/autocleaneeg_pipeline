Command Reference
=================

This page documents the current ``autocleaneeg-pipeline`` command surface.
Examples use common paths and task names; replace them with values from your
own environment.

Command model
-------------

Use these public entrypoints:

- ``autocleaneeg-pipeline`` for processing, workspace, task, config, events,
  reports, and Serve control commands
- ``autocleaneeg-serve`` for the normal Serve launcher lifecycle
- ``autocleaneeg-tui`` for the standalone TUI entrypoint when applicable

For Serve-specific commands, see :doc:`serve_command_reference`.

Global help and status
----------------------

.. code-block:: bash

   autocleaneeg-pipeline --help
   autocleaneeg-pipeline help
   autocleaneeg-pipeline help serve
   autocleaneeg-pipeline version
   autocleaneeg-pipeline tutorial

Workspace bootstrap
-------------------

.. code-block:: bash

   autocleaneeg-pipeline wizard
   autocleaneeg-pipeline config setup
   autocleaneeg-pipeline config show
   autocleaneeg-pipeline workspace show
   autocleaneeg-pipeline workspace set /path/to/workspace
   autocleaneeg-pipeline workspace default
   autocleaneeg-pipeline workspace explore
   autocleaneeg-pipeline workspace size
   autocleaneeg-pipeline workspace cd --print zsh

Processing commands
-------------------

Simple processing:

.. code-block:: bash

   autocleaneeg-pipeline process RestingEyesOpen /path/to/file.raw
   autocleaneeg-pipeline process --task RestingEyesOpen --file /path/to/file.raw
   autocleaneeg-pipeline process --task-file /path/to/MyTask.py --file /path/to/file.raw
   autocleaneeg-pipeline process --task RestingEyesOpen --dir /path/to/data --format "*.set" --recursive
   autocleaneeg-pipeline process --dry-run RestingEyesOpen /path/to/file.raw

ICA control-sheet application:

.. code-block:: bash

   autocleaneeg-pipeline process ica --metadata-dir /path/to/run/metadata

Task discovery and management
-----------------------------

List and inspect tasks:

.. code-block:: bash

   autocleaneeg-pipeline list-tasks
   autocleaneeg-pipeline list-tasks --overrides
   autocleaneeg-pipeline task list
   autocleaneeg-pipeline task list --source builtin
   autocleaneeg-pipeline task list --status outdated
   autocleaneeg-pipeline task search resting
   autocleaneeg-pipeline task diagnose
   autocleaneeg-pipeline task diff RestingEyesOpen

Install, create, copy, edit, and delete:

.. code-block:: bash

   autocleaneeg-pipeline task create MyCustomTask
   autocleaneeg-pipeline task install /path/to/MyCustomTask.py --source file --activate
   autocleaneeg-pipeline task install RestingEyesOpen --source builtin --activate
   autocleaneeg-pipeline task use RestingEyesOpen
   autocleaneeg-pipeline task copy RestingEyesOpen --name RestingEyesOpenCopy
   autocleaneeg-pipeline task edit MyCustomTask
   autocleaneeg-pipeline task delete MyCustomTask
   autocleaneeg-pipeline task explore

Active task and schema commands:

.. code-block:: bash

   autocleaneeg-pipeline task set RestingEyesOpen
   autocleaneeg-pipeline task show
   autocleaneeg-pipeline task unset
   autocleaneeg-pipeline task schema export --output task-schema.json
   autocleaneeg-pipeline task sync
   autocleaneeg-pipeline task sync --update
   autocleaneeg-pipeline task update

Montage commands
----------------

.. code-block:: bash

   autocleaneeg-pipeline montage list
   autocleaneeg-pipeline montage set GSN-HydroCel-129
   autocleaneeg-pipeline montage test

Block commands
--------------

.. code-block:: bash

   autocleaneeg-pipeline blocks list
   autocleaneeg-pipeline blocks info matlab_fooof
   autocleaneeg-pipeline blocks deps matlab_fooof
   autocleaneeg-pipeline blocks update
   autocleaneeg-pipeline blocks install matlab_fooof
   autocleaneeg-pipeline blocks install --locked
   autocleaneeg-pipeline blocks lock --output blocks.lock

Input and source commands
-------------------------

Preferred input commands:

.. code-block:: bash

   autocleaneeg-pipeline input set /path/to/file.raw
   autocleaneeg-pipeline input show
   autocleaneeg-pipeline input unset

Deprecated source alias:

.. code-block:: bash

   autocleaneeg-pipeline source set /path/to/file.raw
   autocleaneeg-pipeline source show
   autocleaneeg-pipeline source unset

Event inspection commands
-------------------------

.. code-block:: bash

   autocleaneeg-pipeline events discover /path/to/file.set
   autocleaneeg-pipeline events discover /path/to/file.xdat --montage GSN-HydroCel-129
   autocleaneeg-pipeline events analyze /path/to/file.set --gap-threshold 45
   autocleaneeg-pipeline events epochs /path/to/file-epo.fif

Configuration commands
----------------------

.. code-block:: bash

   autocleaneeg-pipeline config show
   autocleaneeg-pipeline config setup
   autocleaneeg-pipeline config reset --confirm
   autocleaneeg-pipeline config export /path/to/export-dir
   autocleaneeg-pipeline config import /path/to/export-dir

Audit and cleanup commands
--------------------------

.. code-block:: bash

   autocleaneeg-pipeline export-access-log --output audit.jsonl
   autocleaneeg-pipeline export-access-log --format csv --output audit.csv
   autocleaneeg-pipeline export-access-log --verify-only
   autocleaneeg-pipeline clean-task RestingEyesOpen --dry-run
   autocleaneeg-pipeline clean-task RestingEyesOpen --force

GUI and review commands
-----------------------

.. code-block:: bash

   autocleaneeg-pipeline review --output /path/to/output
   autocleaneeg-pipeline exclude /path/to/run-or-exports
   autocleaneeg-pipeline view /path/to/file.raw
   autocleaneeg-pipeline view /path/to/file.raw --no-view

Reporting commands
------------------

.. code-block:: bash

   autocleaneeg-pipeline report create --run-id RUN123 --context-json /path/to/context.json --out-dir /path/to/reports
   autocleaneeg-pipeline report chat --context-json /path/to/context.json

Authentication and compliance commands
--------------------------------------

Direct auth helpers:

.. code-block:: bash

   autocleaneeg-pipeline login
   autocleaneeg-pipeline logout
   autocleaneeg-pipeline whoami
   autocleaneeg-pipeline auth0-diagnostics

Grouped auth commands:

.. code-block:: bash

   autocleaneeg-pipeline auth login
   autocleaneeg-pipeline auth logout
   autocleaneeg-pipeline auth whoami
   autocleaneeg-pipeline auth diagnostics
   autocleaneeg-pipeline auth setup
   autocleaneeg-pipeline auth enable
   autocleaneeg-pipeline auth disable

MATLAB commands
---------------

.. code-block:: bash

   autocleaneeg-pipeline matlab doctor
   autocleaneeg-pipeline matlab doctor --skip-start
   autocleaneeg-pipeline matlab test-engine

Settings commands
-----------------

.. code-block:: bash

   autocleaneeg-pipeline settings theme
   autocleaneeg-pipeline settings theme dark
   autocleaneeg-pipeline settings theme --clear

Recommended everyday commands
-----------------------------

If you only need the common workflow, start here:

.. code-block:: bash

   autocleaneeg-pipeline version
   autocleaneeg-pipeline config setup
   autocleaneeg-pipeline list-tasks
   autocleaneeg-pipeline process RestingEyesOpen /path/to/file.raw
   autocleaneeg-pipeline review --output /path/to/output

