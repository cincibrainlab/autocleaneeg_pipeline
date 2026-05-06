Serve Command Reference
=======================

This page documents the current Serve command surface. Each entry states what
the command is for and gives one example usage.

Use the Serve commands with this split:

- ``autocleaneeg-serve`` for the normal launcher lifecycle
- ``autocleaneeg-pipeline serve ...`` for workspace, routes, validation,
  deployment, queue inspection, dispatcher control, API/TUI/worker commands,
  and advanced operator actions

Launcher Commands
-----------------

``autocleaneeg-serve``
   Start Serve in the foreground with the current workspace and default host
   settings.

   Example:

   .. code-block:: bash

      autocleaneeg-serve

``autocleaneeg-serve --path /path/to/serve-workspace``
   Start Serve in the foreground with an explicit workspace path.

   Example:

   .. code-block:: bash

      autocleaneeg-serve --path /path/to/serve-workspace

``autocleaneeg-serve --port 8000 --host 127.0.0.1``
   Start Serve in the foreground on a specific host and port.

   Example:

   .. code-block:: bash

      autocleaneeg-serve --port 8000 --host 127.0.0.1

``autocleaneeg-serve up``
   Start the normal Serve daemon lifecycle in the background.

   Example:

   .. code-block:: bash

      autocleaneeg-serve up

``autocleaneeg-serve up --path /path/to/serve-workspace --mode test``
   Start the Serve daemon with an explicit workspace and mode.

   Example:

   .. code-block:: bash

      autocleaneeg-serve up --path /path/to/serve-workspace --mode test

``autocleaneeg-serve status``
   Show whether the Serve UI and dispatcher are operational.

   Example:

   .. code-block:: bash

      autocleaneeg-serve status

``autocleaneeg-serve restart``
   Restart the Serve daemon lifecycle.

   Example:

   .. code-block:: bash

      autocleaneeg-serve restart

``autocleaneeg-serve down``
   Stop the Serve daemon lifecycle.

   Example:

   .. code-block:: bash

      autocleaneeg-serve down

Share Commands
--------------

``autocleaneeg-serve share``
   Start or request a shared tunnel for the current Serve session.

   Example:

   .. code-block:: bash

      autocleaneeg-serve share

``autocleaneeg-serve share status``
   Show the current tunnel-sharing status.

   Example:

   .. code-block:: bash

      autocleaneeg-serve share status

``autocleaneeg-serve share setup``
   Configure tunnel-sharing prerequisites.

   Example:

   .. code-block:: bash

      autocleaneeg-serve share setup

``autocleaneeg-serve share stop``
   Stop the active Serve tunnel.

   Example:

   .. code-block:: bash

      autocleaneeg-serve share stop

``autocleaneeg-serve share clear``
   Clear saved tunnel-sharing state.

   Example:

   .. code-block:: bash

      autocleaneeg-serve share clear

Workspace Commands
------------------

``autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace``
   Create a new Serve workspace at the target path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace

``autocleaneeg-pipeline serve workspace --mode existing --path /path/to/serve-workspace``
   Adopt an existing directory as the current Serve workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve workspace --mode existing --path /path/to/serve-workspace

``autocleaneeg-pipeline serve workspace status``
   Show the workspace currently selected for Serve.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve workspace status

``autocleaneeg-pipeline serve workspace doctor``
   Check the selected Serve workspace for structural problems.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve workspace doctor

``autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace``
   Switch Serve to a different workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace

Overview And Docs Commands
--------------------------

``autocleaneeg-pipeline serve list --path /path/to/serve-workspace``
   Summarize the Serve state for a workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve list --path /path/to/serve-workspace

``autocleaneeg-pipeline serve docs --port 7933 --host 127.0.0.1``
   Serve the operator docs locally.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve docs --port 7933 --host 127.0.0.1

Route Commands
--------------

``autocleaneeg-pipeline serve route list``
   List configured routes in the current Serve workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route list

``autocleaneeg-pipeline serve route list --path /path/to/serve-workspace``
   List routes for a specific workspace path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route list --path /path/to/serve-workspace

``autocleaneeg-pipeline serve route list --mode test``
   List routes for one deployment mode.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route list --mode test

``autocleaneeg-pipeline serve route list --include-archived``
   List active and archived routes together.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route list --include-archived

``autocleaneeg-pipeline serve route upsert resting-test --path /path/to/serve-workspace --taskfile /path/to/task.py --montage GSN-HydroCel-129 --ingestion-folder /path/to/input-folder --file-glob "*.set" --recursive --enabled``
   Create or update one route definition.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route upsert resting-test \
        --path /path/to/serve-workspace \
        --taskfile /path/to/task.py \
        --montage GSN-HydroCel-129 \
        --ingestion-folder /path/to/input-folder \
        --file-glob "*.set" \
        --recursive \
        --enabled

``autocleaneeg-pipeline serve route promote resting-test --path /path/to/serve-workspace``
   Promote one route within the current route set.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route promote resting-test --path /path/to/serve-workspace

``autocleaneeg-pipeline serve route archive resting-test --path /path/to/serve-workspace``
   Archive a route without deleting it.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route archive resting-test --path /path/to/serve-workspace

``autocleaneeg-pipeline serve route unarchive resting-test --path /path/to/serve-workspace``
   Restore an archived route.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route unarchive resting-test --path /path/to/serve-workspace

``autocleaneeg-pipeline serve route delete resting-test --workspace /path/to/serve-workspace --force``
   Delete a route from a workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route delete resting-test --workspace /path/to/serve-workspace --force

``autocleaneeg-pipeline serve route sync --path /path/to/serve-workspace``
   Sync route state from the workspace files.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve route sync --path /path/to/serve-workspace

Validation And Deployment
-------------------------

``autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode test``
   Validate the Serve configuration in test mode.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode test

``autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode live``
   Validate the Serve configuration in live mode.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode live

``autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode test``
   Deploy the current draft configuration for test-mode processing.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode test

``autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode live``
   Deploy the current draft configuration for live-mode processing.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode live

Runtime Commands
----------------

``autocleaneeg-pipeline serve run --path /path/to/serve-workspace --mode test``
   Start the dispatcher loop directly in test mode.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve run --path /path/to/serve-workspace --mode test

``autocleaneeg-pipeline serve run --path /path/to/serve-workspace --mode live --max-cycles 0 --idle-limit 0``
   Start the dispatcher loop for continuous live operation.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve run --path /path/to/serve-workspace --mode live --max-cycles 0 --idle-limit 0

``autocleaneeg-pipeline serve run --path /path/to/serve-workspace --dry-run``
   Simulate dispatcher work without executing real processing.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve run --path /path/to/serve-workspace --dry-run

``autocleaneeg-pipeline serve run --path /path/to/serve-workspace --no-watch``
   Run the dispatcher loop with file watching disabled.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve run --path /path/to/serve-workspace --no-watch

``autocleaneeg-pipeline serve tui --path /path/to/serve-workspace --mode test``
   Start the Serve TUI for a workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve tui --path /path/to/serve-workspace --mode test

``autocleaneeg-pipeline serve api --path /path/to/serve-workspace --mode test --api-port 8000``
   Start the Serve API explicitly.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve api --path /path/to/serve-workspace --mode test --api-port 8000

``autocleaneeg-pipeline serve worker --redis-url redis://localhost:6379``
   Start the Serve worker process with an explicit Redis backend.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve worker --redis-url redis://localhost:6379

Service Commands
----------------

``autocleaneeg-pipeline serve service status``
   Show dispatcher status through the running service API.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve service status

``autocleaneeg-pipeline serve service start``
   Start dispatcher processing through the service API.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve service start

``autocleaneeg-pipeline serve service start --max-cycles 0 --idle-limit 0``
   Start the dispatcher service for continuous operation.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve service start --max-cycles 0 --idle-limit 0

``autocleaneeg-pipeline serve service stop``
   Stop dispatcher processing through the service API.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve service stop

Mode Commands
-------------

``autocleaneeg-pipeline serve mode``
   Show or inspect the current Serve mode.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve mode

``autocleaneeg-pipeline serve mode status``
   Print the current mode explicitly.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve mode status

``autocleaneeg-pipeline serve mode test``
   Switch the running Serve session to test mode.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve mode test

``autocleaneeg-pipeline serve mode live``
   Switch the running Serve session to live mode.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve mode live

Queue Commands
--------------

``autocleaneeg-pipeline serve queue status``
   Show queue health and activity.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve queue status

``autocleaneeg-pipeline serve queue list``
   List queued items.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve queue list

``autocleaneeg-pipeline serve queue list --status failed --route-id resting-test --limit 20``
   List queue items filtered by state and route.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve queue list --status failed --route-id resting-test --limit 20

``autocleaneeg-pipeline serve queue retry-failed``
   Retry all failed queue items.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve queue retry-failed

``autocleaneeg-pipeline serve queue retry-failed /full/path/to/file.set``
   Retry one failed queue item by file path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve queue retry-failed /full/path/to/file.set

``autocleaneeg-pipeline serve queue clear-processed``
   Remove processed items from the queue history.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve queue clear-processed

``autocleaneeg-pipeline serve queue remove /full/path/to/file.set``
   Remove one queue item by file path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve queue remove /full/path/to/file.set

Legacy Aliases
--------------

``autocleaneeg-pipeline serve up``
   Start Serve through the older alias under the main CLI.

   This command uses the workspace already selected through
   ``autocleaneeg-pipeline serve workspace use --path ...`` and does not accept
   ``--path`` itself.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve up

   Persist the workspace first:

   .. code-block:: bash

      autocleaneeg-pipeline serve workspace use --path /path/to/serve-workspace
      autocleaneeg-pipeline serve up

   If you need an explicit-path launcher in one command, use:

   .. code-block:: bash

      autocleaneeg-serve up --path /path/to/serve-workspace

``autocleaneeg-pipeline serve down``
   Stop Serve through the older alias under the main CLI.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve down

``autocleaneeg-pipeline serve restart``
   Restart Serve through the older alias under the main CLI.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve restart

``autocleaneeg-pipeline serve status``
   Show Serve status through the older alias under the main CLI.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve status

``autocleaneeg-pipeline serve share status``
   Show tunnel-sharing status through the legacy alias path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve share status

``autocleaneeg-pipeline serve share``
   Start tunnel sharing through the legacy alias path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve share

``autocleaneeg-pipeline serve share setup``
   Configure tunnel sharing through the legacy alias path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve share setup

``autocleaneeg-pipeline serve share stop``
   Stop tunnel sharing through the legacy alias path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve share stop

``autocleaneeg-pipeline serve share clear``
   Clear saved tunnel-sharing state through the legacy alias path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline serve share clear
