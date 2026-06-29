Command Reference
=================

This page documents the current ``autocleaneeg-pipeline`` command surface.
Each entry states what the command is for and gives one example usage.

Use these public entrypoints:

- ``autocleaneeg-pipeline`` for processing, workspace, task, config, events,
  reports, and Serve control commands
- ``autocleaneeg-serve`` for the normal Serve launcher lifecycle
- ``autocleaneeg-tui`` for the standalone TUI entrypoint when applicable

For Serve-specific commands, see :doc:`serve_command_reference`.

``autocleaneeg-tui``
   Start the standalone TUI entrypoint when that operator surface is needed.

   Example:

   .. code-block:: bash

      autocleaneeg-tui

Global Help And Status
----------------------

``autocleaneeg-pipeline --help``
   Show top-level CLI help.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline --help

``autocleaneeg-pipeline help``
   Show built-in help through the explicit ``help`` command.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline help

``autocleaneeg-pipeline help serve``
   Show help for the Serve command family.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline help serve

``autocleaneeg-pipeline version``
   Print the installed CLI version.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline version

``autocleaneeg-pipeline tutorial``
   Open the built-in tutorial entrypoint.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline tutorial

Workspace And Setup
-------------------

``autocleaneeg-pipeline wizard``
   Run the interactive setup wizard for the local workspace flow.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline wizard

``autocleaneeg-pipeline config setup``
   Configure the active workspace and initial local settings.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline config setup

``autocleaneeg-pipeline config show``
   Display the current saved configuration, including the active workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline config show

``autocleaneeg-pipeline workspace show``
   Show the workspace path currently used by the CLI.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline workspace show

``autocleaneeg-pipeline workspace set /path/to/workspace``
   Set the active workspace path explicitly.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline workspace set /path/to/workspace

``autocleaneeg-pipeline workspace unset``
   Clear the currently assigned workspace from saved configuration.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline workspace unset

``autocleaneeg-pipeline workspace default``
   Reset the workspace to the default location for the current environment.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline workspace default

``autocleaneeg-pipeline workspace explore``
   Open or print the workspace in an explorer-friendly way.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline workspace explore

``autocleaneeg-pipeline workspace size``
   Report the current workspace size on disk.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline workspace size

``autocleaneeg-pipeline workspace cd --print zsh``
   Print a shell-compatible command for changing into the workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline workspace cd --print zsh

Processing
----------

``autocleaneeg-pipeline process RestingEyesOpen /path/to/file.raw``
   Process a single file with a built-in task using positional arguments.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline process RestingEyesOpen /path/to/file.raw

``autocleaneeg-pipeline process --task RestingEyesOpen --file /path/to/file.raw``
   Process a single file with explicit flags instead of positional arguments.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline process --task RestingEyesOpen --file /path/to/file.raw

``autocleaneeg-pipeline process --task-file /path/to/MyTask.py --file /path/to/file.raw``
   Process a file with a custom task file from disk.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline process --task-file /path/to/MyTask.py --file /path/to/file.raw

``autocleaneeg-pipeline process --task RestingEyesOpen --dir /path/to/data --format "*.set" --recursive``
   Process every matching file in a directory tree.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline process --task RestingEyesOpen --dir /path/to/data --format "*.set" --recursive

``autocleaneeg-pipeline process --dry-run RestingEyesOpen /path/to/file.raw``
   Validate what would be processed without starting the real run.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline process --dry-run RestingEyesOpen /path/to/file.raw

``autocleaneeg-pipeline process ica --metadata-dir /path/to/run/metadata``
   Apply ICA control-sheet decisions to an existing run directory.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline process ica --metadata-dir /path/to/run/metadata

Task Commands
-------------

``autocleaneeg-pipeline list-tasks``
   List the currently available tasks.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline list-tasks

``autocleaneeg-pipeline list-tasks --overrides``
   List tasks and include override information where applicable.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline list-tasks --overrides

``autocleaneeg-pipeline task list``
   List tasks through the grouped ``task`` command family.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task list

``autocleaneeg-pipeline task list --source builtin``
   List only tasks from a specific source such as built-in tasks.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task list --source builtin

``autocleaneeg-pipeline task list --status outdated``
   Filter the task list by task status.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task list --status outdated

``autocleaneeg-pipeline task search resting``
   Search task names and metadata by keyword.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task search resting

``autocleaneeg-pipeline task diagnose``
   Check task-library health and common task resolution issues.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task diagnose

``autocleaneeg-pipeline task diff RestingEyesOpen``
   Compare the active or installed task against another task definition.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task diff RestingEyesOpen

``autocleaneeg-pipeline task create MyCustomTask``
   Create a new task scaffold in the current workspace.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task create MyCustomTask

``autocleaneeg-pipeline task install /path/to/MyCustomTask.py --source file --activate``
   Install a task from a local file and activate it immediately.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task install /path/to/MyCustomTask.py --source file --activate

``autocleaneeg-pipeline task install RestingEyesOpen --source builtin --activate``
   Install a task from the built-in task library and activate it.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task install RestingEyesOpen --source builtin --activate

``autocleaneeg-pipeline task use RestingEyesOpen``
   Set a task as the currently selected task for task-aware operations.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task use RestingEyesOpen

``autocleaneeg-pipeline task copy RestingEyesOpen --name RestingEyesOpenCopy``
   Copy an existing task into a new editable task file.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task copy RestingEyesOpen --name RestingEyesOpenCopy

``autocleaneeg-pipeline task edit MyCustomTask``
   Open a task for editing through the configured editor flow.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task edit MyCustomTask

``autocleaneeg-pipeline task delete MyCustomTask``
   Delete a task from the active workspace or task store.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task delete MyCustomTask

``autocleaneeg-pipeline task explore``
   Open or reveal the task storage location.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task explore

``autocleaneeg-pipeline task set RestingEyesOpen``
   Set the active task explicitly.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task set RestingEyesOpen

``autocleaneeg-pipeline task show``
   Show the currently active task.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task show

``autocleaneeg-pipeline task unset``
   Clear the current active task selection.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task unset

``autocleaneeg-pipeline task schema export --output task-schema.json``
   Export the task schema used for validation and generation tools.
   Set ``AUTOCLEAN_CONFIG_DEBUG=1`` to include raw schema errors when debugging task validation failures.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task schema export --output task-schema.json

``autocleaneeg-pipeline task sync``
   Sync task metadata with the current task library state.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task sync

``autocleaneeg-pipeline task sync --update``
   Sync task metadata and refresh it from the upstream task source.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task sync --update

``autocleaneeg-pipeline task update``
   Refresh the task library state used by the CLI and task browser flows.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline task update

Montage Commands
----------------

``autocleaneeg-pipeline montage list``
   List the available montages.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline montage list

``autocleaneeg-pipeline montage set GSN-HydroCel-129``
   Set the active montage for task and processing workflows.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline montage set GSN-HydroCel-129

``autocleaneeg-pipeline montage test``
   Run the built-in montage test flow.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline montage test

Block Commands
--------------

``autocleaneeg-pipeline blocks list``
   List available processing blocks.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline blocks list

``autocleaneeg-pipeline blocks info matlab_fooof``
   Show metadata and details for one block.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline blocks info matlab_fooof

``autocleaneeg-pipeline blocks deps matlab_fooof``
   Show dependency information for a block.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline blocks deps matlab_fooof

``autocleaneeg-pipeline blocks update``
   Refresh the local block cache from the configured block source.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline blocks update

``autocleaneeg-pipeline blocks install matlab_fooof``
   Install one named block into the local block cache.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline blocks install matlab_fooof

``autocleaneeg-pipeline blocks install --locked``
   Install blocks using an existing lock file.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline blocks install --locked

``autocleaneeg-pipeline blocks lock --output blocks.lock``
   Write a block lock file for reproducible block installs.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline blocks lock --output blocks.lock

Input And Source Commands
-------------------------

``autocleaneeg-pipeline input set /path/to/file.raw``
   Save a preferred input path using the supported ``input`` command family.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline input set /path/to/file.raw

``autocleaneeg-pipeline input show``
   Show the currently saved input path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline input show

``autocleaneeg-pipeline input unset``
   Clear the saved input path.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline input unset

``autocleaneeg-pipeline source set /path/to/file.raw``
   Set a source path through the older compatibility alias.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline source set /path/to/file.raw

``autocleaneeg-pipeline source show``
   Show the saved source path through the deprecated alias.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline source show

``autocleaneeg-pipeline source unset``
   Clear the saved source path through the deprecated alias.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline source unset

Event Commands
--------------

``autocleaneeg-pipeline events discover /path/to/file.set``
   Discover events in a supported EEG file.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline events discover /path/to/file.set

``autocleaneeg-pipeline events discover /path/to/file.xdat --montage GSN-HydroCel-129``
   Discover events while forcing a montage during import.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline events discover /path/to/file.xdat --montage GSN-HydroCel-129

``autocleaneeg-pipeline events analyze /path/to/file.set --gap-threshold 45``
   Analyze event timing and gaps for a raw file.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline events analyze /path/to/file.set --gap-threshold 45

``autocleaneeg-pipeline events epochs /path/to/file-epo.fif``
   Inspect an epochs file through the events CLI family.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline events epochs /path/to/file-epo.fif

Configuration Commands
----------------------

``autocleaneeg-pipeline config reset --confirm``
   Reset the saved CLI configuration.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline config reset --confirm

``autocleaneeg-pipeline config export /path/to/export-dir``
   Export the current configuration to a directory.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline config export /path/to/export-dir

``autocleaneeg-pipeline config import /path/to/export-dir``
   Import a previously exported configuration bundle.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline config import /path/to/export-dir

Audit And Cleanup Commands
--------------------------

``autocleaneeg-pipeline export-access-log --output audit.jsonl``
   Export the access log in JSONL form.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline export-access-log --output audit.jsonl

``autocleaneeg-pipeline export-access-log --format csv --output audit.csv``
   Export the access log in CSV form.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline export-access-log --format csv --output audit.csv

``autocleaneeg-pipeline export-access-log --verify-only``
   Verify the access log without exporting a new file.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline export-access-log --verify-only

``autocleaneeg-pipeline clean-task RestingEyesOpen --dry-run``
   Preview what would be cleaned for a task without deleting anything.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline clean-task RestingEyesOpen --dry-run

``autocleaneeg-pipeline clean-task RestingEyesOpen --force``
   Clean task-generated state for a task immediately.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline clean-task RestingEyesOpen --force

Review And GUI Commands
-----------------------

``autocleaneeg-pipeline review --output /path/to/output``
   Open the review flow for an output directory.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline review --output /path/to/output

``autocleaneeg-pipeline exclude /path/to/run-or-exports``
   Open the exclusion workflow for a run or export directory.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline exclude /path/to/run-or-exports

``autocleaneeg-pipeline view /path/to/file.raw``
   Open a supported EEG file in the viewing flow.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline view /path/to/file.raw

``autocleaneeg-pipeline view /path/to/file.raw --no-view``
   Load a file through the view flow without launching the viewer UI.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline view /path/to/file.raw --no-view

Reporting Commands
------------------

``autocleaneeg-pipeline report create --run-id RUN123 --context-json /path/to/context.json --out-dir /path/to/reports``
   Generate a report bundle for one run from a context file.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline report create --run-id RUN123 --context-json /path/to/context.json --out-dir /path/to/reports

``autocleaneeg-pipeline report chat --context-json /path/to/context.json``
   Start the report chat flow from a report context file.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline report chat --context-json /path/to/context.json

Authentication Commands
-----------------------

``autocleaneeg-pipeline login``
   Run the direct login helper.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline login

``autocleaneeg-pipeline logout``
   Run the direct logout helper.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline logout

``autocleaneeg-pipeline whoami``
   Show the current authenticated identity.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline whoami

``autocleaneeg-pipeline auth0-diagnostics``
   Run the direct Auth0 diagnostic helper.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth0-diagnostics

``autocleaneeg-pipeline auth login``
   Run login through the grouped ``auth`` command family.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth login

``autocleaneeg-pipeline auth logout``
   Run logout through the grouped ``auth`` command family.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth logout

``autocleaneeg-pipeline auth whoami``
   Show the current authenticated identity through the grouped auth family.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth whoami

``autocleaneeg-pipeline auth diagnostics``
   Run grouped authentication diagnostics.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth diagnostics

``autocleaneeg-pipeline auth setup``
   Configure authentication support for the current environment.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth setup

``autocleaneeg-pipeline auth enable``
   Enable the configured authentication flow.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth enable

``autocleaneeg-pipeline auth disable``
   Disable the configured authentication flow.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline auth disable

MATLAB Commands
---------------

``autocleaneeg-pipeline matlab doctor``
   Check the local MATLAB integration state.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline matlab doctor

``autocleaneeg-pipeline matlab doctor --skip-start``
   Run MATLAB diagnostics without trying to start the engine.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline matlab doctor --skip-start

``autocleaneeg-pipeline matlab test-engine``
   Test whether the MATLAB engine can be started successfully.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline matlab test-engine

Settings Commands
-----------------

``autocleaneeg-pipeline settings theme``
   Show the current theme setting.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline settings theme

``autocleaneeg-pipeline settings theme dark``
   Set the CLI theme explicitly.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline settings theme dark

``autocleaneeg-pipeline settings theme --clear``
   Clear any saved theme override.

   Example:

   .. code-block:: bash

      autocleaneeg-pipeline settings theme --clear
