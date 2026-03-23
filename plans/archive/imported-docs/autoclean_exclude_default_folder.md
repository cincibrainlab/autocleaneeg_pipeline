# Where the Exclusion Tool’s Default Folder Comes From

This explains how the Inclusion/Exclusion review tool decides which folder to open by default, and why it may feel “wrong”. It covers both the initial folder used when the app starts and the default shown in the “Choose Folder…” dialog.

## TL;DR
- If you start the tool without arguments, it prefers your global workspace’s most recently modified task under `…/Autoclean-EEG/output/<task>/exports`.
- If that can’t be found, it falls back to `./exports` (inside your current working directory), then to the current working directory itself.
- The “Choose Folder…” dialog starts at whatever the current folder is in the UI (`self.current_dir`).
- To avoid guessing, pass `--exports <path>` or a positional path argument when launching the tool.

## Resolution Order (initial folder at launch)
The main entry path resolution happens in `determine_paths`:
- If `--exports` is provided, use that as the exports folder. If `--task-root` is not given, the task root defaults to the parent folder of `--exports`.
  - Code: `src/autoclean/tools/autoclean_exclude.py:1048`
- Else, if a positional `path` is provided:
  - If `<path>/exports` exists, treat `<path>` as the task root and `<path>/exports` as the exports folder.
  - Otherwise treat `<path>` itself as the exports folder; task root becomes `<path>/..` if it exists.
  - Code: `src/autoclean/tools/autoclean_exclude.py:1057`
- Else (no args): prefer the most recent task under the workspace output, then `./exports`, then the current working directory.
  - Code: `src/autoclean/tools/autoclean_exclude.py:1065`

### What is the “workspace output” path?
The tool uses the workspace manager’s default output directory:
- `user_config.get_default_output_dir()` returns `<workspace>/output`.
- The workspace base path defaults to your OS documents folder, e.g. `~/Documents/Autoclean-EEG`, unless you configured it via the CLI (which writes a `setup.json`).
  - Code: `src/autoclean/utils/user_config.py:60`, `src/autoclean/utils/user_config.py:99`

Within that `<workspace>/output` folder, the tool finds the most recent subdirectory (by modification time), and if it contains an `exports/` folder, that becomes the start location.
- Code: `src/autoclean/tools/autoclean_exclude.py:1069`

## How the UI picks the default in “Choose Folder…”
When you click “Choose Folder…”, the dialog opens at the current review folder if set, otherwise at the process’s current working directory:
- Code: `src/autoclean/tools/autoclean_exclude.py:785`

`self.current_dir` itself is set when the window is constructed:
- The constructor passes the resolved exports directory (from `determine_paths`) into the base review widget: `src/autoclean/tools/autoclean_exclude.py:137`.
- Then `_configure_directory` normalizes and assigns it to `self.current_dir` (and prepares files for saving decisions): `src/autoclean/tools/autoclean_exclude.py:682`.

## Why it might feel “wrong”
- Workspace wins over local folders: If you run the tool with no arguments, the most recently modified task in your global workspace output is picked before `./exports` in your current directory. If you expected the local `./exports` to be preferred, this ordering can be surprising.
- “Most recent” is by filesystem mtime: Copying or touching a task directory in the workspace can bump it to the top even if it’s not the run you intended to review.
- Missing `exports/` in the latest workspace task: Then it falls back to your shell’s `cwd` or `./exports` if present; that may not be where your data is.
- Different shell `cwd`: Launching from a different directory than the task directory means the fallback to `cwd` is somewhere else than expected.

## How to control it (recommended)
- Pass an explicit exports folder every time:
  - `python -m autoclean.tools.autoclean_exclude --exports /path/to/<task>/exports`
- Or pass the task root (if it contains an `exports/`):
  - `python -m autoclean.tools.autoclean_exclude /path/to/<task>`
- Or run from the task root and rely on `./exports` being present:
  - `cd /path/to/<task>` then `python -m autoclean.tools.autoclean_exclude`
- Configure your workspace so the default points where you expect, or avoid the workspace default by always passing a path/flag.

## Related code paths
- Start-up path inference: `src/autoclean/tools/autoclean_exclude.py:1045`
- Dialog default folder: `src/autoclean/tools/autoclean_exclude.py:785`
- Directory normalization: `src/autoclean/tools/autoclean_exclude.py:682`
- Workspace output root: `src/autoclean/utils/user_config.py:99`

If you want the tool to prefer `./exports` over the workspace by default, that’s a one-line change to swap the order in `determine_paths` (we can do that if desired).
