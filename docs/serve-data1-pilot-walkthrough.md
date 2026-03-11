# Building a Fresh AutoClean Serve Pilot on `data1`

This is a strict microstep walkthrough for building a brand-new `autoclean serve` pilot on `data1` from `main`, using an isolated `uv` environment.

Rules for this walkthrough:
- one microstep at a time
- plain-English explanation after each step
- no next step until you confirm understanding

## Step 1: Reserve a fresh empty pilot root

Goal:
Create a brand-new empty folder on `data1` that will hold this experiment.

Why this matters:
This gives us an isolated place to build without touching the existing `workspace` or `workspace-assr` deployments.

Result:
- Pilot root: `/Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107`
- Nothing has been installed yet.
- No code has been copied yet.
- No runtime has been created yet.

## Step 2: Create the workspace directory inside the pilot root

Goal:
Create the specific directory that will become the serve workspace.

Why this matters:
The pilot root is just the outer experiment folder. The `workspace/` directory is where the actual serve configs, routes, queue files, runtimes, and outputs will live.

Result:
- Pilot root: `/Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107`
- Serve workspace path: `/Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/workspace`

## Step 3: Clone the repository into the pilot root

Goal:
Create a real Git checkout inside the pilot so the build uses traceable code from `main`.

Why this matters:
This is different from copying files. A clone gives us provenance: we can always answer which branch and commit were used to build the pilot.

Result:
- Repo checkout path: `/Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/repo`
- The pilot is now tied to a real Git history.

## Step 4: Create an isolated `uv` environment inside the pilot repo

Goal:
Create a fresh Python virtual environment for this pilot.

Why this matters:
We do not want this pilot relying on whatever Python environment happens to exist elsewhere on `data1`. The `.venv` gives this pilot its own isolated runtime.

Result:
- Virtual environment path: `/Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/repo/.venv`
- The pilot now has an isolated Python environment, but the package is not installed into it yet.

## Step 5: Install the project into the pilot `.venv`

Goal:
Install the cloned project into the pilot's isolated Python environment.

Why this matters:
The `.venv` exists now, but until the package is installed, it cannot actually run `autocleaneeg-pipeline`. This step turns the empty environment into a runnable tool environment.

Result:
- The pilot `.venv` now has `autocleaneeg-pipeline` installed from the cloned `main` checkout.
- The install source is Git-traceable because the repo checkout is on commit `21bca23`.
- We still have not created the serve workspace contents yet.

## Step 6: Initialize the serve workspace with the installed tool

Goal:
Use the installed `autocleaneeg-pipeline` tool to scaffold a brand-new serve workspace inside the pilot.

Why this matters:
This is the first step where the pilot becomes a real serve setup instead of just a repo plus a Python environment.

Important isolation detail:
The command was run with `HOME` pointed at a pilot-local folder. That prevents this experiment from modifying the normal user-level serve workspace setting on `data1`.

Command shape:
```bash
HOME=/Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/home \
  /Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/repo/.venv/bin/autocleaneeg-pipeline \
  serve workspace \
  --mode new \
  --path /Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/workspace \
  --package /Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/repo \
  --skip-uv \
  --no-test
```

Result:
- The serve workspace skeleton now exists under `/Volumes/ac_serve/autoclean-serve/pilot-uv-20260308-091107/workspace`
- The runtime package is pinned to the cloned repo path
- Runtime bootstrapping was intentionally deferred by using `--skip-uv`

## Step 7: Inspect the scaffolded workspace layout

Goal:
See exactly what files and directories the workspace initializer created.

Why this matters:
This is the first point where you can inspect the concrete shape of a brand-new serve workspace instead of thinking about it abstractly.

What to look for:
- `serve-test.yaml` and `serve-live.yaml`: the top-level mode configs
- `routes/`: where route-first specs live
- `deploy/`: where deployed configs are written
- `runtimes/`: per-mode runtime directories
- `automations/`: where per-route output workspaces will be created
- helper files like `Dockerfile`, compose files, `Makefile`, and `workspace.env`

## Step 8: Check for the two top-level mode config files

Goal:
Verify that the workspace initializer actually created the two top-level serve config files.

Why this matters:
If `serve-test.yaml` and `serve-live.yaml` do not exist, then the workspace scaffold is not really in place yet.

Files checked:
- `serve-test.yaml`
- `serve-live.yaml`
