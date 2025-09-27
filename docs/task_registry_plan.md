# Task Registry MVP Plan

Below is the concrete MVP plan for migrating the built-in task files into a public repository and updating the pipeline CLI to manage them.

## MVP scope (strictly minimal)

### Goal

Move the current built-in task files into a public repo (org-owned).

Pipeline can: list, fetch/sync, and copy/install those tasks into a workspace.

If network is down or GitHub isn’t reachable, pipeline falls back to the baked-in copies (identical to today).

### Out of scope (for now)

Community submissions, signing/AST scans, declarative DSL, per-task dependency handling, PR automation, or personal repos.

## Repository layout (public, org-owned)

Propose a new repo (name is up to you; two options):

- `cincibrainlab/autoclean-builtins` (clear it’s only the built-ins for now)
- `cincibrainlab/autoclean-task-registry` (future-proof name; still fine to start with built-ins only)

```
autoclean-builtins/
├── README.md
├── registry.json              # tiny index (name -> path)
└── tasks/
    ├── resting/
    │   ├── RestingEyesOpen.py
    │   └── RestingEyesClosed.py
    ├── auditory/
    │   ├── ASSR_40Hz.py
    │   └── MMN_Standard.py
    └── ...
```

### Minimal registry.json (no metadata yet)

```json
{
  "version": 1,
  "commit": "<filled-by-CI-or-manually>",
  "tasks": [
    {"name": "RestingEyesOpen",   "path": "tasks/resting/RestingEyesOpen.py"},
    {"name": "RestingEyesClosed", "path": "tasks/resting/RestingEyesClosed.py"},
    {"name": "ASSR_40Hz",         "path": "tasks/auditory/ASSR_40Hz.py"},
    {"name": "MMN_Standard",      "path": "tasks/auditory/MMN_Standard.py"}
  ]
}
```

#### Why keep this file?

It gives you a single stable endpoint to list tasks and their relative paths. It’s trivial to maintain and keeps your pipeline code simple.

## Fallback order

When the user asks for built-ins, the pipeline tries in this order:

1. Local workspace (user may have already copied a built-in and customized it)
2. Local cache (`~/.config/autocleaneeg/.builtin_cache`) pulled from GitHub
3. Packaged fallback (files shipped inside the wheel/sdist)

That ensures zero regressions if GitHub is down or the user is offline.

## CLI (tiny additions)

Add one namespaced group for clarity:

```
autocleaneeg-pipeline task builtins update

autocleaneeg-pipeline task builtins list

autocleaneeg-pipeline task builtins install RestingEyesOpen
```

Users still run the pipeline the same way as today once a task is in their workspace.

## Code: drop-in skeletons

These are deliberately dependency-light (stdlib only). Swap to httpx later if needed.

### 1) Packaged fallback data

Add a folder to your package (e.g., `autoclean/data/builtins/`) that mirrors the GitHub repo structure:

```
autoclean/data/builtins/
├── registry.json
└── tasks/...
```

Ensure it’s included in the build:

- `pyproject.toml`: `include = ["autoclean/data/builtins/**"]` (or use package-data)
- Or a `MANIFEST.in` with: `recursive-include autoclean/data/builtins *`

### 2) Minimal built-ins connector

```python
from __future__ import annotations
import json, shutil, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import importlib.resources as pkg_resources

RAW_BASE = "https://raw.githubusercontent.com/cincibrainlab/autoclean-builtins/main"
CACHE_ROOT = Path.home() / ".config" / "autocleaneeg" / ".builtin_cache"

@dataclass(frozen=True)
class BuiltinTask:
    name: str
    path: str

class BuiltinRegistry:
    ...
```

### 3) CLI bindings (Typer/Click-style)

```python
import typer
from pathlib import Path
from autoclean.utils.builtins import BuiltinRegistry
from .helpers import get_workspace_dir

app = typer.Typer(help="Manage built-in tasks (from the public GitHub repo with offline fallback)")

@app.command("update")
def update():
    ...
```

Integrate this subcommand into your main CLI:

```python
from . import builtins as builtins_cmd
app.add_typer(builtins_cmd.app, name="task builtins")
```

## How this maps to your codebase today

Task discovery doesn’t change: users still point the pipeline at a workspace with `tasks/*.py`.

This MVP simply gives them a new way to populate that folder from an authoritative, version-controlled source (your org’s GitHub).

No behavior changes to processing/execution.

## Simple CI for the new repo (optional but tiny)

Add a workflow that:

- Validates that every `tasks/*/*.py` path appears once in `registry.json`.
- Writes the commit hash into the `commit` field on `registry.json` (or a small script updates it when you merge).

## Developer ergonomics

- Zero background networking: All network access happens only on explicit update/install.
- One source of truth: Built-ins live in the GitHub repo; the package includes a snapshot in `autoclean/data/builtins/` as the offline fallback.
- Fast iteration: Updating built-ins doesn’t require cutting a new package—users just run `task builtins update` to sync the latest (or they continue using the packaged snapshot if they prefer stability).

## Immediate action plan

1. Create the repo with `tasks/` and `registry.json` (as shown).
2. Copy today’s built-in tasks from your codebase into that repo, preserving subfolders.
3. Add the small `BuiltinRegistry` + CLI in the pipeline repo.
4. Package fallback: vendor the same `tasks/ + registry.json` into `autoclean/data/builtins/` and include it in the build.

## Smoke test

With internet:

```
autocleaneeg-pipeline task builtins update && \
  autocleaneeg-pipeline task builtins list && \
  autocleaneeg-pipeline task builtins install RestingEyesOpen
```

Airplane mode:

```
autocleaneeg-pipeline task builtins list && \
  autocleaneeg-pipeline task builtins install RestingEyesOpen
```

## Future enhancements

- richer metadata in `registry.json`
- “safe” declarative tasks
- signatures/AST scan for community Python
- personal repositories + publishing

