# Serve Route-First Devlog

This note is the technical companion to `docs/serve-route-first-operator-guide.html`.

Use the HTML guide first if you are training operators or documenting handoff. Use this devlog when you need the backend model, the exact route-management commands, or the implementation boundaries that should stay stable while the TUI keeps improving.

## What changed

The older serve workflow could process files, but it exposed the wrong things to the wrong people.

- Operators had to reason about large YAML files instead of individual automation routes.
- The runtime already had one lane per mode, but the product did not explain that cleanly.
- The system needed a safer way to support many task-and-montage combinations without spawning many queue files.
- Re-running setup needed to be safe and boring rather than feeling like a destructive reset.

The route-first direction fixes that by separating configuration units from execution lanes.

## Stable model in this branch

The operating model is now:

- one route spec per route under `routes/*.yaml`
- one Draft lane compiled into `serve-test.yaml`
- one Production lane compiled into `serve-live.yaml`
- one queue file for Draft: `queue-test.json`
- one queue file for Production: `queue-live.json`

For operators, `test` maps to Draft and `live` maps to Production.

The important point is that a task-and-montage combination is modeled as a route, not as its own queue.

## Stable operator promises

These are the promises the backend should keep regardless of final TUI design.

### 1. Draft and Production stay explicit

A route should be proven in Draft before it is promoted into Production. Production should never feel like an experimental lane.

### 2. Route setup is idempotent

Re-running route setup should be safe.

That means:

- create the route if it does not exist
- update only the fields that changed
- do nothing visible if the route definition is unchanged
- never duplicate jobs just because setup ran again
- never wipe queue history or prior outputs

### 3. Promotion is additive, not copy-paste heavy

The same route should move from Draft to Production. Operators should not have to rebuild a second near-duplicate configuration for live use.

### 4. Compiled configs are deterministic

The route registry is the source of truth. The generated `serve-test.yaml` and `serve-live.yaml` should be reproducible outputs of that registry, not hand-edited mystery files.

## Commands that define the current backend contract

These are the commands that matter for the route-first backend as it stands now.

### Create or update a route in Draft

```bash
autocleaneeg-pipeline serve route upsert \
  resting-biosemi64 \
  --path /path/to/workspace \
  --mode test \
  --taskfile /path/to/RestingEyesOpen.py \
  --montage biosemi64 \
  --ingestion-folder /data/incoming/resting \
  --file-glob "*.set" \
  --recursive \
  --enabled
```

This should write or update one route spec and recompile the generated configs.

Expected artifacts:

- `routes/resting-biosemi64.yaml`
- `serve-test.yaml`
- `serve-live.yaml`

If the command is run again with the same values, the expected behavior is a no-op.

### List routes

```bash
autocleaneeg-pipeline serve route list --path /path/to/workspace
```

### Validate Draft

```bash
autocleaneeg-pipeline serve validate --path /path/to/workspace --mode test
```

### Promote a route into Production

```bash
autocleaneeg-pipeline serve route promote \
  resting-biosemi64 \
  --path /path/to/workspace
```

Promotion should add Production membership to the same route definition and recompile `serve-live.yaml`.

### Rebuild generated configs after manual spec edits

```bash
autocleaneeg-pipeline serve route sync --path /path/to/workspace
```

## What this means for operators

The operator-facing abstraction is now cleaner:

- manage routes
- validate in Draft
- promote into Production
- monitor work states, not queue internals

That is the right model for 10 to 20 task-and-montage combinations and still leaves room to scale far beyond that.

## What this deliberately does not depend on

This branch now has a working TUI route-management layer for safe operator actions, but the route-first model should still outlive any specific screen wording.

The guide and backend contract above do **not** depend on:

- the exact final wording of TUI buttons
- the final arrangement of setup screens
- whether the operator creates a route through a wizard, form, or CLI wrapper

The behavior that should stay stable is the backend behavior:

- route registry as source of truth
- deterministic config compilation
- explicit Draft vs Production lanes
- safe reruns
- explicit promotion

## Why this is safer than the old model

- Small route specs are easier to review than one giant config file.
- Updating one route does not require rebuilding the whole mental model of the workspace.
- Operators can reason about one route at a time.
- Repeated setup becomes safe enough for routine maintenance, not just first-time installation.
- The system can scale to many routes without asking operators to manage many queue files.

## What still needs work

These are still valid follow-up goals, but they sit on top of the route-first backend rather than replacing it.

- The TUI still needs a first-class create/edit flow so operators can add new routes without dropping to the CLI.
- Operator status views should use human language such as Waiting, Running, Needs attention, and Completed.
- A first-run guided setup flow should sit on top of `serve route upsert`.
- If the product ever truly needs many independent execution lanes, the next step should be SQLite or Redis-backed queue storage rather than multiplying JSON queue files.

## Documentation stance for now

Until the final TUI flow lands, the safest documentation posture is:

- keep the operator guide conceptual and task-oriented
- keep this devlog concrete about the backend contract
- document the current TUI actions only where they are already stable and operator-safe
