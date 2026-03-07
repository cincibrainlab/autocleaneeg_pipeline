# Serve Route-First Devlog

## Why this changed

The original serve workflow was technically capable, but it exposed the wrong abstractions.

- Operators had to think about YAML files instead of automation routes.
- The product had one queue per mode already, but the setup experience did not make that clear.
- The TUI had drifted away from the real runtime model.
- The safest scaling model for 10-20 montage/task combinations is many routes inside `test` and `live`, not many queue files.

The new direction is:

- one queue for `test`
- one queue for `live`
- one route spec per automation route
- deterministic compile back into `serve-test.yaml` and `serve-live.yaml`

That gives us idempotent setup, safer reruns, and a path to hundreds of routes without making the operator manage hundreds of queues.

## What shipped in this slice

### Earlier hardening

- TUI service launch now uses the real mode-specific queue files.
- TUI deploy follows the stricter CLI deploy path.
- Service settings screen now actually controls the launched command.
- Dry-run API tasks no longer require a fully installed runtime before building commands.

### New route-first layer

- Added a route registry under `routes/*.yaml`.
- Added `autocleaneeg-pipeline serve route upsert`.
- Added `autocleaneeg-pipeline serve route promote`.
- Added `autocleaneeg-pipeline serve route list`.
- Added `autocleaneeg-pipeline serve route sync`.
- Compiled configs are now generated deterministically from the route registry.

## The operating model

### Draft

`test` is the draft lane.

Use it to:

- define a route
- validate the compiled config
- dry run or exercise the route
- confirm outputs and logs look right

### Production

`live` is the production lane.

Do not rebuild the route from scratch. Promote the same route into `live` once it is behaving correctly in `test`.

### Queues

Queues are intentionally boring now:

- `queue-test.json`
- `queue-live.json`

Operators should mostly ignore the queue implementation details. The important unit is the route.

## Route setup workflow

### 1. Create or update a route in draft

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

This writes one route spec and recompiles:

- `routes/resting-biosemi64.yaml`
- `serve-test.yaml`
- `serve-live.yaml`

If you run the same command again with the same values, it is a no-op.

### 2. Review the route registry

```bash
autocleaneeg-pipeline serve route list --path /path/to/workspace
```

### 3. Validate the draft config

```bash
autocleaneeg-pipeline serve validate --path /path/to/workspace --mode test
```

### 4. Promote the route into production

```bash
autocleaneeg-pipeline serve route promote \
  resting-biosemi64 \
  --path /path/to/workspace
```

That adds `live` to the route spec and recompiles `serve-live.yaml`.

### 5. Rebuild compiled configs after manual spec edits

```bash
autocleaneeg-pipeline serve route sync --path /path/to/workspace
```

## Why this is safer

- Route specs are small and independent.
- Re-running setup does not touch queue history.
- Re-running setup does not touch output workspaces.
- Promotion is additive instead of copy-paste heavy.
- Compiled configs are deterministic, so drift is easier to spot.

## What still needs work

- TUI route editing should call the same backend instead of asking people to touch YAML.
- The service view should explain current route health in operator language.
- A first-run guided setup flow should sit on top of `serve route upsert`.
- If the product ever needs many independent execution lanes, the next step is SQLite or Redis-backed queue storage, not more JSON queue files.
