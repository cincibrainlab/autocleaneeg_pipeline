# Directory Backup Resolution Plan

Status: Proposal + Phase 1 implementation beginning in this PR

Owner: AutoCleanEEG Core Team

Audience: Engineering, Docs, Support

---

## Objectives

- Preserve audit integrity by not mutating historical run records.
- Make older outputs discoverable after auto‑backups/renames.
- Keep Phase 1 as a zero‑schema‑change rollout with minimal code paths touched.
- Provide a clean Phase 2 that scales and simplifies path resolution.

---

## Background

When a new run targets a task/dataset folder that already exists, the software moves the existing folder to a timestamped backup (e.g., `<name>_backup_YYYYMMDD_HHMMSS`) and re‑creates a fresh folder for the new run. Today, older DB records still point to the pre‑backup paths and don’t automatically reflect the new backup location. This makes “open an old run’s outputs” unreliable unless the user navigates manually.

---

## Plan Overview

### Phase 1: Metadata + Audit Only (no schema changes)

- Emit a `directory_backup` metadata object on the current run at the exact moment a backup is made.
- Log an audit entry for the backup event with the same fields.
- Add a path resolver that derives relocation mappings by scanning a bounded window of recent runs’ metadata, applying reverse‑chronological prefix rewrites from `moved_from` → `moved_to`.
- Apply this resolver in targeted places (e.g., CLI “report chat” default) to ensure last outputs can be found even after subsequent backups.

### Phase 2: Dedicated `directory_backups` Table

- Add a first‑class table recording each backup move: `moved_from`, `moved_to`, `effective_at`, `initiated_by_run_id`, and optional `task`/`dataset_name`.
- Update the resolver to read directly from this table (fast, deterministic) while still writing the backup info into run metadata for human readability.
- Optionally backfill from recent run metadata to seed the table.

---

## Data Captured at Backup Time

Minimal, machine‑actionable fields:

- `moved_from`: Absolute path of the previous root (the one that existed)
- `moved_to`: Absolute path of the timestamped backup
- `effective_at`: ISO timestamp of the move
- `initiated_by_run_id`: ID of the run that triggered the move
- `scope.task_root`: The root path that was moved
- `reason`: e.g., "existing directory found; moved to backup"

Storage locations:

- Phase 1: `metadata.directory_backup` on the current run; one audit log entry
- Phase 2: Also insert into `directory_backups` table; still echo into run metadata

---

## Resolver Behavior

Inputs: `stored_path` (path read from an older run record)

Algorithm:

1. If `stored_path` exists → return as‑is.
2. Else, list a bounded window of recent runs (e.g., last 50–200 or last 7–30 days).
3. Collect all `directory_backup` entries and sort by `effective_at` DESC.
4. For each entry:
   - If `stored_path.startswith(moved_from)`: compute
     `candidate = moved_to + stored_path[len(moved_from):]`.
   - If `candidate` exists → return it.
5. If none match, return `stored_path` (and log a friendly warning higher up).

Notes:

- If Phase 1 stores additional anchors under `scope` later, try most specific anchors before broad root rewrites.
- Cache results (optional) to avoid repeated scans during a single CLI invocation.

---

## Application Points

- CLI `report chat` default path (when no `--context-json` provided)
- Any future “open last outputs” commands
- Any UI panels or utilities that dereference historical run record paths

---

## Acceptance Criteria

- When a backup occurs:
  - Current run metadata includes `directory_backup` with expected fields.
  - Audit log contains a `directory_backup` event.
- For an older run whose stored paths point to pre‑backup locations:
  - The resolver returns the correct existing path (in ≥95% realistic cases).
  - Legacy run records remain immutable.
- If resolution fails:
  - Users see a clear message suggesting the backup path pattern to search for.

---

## Risks & Mitigations

- Multiple relocations: Apply newest mapping first; stop at the first existing candidate.
- User‑moved/deleted files: Resolution is best‑effort; log a helpful message and avoid exceptions.
- DB unavailability: Resolver degrades gracefully (no change to current behavior).

---

## Rollout

- Phase 1
  - Add metadata object & audit entry at backup time.
  - Implement resolver using recent run metadata.
  - Wire resolver in `report chat` default path.
- Phase 2
  - Add `directory_backups` table + writes at backup time.
  - Point resolver to table; keep metadata fallback.
  - Optional backfill.

