# Design: Idempotent ICA Re‑apply (`--changed`) + Consolidated Reporting

Authors:
- Engineering: Core developer familiar with current codebase internals
- Product: PM review focused on UX, safety, and auditability

Status: Approved (Phase 1)
Target release: 2.3.x

## Summary
Add an idempotent bulk mode to the existing ICA re‑apply workflow so that only rows whose control‑sheet edits would change `final_removed` are processed. Provide robust reporting (JSON/CSV/TXT) that summarizes deltas, affected files, run linkage, and obsolescence of prior artifacts. Keep single‑file behavior unchanged and continue to honor workspace task overrides and task‑defined tails via `resume_after_ica`.

## Current Behavior (Baseline)
- CLI `process ica` operates on a single file (active input or `--file`).
- It reads `metadata/ica_control_sheet.csv`, merges `manual_add`/`manual_drop` into `final_removed`, clears manual fields, re‑applies ICA, and auto‑resumes the tail for the one file unless `--no-continue`.
- Paths:
  - `pre_ica_path`: prefer stage FIF under `intermediate/*_pre_ica/…_pre_ica_raw.fif` (fallback to subject/eeg).
  - `ica_fif`: per‑subject under `…/derivatives/autoclean-vX.Y/sub-<id>/eeg/<basename>-ica.fif` (normalized; no duplicate segments).
- Tail resume:
  - Uses workspace override of the active task (discovery first, fallback session registry).
  - Binds module config to `self.settings` and injects YAML fallback for `_check_step_enabled`.
  - Delegates to `task.resume_after_ica` if present; else falls back to `create_regular_epochs(export=True)` → `create_eventid_epochs()` → `generate_reports()`.
  - Creates a fresh tail run linked to the manual re‑apply run (no mutation of completed parent run).

## Goals
- Batch mode that handles the entire control sheet but only touches rows that would change (`--changed`).
- Dry‑run preview (`--dry-run`) for safety.
- Consolidated report artifacts for sign‑off and addendum generation.
- Zero regressions for single‑file mode.

## CLI UX & Flags
Extend `autocleaneeg-pipeline process ica` with:
- `--changed`: Iterate all rows in the task’s control sheet; for each row, re‑apply ICA only if a pure pre‑check indicates `final_removed` would change. Otherwise skip without writes or DB runs.
- `--dry-run`: With `--changed`, print and write a report showing which rows would change and how; perform no writes/DB changes and do not resume tails.
- `--report-only`: Do not process. Generate the consolidated report from current sheet values (and pending manual fields) to show what would change.
- `--report-after`: After processing (single‑file or batch), generate the consolidated report.
- `--report-out <path>`: Optional base path (without extension) for the report artifacts; defaults under `metadata/`.

Mutual exclusivity / precedence:
- `--report-only` is exclusive with `--changed` and single‑file operation (no processing).
- If neither `--changed` nor `--report-only` is specified, behavior remains single‑file.
- `--dry-run` has effect only with `--changed` or `--report-only`.

Help text will be updated accordingly in `src/autoclean/cli.py`.

## Data and Authority
- Single authority for changes: `metadata/ica_control_sheet.csv` with columns:
  - `original_file, ica_fif, pre_ica_path, post_ica_path, auto_initial, final_removed, manual_add, manual_drop, status, last_run_iso, run_id`
- Path normalization rules (already in code):
  - Pre‑ICA source: prefer `intermediate/*_pre_ica/…_pre_ica_raw.fif`; fallback to `sub-<id>/eeg`.
  - Avoid repeated `sub-<id>/eeg` segments when saving/reading FIF paths.

## Change Detection (Pure, Idempotent)
For each row:
- Parse components: `final_prev = parse(final_removed)`, `add = parse(manual_add)`, `drop = parse(manual_drop)`.
- Compute proposed: `final_proposed = (final_prev ∪ add) − drop`.
- A row “would change” iff `final_proposed != final_prev`.
- If invalid tokens in manual fields (non‑ints), warn and skip; include in report.
- If `status == 'pending'` and manual fields are empty → treat as no‑op; list in report separately.

This pre‑check is used for both `--dry-run` and `--changed` to guarantee idempotency.

## Batch Flow (`--changed`)
1) Load control sheet rows.
2) For each row, run the pure pre‑check to compute deltas and `final_after_str`.
3) If `--dry-run`: accumulate rows with deltas and write the report; exit without writes.
4) Otherwise, for rows with deltas:
   - Resolve `pre_ica_path` / `ica_fif` using normalized discovery.
   - Re‑apply ICA using existing single‑file code path (no changes to the actual computation).
   - Update the control sheet atomically: clear `manual_*`, set `final_removed`, set `post_ica_path`, `last_run_iso`, and `status='applied'`.
   - Auto‑resume the tail for that file (delegates to `resume_after_ica` when present). Create fresh “manual_ica_tail” run linked to the manual re‑apply run.
5) At end of batch: if `--report-after`, generate the consolidated report from the updated sheet.

## Reporting (JSON/CSV/TXT)
Artifacts written to `metadata/` by default (or `--report-out` path base):
- JSON: `ica_changes_<timestamp>.json`
- CSV:  `ica_changes_<timestamp>.csv`
- TXT:  optional human summary (`ica_changes_<timestamp>.txt`) and always printed to stdout

Per‑file fields in report:
- `original_file`
- `final_removed_before`, `final_removed_after`
- `delta_added`, `delta_removed`
- `manual_add`, `manual_drop` (as consumed)
- `status_before`, `status_after`
- timestamps: `last_run_iso_before`, `last_run_iso_after`
- paths: `pre_ica_path`, `post_ica_path_after`
- run linkage (if processed): `parent_run_id` (manual re‑apply), `tail_run_id`
- `superseded_artifacts`: list of prior `post_ica_path` values superseded by this batch

`--report-only`:
- Same schema, using the “would change” computation without modifying the sheet or files.

## Implementation Details (Code‑level)

### CLI (`src/autoclean/cli.py`)
- Parser additions on `process ica`:
  - `--changed` (store_true)
  - `--dry-run` (already present)
  - `--report-only` (store_true)
  - `--report-after` (store_true)
  - `--report-out` Path (optional)
- In `cmd_process_ica`:
  - If `--report-only`: call `generate_ica_change_report(sheet, out_path, mode='report-only')`; return 0.
  - If `--changed`: call `run_ica_batch_changed(args, control_sheet, task_name, output_dir, derivatives_root, metadata_dir)`.
  - Else: existing single‑file flow.

### ICA helpers (`src/autoclean/functions/ica/ica_processing.py`)
New pure utilities:
- `def parse_components(s: str) -> set[int]`
- `def format_components(vals: set[int]) -> str`
- `def compute_proposed(final_prev: set[int], add: set[int], drop: set[int]) -> set[int]`
- `def row_would_change(row: dict) -> tuple[bool, list[int], list[int], str]` → `(would_change, added, removed, final_after_str)`

Batch runner:
- `def run_ica_batch_changed(context: dict, sheet_path: Path, ...) -> dict`:
  - Iterates rows, runs `row_would_change`, populates a list of candidates.
  - Honors `--dry-run` and returns early if requested.
  - For each candidate:
    - Resolve paths, call existing per‑file `process_ica_control_sheet(autoclean_dict)` (after injecting `autoclean_dict` from batch context with the specific row’s paths), then auto‑resume tail.
  - Returns a structured result for reporting (list of per‑file items: before/after, deltas, errors, run_ids).

Control sheet updates:
- Must remain atomic per row. Use temp write + replace or `pandas` write with caution; re‑read after write for post‑apply reporting.

### Reporting (`src/autoclean/functions/ica/reporting.py` or co‑located)
- `collect_change_rows(sheet, detection_fn, mode)` → returns list of structured deltas & warnings
- Writers:
  - `write_json(changes, out_json)`
  - `write_csv(changes, out_csv)`
  - `write_txt(changes, out_txt)` (compact human summary)

### DB & Audit
- No schema changes.
- Only create manual and tail runs when a row actually changes; no runs on `--dry-run`, `--report-only`, or unchanged rows.
- Linkage fields already available; just populate in the per‑file result for reports.

### Logging
- Batch header: `Batch ICA --changed: N total rows; M rows would change`.
- Per‑file (when processing): the same compact delta summary used in single‑file: `ICA changes: +{added or ∅} -{removed or ∅} final={after}`.
- Tail: keep existing logs (task file path used, settings gate status, resume hook delegation).
- End summary: processed/skipped/failures; report artifact paths.

## Edge Cases
- Invalid component tokens in manual fields → warn and skip row; add to report warnings.
- Missing `pre_ica_path` or `ica_fif` → attempt normalized discovery; if still missing, skip and warn (include in report).
- `status == 'pending'` with no manual fields → no‑op; list as pending‑noop in report.
- Conflict where the proposed set matches current (e.g., add & drop cancel) → skip, report as no‑op.

## Idempotency & Safety
- Pre‑check guarantees no action on unchanged rows.
- `--dry-run` performs zero writes/DB operations.
- Single‑file behavior remains unchanged.

## Testing & Validation
- Unit tests (where feasible) for pure functions: parse/format, compute_proposed, row_would_change.
- Integration tests/manual validation:
  - `--changed --dry-run`: prints and writes report; no artifacts/DB changes.
  - `--changed` then re‑run with no further edits: should perform zero work.
  - `--report-only`: writes report correctly from current sheet.
- Path normalization regression tests:
  - Ensure pre‑ICA FIF is discovered in stage first.
  - Confirm no duplicate `sub-<id>/eeg` segments appear in ica_fif or pre_ica FIF writes.

## Rollout (Phase 1)
- Implement flags, detection, batch runner, and reporting (JSON/CSV/TXT).
- Update CLI help and docs/tutorials accordingly.
- Leave PDF addendum and file‑pattern filters to Phase 2.

## PM Notes
- `--changed` should be the recommended bulk mode for labs making manual corrections.
- `--report-only` provides a safe preview for sign‑off meetings.
- Keep runs lean: only generate run records when actual changes happen.
- Ensure report language clearly labels previous artifacts as “superseded” where applicable, without deleting them.

## Open Questions
- Do we want a `--force-all` for “recompute everything” explicitly (dangerous; default off)?
- Should we optionally clear manual fields that have no effect (e.g., dropping a component not in final_removed) to reduce confusion? (Default: warn, do not auto‑clear.)
