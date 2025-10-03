# Channel Count Accuracy Plan (Revised)

## Summary
The report correctly lists manually detected bad channels, but the channel totals stay anchored to the original import count (e.g., 32→30) even when EOG or outer-layer channels were removed. Root causes:
- `generate_bad_channels_tsv` (`src/autoclean/step_functions/reports.py:1648`) only exports categories populated by automated detection (`noisy`, `uncorrelated`, etc.), so EOG removals from `drop_eog_channels` (`src/autoclean/mixins/signal_processing/channels.py:378`) never reach `FlaggedChs.tsv`.
- The fallback in `reports.py:1454` subtracts only the channels present in `channel_dict["removed_channels"]`, which is sourced from `FlaggedChs.tsv`, leaving out unrecorded EOG drops.
- Export metadata (`save_epochs_to_set` in `src/autoclean/io/export.py:456`) does contain the verified channel count, but the report falls back before checking for that key in some runs. This plan aligns metadata and reporting so totals always reconcile with the actual saved data.

### New Review Notes
- No other in-repo module consumes `*_flagged_channels.tsv`; however, the format is documented in `README.md:117` and exposed to external tooling, so changes must stay backward compatible (header + two columns, tab separated).
- The `autoclean_review` and `autoclean_exclude` CLI tools rely on `save_epochs_to_set` metadata for post-edit exports, so any schema updates should remain JSON serializable and avoid mutating existing keys unexpectedly.
- `pipeline_runs.metadata` in the SQLite database (see `src/autoclean/utils/database.py:312`) stores a JSON dump of run metadata. Any additions to the channel-removal schema must serialize cleanly and remain parseable for compliance exports.

## Stage-Based Execution Plan

### Stage 0 – Discovery & Alignment
- **Tasks**
  - Trace `channel_dict` construction in `reports.py:1298-1350` and note current metadata dependencies.
  - Inventory all channel-removal pathways (`drop_eog_channels`, outer-layer trimming, template excludes) and document existing metadata writes.
  - Capture current outputs: `FlaggedChs.tsv`, JSON summary, `pipeline_runs.metadata`, and exported SET headers for a representative run.
  - Verify compliance mode behavior by reviewing `manage_database_conditionally` and backup logs.
- **Deliverables**
  - Short findings doc or PR comment summarizing metadata gaps and current artifacts.
  - Example artifacts archived under `test_outputs/` (or similar) for before-state reference.
- **Validation**
  - Senior reviewer signs off on discovery notes.
  - Confirm artifacts reproduce the reported mismatch (e.g., 32→30 with missing EOG drops).

### Stage 1 – Unified Removal Schema
- **Tasks**
  - Propose schema (`channel`, `reason`, `source_step`, optional `stage_timestamp`) and review with senior dev.
  - Implement a shared helper (likely in `mixins/base.py` or new utility) that appends removal entries to run metadata.
  - Update channel-removal steps (`drop_eog_channels`, outer-layer drop, manual exclude hooks) to call the helper.
- **Deliverables**
  - Schema documented in code comments and `docs/channel_count_accuracy_plan.md`.
  - Helper function with unit tests verifying serialization and deduplication.
- **Validation**
  - Run targeted unit tests for the helper.
  - Manual check that metadata now includes removal entries after each step (e.g., via debugger or logging in dev run).

### Stage 2 – Reporting & TSV Integration
- **Tasks**
  - Extend `generate_bad_channels_tsv` to merge unified removal entries, respecting existing header/format.
  - Rework `reports.py` fallback logic to prioritize exported `n_channels` and use the unified list when needed; surface mismatches as warnings.
  - Ensure JSON summary captures both the raw list and any derived counts.
- **Deliverables**
  - Updated TSV output demonstrating new reason labels (e.g., `EOG_DROPPED`) while remaining backward compatible.
  - Revised report showcasing accurate channel counts and warnings when data is missing.
- **Validation**
  - Regenerate report for the reference run and confirm totals now match expectations.
  - Senior dev review of TSV diff to confirm compatibility.

### Stage 3 – Database Propagation
- **Tasks**
  - Update the pipeline completion path (`pipeline.py:516` onwards) so the unified removal data and final counts land in `pipeline_runs.metadata`.
  - Confirm compliance mode backups succeed with the expanded metadata payload.
  - Add instrumentation or logging to flag when database updates fail.
- **Deliverables**
  - Sample `pipeline.db` entry showing the new metadata fields.
  - Notes on compliance-mode verification (e.g., backup created without errors).
- **Validation**
  - Automated or scripted check reading `pipeline.db` to verify fields.
  - Triage log review ensuring no warnings/errors in compliance runs.

### Stage 4 – Testing & Tooling
- **Tasks**
  - Add unit/integration tests covering detection-only, detection+EOG, detection+outer-layer, and exporter-missing scenarios.
  - Smoke-test CLI tooling (`autoclean_review`, `autoclean_exclude`) against runs containing the new metadata.
  - Consider lightweight regression script that runs pipeline end-to-end and asserts on report + DB outputs.
- **Deliverables**
  - Test cases under `tests/unit/` or `tests/integration/` with clear fixtures.
  - CLI smoke-test checklist with results recorded in PR.
- **Validation**
  - `make test` / `make check` passing.
  - Peer review ensuring test coverage is meaningful and non-flaky.

### Stage 5 – Documentation & Rollout
- **Tasks**
  - Update docs (`docs/tutorials/understanding_results.rst`, README channel-count section) to reflect new behavior.
  - Draft release notes covering TSV changes, JSON summary additions, and DB schema impacts.
  - Capture before/after report snippets for QA records.
- **Deliverables**
  - Doc PR sections with screenshots or code blocks illustrating new tables/counts.
  - Release-note draft ready for maintainers.
- **Validation**
  - Documentation review sign-off from senior dev or tech writer.
  - QA approval after reviewing artifacts and tests.

## Stage Dependencies
- Stage 1 depends on Stage 0 findings.
- Stage 2 requires the unified schema from Stage 1.
- Stage 3 builds on Stages 1–2 to ensure database consistency.
- Stage 4 should commence once core code changes (Stages 1–3) stabilize.
- Stage 5 follows successful testing and stakeholder sign-off.

## Stage Gate Checklist
- **After Stage 0**: Findings doc approved; baseline artifacts archived.
- **After Stage 1**: Helper API merged with tests; metadata shows new removal entries in a sample run.
- **After Stage 2**: Updated report + TSV validated; counts match expectations on reference dataset.
- **After Stage 3**: Database entry inspected; compliance-mode logs show clean backups.
- **After Stage 4**: Full test suite and CLI smoke tests pass; regression script (if added) green.
- **After Stage 5**: Docs merged; release notes circulated; QA checklist signed.

## Open Questions
- Should automatically dropped channels share the same TSV/table as manually flagged ones or remain separate with explicit reason codes?
- Do downstream analytics expect a flattened `removed_channels` list, or can we introduce nested structures without breaking consumers?
- What warning or error level is appropriate when exporter metadata is missing—silent fallback or explicit report banner?
