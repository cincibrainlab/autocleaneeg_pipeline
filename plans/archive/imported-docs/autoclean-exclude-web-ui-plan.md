# AutoClean Exclude Web UI Integration Plan

## Purpose

Integrate the existing AutoClean Exclude workflow into the new React-based web UI on branch `feature/web-ui-enhancements-autoclean-ui-autoclean-exclude`, while preserving the behavior that Grace described in the March 16, 2026 SME review.

This plan is based on:

- The current web UI foundation added in commit `4be5ca8` (`Web UI: full Serve dashboard...`)
- Existing results/review infrastructure in `web/src/pages/Results.tsx` and `src/autoclean/api/routes/results.py`
- Existing PDF extraction support in `src/autoclean/api/pdf_extractor.py`
- Prior docs for the standalone exclude tool in `docs/mdx/run-autoclean-exclusion-tool.mdx` and `docs/mdx/standalone-autoclean-exclude-upgrade.mdx`
- The SME transcript from Grace Westerkamp on March 16, 2026

## Product Direction

### Core principle

The new Exclude experience should preserve the parts of the workflow Grace explicitly relies on, while adopting the visual language, interaction patterns, and overall aesthetic of the current web UI.

This is not a requirement to copy the legacy tool wholesale. The transcript supports preserving specific review and reprocess behaviors, but it does not require duplicating the old desktop layout or interaction model everywhere.

### What the transcript clearly supports preserving

- Start from already preprocessed exports
- Default to the workspace `exports/` area without forcing the user to browse manually
- Let the user choose one file and review it in detail
- Preserve access to the review surfaces Grace named:
  - Time series
  - PSD overview
  - Run report PDF
  - ICA components PDF
  - Reprocess / overrides
- Keep the PDF-driven ICA review workflow
- Keep manual override behavior for:
  - ICA components to remove
  - Bad channels to remove
- Keep processing metrics visible for the selected file
- Keep notes for unusual files
- Keep reprocessing semantics the same: rerun using the same task pipeline, but apply the reviewer overrides at the relevant stage

### What should be modernized

- The overall page layout can differ to fit the new app shell
- The experience can be one-file-at-a-time instead of a separate desktop-style multi-pane app
- The UI should match the current web app's aesthetic and interaction model rather than mimic the desktop tool mechanically
- All interactions currently implemented in the Qt tool need browser-native equivalents
- Dense legacy controls should be simplified where that does not remove expert capability

### Explicit migration requirement

The Exclude workflow must move into the browser completely.

That includes:

- EEG review and bad-epoch marking
- PDF review
- ICA override editing
- Bad-channel override editing
- Notes
- Reprocess triggering and status

The browser-native EEG review surface is not optional or a later nice-to-have. It is a core deliverable.

### What Grace explicitly said about look and feel

- She wants the reprocess workflow to remain basically the same
- She does not want the PDFs changed in any major way
- She said the tool "could be exactly the same" as a preference, but also accepted that the new application would look different

Interpretation:

The strongest requirement is workflow continuity for expert review and reprocessing, not visual duplication of the old tool.

## Current State

### Existing web UI assets

- The new React app already has routing, layout, theming, keyboard shortcuts, and a substantial results page
- `web/src/pages/Results.tsx` already supports:
  - Run list
  - Run detail
  - Metric summaries
  - Run report viewing
  - ICA report viewing
  - Notes for pass/fail/review decisions
- `src/autoclean/api/routes/results.py` already aggregates outputs and serves artifacts
- `src/autoclean/api/pdf_extractor.py` already parses ICA PDFs into structured component summaries and page mappings

### Key gap

The current web UI supports review of completed runs, but not the full AutoClean Exclude editing loop:

- No file-centric Exclude workspace
- No explicit override editor for ICA or bad channels
- No time-series rejection workflow equivalent to the old tool
- No reprocess-with-overrides action wired into the web app
- No explicit "open at exports by default" Exclude flow

## Proposed Integration Shape

### Recommended approach

Implement AutoClean Exclude as a first-class web workflow, not as a thin wrapper around the old standalone tool.

Important distinction:

- The UI should be rebuilt as a browser-native experience in `web/`
- The backend can be rewritten, but it should reproduce the behavior and outputs of the current `src/autoclean/tools/autoclean_exclude.py` workflow

That existing tool already contains real domain logic for:

- decision persistence
- file-key and relative-path handling
- bad epoch capture and derived epoch metrics
- metadata parsing for bad channels and rejected ICA
- related-file discovery
- manual-fix payload generation
- AST-based reprocess task generation
- reprocess result copy/merge behavior
- QA export and QA preprocessing log generation

The plan treats those behaviors as the reference implementation for correctness. The new backend does not need to reuse the same code paths, but it does need a deliberate equivalence plan so that the browser version behaves the same way.

### Recommended navigation model

Add a dedicated Exclude area in the web app with one-file-at-a-time detail review.

Two viable placement options:

1. Add a new top-level page: `Exclude`
2. Add an `Exclude` mode inside `Results`

### Recommendation

Prefer a top-level `Exclude` page.

Reasoning:

- The workflow is deeper than the current pass/fail/review decision model
- It needs file selection, tabbed review, override editing, and reprocess actions
- Keeping it separate avoids overloading `Results.tsx`, which is already large and multi-purpose
- The mental model is cleaner: `Results` for viewing and high-level review, `Exclude` for surgical override editing and reruns
- A dedicated page gives us room to make the workflow feel modern instead of forcing it into the old desktop structure

## Target User Flow

1. User opens the web app
2. User navigates to `Exclude`
3. The page auto-resolves the current workspace `exports/` directory when available
4. The page lists exported/preprocessed files
5. User selects one file
6. The detail panel opens with familiar tabs
7. User reviews:
   - Time series
   - PSD overview
   - Run report
   - ICA report
   - Processing metrics
   - Notes
8. User adds or removes override entries:
   - ICA components to reject
   - Bad channels to reject
9. User optionally marks additional bad epochs from the time-series view
10. User clicks `Reprocess with Overrides`
11. The system runs the same processing path with updated manual overrides
12. The UI updates status, progress, and final output for that file

## UI Integration Plan

The Exclude workflow should be integrated as a normal page inside the existing web application shell.

### Integration principles

- Use the existing app shell:
  - sidebar
  - top bar
  - workspace context
  - current theme and page spacing
- Do not launch a separate tool window
- Do not recreate the old Qt layout literally
- Keep Exclude visually consistent with the current Serve UI
- Give EEG review the main visual priority on the page

### Route placement

Recommended route:

- `/exclude`

Recommended nav placement:

- Top-level sidebar item near `Results`

Recommended relationship to existing pages:

- `Results`: review completed runs broadly
- `Exclude`: do deep file-level review, manual epoch rejection, overrides, and reprocess

### Page regions

The page should be organized into three primary regions inside the standard app shell.

#### 1. Left Rail

Purpose:
File selection and lightweight review status.

Contents:

- file search
- file list
- note indicator
- epoch-review indicator
- override indicator
- reprocess status indicator

#### 2. Main Review Panel

Purpose:
Primary review workspace for the selected file.

Contents:

- selected file header
- review tabs
- EEG review view
- PDF views
- PSD view

This is the dominant visual region of the page.

#### 3. Inspector Panel

Purpose:
Persistent file-specific context and actions.

Contents:

- processing metrics
- notes
- ICA overrides
- bad-channel overrides
- reprocess action
- save/reprocess status

This can live as:

- a right rail on wider screens
- a lower stacked section on narrower screens

### Wireframe

Desktop concept:

```text
+---------------------------------------------------------------+
| App Shell: Sidebar | Top Bar                                  |
+---------------------------------------------------------------+
| LeftRail           | MainReviewPanel            | Inspector    |
|                    |                            | Panel        |
| Search             | FileHeader                 | Metrics      |
| File list          | -------------------------  | Notes        |
| - file A           | Tabs: EEG | PSD | Report  | ICA Overrides|
| - file B           |       | ICA | Metadata     | Ch Overrides |
| - file C           |                            | Reprocess    |
|                    | Active tab content         | Status       |
| status chips       |                            |              |
+---------------------------------------------------------------+
```

Tablet/mobile concept:

```text
+--------------------------------------+
| App Shell                            |
+--------------------------------------+
| FileHeader                           |
| Tabs                                 |
| Active review content                |
| -----------------------------------  |
| Inspector sections                   |
| Metrics                              |
| Notes                                |
| Overrides                            |
| Reprocess                            |
| -----------------------------------  |
| File drawer / file picker            |
+--------------------------------------+
```

### Tab model

Recommended tabs in the main review panel:

- `EEG`
- `PSD`
- `Run Report`
- `ICA`
- `Metadata` or `Details`

Notes:

- `EEG` should be the default tab
- `ICA` should be easy to reach because it is central to override review
- `Metadata` is lower priority and should not compete visually with EEG/ICA

### Inspector panel model

Recommended inspector sections:

- `Processing Metrics`
- `Notes`
- `ICA Overrides`
- `Bad Channel Overrides`
- `Reprocess`

Recommended behavior:

- Keep inspector state tied to the selected file
- Preserve unsaved edits when switching tabs within the same file
- Force save or confirm before switching files if needed

### Integration checklist

- [x] Add `Exclude` as a top-level route in the current web app
- [x] Add `Exclude` to the sidebar near `Results`
- [x] Use the existing layout shell rather than a custom full-screen tool page
- [x] Make the EEG view the main review surface
- [x] Keep file selection visible without overwhelming the page
- [x] Keep notes, metrics, overrides, and reprocess controls in a persistent inspector area
- [x] Ensure the layout collapses cleanly on smaller screens

## Views

### 1. Exclude File Browser View

Purpose:
Present the exports-root default and let the user select a file quickly.

Checklist:

- [x] Resolve workspace-aware default exports path
- [x] Show current workspace and exports root clearly
- [x] List eligible files only
- [x] Support filename search/filter
- [x] Show quick status badges if override or notes already exist
- [x] Preserve selected file while background refreshes happen
- [x] Handle missing or empty exports folders cleanly

Suggested fields per row:

- Filename
- Task or pipeline label if known
- Last processed timestamp
- Existing override state
- Existing note indicator
- Reprocess status

### 2. Exclude Detail Workspace

Purpose:
Provide the single-file review experience.

Checklist:

- [x] Keep one selected file active at a time
- [x] Make file identity obvious at the top of the panel
- [x] Show processing status and provenance
- [x] Keep primary actions visible without scrolling too far
- [x] Preserve state when switching tabs

Suggested layout:

- Left rail: file list
- Main panel: tabbed content
- Right rail or lower panel: overrides, metrics, notes, reprocess action

Design note:

This should look like a native extension of the current Serve web UI, not like a browser recreation of a Qt window.

If mobile support is required for this workflow, collapse the file list into a drawer and stack detail sections vertically.

### 3. Time Series View

Purpose:
Provide a browser-native EEG review surface that replaces the embedded Qt/MNE epoch browser.

Checklist:

- [x] Load epoch-level EEG data in a browser-consumable format
- [x] Render channel traces in an interactive web view
- [x] Support epoch-to-epoch navigation
- [x] Support keyboard-driven review
- [x] Support click or key toggle for bad/unbad epoch marking
- [x] Show which epochs are already marked bad
- [x] Autosave epoch edits when switching files or leaving the view
- [x] Preserve the current `postedit` semantics or define the browser-native equivalent explicitly
- [ ] Keep interaction latency acceptable on realistic datasets

Legacy behavior to preserve from the current tool:

- `Review Selected File` opens the EEG review surface
- Left/right navigation through epochs
- `Space` toggles bad epoch state
- Marked bad epochs persist automatically
- Edited outputs feed the downstream exclusion/reprocess flow

Non-goal:

Do not embed the old MNE/Qt browser inside the web app. Build a web-native EEG viewer.

### Browser-native EEG view implementation plan

#### Phase TS-1: Define the browser data contract

- [x] Identify the canonical source for epoch data in the current Exclude flow
- [x] Decide whether to read from `.set`, cached metadata, or a derived lightweight payload
- [x] Define an API payload for:
  - channel labels
  - sampling rate
  - epoch count
  - epoch duration
  - visible trace data
  - existing bad epoch indices
- [x] Define how paging, windowing, or downsampling works for performance

#### Phase TS-2: Build the web EEG viewer foundation

- [x] Choose rendering strategy for dense traces
- [x] Implement channel stack rendering
- [x] Implement epoch navigation controls
- [x] Implement keyboard shortcuts
- [x] Implement bad epoch toggle interactions
- [x] Implement visible markers for selected and rejected epochs

#### Phase TS-3: Persist manual epoch edits

- [x] Define API for saving bad epoch indices
- [x] Save automatically on file switch, tab switch, and explicit reprocess
- [x] Handle conflicts between existing saved state and unsaved browser edits
- [x] Expose save status in the UI

#### Phase TS-4: Connect to downstream outputs

- [x] Confirm how the current tool writes `postedit` outputs
- [x] Decide whether the browser path writes the same artifact or stores epoch overrides separately until rerun
- [x] Ensure manual epoch edits are included in QA/export/reprocess flows
- [ ] Validate parity on a real reviewed file

### 4. PSD Overview View

Purpose:
Show the PSD summary Grace expects during exclusion review.

Checklist:

- [x] Reuse existing output artifacts if they already exist
- [x] Prefer static rendered figures first
- [x] Support zoom/open-in-new-tab if needed
- [x] Handle missing PSD outputs without breaking the page

### 5. Run Report PDF View

Purpose:
Keep the current PDF review experience intact.

Checklist:

- [x] Reuse existing results/PDF serving endpoints where possible
- [x] Start at page 1 by default
- [x] Preserve the current PDF appearance
- [x] Show a clear empty state when missing

### 6. ICA Components View

Purpose:
Preserve the PDF-based ICA review flow and expose override editing next to it.

Checklist:

- [x] Render the ICA PDF in-app
- [x] Reuse extracted component summary data from `pdf_extractor.py`
- [x] Show component list with:
  - label
  - type
  - confidence
  - currently rejected state
- [x] Allow user to add a kept component to the reject override list
- [x] Allow user to remove a previously rejected override
- [x] Keep the override list visible while reading the PDF
- [x] Preserve the "look at topography first" workflow as much as possible

Recommended enhancement:

Link summary rows to the PDF page for that component when page mappings are available.

Design note:

The PDF itself should remain familiar, but the surrounding controls can be cleaner and more modern than the legacy tool.

### 7. Reprocess / Overrides View

Purpose:
Make the manual override workflow feel nearly identical to the current tool.

Checklist:

- [x] Separate sections for ICA overrides and bad channel overrides
- [x] Show current pipeline-rejected items
- [x] Show manual additions/removals distinctly
- [x] Support add/remove interactions with minimal clicks
- [x] Prevent invalid duplicate entries
- [x] Explain exactly what reprocessing will do
- [x] Show last reprocess status and timestamp

Required actions:

- Add ICA component to removal list
- Remove ICA component from removal list
- Add bad channel to removal list
- Remove bad channel from removal list
- Trigger reprocess with overrides

### 8. Metrics and Notes View

Purpose:
Retain the small but important context Grace uses during review.

Checklist:

- [x] Show processing metrics for the selected file
- [x] Include retained channels, epochs kept, ICA removed, and similar summary values
- [x] Keep a lightweight free-text notes field
- [x] Persist notes independently from reprocess execution
- [x] Make notes visible in the file list as an indicator only, not full text

## Backend Work Plan

### Phase 1: Data contract and discovery endpoints

Goal:
Create the minimum backend needed to drive the Exclude page cleanly.

Checklist:

- [x] Define canonical Exclude file identity
- [x] Add endpoint to resolve the default exports root from the active workspace
- [x] Add endpoint to list Exclude-eligible files
- [x] Add endpoint to fetch Exclude detail for one file
- [x] Decide whether to extend `results.py` or add a dedicated `exclude.py` route module
- [x] Reuse existing artifact discovery logic where possible

Recommended output model for file detail:

- File metadata
- Available artifacts
- Processing metrics
- Existing notes
- Existing ICA override state
- Existing bad-channel override state
- Reprocess status

### Phase 2: Artifact and review data access

Goal:
Expose the data needed for the tabbed review views.

Checklist:

- [x] Reuse run report PDF serving
- [x] Reuse ICA PDF serving
- [x] Expose structured ICA component summaries from `pdf_extractor.py`
- [x] Expose PDF page mappings for component navigation
- [x] Expose PSD artifact URLs or generated previews
- [x] Add a browser-native EEG review data endpoint instead of relying on desktop viewer behavior
- [x] Define how bad epoch state is loaded into the browser EEG viewer

### Phase 3: Override persistence

Goal:
Persist reviewer edits in a way that is explicit, inspectable, and safe.

Checklist:

- [x] Define override storage format
- [x] Key overrides by stable file identity, not fragile UI state
- [x] Persist ICA override additions/removals
- [x] Persist bad-channel override additions/removals
- [x] Persist notes
- [x] Track modified timestamp and source
- [x] Decide whether overrides live near outputs, in workspace metadata, or in run DB adjunct storage

Recommended principle:

Keep override persistence logically separate from pass/fail/review decisions. They serve different purposes and should not be conflated in the API contract.

For v1, it is acceptable to keep using the existing `autoclean_exclusion_decisions.json/csv` storage model if the new backend writes equivalent data. The requirement is behavioral compatibility and clean API ownership, not preservation of the current Qt code structure.

### Phase 4: Reprocess execution path

Goal:
Run the original pipeline logic with reviewer-specified overrides.

Checklist:

- [x] Identify the existing CLI or task-entry point used by the standalone exclude flow
- [x] Confirm how manual ICA and bad-channel overrides are currently passed into processing
- [x] Add an API action to trigger rerun for a selected file with overrides
- [x] Return job/task identifiers so the web app can track progress
- [x] Surface progress and completion in the UI
- [x] Handle rerun failures with actionable error messages
- [x] Define whether reprocessing replaces outputs or creates a new derivative version

Critical design decision:

Do not guess at reprocess semantics. Match the current standalone behavior exactly unless there is a strong reason to change it.

## Frontend Work Plan

### Phase 5: Exclude page shell

Checklist:

- [x] Add route and sidebar entry
- [x] Build file list panel
- [x] Build selected-file header
- [x] Build shared loading and error states
- [x] Add polling or refresh behavior where appropriate
- [x] Keep styling aligned with the existing `web/` visual system
- [x] Deliberately avoid porting the legacy desktop layout one-to-one

### Phase 6: Review tabs

Checklist:

- [x] Implement summary/metrics panel
- [x] Implement run report tab
- [x] Implement ICA tab
- [x] Implement PSD tab
- [x] Implement notes panel
- [x] Implement a real browser-native EEG review tab

### Phase 7: Override editor UX

Checklist:

- [x] Build ICA override controls
- [x] Build bad-channel override controls
- [x] Show baseline pipeline decisions separately from manual edits
- [x] Show dirty state before save/reprocess
- [x] Prevent accidental override loss when switching files
- [x] Add confirmations only where destructive or ambiguous

### Phase 8: Reprocess UX

Checklist:

- [x] Add `Reprocess with Overrides` primary action
- [x] Disable action when no file is selected
- [x] Disable or guard action during an active rerun
- [x] Show success, failure, and in-progress states clearly
- [x] Refresh file detail automatically after completion

## Delivery Phases

### Phase A: Foundation

- [x] Confirm integration placement: top-level `Exclude` page
- [x] Confirm file identity and storage model
- [x] Confirm reprocess entry point
- [x] Add backend listing/detail endpoints
- [x] Add frontend page shell

### Phase B: Preserve the familiar review experience

- [x] Browser-native EEG review tab
- [x] Run report PDF tab
- [x] ICA PDF tab with component summary
- [x] Metrics panel
- [x] Notes panel
- [x] Default exports-root behavior

### Phase C: Add real Exclude editing

- [x] ICA override editing
- [x] Bad-channel override editing
- [x] Reprocess action
- [x] Job progress/status feedback

### Phase D: Close the parity gap

- [ ] Validate browser EEG viewer against current Exclude workflow
- [x] Epoch rejection editing
- [x] PSD overview parity
- [x] Keyboard shortcuts tuned for high-throughput review

Note:

Parity here should mean capability parity where it matters, not visual parity for its own sake.

### Phase E: Hardening

- [x] Error handling
- [x] State persistence
- [x] Integration tests
- [ ] User walkthrough with Grace or another power user

## Testing Plan

### Backend

- [x] Unit tests for Exclude file discovery
- [x] Unit tests for override serialization/deserialization
- [x] Unit tests for ICA summary extraction and page mapping reuse
- [x] API tests for file detail and reprocess endpoints
- [x] API tests for empty/missing workspace edge cases

### Frontend

- [x] Component tests for file list and detail loading states
- [x] Component tests for override add/remove flows
- [x] Component tests for notes persistence behavior
- [x] UI tests for tab switching and selected-file state retention
- [x] UI tests for reprocess status updates

### Workflow validation

- [ ] Validate against a real workspace with exports already present
- [ ] Validate that the default folder lands in the expected exports location
- [ ] Validate ICA override behavior on a file where the user disagrees with the classifier
- [ ] Validate bad-channel override behavior from the same file
- [ ] Validate that rerun output reflects the manual overrides

## Risks

### Browser EEG risk

The current tool depends on a Qt/MNE epoch viewer with fast keyboard-driven review. Replacing that in the browser is feasible, but it needs a deliberate rendering and data-loading strategy rather than being treated as a simple port.

### Reprocess semantics risk

If the current standalone tool has undocumented assumptions about task files, workspace layout, or intermediate data reuse, a naive web-triggered rerun could behave differently.

### Artifact discovery risk

The Exclude workflow assumes reports, exports, and metadata can all be resolved reliably from the selected file. Historical or partially processed workspaces may break that assumption.

### Scope creep risk

Trying to merge pass/fail/review decisions, exclude overrides, notes, and rerun orchestration into one large page could produce a brittle UI. Keeping Exclude separate reduces this risk.

### Legacy-copy risk

If the team treats "preserve workflow" as "clone the old tool," the result may fight the current web app architecture and aesthetic instead of benefiting from it.

### Rewrite risk

If the team rewrites Exclude backend behavior without continuously checking it against the current `src/autoclean/tools/autoclean_exclude.py` behavior, the web flow is likely to drift from the existing workflow in subtle but important ways.

## Resolved Decisions

### Product and UX

- [x] Lock the plan to a top-level `Exclude` page
- [x] Use one-file-at-a-time review inside the current shell
- [x] Keep the file list visible on desktop while reviewing
- [x] Preserve `Space` plus left/right navigation as the core EEG shortcuts
- [x] Keep notes as lightweight free text for v1
- [x] Drop the legacy Qt layout while preserving the legacy workflow

### Data and storage

- [x] Use `file_key` plus `relative_path` as the reviewed-export identity
- [x] Store v1 override state in `autoclean_exclusion_decisions.json/csv` under the exports root
- [x] Use current-state persistence plus a monotonic revision token for saves
- [x] Reprocessing writes replacement outputs into the original task folder after completion

### Reprocessing

- [x] Use the standalone-style `autocleaneeg-pipeline process` path for web-triggered reprocess
- [x] Trigger reprocess for a single selected file
- [x] Represent epoch rejections as persisted `bad_epoch_indices`, `bad_epoch_times`, and `bad_epoch_events`
- [x] Support manual channel and ICA overrides independently or together

### Technical implementation

- [x] Use a browser-native canvas renderer for dense EEG traces
- [x] Serve epoch data as paged, downsampled windows
- [x] Save manual bad epochs in the same decision record shape the legacy tool uses
- [x] Put the new endpoints in a dedicated `exclude.py` route module
- [x] Use polling for rerun progress in v1
- [x] Reuse the app shell and patterns from `Results.tsx` without forcing deep component extraction first

## Legacy Logic Reference Map

The old Exclude tool should be treated as the behavioral reference while we rewrite the backend and replace the UI.

### Legacy behaviors that must be matched

- Decision persistence:
  - `_load_decisions`
  - `_commit_decisions`
- Stable file identity:
  - `_record_key`
  - `_relative_path`
- Epoch review state capture:
  - `_capture_bad_epochs_for_current_file`
- Processing metrics derivation:
  - `_read_processing_log_file`
  - `_build_processing_metrics`
  - `_update_processing_metrics_for_file`
- Metadata parsing:
  - `_parse_metadata_json`
- Related-file discovery:
  - `_gather_related_files`
- Manual fix payload generation:
  - `_save_reprocess_payload`
- Reprocess orchestration:
  - `_trigger_reprocess_with_overrides`
  - `_generate_reprocess_task_from_original`
  - `_merge_reprocess_database`
- QA export and QA log behavior:
  - `_batch_export_to_qa`
  - `_create_qa_preprocessing_log`

### How to use the legacy logic in this project

- Do not copy the Qt UI structure forward
- Do not assume the exact existing methods are the right backend abstraction boundaries
- Do use the old logic as the source of truth for:
  - inputs
  - outputs
  - side effects
  - saved fields
  - file locations
  - reprocess behavior

### Equivalence checklist

- [x] Document the current inputs and outputs for decision save/load
- [x] Document the exact saved fields for epoch review state
- [x] Document the exact metadata fields used for bad channels and rejected ICA
- [x] Document the manual-fix payload format produced today
- [x] Document the current reprocess side effects on outputs, metadata, and databases
- [x] Document the current QA export side effects and generated files
- [x] Add tests or fixtures that compare rewritten backend behavior against known legacy outputs

## Browser EEG Viewer Spec

This section translates the current Qt/MNE review behavior into a concrete browser-native implementation target.

### Current behavior we are replacing

From the current desktop implementation in [src/autoclean/tools/autoclean_review.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/tools/autoclean_review.py) and [src/autoclean/tools/autoclean_exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/tools/autoclean_exclude.py):

- The reviewer opens one selected export for epoch review
- Epoch data is loaded from the `.set` file
- The viewer shows multiple epochs at once
- The reviewer navigates across epochs
- `Space` toggles bad epoch state
- Previously marked bad epochs are restored into the viewer
- Bad epoch edits are persisted automatically
- Bad epoch metadata is stored separately from pass/fail/review status
- Batch export and QA reporting later consume the persisted bad-epoch indices

### Browser viewer goals

- Replace the embedded MNE/Qt browser with a native web EEG review surface
- Preserve high-throughput expert review behavior
- Make browser state explicit and API-driven
- Keep persistence compatible with downstream QA export and reprocess flows

### Viewer scope for v1

Required in v1:

- Epoch-level EEG review
- Multi-channel trace display
- Epoch navigation
- Keyboard shortcuts
- Toggle bad/unbad epoch
- Restore previously marked bad epochs
- Autosave edits
- Dirty/saving/saved UI state

Can wait until later if necessary:

- Raw continuous review mode
- Advanced scaling controls
- Per-channel hide/show customization beyond basic visibility
- Rich annotation overlays beyond bad-epoch marking

### Canonical file identity

Use the same identity model the current Exclude tool already uses:

- `file_key`: file path relative to the exports root, without extension, used as the stable decision key
- `relative_path`: file path relative to the exports root, with extension, used to resolve the source file

Recommendation:

- Keep `file_key` as the primary frontend/backend identifier for Exclude state
- Keep `relative_path` in API payloads for traceability and file access

### Canonical epoch-review state

The browser viewer should treat epoch edits as first-class persisted state, not as a side effect of pass/fail/review.

Recommended persisted shape:

```json
{
  "file_key": "subject01/sessionA/file_comp_epo",
  "relative_path": "subject01/sessionA/file_comp_epo.set",
  "epochs_reviewed": true,
  "bad_epoch_indices": [1, 7, 9],
  "bad_epochs_count": 3,
  "total_epochs": 120,
  "epoch_rejection_rate": 2.5,
  "bad_epoch_times": [12.5, 52.5, 67.5],
  "bad_epoch_events": ["101", "101", "101"],
  "last_updated": "2026-03-16 16:45:00"
}
```

Storage note:

- The existing JSON/CSV decisions store can be reused initially
- Longer term, epoch review state and reviewer status/notes should probably be separate backend records even if they are exported together

### EEG data API

#### 1. Viewer manifest endpoint

Purpose:
Return lightweight metadata needed to initialize the browser viewer before trace payloads are requested.

Suggested route:

- `GET /api/exclude/files/{file_key}/eeg/manifest`

Suggested response:

```json
{
  "file_key": "subject01/sessionA/file_comp_epo",
  "relative_path": "subject01/sessionA/file_comp_epo.set",
  "mode": "epochs",
  "sampling_rate": 250.0,
  "channel_names": ["Fp1", "Fp2", "F3"],
  "n_channels": 64,
  "n_epochs": 120,
  "epoch_length_samples": 500,
  "epoch_duration_seconds": 2.0,
  "existing_bad_epoch_indices": [1, 7, 9],
  "default_scaling_uv": 25.0,
  "visible_epoch_count": 10
}
```

#### 2. Epoch data endpoint

Purpose:
Return only the epoch traces needed for the current visible window.

Suggested route:

- `GET /api/exclude/files/{file_key}/eeg/epochs?start=0&count=10&channels=Fp1,Fp2,F3`

Suggested response:

```json
{
  "file_key": "subject01/sessionA/file_comp_epo",
  "start_epoch": 0,
  "count": 10,
  "channel_names": ["Fp1", "Fp2", "F3"],
  "sampling_rate": 250.0,
  "epoch_duration_seconds": 2.0,
  "epochs": [
    {
      "epoch_index": 0,
      "event_code": "101",
      "start_time_seconds": 0.0,
      "is_bad": false,
      "traces_uv": {
        "Fp1": [0.1, 0.2, 0.3],
        "Fp2": [0.1, 0.2, 0.3],
        "F3": [0.1, 0.2, 0.3]
      }
    }
  ]
}
```

Implementation notes:

- Use server-side downsampling if needed to keep payload size bounded
- Return microvolt-scaled values so the frontend can stay simple
- The endpoint should support channel subsets to prevent oversized payloads

#### 3. Epoch review state endpoint

Purpose:
Load and save the canonical bad-epoch state independently of the trace payload.

Suggested routes:

- `GET /api/exclude/files/{file_key}/epoch-review`
- `PUT /api/exclude/files/{file_key}/epoch-review`

Suggested save payload:

```json
{
  "bad_epoch_indices": [1, 7, 9],
  "client_revision": 3
}
```

Suggested save response:

```json
{
  "saved": true,
  "bad_epochs_count": 3,
  "total_epochs": 120,
  "epoch_rejection_rate": 2.5,
  "last_updated": "2026-03-16 16:45:00",
  "server_revision": 4
}
```

### Frontend EEG component architecture

Recommended React structure:

- `ExcludePage`
- `ExcludeFileList`
- `ExcludeFileHeader`
- `ExcludeTabs`
- `EegReviewPanel`
- `EegTraceCanvas`
- `EpochOverviewRail`
- `EpochReviewToolbar`
- `EpochSaveStatus`

Responsibilities:

- `EegReviewPanel`: orchestration, loading, keyboard events, save lifecycle
- `EegTraceCanvas`: trace rendering only
- `EpochOverviewRail`: compact epoch strip showing good/bad/selected states
- `EpochReviewToolbar`: navigation, scale, channel density, reset actions
- `EpochSaveStatus`: dirty/saving/saved/conflict state

### Interaction model

#### Required keyboard shortcuts

- `Left Arrow`: previous epoch window or previous epoch focus
- `Right Arrow`: next epoch window or next epoch focus
- `Space`: toggle bad state for focused epoch
- `Shift+Left/Right`: jump by larger review window
- `Cmd/Ctrl+S`: force save current epoch-review state

#### Required mouse interactions

- Click epoch in overview rail to jump to it
- Click epoch card/header to focus it
- Click bad-epoch badge or action to toggle state
- Optional: drag or marquee interactions only if they add value without slowing users down

#### Required visual cues

- Focused epoch is obvious
- Bad epochs are obvious
- Unsaved changes are obvious
- Save failures are obvious
- Current file and epoch range are always visible

### Rendering strategy

Recommendation:

- Render traces in a canvas-based component rather than SVG DOM nodes
- Keep axes/grid lightweight
- Render only visible epochs and visible channels
- Support channel virtualization for high-channel-count files if needed

Performance targets:

- Initial viewer load under 1 second for a typical file once manifest is available
- Epoch-window navigation should feel immediate
- Toggling bad state should update instantly in the UI before persistence returns

### Save model

Recommendation:

- Optimistic UI for epoch toggles
- Debounced autosave after edits
- Forced save on file switch, tab switch, and reprocess trigger
- Background conflict detection via `server_revision`

Save triggers:

- Toggle bad epoch
- Leave selected file
- Leave EEG tab
- Trigger reprocess
- Explicit save shortcut

### Data derivations the backend should own

The backend should compute and persist:

- `bad_epochs_count`
- `total_epochs`
- `epoch_rejection_rate`
- `bad_epoch_times`
- `bad_epoch_events`
- `last_updated`

The frontend should own only transient UI state such as:

- currently visible epoch range
- selected epoch
- current channel subset
- local dirty state
- save-in-flight state

### Integration with existing Exclude state

The browser EEG viewer must integrate with the rest of the Exclude record for the same file:

- notes
- reviewer status
- bad-channel overrides
- ICA overrides
- reprocess state

Recommendation:

- Keep a single Exclude file detail query that returns all current review state
- Use targeted mutation endpoints for:
  - epoch review
  - notes
  - overrides
  - reprocess action

### QA export and postedit implications

The current tool uses persisted `bad_epoch_indices` to drive QA export and related metrics updates.

That means the browser implementation must preserve one of these models explicitly:

1. Persist bad-epoch indices only, and derive postedit output during QA export or rerun
2. Persist bad-epoch indices and also write a `postedit` artifact immediately after save

Recommendation:

Start with option 1 unless there is an external dependency on immediate `postedit` file creation.

Reasoning:

- It keeps the browser save path fast
- It avoids writing large EEG artifacts on every toggle
- It centralizes artifact generation in QA export and reprocess paths

Open question:

- Do any existing workflows outside the desktop tool expect the `postedit` artifact to appear immediately after review?

### Validation checklist for the EEG viewer

- [x] A reviewed file restores its prior bad-epoch state correctly
- [x] `Space` toggles bad state for the focused epoch
- [x] Left/right navigation matches user expectation
- [x] Autosave occurs on file switch and tab switch
- [x] Saved state survives page refresh
- [x] QA export reads the same bad-epoch indices the viewer saved
- [x] Reprocess uses the saved epoch review state correctly
- [ ] The viewer remains usable on realistic channel and epoch counts

## Engineering Task Tracker

Use this section as the implementation checklist. Items are grouped by execution order and include likely files to touch.

### Phase 0: Discovery and alignment

Goal:
Lock the technical decisions that affect every later implementation step.

Files to inspect:

- [src/autoclean/tools/autoclean_exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/tools/autoclean_exclude.py)
- [src/autoclean/tools/autoclean_review.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/tools/autoclean_review.py)
- [web/src/pages/Results.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Results.tsx)
- [src/autoclean/api/routes/results.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/results.py)

Checklist:

- [x] Confirm `Exclude` will be a dedicated top-level web page
- [x] Confirm the canonical file identifier will be `file_key` plus `relative_path`
- [x] Confirm whether epoch-review state stays in `autoclean_exclusion_decisions.json/csv` for v1
- [x] Confirm whether browser saves should create `postedit` artifacts immediately or defer artifact creation to QA export/reprocess
- [x] Confirm whether reprocess endpoints belong in a new `exclude.py` API route module
- [x] Confirm acceptable performance target for the EEG viewer on typical files
- [x] Write down the legacy input/output contract for the old Exclude backend behaviors

### Phase 1: Backend Exclude API foundation

Goal:
Create the backend API surface for the Exclude page and browser EEG viewer.

Likely files to touch:

- [src/autoclean/api/routes/__init__.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/__init__.py)
- [src/autoclean/api/server.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/server.py)
- [src/autoclean/api/routes/results.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/results.py)
- [src/autoclean/api/routes/exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/exclude.py)
- [src/autoclean/api/models.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/models.py)

Checklist:

- [x] Add Exclude route module or extend route registration cleanly
- [x] Add endpoint to resolve default exports root from current workspace
- [x] Add endpoint to list Exclude files
- [x] Add endpoint to fetch Exclude file detail
- [x] Add endpoint to load notes/status/override/epoch-review state for one file
- [x] Add endpoint to save notes/status state
- [x] Add endpoint to save bad-channel and ICA override state
- [x] Add endpoint to trigger reprocess with overrides
- [x] Add endpoint to fetch reprocess status for a file or job
- [x] Ensure the rewritten API behavior is checked against the legacy Exclude outputs

### Phase 2: Backend EEG data endpoints

Goal:
Serve browser-native EEG review data instead of relying on the Qt viewer.

Likely files to touch:

- [src/autoclean/api/routes/exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/exclude.py)
- [src/autoclean/api/models.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/models.py)
- [src/autoclean/io/export.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/io/export.py)
- [src/autoclean/utils/path_resolution.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/utils/path_resolution.py)

Checklist:

- [x] Add EEG manifest endpoint
- [x] Add paged epoch-trace endpoint
- [x] Add epoch-review state load endpoint
- [x] Add epoch-review save endpoint
- [x] Add backend logic to compute `bad_epochs_count`
- [x] Add backend logic to compute `total_epochs`
- [x] Add backend logic to compute `epoch_rejection_rate`
- [x] Add backend logic to compute `bad_epoch_times`
- [x] Add backend logic to compute `bad_epoch_events`
- [x] Add revision field or equivalent conflict token for saves
- [x] Add server-side downsampling/windowing strategy for trace payloads

### Phase 3: Backend persistence and compatibility

Goal:
Keep browser-native state compatible with the current Exclude persistence and QA flows.

Likely files to touch:

- [src/autoclean/tools/autoclean_exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/tools/autoclean_exclude.py)
- [src/autoclean/api/routes/exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/exclude.py)
- [src/autoclean/utils/database.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/utils/database.py)

Checklist:

- [x] Match the current decision-record schema and saved fields instead of inventing a new shape ad hoc
- [x] Centralize read/write logic for `autoclean_exclusion_decisions.json`
- [x] Centralize CSV export/update logic for `autoclean_exclusion_decisions.csv`
- [x] Ensure browser-saved bad-epoch indices are readable by current QA export logic
- [x] Ensure browser-saved notes/status remain readable by any existing desktop flows if needed
- [x] Rewrite persistence code with equivalent outputs to the current tool

### Phase 4: Frontend API client and types

Goal:
Add the web-side API layer for Exclude.

Likely files to touch:

- [web/src/lib/api.ts](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/lib/api.ts)
- [web/src/vite-env.d.ts](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/vite-env.d.ts)

Checklist:

- [x] Add TypeScript types for Exclude file list items
- [x] Add TypeScript types for Exclude file detail
- [x] Add TypeScript types for EEG manifest
- [x] Add TypeScript types for epoch window payloads
- [x] Add TypeScript types for epoch-review save/load payloads
- [x] Add Exclude API client methods
- [x] Add reprocess trigger/status client methods

### Phase 5: Frontend route and page shell

Goal:
Create the dedicated Exclude page in the current app shell.

Likely files to touch:

- [web/src/App.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/App.tsx)
- [web/src/components/Sidebar.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/Sidebar.tsx)
- [web/src/components/TopBar.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/TopBar.tsx)
- [web/src/pages/Exclude.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.tsx)

Checklist:

- [x] Add `/exclude` route
- [x] Add sidebar nav entry for `Exclude`
- [x] Add page shell with loading/error/empty states
- [x] Add exports-root summary at top of page
- [x] Add selected-file header with key metadata

### Phase 6: File list and detail workspace

Goal:
Build the main one-file-at-a-time Exclude workspace.

Likely files to touch:

- [web/src/pages/Exclude.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.tsx)
- [web/src/components/FolderBrowser.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/FolderBrowser.tsx)
- [web/src/components/StatusBadge.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/StatusBadge.tsx)
- [web/src/components/DataTable.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/DataTable.tsx)

Checklist:

- [x] Build Exclude file list
- [x] Add search/filter by filename
- [x] Show note indicator in file list
- [x] Show epoch-review indicator in file list
- [x] Show override/reprocess indicator in file list
- [x] Preserve selected file during polling/refresh
- [x] Build tabbed detail workspace container

### Phase 7: Browser-native EEG viewer

Goal:
Implement the core EEG review experience in the browser.

Likely files to touch:

- [web/src/pages/Exclude.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.tsx)
- [web/src/components/exclude/EegReviewPanel.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/exclude/EegReviewPanel.tsx)
- [web/src/components/exclude/EegTraceCanvas.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/exclude/EegTraceCanvas.tsx)
- [web/src/components/exclude/EpochOverviewRail.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/exclude/EpochOverviewRail.tsx)
- [web/src/components/exclude/EpochReviewToolbar.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/exclude/EpochReviewToolbar.tsx)
- [web/src/components/exclude/EpochSaveStatus.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/exclude/EpochSaveStatus.tsx)
- [web/src/hooks/useKeyboardShortcuts.ts](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/hooks/useKeyboardShortcuts.ts)

Checklist:

- [x] Create EEG viewer panel component
- [x] Render channel traces in a browser-native canvas view
- [x] Load EEG manifest on file selection
- [x] Load visible epoch windows on demand
- [x] Support left/right keyboard navigation
- [x] Support `Space` to toggle bad epoch
- [x] Support click-to-focus epoch
- [x] Support click-to-toggle bad epoch
- [x] Restore previously saved bad epochs into the viewer
- [x] Show dirty/saving/saved/conflict states
- [x] Autosave on edit
- [x] Force save on file switch and tab switch
- [ ] Keep interaction latency acceptable on realistic datasets

### Phase 8: Notes, metrics, and artifact tabs

Goal:
Bring over the surrounding review surfaces Grace actually uses.

Likely files to touch:

- [web/src/pages/Exclude.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.tsx)
- [web/src/pages/Results.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Results.tsx)
- [web/src/components/CodeViewer.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/CodeViewer.tsx)
- [src/autoclean/api/pdf_extractor.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/pdf_extractor.py)

Checklist:

- [x] Add metrics panel for selected file
- [x] Add notes editor with autosave
- [x] Add run report PDF tab
- [x] Add ICA PDF tab
- [x] Add structured ICA component summary table
- [x] Link ICA summary entries to PDF pages when available
- [x] Add PSD overview tab
- [x] Handle missing artifacts gracefully

### Phase 9: Override editor and reprocess flow

Goal:
Port the real Exclude override workflow, not just the review surfaces.

Likely files to touch:

- [web/src/pages/Exclude.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/pages/Exclude.tsx)
- [web/src/components/exclude/ReprocessPanel.tsx](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/src/components/exclude/ReprocessPanel.tsx)
- [src/autoclean/api/routes/exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/exclude.py)
- [src/autoclean/tools/autoclean_exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/tools/autoclean_exclude.py)

Checklist:

- [x] Build bad-channel override editor
- [x] Build ICA override editor
- [x] Show original pipeline values separately from manual overrides
- [x] Show added/removed diff clearly
- [x] Persist override edits
- [x] Add `Reprocess with Overrides` action
- [x] Show in-progress reprocess state
- [x] Show completion and failure states
- [x] Refresh file detail after reprocess
- [x] Match the current manual-fix payload and reprocess side effects unless intentionally changed

### Phase 10: QA export and downstream compatibility

Goal:
Ensure browser-native review still drives the existing QA and output flows correctly.

Likely files to touch:

- [src/autoclean/tools/autoclean_exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/tools/autoclean_exclude.py)
- [src/autoclean/api/routes/exclude.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/api/routes/exclude.py)

Checklist:

- [x] Verify browser-saved `bad_epoch_indices` can drive QA export
- [x] Verify browser-saved notes appear in QA preprocessing log
- [x] Verify browser-saved reviewer status appears in QA preprocessing log
- [x] Verify browser-saved overrides survive reprocess and reload
- [x] Verify any `postedit` assumptions are either preserved or intentionally replaced
- [ ] Verify rewritten backend outputs against the legacy Exclude workflow on the same files

### Phase 11: Tests

Goal:
Cover the new backend and frontend behavior with targeted tests.

Likely files to touch:

- [tests](/Users/sueo8x/Documents/Github/autoclean_pipeline/tests)
- [web](/Users/sueo8x/Documents/Github/autoclean_pipeline/web)

Checklist:

- [x] Add backend tests for Exclude file discovery
- [x] Add backend tests for EEG manifest endpoint
- [x] Add backend tests for epoch-window endpoint
- [x] Add backend tests for epoch-review save/load
- [x] Add backend tests for override save/load
- [x] Add backend tests for reprocess trigger/status
- [x] Add frontend tests for Exclude page loading states
- [x] Add frontend tests for file selection and state retention
- [x] Add frontend tests for EEG keyboard navigation
- [x] Add frontend tests for bad-epoch toggle and autosave
- [x] Add frontend tests for notes and override persistence

### Phase 12: Validation with real data

Goal:
Confirm the browser-native workflow holds up in a real workspace and not just mocked tests.

Checklist:

- [ ] Run against a real workspace with preprocessed exports
- [ ] Validate that default folder resolution lands in the expected `exports/` path
- [ ] Validate EEG review on a realistic epoched file
- [ ] Validate restoring prior bad-epoch state from saved review data
- [ ] Validate ICA override editing on a file with meaningful components
- [ ] Validate bad-channel override editing on a file with known issues
- [ ] Validate reprocess end-to-end on one reviewed file
- [ ] Validate QA export or equivalent downstream output on reviewed files
- [ ] Review the flow with Grace or another expert user

## Recommended Build Order

If implementation starts now, the recommended order is:

- [x] Phase 0: Discovery and alignment
- [x] Phase 3: Backend persistence and compatibility hardening
- [x] Phase 1: Backend Exclude API foundation
- [x] Phase 2: Backend EEG data endpoints
- [x] Phase 4: Frontend API client and types
- [x] Phase 5: Frontend route and page shell
- [x] Phase 6: File list and detail workspace
- [x] Phase 7: Browser-native EEG viewer
- [x] Phase 8: Notes, metrics, and artifact tabs
- [x] Phase 9: Override editor and reprocess flow
- [x] Phase 10: QA export and downstream compatibility
- [x] Phase 11: Tests
- [ ] Phase 12: Validation with real data

## Recommended First Slice

The fastest defensible first implementation is:

- A top-level `Exclude` page
- Auto-discovery of the exports folder
- File list + selected-file detail
- A browser-native EEG review tab
- Metrics
- Notes
- Run report PDF tab
- ICA PDF tab
- ICA component summary table
- Editable ICA and bad-channel override lists
- `Reprocess with Overrides`

This omits, for the first slice only:

- Perfect PSD parity

That slice preserves the most important expert workflow while still allowing the page to feel like part of the modern web UI: review EEG in-browser, inspect file outputs, review ICA PDF, adjust component/channel overrides, and rerun.

## Definition of Done

- [x] A user can open the web UI and reach an `Exclude` workflow without hunting for folders
- [x] A user can select a preprocessed export and see the same major review materials they rely on now
- [x] A user can review EEG epochs and mark bad epochs entirely in the browser
- [x] A user can edit ICA and bad-channel overrides from the browser
- [x] A user can trigger reprocessing with those overrides
- [x] The rerun result is visible in the web UI
- [ ] Grace can complete a realistic review task without needing the old standalone tool for the covered features
