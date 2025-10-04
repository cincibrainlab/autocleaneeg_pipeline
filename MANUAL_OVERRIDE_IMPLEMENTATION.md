# Manual Override & Reprocessing Implementation

## Goal
Enable users to use the exclude GUI as a QA tool to manually select/modify bad channels and ICA components, then reprocess data with those overrides applied.

## Workflow
1. User reviews processed EEG data in exclude GUI
2. User modifies bad channels and/or ICA components in Reprocess tab
3. Changes are saved to JSON payload in `qa/manual_fixes/{stem}_manual_fix.json`
4. User clicks "Reprocess with Overrides" button
5. System generates task file with manual overrides using Jinja template
6. Pipeline runs asynchronously with overrides applied
7. User reviews reprocessed results

## Implementation Steps Completed

### 1. Payload System
- Manual override changes saved to structured JSON in `qa/manual_fixes/`
- Payload includes: modifications (added/removed channels/components), validation flags, task file hash, timestamps
- Tracks both channel and ICA component changes with diff tracking (original, modified, added, removed)

### 2. Reprocess Template
- Template: `src/autoclean/templates/reprocess_with_overrides.jinja`
- Renders task files with `manual_bad_channels` and `manual_rejected_components` parameters
- Template function: `render_reprocess_task_from_json()` in `template_renderer.py`
- Skips automatic detection when manual overrides provided

### 3. Reprocess Button UI
- Added "Reprocess with Overrides" button to ReprocessWidget
- Button enables only when user makes changes
- Shows confirmation dialog with override summary before processing

### 4. Async Processing
- Uses QProcess for non-blocking pipeline execution
- Simple dialog with cancel button (non-modal, keeps GUI responsive)
- Captures stdout/stderr for error reporting
- Displays pipeline errors in message dialogs

### 5. File Backup & Copy System
- **Automatic backup**: Original comp files backed up to `exports/backups/` with timestamp before reprocessing
- **Temporary folder**: Reprocess creates task with unique class name (e.g., `Task_1807_rest_Reprocess`) to avoid file conflicts
- **Post-process copy**: After successful reprocess, results automatically copied from temp folder to original task folder
- **Folder preservation**: User sees updated results in same folder structure without manual file management
- **Folders copied**: exports/, reports/, and ica/ contents transferred to original locations

### 6. Bug Fixes
- **Class name sanitization**: Prefixes task names starting with digits with `Task_` (e.g., `1807_rest` → `Task_1807_rest_Reprocess`)
- **Directory tracking**: Fixed `task_root` to update when user selects new directory in GUI
- **Output location**: Added `--output` flag to ensure reprocessing targets same directory structure

## Current Status
✅ Basic reprocessing workflow functional
✅ Manual overrides correctly applied
✅ GUI remains responsive during processing
✅ Error reporting implemented
✅ Automatic backup and file copy system working
✅ Results appear in original folder structure
✅ Windows file locking avoided via temporary folder

## File Management
- **Before reprocessing**: Original file `exports/{stem}_comp_epo.fif` → `exports/backups/{stem}_comp_epo_{timestamp}.fif`
- **During reprocessing**: Pipeline creates temporary folder (e.g., `Task_1807_rest_Reprocess/`)
- **After reprocessing**: Results copied from temp folder to original task folder
- **User experience**: Reprocessed files appear in original location; temp folder can be manually deleted
- **Reprocess task file**: Stored in `status/{stem}_Reprocess.py` for reference
