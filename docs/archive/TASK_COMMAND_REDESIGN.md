# Task Command Ecosystem: Holistic Redesign Proposal

**Date**: 2025-09-29
**Status**: PROPOSAL - Under Review
**Author**: Analysis based on codebase review
**Repository**: autocleaneeg_pipeline

---

## Executive Summary

The `task` command in AutoCleanEEG Pipeline currently suffers from poor integration between three distinct task sources (built-in, library, workspace). Commands were added incrementally without reconciliation, leading to user confusion, redundant functionality, and a fragmented mental model.

**Key Issues:**
- `task library` feels "bolted on" rather than integrated
- Overlapping commands (`add`/`import`/`copy`, `remove`/`delete`)
- No unified view of all available tasks
- Multi-step workflows where single-step should suffice
- Unclear source attribution (where does this task come from?)

**Proposed Solution:**
Redesign `task` command with unified taxonomy, consolidated commands, and one-step workflows while maintaining backward compatibility during transition.

---

## Current State Analysis

### Three Task Sources (Poorly Integrated)

```
┌─────────────────────────────────────────────────────────┐
│ 1. BUILT-IN TASKS (Package)                            │
│    Location: src/autoclean/tasks/*.py                   │
│    Purpose: Shipped with pip install                    │
│    Examples: ~5-10 core tasks in package                │
│    Access: Automatic, no installation needed            │
│    Discovery: task_discovery._discover_builtin_tasks()  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 2. LIBRARY TASKS (GitHub Registry)                     │
│    Location: github.com/.../autocleaneeg-task-registry  │
│    Purpose: Curated official templates (14 tasks)       │
│    Cache: ~/.config/autocleaneeg/.builtin_cache/        │
│    Access: Must run `task library install`              │
│    Management: BuiltinRegistry class                    │
│    Registry: registry.json with commit tracking         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 3. WORKSPACE TASKS (User Custom)                       │
│    Location: ~/Documents/Autoclean-EEG/tasks/*.py       │
│    Purpose: User customizations & edits                 │
│    Priority: OVERRIDES built-in tasks with same name    │
│    Access: Direct editing, copied from library/builtin  │
│    Discovery: task_discovery._discover_custom_tasks()   │
└─────────────────────────────────────────────────────────┘
```

### Current Command Structure (Complete Map)

```bash
# ========= WORKSPACE TASK MANAGEMENT =========
task list              # Shows workspace + built-in (merged)
  --overrides          # Shows which workspace override built-in
task explore           # Open workspace folder in file browser

task add <file>        # Copy external file → workspace
task import <path>     # Copy external file → workspace (DUPLICATE?)
task copy [source]     # Copy workspace/built-in task → new workspace task
task delete [name]     # Delete workspace task file
task remove <name>     # Remove workspace task (DUPLICATE?)

task edit [name]       # Edit workspace task (or copy built-in if needed)
task set [name]        # Set active task (workspace only)
task unset             # Clear active task
task show              # Show current active task

# ========= LIBRARY TASK MANAGEMENT =========
task library           # Separate namespace (GitHub registry)
task library update    # Fetch registry.json from GitHub
task library list      # Show library tasks with sync status
task library install <name>  # Copy library task → workspace

# ========= DEVELOPER TOOLS =========
task schema export     # Export task schema JSON
```

**Implementation Files:**
- `src/autoclean/cli.py` (lines 877-1153) - Command parsers
- `src/autoclean/utils/builtins.py` - BuiltinRegistry class
- `src/autoclean/utils/task_discovery.py` - Task discovery logic
- `src/autoclean/utils/user_config.py` - Workspace management

---

## Critical Problems (User Confusion)

### Problem 1: Overlapping Commands with Unclear Purposes

| Command | What it does | Source |
|---------|-------------|--------|
| `task add <file>` | Copy external file to workspace | External file |
| `task import <path>` | Copy external file to workspace | External file |
| `task copy [source]` | Copy task to new workspace file | Workspace/built-in |
| `task library install <name>` | Copy library task to workspace | Library (GitHub) |

**User confusion**: "Why do I have `add`, `import`, `copy`, AND `library install`? They all put files in workspace!"

**Current behavior:**
```bash
# Four different ways to get a task into workspace:
$ task add /path/to/external.py
$ task import /path/to/external.py     # Same as add?
$ task copy RestingEyesOpen            # From built-in
$ task library install ASSR_40Hz       # From library

# User thinks: "Which one should I use?"
```

---

### Problem 2: `remove` vs `delete`

```bash
$ task remove MyTask       # What does this do?
$ task delete MyTask       # What's the difference?
```

**Investigation:**
- Both appear to delete workspace tasks
- Naming suggests different purposes but implementation may be identical
- Creates cognitive overhead for users

---

### Problem 3: `task list` vs `task library list`

```bash
# User wants to see "all available tasks"
$ task list
→ Shows: workspace tasks + built-in tasks (merged)
→ Does NOT show: library tasks available for install

$ task library list
→ Shows: library tasks (14 from registry)
→ Does NOT show: workspace or built-in tasks

# User thinks: "Where's RestingEyesOpen? I know it exists!"
# Answer: It's in library, must use separate command
```

**No unified view** of all tasks from all sources.

**Example confusion:**
```bash
$ task list
  MyCustomTask     User-created task
  BiotrialResting  Built-in task

$ task library list
  RestingEyesOpen  Standard resting-state
  ASSR_40Hz        Auditory steady-state

# User question: "Wait, can I use BiotrialResting? Is it installed?"
# User question: "How do I see EVERYTHING in one list?"
```

---

### Problem 4: Library is Bolted On

The `library` subcommand feels like an afterthought:

**Evidence:**
1. Separate namespace (`task library`) instead of integrated with main `task` commands
2. No interaction with `task set` (can't install-and-activate in one step)
3. No awareness in `task list` (doesn't show installable library tasks)
4. Unclear relationship with built-in tasks (are they overlapping? separate?)
5. Different sync semantics (library tracks remote hash, built-in doesn't)

**User workflow friction:**
```bash
# Current: 5 separate steps to use a library task
$ task library update                  # 1. Check for updates
$ task library list                    # 2. Browse available
$ task library install HBCD_VEP       # 3. Copy to workspace
$ task set HBCD_VEP                   # 4. Set as active
$ process data.raw                     # 5. Finally process

# Should be:
$ task use HBCD_VEP                   # Install + activate
$ process data.raw                     # Process immediately
```

---

### Problem 5: No Source Visibility

```bash
$ task list
  RestingEyesOpen    Standard resting-state with eyes open
  ASSR_40Hz          Auditory steady-state response
  MyCustomTask       My custom preprocessing

# User doesn't know:
# - Is RestingEyesOpen from built-in, library, or workspace?
# - Is ASSR_40Hz customized or original?
# - Are newer versions available in library?
# - Which ones can I safely edit without losing updates?
```

**Missing information:**
- Source attribution (library/built-in/user)
- Sync status (synced/customized/outdated)
- Update availability
- Override indicators

---

### Problem 6: Multi-Step Workflows

**Scenario 1: Try a library task**
```bash
# Current (3 commands)
$ task library install RestingEyesOpen
$ task set RestingEyesOpen
$ process data.raw

# Ideal (1 command)
$ task use RestingEyesOpen
$ process data.raw
```

**Scenario 2: Update customized task**
```bash
# Current (manual, risky)
$ task library list                    # Check if updated
$ # Manually backup my changes
$ cp tasks/ASSR.py tasks/ASSR.backup.py
$ task library install ASSR --force    # Destroys customizations!
$ # Manually re-apply changes

# Ideal (automated, safe)
$ task sync                            # Shows updates available
$ task diff ASSR                       # Preview changes
$ task sync --update                   # Auto-backup + merge
```

---

## Conceptual Model Issues

The CLI reflects **two competing mental models** that were never reconciled:

### Model A (Original): Built-in + Workspace

**Timeline**: Initial implementation (v1.x)

```
Package Installation
        ↓
Built-in Tasks (autoclean.tasks.*)
        ↓
User copies to workspace to customize
        ↓
Workspace Tasks (~/Documents/Autoclean-EEG/tasks/)
        ↓
Workspace overrides built-in (same name = override)
```

**Commands**: `list`, `add`, `import`, `copy`, `edit`, `set`, `delete`

**Mental model**: "Tasks come with the package, I can customize them in my workspace"

---

### Model B (Added Later): GitHub Library

**Timeline**: Added in v2.x alongside task registry

```
GitHub Registry (public repo)
        ↓
Local Cache (~/.config/autocleaneeg/.builtin_cache/)
        ↓
User must explicitly "install" to workspace
        ↓
Updates tracked via remote hash comparison
        ↓
User can force-update (destructive to customizations)
```

**Commands**: `library update`, `library list`, `library install`

**Mental model**: "Tasks are on GitHub, I pull them down when needed"

---

### The Problem: Models Never Reconciled

**Questions with unclear answers:**

1. **Are built-in tasks and library tasks the same thing?**
   - Sometimes yes (overlap), sometimes no (unique tasks)
   - No documentation clarifies relationship

2. **Do I need to install built-in tasks?**
   - No, they're already available
   - But library tasks require installation
   - User confusion: "Why are some tasks ready and others need install?"

3. **If I install a library task, does it become a built-in task?**
   - No, it becomes a workspace task
   - But then how do I update it?
   - And does it override built-in if names match?

4. **How do I know which tasks are outdated?**
   - Library tasks: check `library list` for sync status
   - Built-in tasks: no update mechanism
   - Workspace tasks: manual comparison

**Root cause**: Two systems (built-in discovery + library registry) operate independently without integration layer.

---

## Holistic Redesign Proposal

### Core Principle: Unified Task Taxonomy

Users should think about tasks in **ONE taxonomy**, not three separate systems:

```
ALL TASKS (Unified View)
│
├── AVAILABLE (Not in workspace)
│   ├── From library (GitHub registry, 14 tasks)
│   │   ├── RestingEyesOpen
│   │   ├── ASSR_40Hz
│   │   ├── Chirp_Default
│   │   └── ... (11 more)
│   │
│   └── From built-in (Packaged, ~5-10 tasks)
│       ├── BiotrialResting1020
│       └── ... (others)
│
├── INSTALLED (In workspace)
│   ├── Unmodified (hash matches source)
│   │   └── Can safely update from source
│   │
│   ├── Customized (user modified)
│   │   └── Update requires merge/review
│   │
│   ├── Outdated (newer version in source)
│   │   └── Update available
│   │
│   └── User-created (no source)
│       └── MyCustomTask.py
│
└── ACTIVE (Currently selected)
    └── One task active for `process` command
```

### Design Goals

1. **Single source of truth**: One command shows ALL tasks
2. **Clear attribution**: Always show source (library/built-in/user)
3. **Unified operations**: Commands work across all sources
4. **Safe updates**: Never lose user customizations
5. **Reduced friction**: One-step workflows where possible
6. **Backward compatible**: Deprecate gradually, don't break existing scripts

---

## Redesigned Command Structure

### Primary Commands (Simplified)

```bash
# ========= DISCOVERY & BROWSING =========
task list                         # Show ALL tasks (unified view)
  --source=[all|library|builtin|workspace|active]
  --status=[all|installed|available|outdated|customized]
  --category=[resting|auditory|visual|rodent]
  --format=[table|json]

task browse                       # Alias of `list` with rich display
task search <query>               # Search across all sources (name, description, tags)

# ========= INSTALLATION & ACTIVATION =========
task install <name>               # Install from library/built-in → workspace
  --set                           # Auto-activate after install
  --force                         # Overwrite existing
  --source=[auto|library|builtin] # Prefer specific source

task use <name>                   # Install (if needed) + set active (ONE STEP!)
  --force                         # Overwrite if customized

# ========= EDITING & REMOVAL =========
task edit [name]                  # Edit workspace task (copies from source if needed)
task delete [name]                # Delete workspace task (with confirmation)
task import <path>                # Import external task file → workspace

# ========= ACTIVATION (State Management) =========
task set [name]                   # Set active task
task unset                        # Clear active task
task show                         # Show active task info

# ========= WORKSPACE MANAGEMENT =========
task explore                      # Open workspace folder in OS
task sync                         # Check all workspace tasks for updates
  --update                        # Auto-update outdated tasks (with backup)
  --dry-run                       # Show what would be updated
task diagnose                     # Health check (broken tasks, orphans, validation errors)
task clean                        # Remove orphaned/invalid tasks (interactive)

# ========= LIBRARY/REMOTE (Reduced Surface Area) =========
task update                       # Update library cache from GitHub (was: task library update)
task diff <name>                  # Show diff between workspace and source
  --color                         # Colorized diff output
  --context=3                     # Lines of context

# ========= DEVELOPER TOOLS =========
task schema export                # Export task schema JSON
task validate <path>              # Validate task file against schema
task info <name>                  # Show detailed task metadata
```

---

## Key Improvements

### 1. Unified `task list` (Show Everything)

**New behavior:**
```bash
$ task list

Task Library: 75c4731 (last checked 2 hours ago) | 14 templates available

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INSTALLED TASKS (in workspace)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ✓ RestingEyesOpen       [library]     synced        Eyes-open resting
 ⚠ ASSR_40Hz             [library]     customized    40Hz steady-state
 ↻ MMN_Standard          [library]     outdated      Mismatch negativity
 ✓ BiotrialResting1020   [builtin]     synced        10-20 montage
 • MyCustomTask          [user]        —             Custom preprocessing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AVAILABLE FOR INSTALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 📦 Chirp_Default        [library]                   Chirp auditory
 📦 HBCD_VEP             [library]                   Visual evoked potential
 📦 Mouse_XDAT_ASSR      [library]                   Mouse ASSR protocol
 📦 RestingEyesClosed    [library]                   Eyes-closed resting
    ... (7 more)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ACTIVE TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 🎯 RestingEyesOpen (ready to process)

Tip: Use 'task use <name>' to install and activate in one command
     Use 'task sync' to check for updates to installed tasks
```

**With filters:**
```bash
$ task list --status=outdated
  MMN_Standard   [library]   outdated   Update available (v2.1.0 → v2.2.0)

$ task list --source=library --category=auditory
  ASSR_40Hz      [installed]   Auditory steady-state at 40Hz
  Chirp_Default  [available]   Chirp auditory stimulation
  BB_Long        [available]   Long-duration broadband
  MMN_Standard   [installed]   Mismatch negativity (oddball)
  HBCD_MMN       [available]   HBCD mismatch negativity protocol

$ task list --format=json > tasks.json   # Machine-readable output
```

---

### 2. One-Step Workflow: `task use`

**Problem**: Users must run 2-3 commands to start using a task

**Solution**: Single command to install (if needed) + activate

```bash
# NEW COMMAND
$ task use HBCD_VEP
→ Checking library for HBCD_VEP...
→ ✓ HBCD_VEP installed from library
→ ✓ Active task set to: HBCD_VEP
→ Ready to process: autocleaneeg-pipeline process <file>

# If already installed
$ task use RestingEyesOpen
→ ✓ RestingEyesOpen is already installed (up to date)
→ ✓ Active task set to: RestingEyesOpen
→ Ready to process

# If customized (requires force)
$ task use ASSR_40Hz
→ ⚠ ASSR_40Hz exists in workspace with customizations
→ Use --force to overwrite, or run 'task set ASSR_40Hz' to use current version

$ task use ASSR_40Hz --force
→ ⚠ Backing up ASSR_40Hz to ASSR_40Hz.backup.2025-09-29.py
→ ✓ ASSR_40Hz reinstalled from library
→ ✓ Active task set to: ASSR_40Hz
```

**Comparison:**
```bash
# BEFORE (3 steps)
$ task library update
$ task library install HBCD_VEP
$ task set HBCD_VEP
$ process data.raw

# AFTER (1 step)
$ task use HBCD_VEP
$ process data.raw
```

---

### 3. Smart `task install`

**Unified command** that works with both library AND built-in tasks:

```bash
# Auto-detect source (library preferred, then built-in)
$ task install RestingEyesOpen
→ ✓ RestingEyesOpen installed from library

# Explicit source
$ task install --source=builtin BiotrialResting1020
→ ✓ BiotrialResting1020 installed from built-in package

$ task install --source=library ASSR_40Hz
→ ✓ ASSR_40Hz installed from library (version 2.1.0)

# Install and activate immediately
$ task install Chirp_Default --set
→ ✓ Chirp_Default installed from library
→ ✓ Active task set to: Chirp_Default

# Force overwrite existing
$ task install MMN_Standard --force
→ ⚠ Backing up MMN_Standard to MMN_Standard.backup.py
→ ✓ MMN_Standard reinstalled from library
```

**Logic:**
1. Check if task exists in workspace → error (unless --force)
2. Try library (if available and not --source=builtin)
3. Try built-in (if available and not --source=library)
4. Fail with helpful message suggesting `task list --source=all`

---

### 4. Health & Sync Commands

#### `task sync` - Check for updates

```bash
$ task sync
→ Checking workspace tasks against sources...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SYNC STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ✓ RestingEyesOpen      [library]   up to date
 ⚠ ASSR_40Hz            [library]   customized (no updates available)
 ↻ MMN_Standard         [library]   outdated (v2.1.0 → v2.2.0)
 ✗ DeletedTask          [orphan]    source no longer exists

Summary:
  • 1 task has updates available
  • 1 orphaned task (source removed from registry)
  • 1 customized task (review before updating)

Next steps:
  • Run 'task sync --update' to apply updates (will backup customized tasks)
  • Run 'task diff MMN_Standard' to preview changes
  • Run 'task clean' to remove orphaned tasks
```

#### `task sync --update` - Apply updates

```bash
$ task sync --update
→ Found 1 task with available updates

Updating MMN_Standard:
  ⚠ Backing up to MMN_Standard.backup.2025-09-29.py
  ↻ Downloading from library (commit 75c4731)
  ✓ Updated successfully

  Changes:
    • config: resample_step 250 → 500
    • Added: clean_bad_channels() method
    • Removed: deprecated legacy_cleanup() method

Summary:
  ✓ 1 task updated
  ⚠ 1 backup created (see tasks/.backups/)

Tip: Review changes with 'task diff MMN_Standard.backup.2025-09-29.py MMN_Standard.py'
```

#### `task diagnose` - Health check

```bash
$ task diagnose
→ Running workspace health check...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WORKSPACE HEALTH: GOOD ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tasks:
  • 5 tasks installed in workspace
  • 1 customized (ASSR_40Hz)
  • 1 outdated (MMN_Standard)
  • 0 with import errors
  • 1 orphan (DeletedTask.py)

Cache:
  • Library cache: 75c4731 (2 hours ago)
  • Network: OK
  • Cache size: 2.3 MB

Configuration:
  • Active task: RestingEyesOpen ✓
  • Workspace: ~/Documents/Autoclean-EEG ✓
  • Tasks directory: ~/Documents/Autoclean-EEG/tasks ✓

Issues:
  ⚠ 1 orphaned task file (source no longer in registry)

Recommendations:
  • Run 'task sync --update' to update MMN_Standard
  • Run 'task clean' to remove orphaned tasks
  • Consider reviewing customizations in ASSR_40Hz

Overall: Your workspace is healthy with minor cleanup needed
```

#### `task diff` - Preview changes

```bash
$ task diff MMN_Standard
→ Comparing workspace copy vs library source

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CONFIGURATION CHANGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 config = {
     "schema_version": "2025.09",
-    "resample_step": {"enabled": True, "value": 250},
+    "resample_step": {"enabled": True, "value": 500},
     "filtering": {
         "enabled": True,
-        "value": {"l_freq": 0.1, "h_freq": 100}
+        "value": {"l_freq": 1.0, "h_freq": 100}
     },
 }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 METHOD CHANGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 + Added: clean_bad_channels(self)
     """Identify and interpolate bad channels."""

 - Removed: legacy_cleanup(self)
     """Deprecated cleanup method."""

Summary:
  • 2 config values changed
  • 1 method added
  • 1 method removed

Use 'task sync --update' to apply changes (backup will be created)
```

---

### 5. Remove Redundant Commands

#### Consolidation Table

| Old Command | New Command | Rationale |
|-------------|-------------|-----------|
| `task add <file>` | `task import <file>` | Clearer intent (importing external) |
| `task import <path>` | `task import <path>` | Keep (clear purpose) |
| `task remove <name>` | `task delete <name>` | More direct, less ambiguous |
| `task delete [name]` | `task delete [name]` | Keep (clear purpose) |
| `task library update` | `task update` | Top-level, no namespace needed |
| `task library list` | `task list --source=library` | Integrated view with filters |
| `task library install` | `task install` | Works for any source |
| `task copy [source]` | `task copy [source]` | Keep (useful for duplication) |

#### NEW Commands

| Command | Purpose | Priority |
|---------|---------|----------|
| `task use <name>` | Install + activate in one step | 🔥 High |
| `task sync` | Check all tasks for updates | 🔥 High |
| `task diagnose` | Workspace health check | 🟡 Medium |
| `task diff <name>` | Compare workspace vs source | 🟡 Medium |
| `task validate <path>` | Schema validation | 🟢 Low |
| `task info <name>` | Detailed task metadata | 🟢 Low |
| `task browse` | Alias for `list` | 🟢 Low |
| `task search <query>` | Search all tasks | 🟡 Medium |
| `task clean` | Remove orphans/invalid | 🟡 Medium |

---

## Implementation Strategy

### Phase 1: Add New Commands (Non-Breaking) ✅

**Goal**: Introduce improvements without changing existing commands

**Changes:**
1. ✅ Add `task use` (install + set in one command)
2. ✅ Add `task sync` (check all tasks for updates)
   - `task sync --update` (apply updates with backup)
   - `task sync --dry-run` (preview only)
3. ✅ Add `task diagnose` (workspace health check)
4. ✅ Enhance `task list` with unified view
   - Add `--source` filter (all|library|builtin|workspace|active)
   - Add `--status` filter (all|installed|available|outdated|customized)
   - Add `--category` filter (resting|auditory|visual|rodent)
   - Add `--format` flag (table|json)
   - Show source indicators ([library], [builtin], [user])
   - Show sync status (synced, customized, outdated)
5. ✅ Add `task diff` (workspace vs source comparison)
6. ✅ Add `task search` (search across all sources)

**Testing:**
- Verify new commands don't break existing workflows
- Test with real task files from registry
- Validate JSON output format
- Check sync status detection logic

**Timeline**: 2-3 days
**Risk**: Low (additive only)

---

### Phase 2: Unify Existing Commands (Minimal Breaking)

**Goal**: Make existing commands work across all sources

**Changes:**
1. ✅ Make `task install` work with library/built-in/external
   - Add `--source` flag (auto|library|builtin)
   - Auto-detect source (library preferred)
   - Add `--set` flag (activate after install)
2. ✅ Add source indicators to ALL command output
   - `task list` shows [library]/[builtin]/[user]
   - `task show` displays source information
   - `task edit` indicates which source will be copied
3. ✅ Enhance `task update` (promote from `task library update`)
   - Top-level command (no `library` namespace)
   - Keep `library update` as deprecated alias
4. ✅ Add `task validate` (schema validation)
5. ✅ Add `task info` (detailed metadata)

**Testing:**
- Backward compatibility checks
- Verify `--source` logic priority
- Test with mixed task sources

**Timeline**: 2-3 days
**Risk**: Low (mostly additive with deprecation warnings)

---

### Phase 3: Deprecate (with Warnings) ⚠️

**Goal**: Warn users about deprecated commands but keep them working

**Changes:**
1. Add deprecation warnings:
   ```bash
   $ task library update
   → ⚠ Warning: 'task library update' is deprecated
   →   Use 'task update' instead
   →   This alias will be removed in v3.0.0
   → Continuing...
   ```

2. Deprecation targets:
   - `task library update` → `task update`
   - `task library list` → `task list --source=library`
   - `task library install` → `task install`
   - `task add` → `task import`
   - `task remove` → `task delete`

3. Update documentation:
   - Add migration guide to CLAUDE.md
   - Update CLI help text
   - Add to CHANGELOG.md

4. Add `--quiet` flag to suppress deprecation warnings (for scripts)

**Testing:**
- Verify warnings appear correctly
- Test that workflows still function
- Check `--quiet` suppression

**Timeline**: 1 day
**Risk**: Very Low (no behavior changes)

---

### Phase 4: Remove (Breaking Change - v3.0.0) 🔴

**Goal**: Clean removal of deprecated commands (major version bump)

**Changes:**
1. Remove deprecated commands:
   - Remove `task library` subcommand entirely
   - Remove `task add` (use `import`)
   - Remove `task remove` (use `delete`)

2. Update all documentation:
   - Remove deprecated commands from help
   - Update examples in README
   - Migration guide in release notes

3. Regression testing:
   - Full CLI test suite
   - Integration tests with real tasks
   - User acceptance testing

**Timeline**: 1-2 days + testing
**Risk**: High (breaking change, requires major version bump)
**Prerequisites**:
- Phase 3 deprecated for at least 2 releases
- User survey confirms low usage of deprecated commands
- Clear migration path documented

---

## Example User Journeys (Before vs After)

### Journey 1: New User Wants to Try ASSR Task

**BEFORE (Current - Confusing)**
```bash
# User doesn't know task exists
$ task list
  BiotrialResting1020   Biotrial resting-state
  MyOldTask             My custom task

# Where's ASSR? Let me try library
$ task library list
  ASSR_40Hz             Auditory steady-state
  RestingEyesOpen       Resting-state eyes open
  ...

# Found it! Now install
$ task library install ASSR_40Hz
→ ASSR_40Hz copied to ~/Documents/Autoclean-EEG/tasks/ASSR_40Hz.py

# Now set as active
$ task set ASSR_40Hz
→ ✓ Active task set to: ASSR_40Hz

# Finally process
$ process data.raw

# Total: 4 commands, user had to know about library subcommand
```

**AFTER (Redesigned - Streamlined)**
```bash
# User sees ALL tasks in one list
$ task list
INSTALLED TASKS
  BiotrialResting1020   [builtin]   synced
  MyOldTask             [user]      —

AVAILABLE FOR INSTALL
  📦 ASSR_40Hz          [library]   Auditory steady-state
  📦 RestingEyesOpen    [library]   Resting-state eyes open
  ...

# Install and activate in one step
$ task use ASSR_40Hz
→ ✓ ASSR_40Hz installed from library
→ ✓ Active task set to: ASSR_40Hz
→ Ready to process

# Process immediately
$ process data.raw

# Total: 2 commands, clear unified view
```

---

### Journey 2: User Wants to Check for Updates

**BEFORE (Current - Manual, Tedious)**
```bash
# Check for library updates
$ task library update
→ Task Library refreshed (version 75c4731)

# List to find outdated tasks
$ task library list
  RestingEyesOpen    [success]up to date[/success]
  ASSR_40Hz          [warning]customized[/warning]
  MMN_Standard       [warning]customized[/warning]     <- Wait, is this outdated?

# No way to know if customized means outdated!
# User must manually check GitHub

# Manually backup before updating
$ cp tasks/MMN_Standard.py tasks/MMN_Standard.backup.py

# Force reinstall (DESTROYS CUSTOMIZATIONS!)
$ task library install MMN_Standard --force
→ MMN_Standard copied to workspace

# Now manually reapply customizations by diffing backup
# User may lose changes if they forget to backup

# Total: 4+ commands, manual diff, risk of data loss
```

**AFTER (Redesigned - Automated, Safe)**
```bash
# Check for updates (clear status)
$ task sync
→ Checking workspace tasks against sources...

 ✓ RestingEyesOpen      [library]   up to date
 ⚠ ASSR_40Hz            [library]   customized (no updates)
 ↻ MMN_Standard         [library]   outdated (v2.1.0 → v2.2.0)

# Preview changes before updating
$ task diff MMN_Standard
→ Comparing workspace vs library:

  config changes:
    - resample_step: 250 → 500
    + Added: clean_bad_channels()
    - Removed: legacy_cleanup()

# Auto-update with backup
$ task sync --update
→ Backing up MMN_Standard to MMN_Standard.backup.2025-09-29.py
→ ✓ MMN_Standard updated from library
→ Tip: Review changes with 'task diff MMN_Standard.backup MMN_Standard'

# Total: 2-3 commands, automatic backup, safe updates
```

---

### Journey 3: User Wants to See ALL Available Tasks

**BEFORE (Current - Impossible with One Command)**
```bash
# See workspace + built-in
$ task list
  BiotrialResting1020   Biotrial resting-state
  MyCustomTask          My custom task

# See library (separate command)
$ task library list
  ASSR_40Hz             Auditory steady-state
  RestingEyesOpen       Resting-state eyes open
  Chirp_Default         Chirp auditory stimulation
  ...

# User must mentally merge two lists
# No way to know:
#   - Which built-in tasks overlap with library?
#   - Which tasks are installed vs available?
#   - What's the complete inventory?

# Total: 2 commands, fragmented view, mental effort required
```

**AFTER (Redesigned - Unified View)**
```bash
# See EVERYTHING in one unified list
$ task list

Task Library: 75c4731 (2 hours ago) | 14 templates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INSTALLED TASKS (in workspace)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ✓ RestingEyesOpen       [library]   synced
 ⚠ ASSR_40Hz             [library]   customized
 ✓ BiotrialResting1020   [builtin]   synced
 • MyCustomTask          [user]      —

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AVAILABLE FOR INSTALL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 📦 Chirp_Default        [library]   Chirp auditory
 📦 HBCD_VEP             [library]   Visual evoked potential
 📦 MMN_Standard         [library]   Mismatch negativity
 📦 Mouse_XDAT_ASSR      [library]   Mouse ASSR protocol
    ... (7 more)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ACTIVE TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 🎯 RestingEyesOpen (ready to process)

# Filter by source
$ task list --source=library
  RestingEyesOpen    [installed]   synced
  ASSR_40Hz          [installed]   customized
  Chirp_Default      [available]   Chirp auditory
  ... (11 more library tasks)

# Filter by status
$ task list --status=available
  Chirp_Default      [library]   Chirp auditory
  HBCD_VEP           [library]   Visual evoked potential
  ... (9 more available)

# Total: 1 command, complete inventory, clear status
```

---

### Journey 4: User Customized a Task, Now Wants Updates

**BEFORE (Current - Dangerous)**
```bash
# User customized ASSR_40Hz months ago
# Now wants to get upstream improvements

$ task library list
  ASSR_40Hz    [warning]customized[/warning]

# No way to see what changed upstream!
# User must:
#   1. Manually visit GitHub
#   2. Find the file
#   3. Compare visually
#   4. Decide whether to update

# If user force-updates:
$ task library install ASSR_40Hz --force
→ ASSR_40Hz copied to workspace

# CUSTOMIZATIONS ARE GONE! No backup was created.
# User must restore from git or manual backup

# Total: High risk of data loss, manual GitHub checking
```

**AFTER (Redesigned - Safe with Preview)**
```bash
# Check sync status
$ task sync
  ⚠ ASSR_40Hz    [library]   customized (no upstream changes)

# Later, when updates are available:
$ task sync
  ↻ ASSR_40Hz    [library]   outdated (v2.0.0 → v2.1.0)

# Preview exactly what changed
$ task diff ASSR_40Hz
→ Comparing workspace vs library (v2.1.0):

  Config changes:
    - filtering.h_freq: 80 → 100
    + Added step: wavelet_denoise

  Method changes:
    + Added: apply_wavelet_filter(self)
    ~ Modified: run() [added wavelet step]

# Decide: merge changes manually or auto-update with backup
$ task sync --update
→ ⚠ ASSR_40Hz has customizations
→ Creating backup: ASSR_40Hz.backup.2025-09-29.py
→ ✓ Updated from library (v2.1.0)
→ ⚠ Review your customizations and reapply if needed

# Or: manual merge with backup
$ task diff ASSR_40Hz.backup.2025-09-29.py ASSR_40Hz.py
→ [Shows differences between backup and new version]

# Total: Safe with automatic backup, clear preview, informed decision
```

---

## Priority Recommendations

### 🔥 High Priority (Implement First)

**Impact**: Maximum user value, minimal breaking changes

1. **`task use`** - Install + activate in one step
   - **Why**: Reduces most common workflow from 3 steps to 1
   - **Effort**: Low (2-3 hours)
   - **Breaking**: None (additive)

2. **`task sync`** - Check all tasks for updates
   - **Why**: Currently no way to know if tasks are outdated
   - **Effort**: Medium (1 day)
   - **Breaking**: None (additive)

3. **Unified `task list`** - Show all tasks in one view
   - **Why**: Most critical UX improvement, eliminates fragmentation
   - **Effort**: Medium (1-2 days)
   - **Breaking**: None (enhances existing)

4. **`task diff`** - Preview changes before updating
   - **Why**: Prevents accidental data loss from blind updates
   - **Effort**: Medium (1 day)
   - **Breaking**: None (additive)

5. **Enhance `task list` with filters**
   - `--source`, `--status`, `--category`, `--format`
   - **Why**: Power users need granular filtering
   - **Effort**: Low (4-6 hours)
   - **Breaking**: None (optional flags)

---

### 🟡 Medium Priority (Do Next)

**Impact**: Quality-of-life improvements

6. **`task diagnose`** - Workspace health check
   - **Why**: Helps users troubleshoot issues
   - **Effort**: Medium (1 day)
   - **Breaking**: None

7. **`task search`** - Search across all sources
   - **Why**: Discovery is hard with 14+ tasks
   - **Effort**: Low (4-6 hours)
   - **Breaking**: None

8. **`task clean`** - Remove orphaned/invalid tasks
   - **Why**: Workspace hygiene, automated cleanup
   - **Effort**: Low (4-6 hours)
   - **Breaking**: None

9. **Make `task install` work with all sources**
   - Add `--source` flag, auto-detection
   - **Why**: Unifies task installation
   - **Effort**: Medium (1 day)
   - **Breaking**: None (enhances existing)

10. **Promote `task update`** to top level
    - Move from `task library update`
    - **Why**: Simpler mental model
    - **Effort**: Low (2-3 hours)
    - **Breaking**: Minor (add deprecation warning)

---

### 🟢 Low Priority (Nice-to-Have)

**Impact**: Polish and completeness

11. **`task validate`** - Schema validation for task files
    - **Why**: Helps developers catch errors early
    - **Effort**: Medium (1 day)
    - **Breaking**: None

12. **`task info`** - Detailed task metadata
    - **Why**: Power users want comprehensive info
    - **Effort**: Low (3-4 hours)
    - **Breaking**: None

13. **`task browse`** - Alias for `list` with rich display
    - **Why**: Aesthetic, better visuals
    - **Effort**: Low (2-3 hours)
    - **Breaking**: None

14. **Backup mechanism** for force updates
    - Auto-create `.backup` files
    - **Why**: Safety net for destructive operations
    - **Effort**: Low (4-6 hours)
    - **Breaking**: None

15. **Registry enhancements** (tags, complexity, version)
    - Add metadata to registry.json
    - **Why**: Better filtering and discovery
    - **Effort**: Medium (depends on registry changes)
    - **Breaking**: None (additive to registry)

---

## Security & Reliability Considerations

### Current Strengths ✅

1. **Hash Verification**: SHA256 hashing for cache integrity
2. **Network Timeouts**: 5s default, configurable via env var
3. **Error Recovery**: Manifest corruption renames to `.broken`
4. **Offline Mode**: 3-tier fallback (network → cache → package)
5. **User-Agent Header**: Identifies requests to GitHub

### Potential Improvements

1. **Signature Verification**: GPG-signed registry.json for supply-chain security
2. **Rollback Mechanism**: No current way to revert to older task versions
3. **Backup Retention**: Automatic cleanup of old `.backup` files
4. **Checksum Display**: Show hash in `task info` for verification
5. **Update Channels**: Stable vs development branches in registry

---

## Open Questions

### Technical Decisions

1. **Should `task use` auto-update outdated tasks?**
   - Option A: Always install latest (may break workflows)
   - Option B: Fail if outdated, require explicit `--force`
   - **Recommendation**: B (safer, explicit intent)

2. **How many backup files to keep?**
   - Option A: Unlimited (clutter)
   - Option B: Last 5 backups
   - Option C: Configurable in user_config
   - **Recommendation**: B with C option

3. **Should `task list` show backups in output?**
   - Option A: Hide by default, show with `--backups`
   - Option B: Always show in separate section
   - **Recommendation**: A (reduces clutter)

4. **Merge strategy for `task sync --update` on customized tasks?**
   - Option A: Always backup + overwrite (current proposed)
   - Option B: Three-way merge with conflict markers
   - Option C: Interactive merge tool
   - **Recommendation**: A for v1, B for v2

5. **Should library tasks eventually replace built-in tasks entirely?**
   - Option A: Keep both (current)
   - Option B: Migrate all built-in to library, deprecate package tasks
   - **Recommendation**: Needs product decision, affects distribution model

---

## Next Steps

### Immediate Actions (This Week)

1. **Review this proposal** with team
2. **Prioritize Phase 1 commands** based on user feedback
3. **Create implementation tickets** for approved features
4. **Set up user testing plan** for unified `task list`

### Implementation Timeline (Estimated)

- **Phase 1**: 1-2 weeks (5 new commands)
- **Phase 2**: 1 week (command unification)
- **Phase 3**: 2-3 days (deprecation warnings)
- **Phase 4**: After 2-3 releases with deprecations active

**Total estimated time**: 3-4 weeks for complete implementation

---

## Appendix: Command Reference Quick Sheet

```bash
# ============ DISCOVERY ============
task list                      # All tasks (unified view)
task list --source=library     # Library tasks only
task list --status=outdated    # Tasks needing updates
task search "resting"          # Search by keyword

# ============ INSTALLATION ============
task install <name>            # Install from any source
task use <name>                # Install + activate (one-step)
task import <path>             # Import external file

# ============ EDITING ============
task edit [name]               # Edit workspace task
task delete [name]             # Delete workspace task

# ============ ACTIVATION ============
task set [name]                # Set active task
task unset                     # Clear active task
task show                      # Show active task

# ============ MAINTENANCE ============
task sync                      # Check for updates
task sync --update             # Apply updates (with backup)
task diagnose                  # Health check
task diff <name>               # Compare workspace vs source
task clean                     # Remove orphaned tasks

# ============ LIBRARY ============
task update                    # Update library cache from GitHub

# ============ DEVELOPER ============
task schema export             # Export schema JSON
task validate <path>           # Validate task file
task info <name>               # Show detailed metadata
```

---

**End of Proposal** | **Status**: DRAFT | **Date**: 2025-09-29