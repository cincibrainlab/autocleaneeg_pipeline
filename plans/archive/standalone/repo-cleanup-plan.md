# Repo Cleanup Plan

## Purpose

This document records a repo-wide cleanup review and the exact cleanup work to do next.

It is intentionally written before making cleanup changes so there is a clear record of:

- what problems were found
- which ones are worth fixing now
- which files should be touched
- what should explicitly be left alone

## Current Repo Review

### 1. Line-ending policy is causing avoidable churn

Relevant file:

- [/.gitattributes](/Users/sueo8x/Documents/Github/autoclean_pipeline/.gitattributes)

Current content:

- `* text=auto`
- `* text eol=crlf`

Impact:

- forces CRLF for all text files
- causes repeated `LF will be replaced by CRLF` warnings during normal edits
- creates noisy diffs in Python, TypeScript, Markdown, HTML, JSON, and generated assets
- makes it harder to distinguish real content changes from formatting churn

Assessment:

- this is the highest-value cleanup item in the repo
- it should be normalized deliberately, not file-by-file ad hoc
- the target should be Linux-style line endings (`LF`) for normal source and documentation files
- the cleanup should be done as a one-time repo-wide normalization pass so future commits do not keep carrying line-ending-only noise

Best-practice rationale:

- most open-source repositories store tracked text files as `LF`
- Linux/macOS tooling expects `LF`
- Python, TypeScript, Markdown, JSON, YAML, and HTML workflows are all cleaner with `LF` in git
- Windows-specific scripts can still be explicitly marked `CRLF` if needed
- forcing `CRLF` for all text files is not the usual open-source default and tends to create avoidable churn

### 2. Stash list is cluttered and operationally risky

Current stash list includes multiple old temporary entries from different branches.

Examples seen during review:

- `serve-auth-resend`
- `main`
- older feature-branch temp stashes

Impact:

- easy to lose track of real work
- easy to apply the wrong stash by mistake
- increases risk during branch switching and cleanup work

Assessment:

- this should be cleaned now
- but only after reviewing each stash briefly so nothing important is lost

### 3. Documentation sprawl is real

Relevant paths:

- [/docs](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs)
- [/docs/from-root](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root)
- [/plans](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans)

Observed issues:

- `docs/from-root` contains many implementation notes and one-off planning docs
- `docs/` also contains plans, summaries, workflow notes, and generated-looking artifacts
- there is no single index for “active engineering plans” vs “historical notes” vs “user-facing docs”
- there are multiple overlapping contributor docs:
  - [/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/CONTRIBUTING.md)
  - [/docs/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/CONTRIBUTING.md)
  - [/docs/from-root/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/CONTRIBUTING.md)

Impact:

- hard to know which docs are canonical
- harder for new contributors to find the right doc
- engineering plans accumulate without lifecycle management

Assessment:

- worth cleaning
- should be done by indexing and classifying before deleting or moving anything

### 4. Root-level frontend workflow is not obvious

Relevant files:

- [/package.json](/Users/sueo8x/Documents/Github/autoclean_pipeline/package.json)
- [/web/package.json](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/package.json)

Observed state:

- root `package.json` is effectively empty
- actual frontend scripts live under `web/`

Impact:

- `npm run build` from repo root fails
- `npm test` from repo root fails
- this is confusing unless you already know the web app lives under `web/`

Assessment:

- this is a documentation and developer-experience issue
- should be addressed in docs only for this cleanup pass
- do not add root proxy scripts for frontend commands

### 5. Ignore rules are doing too much and hiding important generated state

Relevant file:

- [/.gitignore](/Users/sueo8x/Documents/Github/autoclean_pipeline/.gitignore)

Observed issues:

- `src/autoclean/api/static/assets/` is ignored
- `src/autoclean/api/static/index.html` is ignored
- but the repo still sometimes needs to commit bundled Serve static assets for deployment consistency
- `scripts/` is globally ignored even though the repo has a real `/scripts` directory
- there are duplicate or overlapping ignore entries

Impact:

- committing legitimate built Serve assets requires `git add -f`
- easy to end up with mismatched `index.html` and asset filenames
- ignore behavior is harder to reason about than it should be

Assessment:

- this is a real cleanup target
- should be handled carefully because it affects build/release workflow

### 6. Root repo has some local-run artifacts that should stay out of version control

Observed examples:

- `.serve-run.pid`
- `.coverage`
- `htmlcov/`
- local logs

Assessment:

- mostly already ignored
- worth checking that the ignore policy is intentional and minimal

### 7. Generated and historical content needs clearer separation

Relevant paths:

- [/docs/archive](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/archive)
- [/plans/archive](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive)
- [/docs/runs](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/runs)
- [/docs/maintenance_logs](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/maintenance_logs)

Assessment:

- the repo already has some archive structure
- cleanup should build on that rather than inventing a new system

## Cleanup Goals

### Goal 1

Normalize line-ending behavior so routine edits stop producing line-ending noise.

### Goal 2

Reduce git operational clutter by cleaning old stashes.

### Goal 3

Make docs easier to navigate by classifying and indexing planning and contributor docs.

### Goal 4

Clarify the frontend build/test workflow at the repo root.

### Goal 5

Review ignore rules so build artifacts and real source assets are handled intentionally.

## Requested Cleanup Items

These are the explicit cleanup actions requested for this pass and are in scope for execution from this plan:

1. Normalize line-ending policy.
2. Clean the stash list.
3. Trim or index the planning docs.

These are not optional side notes. They are primary deliverables for the cleanup pass.

For line endings specifically, the expected direction is:

- normalize tracked source and documentation files to Linux-style `LF`
- do a deliberate repository-wide touch/renormalization pass
- avoid leaving the repo in a state where future line-ending-only commits are still likely

## Planned Cleanup Work

### Phase 1: Policy and Git Hygiene

Files to review/update:

- [/.gitattributes](/Users/sueo8x/Documents/Github/autoclean_pipeline/.gitattributes)
- [/.gitignore](/Users/sueo8x/Documents/Github/autoclean_pipeline/.gitignore)
- optional new file: [/.editorconfig](/Users/sueo8x/Documents/Github/autoclean_pipeline/.editorconfig)

Planned work:

- replace the global forced-CRLF policy with a more defensible text normalization policy
- add file-type-specific behavior only if needed
- consider introducing `.editorconfig` so editors share newline/final-newline/indent rules
- review `.gitignore` for duplicate or misleading entries
- explicitly decide how `src/autoclean/api/static/index.html` and `src/autoclean/api/static/assets/` should be handled
- normalize source, docs, and other tracked text files to `LF`
- do the normalization as a single deliberate pass across the repository instead of waiting for individual files to be touched later

Expected result:

- fewer line-ending warnings
- cleaner diffs
- less ambiguity around bundled Serve frontend assets
- future commits should stop needing line-ending-only cleanup

Primary requested item covered here:

- normalize line-ending policy

### Phase 2: Stash Cleanup

Files touched:

- none in the repo tree

Git objects to review:

- current stash entries from `git stash list`

Planned work:

- review each stash entry briefly
- keep only entries that correspond to intentionally preserved unfinished work
- drop obsolete temp stashes

Expected result:

- lower risk when switching branches
- easier to reason about unfinished work

Primary requested item covered here:

- clean stash list

### Phase 3: Documentation Indexing and Normalization

Files to review/update:

- [/README.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/README.md)
- [/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/CONTRIBUTING.md)
- [/docs/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/CONTRIBUTING.md)
- [/docs/from-root/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/CONTRIBUTING.md)
- optional new file: [/plans/README.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/README.md)
- optional new file: [/docs/INDEX.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/INDEX.md)

Planned work:

- classify docs into:
  - canonical contributor docs
  - active engineering plans
  - historical implementation notes
  - archived material
- add an index for `docs/from-root`
- decide whether duplicate `CONTRIBUTING` files should be consolidated or cross-linked
- avoid deleting docs in the first pass unless they are obvious duplicates

Expected result:

- easier navigation
- less duplication
- clearer “where should new planning docs go?” guidance

Primary requested item covered here:

- trim or index the planning docs

### Phase 4: Frontend Workflow Clarity

Files to review/update:

- [/README.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/README.md)
- [/package.json](/Users/sueo8x/Documents/Github/autoclean_pipeline/package.json)
- [/web/package.json](/Users/sueo8x/Documents/Github/autoclean_pipeline/web/package.json)

Planned work:

- document clearly that the frontend app lives under `web/`
- do not add root proxy scripts for frontend build or test commands
- keep this small and documentation-only

Expected result:

- fewer failed root-level commands
- better contributor onboarding

## Explicit Non-Goals For First Pass

- no accidental deletion of canonical documentation
- no repo-wide reformat of every tracked file
- no mass line-ending rewrite of the whole repository unless the policy change requires a carefully managed normalization pass
- no build system redesign
- no migration of all planning docs out of current locations in one shot

## Risks

### Risk 1

Changing `.gitattributes` can create large diffs if followed by an uncontrolled renormalization.

Mitigation:

- update policy first
- then do one deliberate, reviewable repository-wide renormalization step
- keep that renormalization isolated so the repo does not need repeat line-ending cleanup later

### Risk 2

Deleting stashes too aggressively could lose useful unfinished work.

Mitigation:

- inspect before dropping
- keep anything that is plausibly still valuable

### Risk 3

Moving docs too quickly can break links or confuse current contributors.

Mitigation:

- prefer indexing and cross-linking before relocation

## Recommended Execution Order

1. finalize cleanup policy in `.gitattributes` and `.gitignore`
2. add `.editorconfig` if we decide to
3. run one repository-wide LF normalization pass on tracked text files
4. clean stash list
5. add docs indexes and normalize contributor docs
6. clarify frontend workflow in root docs

## Deliverables

Planned end-state deliverables:

- cleaned `.gitattributes`
- cleaned `.gitignore`
- optional `.editorconfig`
- repository-wide LF normalization completed for tracked text files
- reduced stash list
- docs index for planning/reference material
- clarified root/frontend workflow documentation

Requested deliverables to explicitly close:

- line-ending policy normalized
- tracked files normalized for Linux-style line endings
- one-time repo-wide touch completed so line-ending-only commits stop recurring
- stash list cleaned
- planning docs trimmed or indexed

## Status

Completed in this cleanup pass:

- line-ending policy changed away from global CRLF forcing
- `.editorconfig` added
- tracked text files normalized to `LF`
- obsolete stash entries cleared
- planning docs indexed in `docs/from-root`
- planning directory indexed with `plans/README.md`
- top-level docs indexed with `docs/INDEX.md`
- duplicate contributor docs reduced to compatibility pointers to the canonical root guide
- root README updated with frontend/doc navigation notes
- tracked session transcript artifact removed and ignored for future runs
- safe junk-doc cleanup executed across `docs/`, `docs/from-root/`, and `plans/`
- generated `plans/_site/` output removed from version control
- `plans/.gitignore` updated so `_site/` stays local
- standalone one-time plan notes moved under `plans/archive/standalone/`

Remaining follow-up work, if desired:

- review whether `main-plan.md` and `main-plan-log.csv` still deserve to remain active
- decide whether the numbered `plans/*/*.md` reasoning files remain part of the canonical plan format
- review the `Needs Judgment Before Deletion` docs list

## Separate Follow-Up: Junk Docs Investigation

This is a separate cleanup track from the main line-ending / stash / indexing work.

### Goal

Identify documentation files that are truly junk, backup artifacts, generated leftovers, or redundant non-canonical copies, and remove them deliberately.

### Examples already identified for review

- duplicate contributor docs outside the canonical root guide
- timestamped backup docs such as CLI help dumps
- generated-orphan docs that are not part of the intended published set
- standalone HTML artifacts that are not linked from the docs entry points

### Rules for this separate pass

- referenced docs should not be deleted casually
- generated docs that are intentionally linked should stay
- obvious backups, scratch exports, and duplicate non-canonical docs can be deleted
- if a doc is historical but still useful, archive it instead of deleting it

### Output of this separate pass

- a short list of docs to delete
- a short list of docs to archive
- a short list of docs to keep because they are still referenced or intentional

### Investigation Method

This review checked:

- the current `docs/` and `plans/` file inventory
- repo references to candidate files with `rg`
- the current docs indexes:
  - [/docs/INDEX.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/INDEX.md)
  - [/plans/README.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/README.md)
  - [/plans/README.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/README.md)
- whether a file appears to be generated output, a backup dump, a scratch note, a compatibility pointer, or an intentionally linked operator doc

### Findings

#### Best-Practice Assessment

By normal repository standards, the current state is too artifact-heavy.

The main repo-cleanliness problems are:

- generated website output is tracked in the main branch
- many planning files exist as source-plus-shadow copies rather than one canonical source
- scratch and prompt-transcript style files are mixed into the real plan set
- some compatibility and backup docs remain in-tree even though they are not canonical

That is not a good long-term repo shape. A cleaner repo should prefer:

- one canonical source per planning document
- generated site output ignored or published elsewhere
- archived material moved out of the active plan tree
- scratch notes, prompt transcripts, and timestamped dumps removed

#### Direction Decision: `plans/` stays, but one-time plans should be archived inside it

Current decision:

- `plans/` remains the home for planning/design artifacts
- plans should stay out of `docs/`
- but one-time or superseded plan files should not sit loose at the top level forever

Best-practice direction:

- keep active structured plan sources in `plans/`
- move one-time or retired plan artifacts into `plans/archive/`
- keep generated output and process residue out of the active surface area

#### Biggest Junk Cluster: `plans/`

This is the strongest cleanup target in the repo.

Observed state:

- `plans/` is about `5.0M`
- `plans/_site/` alone is about `4.7M`
- `plans/_site/` contains `47` tracked generated files
- `23` numbered plan folders contain both a `.md` reasoning file and a `.qmd` executed file with the same basename
- [/plans/main-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan.md) is largely a preserved instruction transcript, not a clean project document
- [/plans/main-plan-log.csv](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan-log.csv) is an execution ledger/export, not a primary engineering artifact

Best-practice conclusion:

- `plans/_site/` should not live in the main repo unless deployment explicitly depends on committed static output
- standalone one-time plan notes should move into `plans/archive/`, not stay loose at the top level
- `main-plan.md` and `main-plan-log.csv` still look like process residue rather than durable project documentation
- the numbered RFC `.md` plus `.qmd` pairing should be treated as a separate policy decision, not an automatic delete target in this cleanup pass

#### Strong Cleanup Candidates In `plans/`

- [/plans/_site](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site)
  - generated Quarto output
  - dominates plan-directory size
  - should be ignored and regenerated, not tracked, unless there is a hard deployment requirement
- [/plans/archive/standalone/scratch.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/standalone/scratch.md)
  - archived one-time note
  - no longer part of the active top-level plan surface
- [/plans/main-plan-log.csv](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan-log.csv)
  - process log/export
  - no repo references outside plan instructions and generated site output
  - not a normal long-lived repo document

#### Likely Delete Or Retire Candidates In `plans/`

- [/plans/main-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan.md)
  - preserved instruction transcript
  - currently rendered into the Quarto site
  - does not read like a durable project plan
- the paired numbered `.md` reasoning plans beside each `.qmd`
  - example: [/plans/001-intro-big-picture/001-intro-big-picture.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/001-intro-big-picture/001-intro-big-picture.md)
  - example: [/plans/001-intro-big-picture/001-intro-big-picture.qmd](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/001-intro-big-picture/001-intro-big-picture.qmd)
  - current pattern stores both a short planning note and an executed document for each RFC
  - best practice would be to keep the canonical executed source and archive or delete the shadow reasoning copy unless it still serves an active workflow

#### Delete Candidates

These are the strongest junk-doc candidates currently in the repo.

- [/docs/cli-help-backup-20250830_162135.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/cli-help-backup-20250830_162135.md)
  - timestamped backup dump
  - no repo references
  - content is a captured CLI help transcript, not maintained documentation
- [/docs/sg_execution_times.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/sg_execution_times.rst)
  - orphan generated Sphinx-gallery timing page
  - no repo references
  - reports `0 files` and does not document product behavior
- [/plans/archive/standalone/scratch.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/standalone/scratch.md)
  - archived raw scratch note
  - content appears already absorbed into the numbered serve/automation planning set

#### Likely Delete Or Archive Candidates

- [/docs/platform-guide.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/platform-guide.html)
  - no repo references
  - standalone HTML artifact with embedded styling
  - should be deleted unless there is an intentional external/manual use that is not captured in repo docs
- [/plans/main-plan-log.csv](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan-log.csv)
  - no repo references
  - looks like an execution log/export rather than a canonical plan source
  - likely archive-or-delete material, not active top-level documentation
- [/plans/main-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan.md)
  - reads like instruction residue rather than a clean project artifact
  - currently kept alive mostly because the Quarto site renders it
- numbered plan `.md` reasoning files in `plans/*/*.md`
  - likely redundant next to the `.qmd` executed documents
  - should be retired if the repo chooses one canonical source format per RFC

#### Keep For Now

These are old or specialized, but not junk based on the current repo state.

- [/docs/serve-docs-portal.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-docs-portal.html)
- [/docs/serve-first-route-tutorial.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-first-route-tutorial.html)
- [/docs/serve-edit-route-tutorial.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-edit-route-tutorial.html)
- [/docs/serve-failures-recovery-tutorial.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-failures-recovery-tutorial.html)
- [/docs/serve-operator-handoff-checklist.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-operator-handoff-checklist.html)
- [/docs/serve-route-first-operator-guide.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-route-first-operator-guide.html)
- [/docs/serve-route-first-devlog.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-route-first-devlog.md)

Reason:

- they are linked from [/docs/index.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/index.rst) and from each other
- they are part of an intentional local Serve docs set, not random leftovers

#### Compatibility-Pointer Docs

- [/docs/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/CONTRIBUTING.md)
- [/docs/from-root/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/CONTRIBUTING.md)

Assessment:

- these are no longer true duplicate guides
- they are tiny compatibility-pointer files to the canonical root guide at [/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/CONTRIBUTING.md)
- they are not high-value docs, but they are also not harmful junk unless we want to aggressively delete compatibility links

#### Broader Cleanup Pattern

A large number of docs in `docs/`, `docs/from-root/`, and `plans/` are not referenced by other repo files. That does not make them junk by itself.

The main categories are:

- canonical source docs that are reached through index pages rather than direct filename mentions
- historical engineering notes that should likely move to `docs/archive/` over time
- generated Quarto output in `plans/_site/`, which is intentional generated content and not junk as long as the project still publishes it
- one-off dumps and scratch artifacts, which are the highest-confidence delete targets

#### `docs/from-root/` Assessment

This folder is much smaller than the broader `plans/` clutter, but the same pattern is present: most files are one-off engineering residue, not durable repository documentation.

Current inventory reviewed:

- `16` files total under `docs/from-root/`
- `2` smoke helper scripts under `docs/from-root/smoke_tools/`
- only a small subset are referenced outside the folder itself

Best-practice conclusion:

- `docs/from-root/` should be reduced to a very small set of durable repo-operational documents
- implementation summaries, agent-specific prompt guides, ad hoc investigations, and one-off issue notes should generally not live here long-term
- if something is truly worth keeping, it should move to the proper canonical location in `docs/`, `plans/`, or `docs/archive/`

#### Keep In `docs/from-root/`

These are the only files that clearly justify staying here for now.

- [/plans/REPO_CLEANUP_PLAN.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/REPO_CLEANUP_PLAN.md)
  - active cleanup plan for the work currently in progress
- [/plans/README.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/README.md)
  - only needed while this folder still exists in any meaningful form
- [/AGENTS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/AGENTS.md)
  - canonical root location for automatic agent discovery
- [/CLAUDE.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/CLAUDE.md)
  - canonical root location for Claude-oriented discovery and workflow references

#### Strong Delete Candidates In `docs/from-root/`

These look like exactly the kind of junk that should not stay in a normal repo.

- [/docs/from-root/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/CONTRIBUTING.md)
  - compatibility pointer only
  - not a canonical doc
- [/docs/from-root/BLOCK_DUPLICATION_ANALYSIS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/BLOCK_DUPLICATION_ANALYSIS.md)
  - one-off analysis note
  - better archived or deleted
- [/docs/from-root/FOOOF_IMPLEMENTATION_SUMMARY.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/FOOOF_IMPLEMENTATION_SUMMARY.md)
  - branch/date implementation summary
  - process residue, not durable docs
- [/docs/from-root/GUI_ENHANCEMENTS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/GUI_ENHANCEMENTS.md)
  - one feature summary
  - not canonical user or developer documentation
- [/docs/from-root/IMPLEMENTATION_SUMMARY.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/IMPLEMENTATION_SUMMARY.md)
  - generic branch-specific implementation summary
  - strong junk candidate
- [/docs/from-root/MANUAL_OVERRIDE_IMPLEMENTATION.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/MANUAL_OVERRIDE_IMPLEMENTATION.md)
  - implementation note, not maintained product docs
- [/docs/from-root/SERVE_EXCLUDE_DRAG_SELECT_IMPLEMENTATION.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/SERVE_EXCLUDE_DRAG_SELECT_IMPLEMENTATION.md)
  - one-off feature plan for already-implemented work
  - should not remain as a permanent root-note artifact
- [/docs/from-root/autoclean_exclude_file_searching_strategies.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/autoclean_exclude_file_searching_strategies.md)
  - focused investigation note
- [/docs/from-root/ic_component_psd_fmax_issue_report.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/ic_component_psd_fmax_issue_report.md)
  - issue investigation writeup
- [/docs/from-root/serve-test-smoke-issue-141.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/serve-test-smoke-issue-141.md)
  - issue-specific runbook
  - should live elsewhere if still needed
- [/docs/from-root/smoke_tools/serve_test_smoke.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/smoke_tools/serve_test_smoke.py)
  - operational script stored under docs
  - wrong location even if retained
- [/docs/from-root/smoke_tools/serve_test_smoke.sh](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/smoke_tools/serve_test_smoke.sh)
  - shell wrapper stored under docs
  - wrong location even if retained

#### Recommended Action For `docs/from-root/`

Best-practice target:

- keep only:
  - [/plans/REPO_CLEANUP_PLAN.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/REPO_CLEANUP_PLAN.md)
  - [/plans/README.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/README.md)
  - [/AGENTS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/AGENTS.md)
  - [/CLAUDE.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/CLAUDE.md)
- delete the rest unless a specific file is promoted into a canonical location first

### Recommended Separate Cleanup Actions

If we execute this separate junk-doc pass, the recommended order is:

1. remove the biggest generated artifact set first:
   - [/plans/_site](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site)
   - update ignore rules so generated Quarto site output is not recommitted by accident
2. delete the clear junk files:
   - [/docs/cli-help-backup-20250830_162135.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/cli-help-backup-20250830_162135.md)
   - [/docs/sg_execution_times.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/sg_execution_times.rst)
   - [/plans/archive/standalone/scratch.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/standalone/scratch.md)
   - [/plans/main-plan-log.csv](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan-log.csv)
3. retire non-canonical planning residue:
   - decide whether [/plans/main-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan.md) stays at all
   - choose one canonical source format for numbered RFC plans and delete or archive the shadow files
4. decide whether to delete or archive:
   - [/docs/platform-guide.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/platform-guide.html)
5. leave the linked Serve HTML docs alone
6. only delete the compatibility-pointer contributor docs if we are willing to update or break any old deep links

### Proposed Delete List

This is the aggressive best-practice delete list based on the current repo state.

The standard applied here is:

- keep canonical docs, indexes, and published docs sources
- delete generated artifacts, scratch notes, backup dumps, prompt residue, shadow copies, and one-off implementation notes
- if a file is not part of an intentional published docs tree and is not a canonical engineering artifact, it should usually not be in the repo

### Safer Execution Split

To avoid deleting real documentation by accident, the delete candidates should be split into two groups:

- `Safe to delete now`
  - generated artifacts
  - scratch notes
  - backup dumps
  - shadow copies with an obvious canonical counterpart
  - compatibility pointers
- `Needs judgment before deletion`
  - documents that may still be the canonical explanation of a feature, architecture decision, or workflow even if they are poorly placed

The lists below follow that rule.

#### Safe To Delete Now: top-level junk in `docs/`

- [/docs/amplitude_quality_coach.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/amplitude_quality_coach.md)
  - standalone one-off note
  - no repo references
  - not part of published docs structure
- [/docs/cli-help-backup-20250830_162135.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/cli-help-backup-20250830_162135.md)
  - timestamped backup dump
  - obvious artifact
- [/docs/directory-backup-resolution.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/directory-backup-resolution.md)
  - one-off resolution note
  - no repo references
- [/docs/event_discovery_feature.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/event_discovery_feature.md)
  - feature note outside canonical docs tree
  - no repo references
- [/docs/platform-guide.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/platform-guide.html)
  - standalone styled HTML artifact
  - no repo references
- [/docs/serve-data1-pilot-walkthrough.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-data1-pilot-walkthrough.md)
  - pilot walkthrough note
  - no repo references
- [/docs/serve_ui_workflow.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve_ui_workflow.md)
  - duplicate-format workflow doc
  - no repo references
- [/docs/serve_ui_workflow.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve_ui_workflow.rst)
  - duplicate-format workflow doc
  - no repo references
- [/docs/sg_execution_times.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/sg_execution_times.rst)
  - generated Sphinx timing page
  - reports no real gallery content
- [/docs/ui_startup_profiling_plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/ui_startup_profiling_plan.md)
  - profiling plan artifact
  - no repo references

#### Safe To Delete Now: compatibility-pointer docs

- [/docs/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/CONTRIBUTING.md)
  - compatibility pointer only
  - canonical file already exists at repo root
- [/docs/from-root/CONTRIBUTING.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/CONTRIBUTING.md)
  - compatibility pointer only
  - canonical file already exists at repo root

#### Safe To Delete Now: `docs/development/` residue

- [/docs/development/BDF_IMPLEMENTATION_SUMMARY.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/development/BDF_IMPLEMENTATION_SUMMARY.md)
  - implementation summary, not durable docs
  - no repo references

#### Safe To Delete Now: `docs/mdx/` one-off run and plan artifacts

- [/docs/mdx/report-restructure-implementation.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/report-restructure-implementation.mdx)
  - one-off implementation note
  - no repo references
- [/docs/mdx/report-restructure-plan.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/report-restructure-plan.mdx)
  - one-off plan artifact
  - no repo references
- [/docs/mdx/run-autoclean-exclusion-tool.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-autoclean-exclusion-tool.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-guided-setup-wizard.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-guided-setup-wizard.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-ica-custom-layout.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-ica-custom-layout.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-processing-status-failure-handling.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-processing-status-failure-handling.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-reporting-template-extension.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-reporting-template-extension.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-template-engine-evaluation.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-template-engine-evaluation.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-template-hardening.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-template-hardening.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-template-validation-safeguards.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-template-validation-safeguards.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-wavelet-erp-upgrade.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-wavelet-erp-upgrade.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/run-wavelet-report-integration.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/run-wavelet-report-integration.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/standalone-autoclean-exclude-upgrade.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/standalone-autoclean-exclude-upgrade.mdx)
  - one-off run artifact
  - no repo references
- [/docs/mdx/wavelet-thresholding.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/mdx/wavelet-thresholding.mdx)
  - one-off run artifact
  - no repo references

#### Safe To Delete Now: `docs/runs/` and maintenance residue

- [/docs/runs/2025-02-16-builtins-registry.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/runs/2025-02-16-builtins-registry.mdx)
  - dated run artifact
  - no repo references
- [/docs/maintenance_logs/2025-09-17-remove-mkdocs.mdx](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/maintenance_logs/2025-09-17-remove-mkdocs.mdx)
  - maintenance log transcript
  - not durable product or contributor documentation

#### Safe To Delete Now: `docs/from-root/` one-off engineering residue

- [/docs/from-root/BLOCK_DUPLICATION_ANALYSIS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/BLOCK_DUPLICATION_ANALYSIS.md)
  - one-off analysis note
- [/docs/from-root/FOOOF_IMPLEMENTATION_SUMMARY.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/FOOOF_IMPLEMENTATION_SUMMARY.md)
  - branch/date implementation summary
- [/docs/from-root/GUI_ENHANCEMENTS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/GUI_ENHANCEMENTS.md)
  - one feature summary
- [/docs/from-root/IMPLEMENTATION_SUMMARY.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/IMPLEMENTATION_SUMMARY.md)
  - generic implementation summary residue
- [/docs/from-root/MANUAL_OVERRIDE_IMPLEMENTATION.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/MANUAL_OVERRIDE_IMPLEMENTATION.md)
  - one-off implementation note
- [/docs/from-root/SERVE_EXCLUDE_DRAG_SELECT_IMPLEMENTATION.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/SERVE_EXCLUDE_DRAG_SELECT_IMPLEMENTATION.md)
  - one-off feature plan for already-implemented work
- [/docs/from-root/autoclean_exclude_file_searching_strategies.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/autoclean_exclude_file_searching_strategies.md)
  - focused investigation note
- [/docs/from-root/ic_component_psd_fmax_issue_report.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/ic_component_psd_fmax_issue_report.md)
  - issue analysis note
- [/docs/from-root/serve-test-smoke-issue-141.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/serve-test-smoke-issue-141.md)
  - issue-specific runbook
- [/docs/from-root/smoke_tools/serve_test_smoke.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/smoke_tools/serve_test_smoke.py)
  - script stored under docs
  - wrong location even if retained
- [/docs/from-root/smoke_tools/serve_test_smoke.sh](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/from-root/smoke_tools/serve_test_smoke.sh)
  - script stored under docs
  - wrong location even if retained

#### Safe To Delete Now: `plans/` generated site output

Every file in `plans/_site/` is tracked generated output and should be deleted from the repo, then ignored.

- [/plans/_site/index.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/index.html)
  - generated Quarto output
- [/plans/_site/main-plan.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/main-plan.html)
  - generated Quarto output
- [/plans/_site/001-intro-big-picture/001-intro-big-picture.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/001-intro-big-picture/001-intro-big-picture.html)
  - generated Quarto output
- [/plans/_site/002-automation-idempotency/002-automation-idempotency.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/002-automation-idempotency/002-automation-idempotency.html)
  - generated Quarto output
- [/plans/_site/003-automation-mode/003-automation-mode.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/003-automation-mode/003-automation-mode.html)
  - generated Quarto output
- [/plans/_site/004-automation-mode-validation/004-automation-mode-validation.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/004-automation-mode-validation/004-automation-mode-validation.html)
  - generated Quarto output
- [/plans/_site/005-serve-workspace/005-serve-workspace.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/005-serve-workspace/005-serve-workspace.html)
  - generated Quarto output
- [/plans/_site/006-serve-command-family/006-serve-command-family.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/006-serve-command-family/006-serve-command-family.html)
  - generated Quarto output
- [/plans/_site/007-ingestion-planning/007-ingestion-planning.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/007-ingestion-planning/007-ingestion-planning.html)
  - generated Quarto output
- [/plans/_site/008-ingestion-prd/008-ingestion-prd.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/008-ingestion-prd/008-ingestion-prd.html)
  - generated Quarto output
- [/plans/_site/009-ingestion-implementation/009-ingestion-implementation.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/009-ingestion-implementation/009-ingestion-implementation.html)
  - generated Quarto output
- [/plans/_site/010-ingestion-provenance/010-ingestion-provenance.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/010-ingestion-provenance/010-ingestion-provenance.html)
  - generated Quarto output
- [/plans/_site/011-ingestion-watcher/011-ingestion-watcher.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/011-ingestion-watcher/011-ingestion-watcher.html)
  - generated Quarto output
- [/plans/_site/012-ingestion-dispatch/012-ingestion-dispatch.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/012-ingestion-dispatch/012-ingestion-dispatch.html)
  - generated Quarto output
- [/plans/_site/013-ingestion-dispatch-execution/013-ingestion-dispatch-execution.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/013-ingestion-dispatch-execution/013-ingestion-dispatch-execution.html)
  - generated Quarto output
- [/plans/_site/014-ingestion-runtime-dispatch/014-ingestion-runtime-dispatch.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/014-ingestion-runtime-dispatch/014-ingestion-runtime-dispatch.html)
  - generated Quarto output
- [/plans/_site/015-ingestion-integration/015-ingestion-integration.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/015-ingestion-integration/015-ingestion-integration.html)
  - generated Quarto output
- [/plans/_site/016-ingestion-loop/016-ingestion-loop.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/016-ingestion-loop/016-ingestion-loop.html)
  - generated Quarto output
- [/plans/_site/017-ingestion-service/017-ingestion-service.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/017-ingestion-service/017-ingestion-service.html)
  - generated Quarto output
- [/plans/_site/019-serve-multi-route-response/019-serve-multi-route-response.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/019-serve-multi-route-response/019-serve-multi-route-response.html)
  - generated Quarto output
- [/plans/_site/020-rfc19-execution-review/020-rfc19-execution-review.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/020-rfc19-execution-review/020-rfc19-execution-review.html)
  - generated Quarto output
- [/plans/_site/021-serve-multi-route-implementation/021-serve-multi-route-implementation.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/021-serve-multi-route-implementation/021-serve-multi-route-implementation.html)
  - generated Quarto output
- [/plans/_site/022-serve-multi-route-enforcement/022-serve-multi-route-enforcement.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/022-serve-multi-route-enforcement/022-serve-multi-route-enforcement.html)
  - generated Quarto output
- [/plans/_site/023-resting-state-automation-test/023-resting-state-automation-test.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/023-resting-state-automation-test/023-resting-state-automation-test.html)
  - generated Quarto output
- [/plans/_site/024-testing-make-targets/024-testing-make-targets.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/_site/024-testing-make-targets/024-testing-make-targets.html)
  - generated Quarto output

#### Needs Judgment Before Deletion: active `plans/` source-format decisions

These files are not obvious junk. They are part of the current numbered RFC planning format, even if that format may be overly redundant.

- `plans/*/*.md` reasoning files beside the numbered `.qmd` executed documents
  - example: [/plans/001-intro-big-picture/001-intro-big-picture.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/001-intro-big-picture/001-intro-big-picture.md)
  - example: [/plans/001-intro-big-picture/001-intro-big-picture.qmd](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/001-intro-big-picture/001-intro-big-picture.qmd)
  - keep for now unless we make an explicit repo policy decision that `.qmd` is the only canonical source format

#### Safe To Delete Now: `plans/` process residue

These are not the active numbered plan sources. They are process residue, generated output, or archived one-off leftovers.

- [/plans/main-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan.md)
  - preserved instruction transcript
  - not a clean durable project doc
- [/plans/main-plan-log.csv](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan-log.csv)
  - process log export
  - not a canonical source document

#### Safe To Delete Now: archived plan residue if we want a truly clean repo

- [/plans/archive/018-serve-routing-sprints/018-serve-routing-sprints.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/018-serve-routing-sprints/018-serve-routing-sprints.md)
  - archived shadow reasoning file
- [/plans/archive/018-serve-routing-sprints/018-serve-routing-sprints.qmd](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/018-serve-routing-sprints/018-serve-routing-sprints.qmd)
  - archived executed plan source
- [/plans/archive/018-serve-routing-sprints/018-serve-routing-sprints.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/018-serve-routing-sprints/018-serve-routing-sprints.html)
  - archived generated output

#### Needs Judgment Before Deletion

These are suspicious, poorly placed, or unreferenced, but they may still be the only meaningful documentation for their topic.

- [/plans/archive/imported-docs/CODEBASE_MAP.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/imported-docs/CODEBASE_MAP.md)
  - could be real architecture documentation
  - currently just not integrated well
- [/plans/archive/imported-docs/MNE_PICK_TYPES_MIGRATION.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/imported-docs/MNE_PICK_TYPES_MIGRATION.md)
  - may be a valid migration note
- [/plans/archive/imported-docs/autoclean-exclude-web-ui-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/imported-docs/autoclean-exclude-web-ui-plan.md)
  - may be the only design record for a real feature
- [/plans/archive/imported-docs/autoclean_exclude_default_folder.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/imported-docs/autoclean_exclude_default_folder.md)
  - could represent still-relevant behavior documentation
- [/plans/archive/imported-docs/channel_count_accuracy_plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/imported-docs/channel_count_accuracy_plan.md)
  - might be the only explanation of a real reporting fix
- [/plans/archive/imported-docs/task_generator_wizard.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/imported-docs/task_generator_wizard.md)
  - might be feature design documentation rather than junk
- [/plans/archive/imported-docs/PLUGIN_REPORT_INTERFACE_PROPOSAL.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/imported-docs/PLUGIN_REPORT_INTERFACE_PROPOSAL.md)
  - proposal doc referenced from the changelog
  - likely real documentation, just niche
- numbered `plans/*/*.md` reasoning files beside `.qmd`
  - these stay for now as part of the current active plan format
  - delete only if we later make an explicit format decision to keep `.qmd` alone
- [/plans/main-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan.md)
  - active top-level planning record today
  - still looks process-oriented, but should not be deleted casually
- [/plans/main-plan-log.csv](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/main-plan-log.csv)
  - likely process residue
  - but tied to the current top-level planning workflow, so it should be a deliberate second-pass decision
- [/plans/archive/standalone/scratch.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/standalone/scratch.md)
  - now correctly archived inside `plans/`
  - keep as archived history unless we later decide archive contents should be pruned
- [/plans/archive/standalone/serve-cli-ux-review-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/standalone/serve-cli-ux-review-plan.md)
  - archived one-time plan note
- [/plans/archive/standalone/serve-ui-route-centric-plan.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/archive/standalone/serve-ui-route-centric-plan.md)
  - archived one-time plan note

#### Recommended Actual Execution Order

1. Delete everything in the `Safe To Delete Now` groups.
2. Do not touch the `Needs Judgment Before Deletion` group in the first destructive pass.
3. After the first pass, review whether any `Needs Judgment` file should be:
   - promoted and linked as canonical docs
   - moved to `docs/archive/`
   - or deleted in a second pass

### Files Explicitly Not In This Delete List

These should stay unless there is a separate deliberate docs redesign:

- canonical published docs sources:
  - [/docs/index.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/index.rst)
  - [/docs/getting_started.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/getting_started.rst)
  - [/docs/development.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/development.rst)
  - [/docs/tutorials/index.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/tutorials/index.rst)
  - [/docs/api_reference/index.rst](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/api_reference/index.rst)
- tutorial and API reference trees
- linked Serve operator HTML docs:
  - [/docs/serve-docs-portal.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-docs-portal.html)
  - [/docs/serve-first-route-tutorial.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-first-route-tutorial.html)
  - [/docs/serve-edit-route-tutorial.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-edit-route-tutorial.html)
  - [/docs/serve-failures-recovery-tutorial.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-failures-recovery-tutorial.html)
  - [/docs/serve-operator-handoff-checklist.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-operator-handoff-checklist.html)
  - [/docs/serve-route-first-operator-guide.html](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-route-first-operator-guide.html)
  - [/docs/serve-route-first-devlog.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/docs/serve-route-first-devlog.md)
- repo-operational guidance files:
  - [/AGENTS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/AGENTS.md)
  - [/CLAUDE.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/CLAUDE.md)
  - [/plans/REPO_CLEANUP_PLAN.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/plans/REPO_CLEANUP_PLAN.md)
