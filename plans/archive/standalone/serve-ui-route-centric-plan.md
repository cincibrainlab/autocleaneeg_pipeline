# Serve UI Route-Centric Plan

## Goal

Refine the AutoClean Serve UI so it matches the agreed product structure:

- route-centric where the workflow is route-specific
- utility-oriented where the workflow is global
- thin frontend over CLI/API behavior
- clearer workspace framing
- more scalable filtering and navigation

This plan is based on the latest product feedback and should be treated as the working implementation plan for the next Serve UI cleanup pass.

## Current Product Direction

### Core framing

- Routes are the primary project-level unit.
- The frontend should not invent business logic.
- The CLI is the real API.
- Anything the UI can do should also be possible through CLI/API behavior.

### Route-management hub

`Routes` is the page where route context is created, managed, and selected.

It is not route-scoped in the same way `Results` and `Exclude` are.

It should:

- define the primary project-level unit
- expose route management actions
- establish route context for downstream workflow pages

### Route-contextual areas

These should depend on the selected route:

- Queue
- Results
- Exclude

### Global utility areas

These should not depend on route:

- Tasks
- Montage
- Events
- Settings

### Workspace

- Workspace needs a clearer visual place in the app shell.
- Workspace labeling should be obvious and persistent near the top.
- Browsing behavior needs to work correctly at the root directory.

## Target Information Architecture

### Sidebar order

Primary workflow group:

1. Dashboard
2. Service
3. Routes
4. Queue
5. Results
6. Exclude

Utility group:

1. Tasks
2. Montage
3. Events
4. Settings

Top workspace area:

- visible workspace title
- current workspace path/name
- workspace switch or open action

## Product Decisions To Implement

### 1. Make Queue route-aware but not route-limited

Queue should:

- show all files currently being processed
- support filtering by route
- keep route identity visible per row
- scale to many routes without becoming noisy

Implication:

- Queue is still a global operational page
- but route filtering must be first-class
- Queue should be described as route-contextual, not route-scoped in the strict sense

### 2. Make Results route-scoped

Results should:

- be viewed in the context of a route
- support route selection explicitly
- avoid reading like one giant global result bucket

Possible UI shapes:

- route filter/dropdown at page top
- route-first grouping within the page
- route in URL/query state if helpful for deep linking

Preferred implementation direction:

- use an explicit route selector/filter
- keep the page usable even when many routes exist
- reflect the workflow ordering that places Results after Queue in the route-centric review flow

### 3. Make Exclude route-scoped

Exclude should:

- start from route selection
- show files and review context for the active route
- avoid acting like a global utility detached from route configuration

Implication:

- route choice should be visible and stable in the Exclude workflow
- route context should control which files are shown

### 4. Keep Events as a utility

Events should:

- stay file-level
- act as a diagnostic/inspection tool
- not depend on route-centric navigation

### 5. Keep Tasks tied to the registry model

Tasks UI should:

- reflect the GitHub-backed task registry model
- avoid pretending tasks are purely local ad hoc objects
- support safe editing/rename behavior without breaking repository linkage

## Engineering Constraints

### CLI parity rule

CLI parity is not a late cleanup item. It is a gating rule for implementation.

For every new or changed frontend action:

- identify the CLI/API operation behind it
- confirm the backend capability already exists or is being added there first
- avoid embedding unique decision-making in the frontend

This check should happen during each implementation phase, not only at the end.

### Frontend role

The frontend should be responsible for:

- presentation
- flow organization
- discoverability
- filtering state
- user feedback

The frontend should not be responsible for:

- hidden processing logic
- business rules that do not exist in CLI/API
- route/task behaviors that only exist in the UI

## Existing Fixes And Known Patches

### In progress

- workspace browse root bug
  - browsing to `/` currently flags incorrectly as a security risk
  - patch needs to be merged and verified

### Fixed or intended fix

- invalid route delete behavior
  - deleting a route currently fails unless archived first
  - current agreed direction is to remove the broken delete path rather than expose invalid behavior

## Open Questions

### Queue / processing

- Has the pre-processing queue already been implemented end to end?
- Is current queue behavior fully backed by CLI/API?
- Are there workflow gaps between queue state and route state?
- What is the default UX when there are no routes, exactly one route, or many routes?

### Smarter route filtering

- Should route filters support task-based narrowing?
- Should route filters support montage/net-based narrowing?
- Should the app proactively show only relevant routes in certain contexts?

### Task editing model

- How much inline editing is appropriate?
- When should the UI direct users back to repository-backed task workflows?
- What metadata can be safely edited in-app without weakening registry integrity?

### Tutorial / onboarding

- How much of the existing tutorial becomes wrong after the navigation changes?
- Which route-centric flows need new walkthrough coverage?

### Future AI assistant direction

- A unified chat or AI assistant experience may be valuable later as a cross-cutting help layer.
- This should be treated as a future enhancement, not part of the current navigation/workflow cleanup.
- It should not be implemented until the route-centric information architecture and CLI-parity work are stable.

## Implementation Plan

### Phase 1: Navigation and workspace cleanup

- [x] Reorganize the sidebar to match the agreed structure
- [x] Separate route-centric pages from utility pages visually
- [x] Add a clearly labeled workspace section near the top of the shell
- [x] Make workspace title/path easier to understand at a glance
- [x] Validate the revised sidebar on narrow and wide displays
- [x] Define route-context rules clearly:
  - `Routes` as the management hub
  - `Queue` as global with route filter
  - `Results` and `Exclude` as route-scoped review pages
- [x] Add CLI-parity check to the implementation checklist for every page touched

### Phase 2: Queue restructuring

- [x] Confirm current queue behavior and data source through CLI/API
- [x] Make Queue show all files being processed
- [x] Add route filter controls to Queue
- [x] Show route identity clearly in queue rows
- [x] Design queue filtering so it scales across many routes
- [x] Preserve useful global visibility while allowing route-specific focus
- [x] Define fallback UX for:
  - no routes
  - one route
  - many routes
- [x] Verify Queue changes do not introduce UI-only logic not backed by CLI/API

### Phase 3: Route-scoped Results

- [x] Make Results route-based
- [x] Add an explicit route selector/filter to Results
- [x] Ensure Results remains usable with large route counts
- [x] Decide whether route selection belongs in URL state
- [x] Make the page structure clearly read as route-contextual, not global
- [x] Clarify whether “under Queue” means:
  - workflow order only
  - or actual navigation nesting
- [x] Verify Results route context is sourced from CLI/API-backed route data

### Phase 4: Route-scoped Exclude

- [x] Make Exclude operate in route context
- [x] Add route selection or route context display in Exclude
- [x] Ensure Exclude file lists are scoped to the active route
- [x] Preserve current Exclude review workflow while clarifying route ownership
- [x] Confirm all Exclude actions map cleanly to backend/CLI behavior

### Phase 5: Utility section cleanup

- [x] Keep Tasks in the utility group
- [x] Keep Montage in the utility group
- [x] Keep Events in the utility group
- [x] Keep Settings in the utility group
- [x] Make Events clearly read as a file inspection tool
- [x] Remove route-scoped visual assumptions from utility pages

### Phase 6: Bug fixes and cleanup

- [x] Merge the root workspace browse fix
- [x] Verify browsing to `/` no longer produces the wrong security error
- [x] Merge the invalid route delete fix
- [x] Remove or disable broken route delete affordances in the UI
- [x] Re-test route archive/delete flows after the patch lands

### Phase 7: Filtering and scalability

- [x] Add smarter route dropdown/filter behavior where needed
- [x] Explore route filtering by task
- [x] Explore route filtering by montage/net
- [x] Prevent route pickers from becoming unusable at scale
- [x] Standardize route filter behavior across Queue, Results, and Exclude

### Phase 8: Task registry UX

- [x] Add a GitHub link in the Tasks UI for the task registry
- [x] Make the registry relationship explicit in the UI
- [x] Design safe rename/edit behavior for tasks
- [x] Preserve repository linkage in any task-edit workflow
- [x] Avoid creating a misleading “fully local task editor” model

### Phase 9: Final CLI parity audit

- [x] Audit implemented frontend actions for CLI parity
- [x] Remove or redesign any UI action that lacks CLI/API support
- [x] Confirm route-scoped behavior is driven by backend capabilities
- [x] Document gaps where backend work is required before UI work
- [x] Keep frontend-specific logic minimal

Notes:

- Results route scoping required backend `route_id` support in the results API rather than frontend-only grouping.
- Exclude route scoping required backend route-specific exports-root resolution rather than a global workspace-wide file list.
- Queue route filtering was implemented against the existing queue API route filter instead of adding UI-only filtering rules.

### Phase 10: Tutorial and onboarding refresh

- [x] Update tutorial flow for the new navigation structure
- [x] Update tutorial flow for route-scoped Queue, Results, and Exclude
- [x] Re-test tutorial flow end to end
- [x] Identify any missing onboarding hints after the IA changes

## Recommended Build Order

1. Land bug fixes that affect navigation and workspace handling.
2. Lock the route-context model:
   - `Routes` = management hub
   - `Queue` = global with route filter
   - `Results` / `Exclude` = route-scoped
3. Reorganize sidebar and workspace framing.
4. Restructure Queue around global view plus route filtering.
5. Convert Results to route-scoped behavior.
6. Convert Exclude to route-scoped behavior.
7. Clean up utility pages so they no longer imply route context.
8. Tighten route filtering/scalability patterns across all affected pages.
9. Finish task-registry UX and tutorial updates.
10. Run final CLI-parity audit before calling the pass complete.

## Validation Checklist

- [x] Sidebar order matches the agreed product structure
- [x] Workspace is clearly visible and understandable from the shell
- [x] Routes reads as the route-management hub, not a route-scoped detail page
- [x] Queue is clearly global while supporting route filtering
- [x] Results is clearly route-scoped
- [x] Exclude is clearly route-scoped
- [x] Tasks, Montage, Events, and Settings read as utilities
- [x] Events does not imply route dependence
- [x] No frontend action introduces logic that does not exist in CLI/API
- [x] Root workspace browsing works correctly
- [x] Invalid route delete behavior is no longer exposed
- [x] Route filters remain usable when many routes exist
- [x] Task registry linkage remains clear in the UI
- [x] Tutorial matches the final navigation and workflow model

## Explicitly Deferred

These items should be tracked but not worked on in the current pass:

- [ ] Unified AI chat / assistant layer across the Serve UI
  - treat this as future product work after the current IA, filtering, route-scoped UX, and CLI-parity cleanup are complete

## Owners / Coordination Notes

- Nate
  - cherry-pick useful commits/patches from current work
  - validate design decisions with stakeholders as needed
  - keep checking UI actions against CLI parity

- Grace
  - identify missing workflows
  - identify blockers in real usage
  - validate route-scoped review flow and utility split

## Status Assessment

Current working assessment:

- core concept and architecture: mostly there
- major UI structure: direction agreed, still needs cleanup
- key bugs: identified, some already patched
- missing work: filtering, route-scoped views, task-registry UX, tutorial refresh

Practical status label:

**Feature-complete directionally, but not yet polished or release-ready.**
