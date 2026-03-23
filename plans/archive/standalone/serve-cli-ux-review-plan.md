# Serve CLI UX Review And Improvement Plan

## Goal

Review the current `autocleaneeg-pipeline serve` user experience and define a concrete plan to make Serve easier to set up, start, and operate from the CLI-first workflow.

This plan assumes:

- users start Serve with the CLI, not the root repo `Makefile`
- the web UI should be a first-class operator surface
- everything the UI can do must also be doable through the CLI
- `serve` should start the right things for the normal user path without requiring users to understand the internal process model

## Scope

### In scope

- CLI-first setup flow for Serve
- how users create or link a Serve workspace
- how users bring Serve up
- how users know whether processing is actually running
- how much of setup can be completed in the UI versus the CLI
- command-family design for `serve`
- onboarding/help/status output for `serve`

### Out of scope

- authentication
- permissions
- email / Resend
- deep redesign of route semantics
- changing the core rule that UI actions must map to CLI/backend capabilities

## Current State

## What exists today

The current Serve command family includes:

- `serve workspace`
- `serve list`
- `serve route ...`
- `serve validate`
- `serve deploy`
- `serve run`
- `serve api`
- `serve up`
- `serve down`
- `serve restart`
- `serve status`
- `serve share`
- `serve tui`

### Current startup model

There are currently two different startup mental models:

#### Low-level model

- `serve api` starts the web API/UI server
- `serve run` starts the ingestion dispatcher
- `serve worker` exists as an advanced path

#### Convenience model

- `serve up` starts the dashboard server
- `serve down` stops it
- `serve status` reports whether it is running

### Current workspace model

Users can:

- create a Serve workspace with `serve workspace --mode new --path ...`
- link an existing Serve workspace with `serve workspace --mode existing --path ...`
- persist the workspace path in user config

### Current route/config model

Users can:

- create and update routes from the CLI with `serve route upsert`
- promote/archive/unarchive routes
- sync routes into compiled Serve YAML
- validate YAML
- deploy YAML

### Current UI model

Once the API is up, the Serve UI can:

- configure the workspace
- inspect routes, queue, results, exclude, settings
- start and stop the dispatcher from the Service page

## Current UX Problems

### 1. `serve up` does not clearly mean "Serve is operational"

This is the biggest UX problem.

Today, `serve up` starts the dashboard/API server, but normal users can easily assume that means:

- the web UI is up
- the watcher/dispatcher is up
- files will be processed automatically

That assumption is not reliably true.

For actual processing, the dispatcher still matters.

This creates a mismatch between:

- what the command sounds like
- what users expect
- what is actually running

### 2. Setup is still fragmented between CLI and UI

A user can get into a state where:

- the UI is open
- but the workspace is only partially configured
- routes are missing
- the service is not actually running
- processing is not happening

That means there is not yet one clean, end-to-end setup path.

### 3. The current command family exposes too many internal layers too early

The current Serve CLI makes advanced concepts visible before the user needs them:

- operator vs deployed config
- validate vs deploy
- api vs run vs worker
- route sync mechanics

These are useful concepts, but they are not the right first-run UX.

### 4. It is not obvious which steps are required versus optional

A new user has to infer:

- whether `serve workspace` is enough
- whether `serve up` is enough
- whether `serve run` is also needed
- whether the UI will start the dispatcher for them
- whether `validate` and `deploy` are mandatory before the UI is usable

That ambiguity is too high.

### 5. The UI is not yet a complete replacement for setup

Right now, it does not appear that a new user can reliably complete the entire happy path only from the Serve UI.

In particular, the path from:

- no Serve workspace
- to configured workspace
- to configured route
- to deployed config
- to active processing

is not yet clearly one continuous UI-driven flow.

### 6. CLI parity is a product requirement, but not yet a polished UX principle

The rule is correct:

- everything the UI can do must also be doable with CLI/backend

But the user experience implication needs to be stronger:

- the CLI should have a clear recommended path
- the UI should reflect that path
- advanced commands should remain available without becoming the default burden

## Desired User Experience

## Primary user journeys

### Journey 1: First-time local lab setup

The user should be able to:

1. create or select a Serve workspace
2. launch Serve
3. complete remaining setup in the UI
4. create at least one route
5. start processing
6. understand whether Serve is actually active

### Journey 2: Returning operator

The user should be able to:

1. run one command
2. know Serve is up
3. know whether processing is active
4. open the UI and continue working

### Journey 3: CLI-only operator

The user should be able to:

1. create a workspace
2. create routes
3. validate and deploy configuration
4. start Serve
5. confirm queue and processing state

without needing the web UI at all

## UX principles

- one obvious way to start Serve
- one obvious way to tell whether Serve is fully operational
- workspace setup should feel guided, not piecemeal
- advanced commands should still exist, but be clearly secondary
- the UI and CLI should describe the same workflow
- if the UI exposes an operator action, add the CLI/backend equivalent rather than deleting the UI action by default

## Recommended Product Direction

### 1. Make `serve up` mean the normal full Serve experience

Recommended change:

- `serve up` should start everything needed for the standard operator path

For the default Serve architecture, that should mean:

- API/UI server is up
- dispatcher is up
- required workspace context is loaded

If something is missing, `serve up` should not fail vaguely. It should guide the user.

Example expectations:

- no workspace configured:
  - explain exactly how to create or select one
- workspace invalid:
  - explain what is missing
- no routes:
  - start UI if appropriate, but say processing cannot begin until a route exists
- config not deployed:
  - either deploy automatically when safe, or clearly explain why not

### 2. Keep `serve api` and `serve run` as advanced commands

Recommended framing:

- `serve up` is the normal operator command
- `serve api` is advanced/debug/development
- `serve run` is advanced/foreground dispatcher control

This matches how real users think better than exposing process boundaries first.

### 3. Add a guided setup path that spans CLI and UI cleanly

Recommended first-run model:

- CLI gets the user into Serve with a workspace
- UI finishes configuration comfortably
- CLI remains capable of doing the same actions directly

Recommended command:

- keep `serve workspace`
- do not add `serve init`
- expand `serve workspace` instead

Recommended `serve workspace` direction:

- keep workspace creation and linking in one obvious place
- make workspace output more guided
- continue printing the next recommended command after setup
- add workspace-focused helpers there in the future if needed
  - `serve workspace status`
  - `serve workspace doctor`
  - `serve workspace use`

Rationale:

- `serve workspace` already owns the workspace lifecycle
- `serve init` would overlap with it and create unnecessary CLI ambiguity
- first-time users should not have to choose between two similar setup commands

### 4. Tighten status language around "running"

Right now, the app can be:

- API running
- dispatcher stopped
- queue idle
- routes missing

and a user may still interpret that as "Serve is running."

Recommended status model:

- `Server`: API/UI process state
- `Dispatcher`: processing loop state
- `Workspace`: configured / incomplete / invalid
- `Routes`: none / present
- `Processing`: active / idle / blocked

Both CLI and UI should use the same status vocabulary.

### 5. Make full setup possible in the UI, but not UI-only in architecture

The UI should be able to complete:

- workspace selection
- route creation
- route editing
- deploy/activate path
- dispatcher start

But each of those actions must still map to CLI/backend behavior.

So the improvement target is:

- complete setup in the UI is possible
- but only because backend/CLI capabilities already exist

### 6. Reduce operator exposure to `validate` and `deploy` if possible

Today those are explicit steps.

That is defensible technically, but not ideal UX if every user has to think about them.

Recommended direction:

- keep `validate` and `deploy` in the CLI
- consider letting `serve up` or UI route changes trigger safe validation automatically
- only expose manual deploy when there is a real draft/live distinction the user must control

If test/live remains central, the UI needs to explain it better.

## Recommended CLI Improvements

## Phase 1: Clarify and simplify startup

- [x] Redefine `serve up` as the standard operator startup path
- [x] Decide whether `serve up` starts dispatcher automatically or prompts/flags clearly when it does not
- [x] Make `serve status` show API, dispatcher, workspace, route, and queue state separately
- [x] Improve top-level `serve` help text so the recommended path is obvious
- [x] Mark `serve api` and `serve run` as advanced in help text and docs

## Phase 2: Improve first-run setup

- [x] Audit whether `serve workspace` already covers everything needed for first-run setup
- [x] Decide whether to add `serve init` or expand `serve workspace`
- [x] Add `serve workspace status` or equivalent workspace inspection command
- [x] Add `serve workspace doctor` or equivalent workspace repair/diagnostic command
- [x] Add `serve workspace use` or equivalent explicit workspace-switch command if current behavior remains too implicit
- [x] Make workspace creation output more action-oriented
- [x] Print the exact next command after workspace creation
- [x] Ensure first-run users are guided toward the recommended path, not advanced subcommands

## Phase 3: Close UI setup gaps

- [x] Audit every setup step that still requires CLI-only work
- [x] Make route creation fully achievable in the UI
- [x] Make route activation/deploy fully achievable in the UI
- [x] Make dispatcher start/stop fully understandable in the UI
- [x] Make the UI explicitly show whether Serve is operational or only partially configured

## Phase 4: Enforce CLI parity explicitly

- [x] List every Serve UI action
- [x] Map each UI action to a CLI command or backend capability
- [x] Add missing CLI commands where the UI currently relies on implicit behavior
- [x] Add a clear CLI command or documented CLI sequence for every remaining UI operator action
- [x] Remove or redesign UI actions only when the action itself is invalid or should not exist
- [x] Document the recommended CLI path next to the UI-driven workflow

### Phase 4 audit notes

- Queue maintenance UI actions now have direct CLI equivalents:
  - `serve queue status`
  - `serve queue list`
  - `serve queue retry-failed`
  - `serve queue clear-processed`
  - `serve queue remove <path>`
- Task Manager workspace-local creation now has a direct CLI equivalent:
  - `task create <ClassName> [--file-name <name>]`
- Other remaining operator surfaces already had clear CLI equivalents before this pass:
  - workspace actions: `serve workspace ...`
  - route actions: `serve route ...`
  - config apply/validate: `serve validate`, `serve deploy`
  - dispatcher actions: `serve service ...`
  - mode switch: `serve mode ...`
  - tunnel/share actions: `serve share ...`
  - task install/update/remove/library refresh: `task install`, `task sync --update`, `task delete`, `task update`
- Review surfaces such as Results and Exclude remain backend-driven. They are not treated as missing operator-action parity in this plan because the product requirement here was to close the operational Serve UI gaps first.
- Invalid UI actions audit:
  - the broken route delete action was already removed in the earlier route-centric cleanup
  - no additional invalid operator actions were found in this audit, so no further UI removals were required

## Phase 5: Documentation and onboarding

- [x] Update tutorial/help text for the Serve command family
- [x] Add a "first-time setup" section to Serve help/docs
- [x] Add a "normal daily use" section to Serve help/docs
- [x] Add a "debug/advanced commands" section for `serve api`, `serve run`, and `serve worker`
- [x] Add a Markdown tutorial in `docs/` for end-to-end Serve usage
- [x] Test the tutorial flow after writing it and verify every documented command/path works as written

### Phase 5 validation notes

- Added tutorial: `docs/serve_ui_workflow.md`
- Verified directly against the CLI in a temporary Serve workspace:
  - `serve`
  - `serve workspace --mode new --path ... --skip-uv --no-test`
  - `serve workspace status --path ...`
  - `serve workspace use --path ...`
  - `serve workspace doctor --path ...`
  - `serve route upsert ...`
  - `serve route list --path ... --mode test`
  - `serve validate --path ... --mode test`
  - `task create ...`
  - `serve service status`
  - `serve queue status`
  - `serve queue list`
- The tutorial was adjusted to match what was validated:
  - `serve workspace --mode new` now explicitly requires an empty target directory
  - explicit `serve deploy` commands were removed from the tutorial path
- Local server startup commands reached the expected startup path, but full bind/listen verification was limited by sandbox restrictions on opening a local port in this environment:
  - `serve up`
  - `serve api`

## Recommended Command Model

### Recommended normal path

For most users, the intended path should become:

1. `autocleaneeg-pipeline serve workspace --mode new --path <dir>`
2. `autocleaneeg-pipeline serve up`
3. finish setup in the UI if needed

For returning users:

1. `autocleaneeg-pipeline serve up`

### Recommended advanced path

Keep these commands for advanced use:

- `serve api`
- `serve run`
- `serve worker`
- `serve validate`
- `serve deploy`
- `serve route ...`

But make their role clearer:

- powerful
- explicit
- not the default first-run experience

## Serve UI To CLI Parity Map

This is the current intended mapping for the normal Serve surfaces.

### Workspace

- choose or create workspace in UI
  - CLI equivalent: `serve workspace --mode new|existing --path ...`

### Routes

- create or edit route in UI
  - CLI equivalent: `serve route upsert ...`
- archive route in UI
  - CLI equivalent: `serve route archive <route-id>`
- restore route in UI
  - CLI equivalent: `serve route unarchive <route-id>`
- promote route to live in UI
  - CLI equivalent: `serve route promote <route-id>`
- enable or disable route in UI
  - CLI equivalent: `serve route upsert <route-id> --enabled|--disabled`
- sync route registry in UI
  - CLI equivalent: `serve route sync`

### Settings

- validate config in UI
  - CLI equivalent: `serve validate --mode <test|live>`
- apply config in UI
  - CLI equivalent: `serve deploy --mode <test|live>`

### Service

- start service in UI
  - CLI equivalent: `serve service start`
  - normal operator shortcut: `serve up`
- stop service in UI
  - CLI equivalent: `serve service stop`
- inspect dispatcher state in UI
  - CLI equivalent: `serve service status`

### Startup and status

- start normal Serve operation
  - CLI equivalent: `serve up`
- stop server/UI
  - CLI equivalent: `serve down`
- inspect operational state
  - CLI equivalent: `serve status`
- switch test/live mode from the UI
  - CLI equivalent: `serve mode test` or `serve mode live`

### Queue

- inspect queue stats in UI
  - CLI equivalent: `serve queue status`
- inspect queue entries in UI
  - CLI equivalent: `serve queue list [--status ...] [--route-id ...]`
- retry failed queue entries in UI
  - CLI equivalent: `serve queue retry-failed [<path> ...]`
- clear processed queue entries in UI
  - CLI equivalent: `serve queue clear-processed`
- remove one queue entry in UI
  - CLI equivalent: `serve queue remove <path>`

### Tasks

- create a workspace-local task in UI
  - CLI equivalent: `task create <ClassName> [--file-name <name>]`
- install a task from the registry in UI
  - CLI equivalent: `task install <task-name>`
- update an installed task in UI
  - CLI equivalent: `task sync --update`
- remove a workspace task in UI
  - CLI equivalent: `task delete <task-name>`
- refresh the shared task registry in UI
  - CLI equivalent: `task update`

### Tunnel / Sharing

- start or stop sharing from the UI
  - CLI equivalent: `serve share start` / `serve share stop`
- inspect sharing state in the UI
  - CLI equivalent: `serve share status`
- set or clear named tunnel config in the UI
  - CLI equivalent: `serve share setup` / `serve share clear`

## Open Questions

- Should `serve up` automatically start the dispatcher every time, or only when a valid deployed config exists?
- If no routes exist, should `serve up` still open the UI and report "setup incomplete" rather than fail?
- Should `serve up` automatically deploy validated operator config, or should deploy remain explicit?
- Should the UI expose draft/live mode more clearly, or should the startup experience hide more of that complexity?
- Is Redis part of the normal Serve path or only an advanced/optional path for specific features?

## Risks

- If `serve up` stays too narrow, users will continue thinking Serve is broken when only the dispatcher is missing
- If `serve up` becomes too magical, it may hide important operational distinctions
- If setup remains split between CLI and UI without a clear handoff, first-time users will continue getting stuck
- If CLI parity is enforced technically but not explained ergonomically, the product will still feel fragmented

## Validation Checklist

- [x] A first-time user can create a workspace and reach a usable UI without reading source code
- [x] A returning user can start normal Serve operation with one CLI command
- [x] Users can tell whether only the UI server is running or whether processing is actually active
- [x] Everything the UI can do is still doable through CLI/backend capabilities
- [x] The CLI help text makes the recommended path obvious
- [x] The UI setup flow does not require `Makefile` usage

## Recommended First Slice

The highest-value first pass is:

1. redefine the intended meaning of `serve up`
2. improve `serve status`
3. improve `serve workspace` or add `serve init`
4. audit the remaining setup steps that still cannot be completed cleanly from the UI

That would resolve the biggest user-experience confusion without redesigning the whole Serve architecture at once.
