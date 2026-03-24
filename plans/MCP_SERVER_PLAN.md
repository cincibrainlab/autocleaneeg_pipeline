# MCP Server Plan

## Intent
Implement a FastMCP server for this repository that can drive the full AutoCleanEEG CLI surface, including Serve commands, rather than a narrow subset of commands.

## Requirement Clarification
- The MCP server should be able to do whatever the CLI and Serve can do.
- The design should not assume only frequently used commands matter.
- The MCP layer is a programmatic control surface over the existing CLI, not a replacement for the CLI.
- FastMCP is the required MCP framework.

## Core Design Decision
Use the existing `autocleaneeg-pipeline` CLI as the execution backend.

Rationale:
- It already encodes the complete supported command surface.
- It avoids duplicating business logic across CLI, API, and MCP.
- It gives the MCP server access to both non-Serve and Serve functionality from one integration point.
- It preserves command parity as the CLI evolves, if the MCP layer is built around a generated or centrally declared command registry.

Execution model decision:
- Use subprocess execution as the default and required backend model for the MCP server.
- Do not use in-process callable dispatch for first implementation.
- Do not rely on `python -m autoclean` as a separate execution mode unless it is the concrete subprocess entrypoint chosen for all commands.
- The MCP layer should treat the CLI as an external command surface and normalize its results.

Implication:
- The rest of this plan assumes subprocess-backed execution, typed MCP tool wrappers, and a managed process/session layer for long-running commands.

## Non-Goals
- Re-implement CLI logic directly inside MCP tools.
- Starting with a small curated tool set and calling it done.
- Restricting the MCP server to Serve-only actions.
- Adding speculative abstractions before command coverage is mapped.

## Constraints
- MCP tools must return structured, machine-usable output where possible.
- MCP tools must expose stable, typed schemas suitable for agent use.
- Mutating commands need explicit safety rules and clear argument validation.
- Long-running commands need predictable timeout and streaming behavior.
- Commands that launch persistent processes need a session model rather than one-shot subprocess handling.
- CLI stderr/stdout parsing must be deterministic enough for agent use.
- Full command-surface parity is required, including commands that are interactive today.

## Proposed Architecture

### 1. MCP Server Layer
- Add a dedicated FastMCP server package in a top-level root directory named `autoclean_mcp/`, separate from `src/autoclean/`.
- Expose tools for command execution, discovery, and session/process control.
- Keep MCP-specific code isolated from CLI implementation details except through a documented adapter.

### 2. CLI Adapter Layer
- Build a thin adapter responsible for:
  - mapping MCP tool inputs to CLI argv
  - invoking the chosen subprocess CLI entrypoint
  - capturing exit code, stdout, stderr, duration, and structured metadata
  - normalizing failures into MCP-friendly errors
- Use one subprocess invocation strategy consistently across the server.
- Initial implementation should target one canonical entrypoint and use it everywhere.

### 3. Tool Registry
- Represent the CLI command tree as MCP tool definitions.
- Split tools into categories:
  - workspace/configuration
  - tasks/montages/events
  - processing/review/reporting
  - serve workspace/routes/settings
  - serve service/queue/mode/share
- Generate tool metadata from a registry rather than hand-writing every tool ad hoc.
- Use a hybrid registry strategy:
  - inventory and extract command/argument structure from the `argparse` tree
  - materialize that into a maintained MCP registry artifact with explicit typing, descriptions, parsing rules, and safety annotations
  - do not generate final MCP tool schemas directly from `argparse` at runtime

Registry implications:
- `argparse` is the source for command discovery and parity auditing.
- The MCP registry is the source for stable tool definitions.
- Any CLI command not yet fully typed must still be represented in the registry, even if initially marked as passthrough or compatibility-wrapped.

### 4. Long-Running Session Model
- Commands like Serve launcher/process control need more than one-shot execution.
- Add MCP resources or tools for:
  - starting a managed session
  - polling session status
  - stopping/killing a managed session
  - reading buffered stdout/stderr
- This is especially important for:
  - `serve up`
  - `serve api`
  - `serve run`
  - `serve worker`
  - possibly review-style interactive or persistent commands
- Session model decisions:
  - every managed execution gets a stable session ID
  - session metadata must include command, argv, cwd, env overrides, pid tree, start time, and current state
  - sessions do not need to survive MCP server restart in the first implementation, but stale child processes must be detected and cleaned up on startup where possible
  - `serve up` should be modeled as one managed top-level command session, with child-process metadata recorded when detectable
  - logs should be retrievable incrementally by session ID
  - stop/kill semantics must be explicit and deterministic

## Tooling Strategy

### Phase 1: Discovery And Skeleton
1. Inventory the complete CLI command tree from `src/autoclean/cli.py`.
2. Group commands by behavior:
   - read-only, one-shot
   - mutating, one-shot
   - long-running/persistent
   - interactive/TTY-sensitive
3. Identify commands that need compatibility wrappers to reach parity.
4. Convert the inventory into a maintained registry/checklist section at the bottom of this plan.

Phase 1 working checklist:
- [x] Extract every CLI leaf command from the parser tree.
- [x] Group each command as read-only, mutating, long-running, or interactive.
- [x] Record argument shape notes for each command family.
- [x] Identify commands that already produce structured output versus rich human output.
- [x] Identify commands that will require special MCP wrappers.
- [x] Identify commands that require typed parsing adapters rather than raw stdout.
- [x] Convert the inventory into the running checklist at the bottom of this plan.

Phase 1 notes to capture while working:
- command family coverage gaps
- commands with unsafe or ambiguous side effects
- commands that depend on TTY behavior
- commands that already map well to typed MCP tools

### Phase 2: Execution Backbone
1. Implement a robust subprocess runner with:
   - cwd control
   - env control
   - timeout support
   - streamed capture
   - explicit exit-code handling
2. Define a normalized result envelope:
   - `command`
   - `argv`
   - `cwd`
   - `exit_code`
   - `stdout`
   - `stderr`
   - `started_at`
   - `finished_at`
   - `duration_ms`
   - `ok`
3. Define the canonical CLI subprocess entrypoint and use it consistently.

Phase 2 working checklist:
- [x] Define the subprocess execution abstraction.
- [x] Lock the canonical subprocess entrypoint.
- [x] Define the normalized result schema.
- [x] Add timeout handling and process cleanup rules.
- [x] Add environment and working-directory control.
- [x] Decide how stdout/stderr buffering and truncation should work.
- [x] Define when raw output is allowed versus when structured parsing is mandatory.
- [x] Record implementation notes and unresolved issues in this plan.

Phase 2 notes to capture while working:
- platform-specific execution issues
- subprocess isolation tradeoffs
- output normalization decisions
- failure-mode handling decisions

### Phase 3: Broad Tool Coverage
1. Add MCP tools for the complete command surface in batches.
2. Start with read-only and low-risk commands, but continue until full command parity is reached.
3. Add explicit confirmation or guardrails for destructive/mutating actions where appropriate.
4. Represent every CLI leaf command in the MCP registry, even when the first version uses a compatibility wrapper.

Phase 3 working checklist:
- [x] Define the MCP tool registry format.
- [x] Define how parser inventory maps into the maintained MCP registry.
- [x] Implement generated or centrally declared typed tools.
- [x] Add a raw escape-hatch CLI execution tool.
- [x] Add guardrails for destructive commands.
- [x] Mark which commands are fully typed, compatibility-wrapped, or passthrough-backed.
- [x] Track command-family completion in the running checklist at the bottom.
- [x] Record tool-shape and validation notes in this plan.

Phase 3 notes to capture while working:
- tool naming conventions
- argument validation patterns
- commands that need custom parsing
- commands that remain unstructured

### Phase 4: Serve Session Support
1. Add managed-process support for persistent Serve commands.
2. Expose status/log retrieval tools.
3. Ensure the MCP layer can supervise Serve processes without hanging the MCP server itself.
4. Define deterministic lifecycle rules for session IDs, stale sessions, and stop/kill behavior.

Phase 4 working checklist:
- [x] Define the managed session model.
- [x] Define session identity and metadata fields.
- [x] Implement session start, inspect, log, and stop behavior.
- [x] Define startup recovery and stale-process cleanup behavior.
- [x] Add session cleanup and stale-process handling.
- [x] Cover `serve up`, `serve api`, `serve run`, and `serve worker`.
- [x] Record lifecycle notes and operational caveats in this plan.

Phase 4 notes to capture while working:
- session persistence rules
- process ownership and cleanup behavior
- log retrieval decisions
- long-running command edge cases

### Phase 5: Testing And Hardening
1. Add unit tests for argv generation and result normalization.
2. Add integration tests for representative commands across the CLI tree.
3. Add Serve-focused tests for long-running process lifecycle.
4. Validate behavior against real workspace fixtures.

Phase 5 working checklist:
- [x] Add unit coverage for command mapping.
- [x] Add unit coverage for result normalization.
- [x] Add integration coverage for representative CLI families.
- [x] Add Serve lifecycle coverage for managed sessions.
- [x] Record testing gaps and follow-up items in this plan.

Phase 5 notes to capture while working:
- flaky or environment-sensitive tests
- fixture requirements
- remaining uncovered command families
- hardening work deferred beyond first implementation

## Key Open Technical Questions

### A. Subprocess vs In-Process Dispatch
- Subprocess is safer for parity and isolation.
- In-process dispatch may be faster but risks hidden coupling to CLI globals and side effects.
- Decision: use subprocess execution for the MCP server.

### B. How To Model The Full Command Surface
- One tool per CLI leaf command is the clearest parity model.
- A single generic “run cli command” tool is too weak for typed MCP usage.
- Decision:
  - one generic escape-hatch tool for raw argv execution
  - plus registry-backed typed tools for each supported CLI leaf command
  - parser inventory feeds the registry, but runtime tool contracts come from the maintained MCP registry

### C. Interactive Commands
- Some commands may assume a TTY or human interaction.
- Full parity still applies to these commands.
- They must be represented in the MCP registry and assigned one of:
  - non-interactive typed wrapper
  - managed-session wrapper
  - compatibility wrapper with explicit limitations
- They must not be silently dropped from scope.
- The plan should identify these early rather than discovering them late.

### D. Output Shape
- Raw stdout alone is not enough for reliable agent consumption.
- When possible, the MCP layer should parse or annotate CLI output.
- Decision:
  - typed MCP tools must return structured payloads for the fields the tool contract promises
  - raw stdout/stderr may still be included for observability
  - passthrough or compatibility tools may return raw output plus metadata only when a stable structured contract does not yet exist
  - the registry must mark each tool as `structured`, `partially_structured`, or `raw_compatible`

## Safety Model
- Read-only tools should be callable directly.
- Mutating tools should validate arguments strictly.
- High-impact commands should include clear descriptions and, where necessary, explicit opt-in flags or confirmation fields.
- Session cleanup must be deterministic so MCP-launched Serve processes do not leak.

## Deliverables
1. A FastMCP server package/module in `autoclean_mcp/`.
2. A command registry covering the full CLI and Serve surface.
3. A subprocess/session adapter for one-shot and long-running commands.
4. Test coverage for command mapping, execution, and session lifecycle.
5. Usage documentation for connecting an MCP client to the server.

## Ordered Steps
1. Inventory every CLI leaf command and classify execution style.
2. Define the MCP tool registry format and normalized result schema.
3. Implement the subprocess adapter for one-shot commands.
4. Expose the first full batch of typed MCP tools for non-interactive commands.
5. Implement managed sessions for persistent Serve commands.
6. Expand coverage until the full CLI and Serve surface is represented.
7. Add test coverage for representative commands in every command family.
8. Document setup, limits, and operational guidance.

## Rationale
This sequence keeps the work grounded in full CLI parity from the start. It uses FastMCP as the control surface, preserves existing CLI behavior by treating the CLI as the system of record, and avoids the common failure mode of shipping a partial MCP server that only covers a convenient subset of commands.

## Running Inventory And Execution Checklist

This section should be updated as work progresses. When a phase produces an inventory, list, checklist, or implementation note, it should be recorded here rather than kept only in scratch notes.

### CLI Command Inventory
- [x] Parser tree inventoried
- [x] Leaf commands listed
- [x] Command families grouped
- [x] Read-only commands marked
- [x] Mutating commands marked
- [x] Long-running commands marked
- [x] Interactive commands marked
- [x] Special-wrapper commands marked
- [x] Structured-output candidates marked
- [x] Compatibility-wrapper commands marked
- [x] Destructive-command guardrails marked

Command inventory notes:
- Initial parser inventory extracted from `src/autoclean/cli.py`
- Current leaf-command count: `110`
- Current top-level families discovered:
  - `auth`
  - `auth0-diagnostics`
  - `blocks`
  - `clean-task`
  - `config`
  - `events`
  - `exclude`
  - `export-access-log`
  - `help`
  - `input`
  - `list-tasks`
  - `login`
  - `logout`
  - `montage`
  - `process`
  - `report`
  - `review`
  - `serve`
  - `settings`
  - `source`
  - `task`
  - `tutorial`
  - `version`
  - `view`
  - `whoami`
  - `wizard`
  - `workspace`
- Current registry summary:
  - mutating commands: `63`
  - destructive commands: `12`
  - interactive commands: `7`
  - long-running commands: `4`
  - wrapper kinds:
    - `typed_wrapper`: `93`
    - `compatibility_wrapper`: `13`
    - `managed_session`: `4`
  - output modes:
    - `partially_structured`: `97`
    - `raw_compatible`: `13`

### Command Family Checklist
- [x] workspace/configuration
- [x] tasks
- [x] montages
- [x] events
- [x] processing/review/reporting
- [x] serve workspace
- [x] serve routes
- [x] serve validate/deploy
- [x] serve service
- [x] serve queue
- [x] serve mode/share
- [x] serve launcher/process commands

Command family notes:
- High-level family grouping is now established from the parser inventory.
- Per-command execution-style classification is partially implemented in code but not yet audited across the full inventory.
- Family counts currently recorded in the maintained registry:
  - `workspace_configuration`: `19`
  - `tasks`: `18`
  - `serve_launcher_process`: `10`
  - `serve_routes`: `7`
  - `processing_review_reporting`: `7`
  - `auth`: `7`
  - `blocks`: `6`
  - `serve_mode_share`: `8`
  - `serve_queue`: `5`
  - `events`: `3`
  - `montages`: `3`
  - `serve_service`: `3`
  - `serve_workspace`: `3`
  - `serve_validate_deploy`: `2`
  - remaining single-command families are tracked in code
- Family-level argument-shape notes:
  - `workspace_configuration`: mostly simple positional path/theme arguments plus a few booleans such as `--clear`, `--confirm`, and `--spawn`
  - `tasks`: highest read-only schema complexity so far, with mixed positional names plus filters/format/source/status/category options and multiple booleans
  - `serve_routes`: highest mutating schema complexity, with route IDs, repeated folder/glob inputs, enable/recursive flags, and mode choices
  - `serve_launcher_process`: long-running command family with mode/path/host/redis settings and multiple booleans such as `--force`, `--reload`, `--dry-run`, and watcher toggles
  - `serve_queue`: read-only and mutating queue commands use straightforward scalar filters (`status`, `route_id`, `limit`, `offset`) plus path/paths for remove/retry flows
  - `serve_workspace`: setup commands mix `--path`, `--mode`, package/runtime options, and bootstrap booleans such as `--skip-uv` and `--no-test`
  - `processing_review_reporting`: heterogeneous family with file/path/task positional arguments and several tool-specific flags; review/exclude remain compatibility-wrapped
  - small singleton families like `version`, `whoami`, `login`, `logout`, and `tutorial` have trivial or no argument shape

### Structured Output Checklist
- [x] Structured tools identified
- [x] Partially structured tools identified
- [x] Raw-compatible tools identified

Structured output notes:
- Initial registry marks interactive commands as `raw_compatible`.
- Non-interactive commands remain registry-marked as `partially_structured`, but every command class now has an explicit MCP execution path.
- First typed MCP wrappers have been added for low-risk commands:
  - `auth0-diagnostics`
  - `help`
  - `list-tasks`
  - `version`
  - `whoami`
  - `source show`
  - `input show`
  - `blocks list`
  - `blocks info`
  - `blocks deps`
  - `events discover`
  - `events analyze`
  - `events epochs`
  - `montage list`
  - `montage test`
  - `workspace show`
  - `workspace size`
  - `config show`
  - `task list`
  - `task search`
  - `task show`
  - `task diagnose`
  - `task diff`
  - `task schema export`
  - `serve docs`
  - `serve list`
  - `serve status`
  - `serve mode status`
  - `serve route list`
  - `serve workspace status`
  - `serve workspace doctor`
  - `serve service status`
  - `serve queue status`
  - `serve queue list`
  - `serve validate`
  - `serve share status`
- Registry-backed argv construction now exists for typed wrappers.
- Named `cli_*` wrappers now exist for every registry command, including typed, compatibility, and managed-session commands.
- Current policy:
  - typed wrappers must provide stable argument schemas and return normalized execution envelopes
  - `run_registered_cli_command` is the generic one-shot path for registry-backed typed commands and enforces `confirm=True` for mutating operations
  - `run_compatibility_cli_command` is the generic one-shot path for compatibility-wrapper commands
  - `start_registered_cli_session` is the generic managed-session path for long-running commands
  - `start_compatibility_cli_session` is the generic managed-session path for compatibility commands that need TTY/GUI/session-style handling
- Current raw-compatible commands identified in Phase 1:
  - `wizard`
  - `review`
  - `exclude`
  - `task set`
  - `montage set`
  - `workspace explore`
  - `serve tui`
- Additional compatibility-wrapper commands identified during Phase 3 implementation:
  - `login`
  - `task explore`
  - `task edit`
  - `workspace cd`
  - `report chat`
  - `view`
- Current typed-parsing-adapter candidates identified in Phase 1:
  - `task list`
  - `task search`
  - `task diff`
  - `serve queue list`
  - `serve validate`
  - `serve route upsert`
  - `serve workspace use`

### Session And Lifecycle Checklist
- [x] Session identity model defined
- [x] Session metadata schema defined
- [x] Serve long-running commands mapped
- [x] Startup recovery behavior defined
- [x] Stop/kill behavior defined

Session notes:
- In-memory session manager implemented for initial long-running subprocess support.
- Current managed-session commands are the long-running CLI commands, including `serve up`, `serve api`, `serve run`, and `serve worker`.
- First implementation does not yet persist sessions across MCP server restart.
- Session stop behavior is deterministic: terminate first, then kill if the process does not exit promptly.
- Startup recovery policy is now explicit:
  - no session reattachment across MCP server restart
  - exited in-memory sessions are pruned during MCP server initialization
  - startup report exposes the current no-persistence policy and prune counts

### Safety And Guardrails Checklist
- [x] Mutating commands classified
- [x] Destructive commands classified
- [x] Guardrail policy defined
- [x] Confirmation strategy defined where needed

Safety notes:
- Registry now carries `mutating` and `destructive` annotations for every discovered CLI leaf command.
- Typed mutating wrappers now require explicit `confirm=True` rather than relying on implicit execution.
- First guarded mutating wrappers have been added for:
  - `serve deploy`
  - `serve route sync`
  - `serve route promote`
  - `serve route archive`
  - `serve route unarchive`
  - `serve route delete`
  - `serve service start`
  - `serve service stop`
  - `serve mode test`
  - `serve mode live`
  - `serve queue retry-failed`
  - `serve down`
  - `serve restart`
- Positional-choice pseudo-subcommands such as `serve workspace status` and `serve mode status/test/live` are now expanded into distinct registry entries.

### Phase Progress Checklist
- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Phase 4 complete
- [x] Phase 5 complete

### Implementation Notes
- Created top-level MCP package at `autoclean_mcp/`
- Added subprocess-backed CLI adapter with canonical command prefix `sys.executable -m autoclean`
- Added timeout normalization for one-shot subprocess execution
- Added bounded stdout/stderr truncation for captured command output
- Added parser inventory extraction from the real `argparse` tree
- Added maintained MCP registry derived from parser inventory with family, safety, wrapper, and output annotations
- Added registry-backed argv builder for structured command invocation
- Added initial in-memory session manager for long-running CLI subprocesses
- Added explicit startup initialization policy for session recovery/pruning
- Added initial FastMCP server surface for inventory, registry queries, raw CLI execution, registry-backed invocation, typed wrappers, and session control
- Added initial unit coverage for inventory, registry, argv builder, adapter, and session manager
- Expanded typed read-only wrapper coverage into the `blocks`, `events`, `serve list`, and `serve route list` command families
- Expanded typed read-only wrapper coverage into `help`, `list-tasks`, `source/input`, `montage`, `events epochs`, `task schema export`, `serve docs`, and `serve mode status`
- Expanded argv-builder unit coverage for repeated options and mixed boolean/optional flag combinations
- Fixed registry safety classification drift for mutating `serve mode`, `serve share`, and `serve queue retry-failed` subcommands
- Reclassified shell/editor/viewer/auth commands into compatibility-wrapped or mutating buckets where the original registry assumptions were too optimistic
- Added the first explicit confirmation guard helper for typed mutating wrappers and covered it with unit tests
- Expanded the confirmation-guarded Serve operational wrapper batch to route lifecycle and daemon control commands
- Added integration coverage for representative CLI families through the subprocess adapter (`version`, `blocks list`, `list-tasks`, `serve workspace status`)
- Added Serve-focused managed-session integration coverage using a real `serve run` lifecycle against a temporary workspace fixture
- Added registry execution-policy helpers so every command class now has an explicit supported MCP execution route
- Added explicit generic parity tools for compatibility-wrapper and managed-session registry commands
- Added named wrapper coverage for every remaining registry command and a regression test that fails if any CLI leaf loses its dedicated `cli_*` wrapper

### Open Issues And Follow-Ups
- Current testing gaps:
  - no FastMCP server-surface tests yet for tool registration or typed tool invocation contracts
  - no managed-session integration coverage yet for `serve up`, `serve api`, or `serve worker`
  - no integration coverage yet for compatibility-wrapped commands with editor/shell/viewer/auth side effects
