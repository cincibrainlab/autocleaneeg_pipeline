# MATLAB Integration Plan

## Intent
Add first-class MATLAB execution support to AutoClean so MATLAB-backed analyses can run inside the existing task, block, route, and serve automation system without introducing a parallel workflow that bypasses provenance, config validation, or route orchestration.

The initial design target is support for MATLAB Engine API for Python via the `matlabengine` package, with the near-term use case informed by the example MATLAB scripts in `temp/` such as [run_fooof_batch.m](/Users/sueo8x/Documents/Github/autoclean_pipeline/temp/run_fooof_batch.m) and the FOOOF batch instructions in [FOOOF_BATCH_INSTRUCTIONS.md](/Users/sueo8x/Documents/Github/autoclean_pipeline/temp/FOOOF_BATCH_INSTRUCTIONS.md).

## Critical Constraint
MATLAB support must remain optional.

This is not just a packaging preference. It is a hard runtime and installation constraint:
- `matlabengine` is tied to a matching installed MATLAB release.
- Installing the Python package can fail before import time if the required MATLAB release is not installed locally.
- On this machine, attempting to install `matlabengine` from PyPI failed because no matching MATLAB installation was present.
- Many AutoClean users, CI environments, route workers, and developer machines will not have MATLAB installed or licensed.

Because of that, MATLAB functionality cannot be treated like an ordinary always-on Python dependency. The plan must preserve a fully working AutoClean system when MATLAB is absent.

## Goals
- Allow Python tasks and bundled blocks to execute MATLAB code through a supported runtime layer.
- Add first-class AutoClean helper functionality that can act as a wrapper around a MATLAB function, so developers can write thin Python wrappers inside task or helper files instead of hand-rolling engine management.
- Make MATLAB-backed steps configurable through the same task config patterns already used for `apply_*`, `clean_*`, and `run_*` processing steps.
- Ensure serve routes, queue workers, and route automation can run MATLAB-backed processing without special-case operator workflows.
- Preserve provenance, logging, output paths, and failure visibility in the same way existing AutoClean blocks do.
- Keep MATLAB support optional so environments without MATLAB remain usable.

## Non-Goals
- Rewriting existing Python-native functionality into MATLAB.
- Supporting arbitrary ad hoc shell execution as a substitute for a typed MATLAB integration layer.
- Depending on interactive MATLAB desktop usage during automated route processing.
- Designing around one-off remote SSHFS container scripts as the primary runtime model.

## Design Principles
- MATLAB must be an optional execution backend, not a hard requirement for core AutoClean installs, tests, workers, or route management.
- The integration should be typed, validated, and route-safe. Raw `subprocess` calls from tasks should not become the default pattern.
- MATLAB-backed steps must behave like existing processing blocks: declarative config, deterministic output locations, structured logs, and clear enable/disable semantics.
- Installation and readiness checks must fail early with actionable guidance because `matlabengine` compatibility depends on the installed MATLAB release and may fail during package build if MATLAB is missing.
- Serve automation must treat MATLAB steps as normal task work, including status updates, worker logging, output discovery, and error reporting.

## What Optional Must Mean
- Base AutoClean install must work without MATLAB.
- Core tasks, routes, serve commands, and tests must continue to work without MATLAB.
- MATLAB dependencies must not be imported at module import time by core paths that non-MATLAB users hit.
- MATLAB-backed blocks and wrappers must fail lazily and clearly when invoked in a non-MATLAB environment.
- Route systems must surface MATLAB readiness as a preflight concern, not as an opaque crash deep into processing.
- CI must not assume MATLAB exists unless a dedicated MATLAB-capable job is configured.
- MATLAB-backed routes are only supported when the CLI process, worker process, and MATLAB engine package all run from the same MATLAB-capable Python environment.

## Supported Execution Modes

### Supported
- Base CLI and route usage without MATLAB in any normal AutoClean install.
- MATLAB-enabled task and route execution from a project-managed virtual environment where:
  - AutoClean is installed
  - `matlabengine` is installed
  - the local MATLAB release matches the installed engine package
  - the MATLAB license is valid for actual startup

### Not Supported for v1
- Assuming `uv tool install autocleaneeg-pipeline` alone is sufficient for MATLAB-backed routes.
- Mixing a non-MATLAB CLI install with a different interpreter or environment that separately has `matlabengine`.
- Treating route workers as MATLAB-capable unless they were launched from the same validated MATLAB-capable environment as the CLI.
- Any install path that requires the default package install to pull in MATLAB automatically.

## Recommended Architecture

### 1. Introduce a MATLAB Runtime Service Layer
Create a dedicated module such as `src/autoclean/utils/matlab_runtime.py` or `src/autoclean/tools/matlab_runtime.py` that owns:
- MATLAB Engine import and startup.
- Engine lifecycle management.
- Environment validation.
- Conversion helpers between Python values and MATLAB values.
- Standardized execution result objects.
- Structured logging around MATLAB calls.

This layer should expose a narrow API such as:
- `detect_matlab_engine()`
- `validate_matlab_environment()`
- `start_matlab_engine()`
- `run_matlab_function(...)`
- `run_matlab_script(...)`
- `shutdown_matlab_engine(...)`
- `call_matlab_function(...)`

It should also centralize path setup so tasks and blocks can register script folders without each implementation manually calling `addpath`.

### 2. Treat MATLAB as a Block/Task Capability, Not a Separate Pipeline
MATLAB execution should plug into the current task lifecycle rather than bypass it. The cleanest fit is:
- New mixin support for MATLAB-backed processing steps.
- Optional bundled analysis blocks whose mixins delegate to the MATLAB runtime service.
- A lightweight Python-side wrapper capability for calling MATLAB functions from normal AutoClean task code.
- Config schema descriptors for MATLAB-backed `apply_*` or `run_*` step definitions.

This keeps the system aligned with existing block patterns under `src/autoclean/blocks/` and task composition via `Task` in [task.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/core/task.py).

### 2a. Support Two MATLAB Integration Modes
The implementation should explicitly support both of these modes:

1. Pipeline execution of MATLAB files or function entrypoints
- AutoClean can execute MATLAB `.m` files or MATLAB functions as part of normal task, block, route, and serve processing.
- This is the production integration path for bundled functionality and automated routes.

2. Loose Python wrappers over MATLAB functions
- AutoClean exposes helper functionality so a task author can write a thin Python method that calls a MATLAB function through a shared runtime layer.
- This avoids copy-pasted engine startup code across tasks and helper files.

Both modes should share the same runtime service, error handling, logging, and path management so they remain one coherent system.

### 3. Use a Typed Config Contract for MATLAB Steps
Add a reusable config shape for MATLAB-backed steps, for example:

```python
{
    "enabled": True,
    "engine": {
        "startup_mode": "shared|new",
        "startup_timeout_sec": 120,
        "shutdown_on_complete": True,
    },
    "script": {
        "kind": "function|script",
        "name": "eeg_htpCalcFooof",
        "paths": ["temp/scripts", "external/vhtp"],
    },
    "inputs": {...},
    "outputs": {
        "capture_stdout": True,
        "artifacts_subdir": "matlab/fooof",
    },
}
```

The exact schema can vary, but the plan should preserve:
- `enabled` semantics consistent with existing task steps.
- Explicit distinction between script execution and function invocation.
- Declarative MATLAB search paths.
- Declarative mapping of inputs and expected outputs.
- Route-safe output directory configuration.

### 4. Default to Engine API, but Design for a Future Fallback
Primary implementation path:
- `import matlab.engine`
- `eng = matlab.engine.start_matlab()`
- Execute functions or scripts from Python.

Design extension point:
- A future adapter could run `matlab -batch` when the engine is not suitable.

The first implementation should not start by supporting both if that complicates the design. It should, however, avoid hard-coding assumptions that make a future `-batch` adapter impossible.

### 5. Execution Isolation and Worker Safety
MATLAB execution must not be treated as an ordinary in-process helper call without lifecycle controls.

The plan should explicitly define:
- where MATLAB engine startup happens
- how long startup is allowed to take
- how execution timeouts are enforced
- how cancellations are handled
- what happens when MATLAB hangs or licensing blocks startup
- how a worker recovers from a failed MATLAB run

Recommendation for v1:
- Use a dedicated execution adapter with explicit timeout and teardown behavior.
- Keep lifecycle ownership at the task-run level, not global process singleton scope.
- If in-process engine reuse is used, wrap it with hard failure handling so a broken MATLAB session cannot silently poison subsequent jobs.
- Treat worker recovery and failure isolation as first-class requirements for serve usage.

## Dependency and Installation Strategy

### Python Packaging
Do not make `matlabengine` a mandatory core dependency in `pyproject.toml`. MATLAB compatibility is release-specific, and many users or CI agents will not have MATLAB installed.

This needs to be treated as a strict requirement, not a recommendation:
- `matlabengine` must not be added to the default dependency set.
- `matlabengine` must not be imported unconditionally from core modules.
- MATLAB-specific code paths must be isolated behind optional helpers, blocks, extras, and runtime checks.

Recommended approach:
- Document supported installation modes:
  - `python -m pip install matlabengine`
  - install from `matlabroot/extern/engines/python`
  - install inside the active `.venv`
- Add a CLI doctor/check command to verify runtime readiness after install.

### Install Strategy for `pip` and `uv`
The install model must be explicit because the normal package-manager path will fail if MATLAB is not already installed and compatible.

Required behavior:
- `pip install autocleaneeg-pipeline` must succeed without MATLAB.
- `uv tool install autocleaneeg-pipeline` must succeed without MATLAB.
- Neither default install path may attempt to install `matlabengine`.
- MATLAB enablement must happen as a separate, documented post-install step on a machine that already has a compatible MATLAB installation.

Recommended user flows:

Base install without MATLAB:

```bash
pip install autocleaneeg-pipeline
```

or

```bash
uv tool install autocleaneeg-pipeline
```

MATLAB enablement on a compatible machine:

```bash
python -m pip install matlabengine
```

or the MATLAB-provided install flow from `matlabroot/extern/engines/python`.

Planning implications:
- AutoClean must not assume that an optional extra such as `[matlab]` is a safe general-purpose install path.
- For v1, do not expose a `[matlab]` extra as an advertised install path.
- The recommended path for MATLAB-enabled development may need to be a dedicated project `.venv`, not the generic `uv tool install` flow.
- The docs must clearly distinguish:
  - base AutoClean install
  - MATLAB enablement
  - MATLAB readiness validation

### `uv tool` Versus Project `.venv`
The plan should assume that MATLAB support may work better in a project-managed virtual environment than in a generic `uv tool` environment.

That means:
- `uv tool install autocleaneeg-pipeline` remains the recommended path for core non-MATLAB CLI usage.
- MATLAB-enabled workflows may require:
  - a local `.venv`
  - AutoClean installed into that environment
  - `matlabengine` installed into that same environment after MATLAB is confirmed present
- The docs and CLI diagnostics must explain this distinction plainly.
- Serve and worker documentation must explicitly state that MATLAB-backed routes are only supported from the MATLAB-capable `.venv` path in v1.

### Environment Validation
The runtime validator should check:
- Python version compatibility with the installed MATLAB release.
- 64-bit Python.
- Importability of `matlab.engine`.
- Whether `matlabengine` installation failed because no matching MATLAB release is installed.
- Ability to start and stop an engine in the current environment.
- Optional presence of required MATLAB toolboxes for a requested block.
- Optional existence of required `.m` files or MATLAB package folders.

### Best-Practice Recommendation
For this repo, MATLAB support should be installed into the same virtual environment that runs AutoClean so the route worker and CLI process see the same `matlabengine` package. That aligns with the user instruction to install via PyPI using:

```bash
python -m pip install matlabengine
```

However, the plan must explicitly assume that this command can fail on machines without a matching local MATLAB install. Documentation and runtime checks need to say that plainly.

## Proposed Code Areas
- `src/autoclean/utils/matlab_runtime.py`
- `src/autoclean/mixins/analysis/` or `src/autoclean/mixins/utils/` for a `MatlabExecutionMixin`
- `src/autoclean/functions/` for user-facing MATLAB wrapper helpers that look like normal AutoClean functionality
- `src/autoclean/blocks/analysis/...` for one or more MATLAB-backed bundled blocks
- `src/autoclean/configkit/schema.py` for MATLAB step descriptors
- `src/autoclean/cli.py` for install/doctor/test commands
- `src/autoclean/api/routes/worker.py` and related serve worker execution paths for environment and recovery handling
- `src/autoclean/utils/serve_routes.py` only if route specs need explicit MATLAB environment hints
- `tests/unit/` and `tests/integration/` for mocked and gated MATLAB coverage
- `docs/` for installation, configuration, and troubleshooting

## Phased Implementation Plan

## Phase 1 Decisions
- v1 support target:
  - Primary tested target is MATLAB R2025b with a matching `matlabengine` release in the same Python environment.
  - The runtime will validate compatibility dynamically rather than pretending every MATLAB/Python combination works.
- v1 deployment model:
  - Local MATLAB only.
  - Remote/container MATLAB execution is explicitly deferred until the local model is stable.
- v1 product scope:
  - Build a general MATLAB runtime layer first.
  - Ship one concrete bundled proof-of-value block on top of it, with the current likely candidate being a MATLAB-backed FOOOF analysis block.
- v1 environment rule:
  - MATLAB-backed routes are supported only from a MATLAB-capable project `.venv`.
  - Generic `uv tool install` remains supported for non-MATLAB usage only.
- v1 MATLAB asset location:
  - Bundled production `.m` assets will live inside this repo in a dedicated MATLAB asset folder.
  - The runtime may accept additional external search paths for research code, but external directories are not the primary production model.
- v1 execution contract:
  - Large EEG data and large result payloads move through files, not direct in-memory engine transfer.
  - Control parameters and small scalar values may be passed through the engine API directly.
  - Production bundled blocks must prefer function entrypoints over monolithic scripts.
- v1 worker contract:
  - MATLAB startup must pass preflight before route execution begins.
  - The worker and CLI must use the same validated interpreter and environment.
  - MATLAB runs must have explicit timeout, teardown, and failure propagation semantics.
  - A failed MATLAB session must not be silently reused by the next job.

### Phase 1. Requirements and Runtime Contract
- [x] Record as a non-negotiable requirement that MATLAB support is optional and cannot be part of the default install path.
- [x] Record as a non-negotiable requirement that both `pip install autocleaneeg-pipeline` and `uv tool install autocleaneeg-pipeline` must remain successful on machines without MATLAB.
- [x] Confirm the supported MATLAB release range for AutoClean.
  - v1 primary target is MATLAB R2025b with matching engine package support in the same Python environment.
  - Broader release coverage is handled through runtime compatibility validation, not broad install promises.
- [x] Confirm whether the required deployment model is local MATLAB only, remote MATLAB only, or both.
  - v1 is local MATLAB only.
- [x] Confirm whether the first supported use case is specifically FOOOF or a broader generic MATLAB execution framework.
  - v1 ships a general runtime layer plus one concrete bundled FOOOF-oriented proof-of-value block.
- [x] Record the v1 support rule that MATLAB-backed routes are only supported from a MATLAB-capable project `.venv`, not generic `uv tool` installs.
- [x] Inventory required toolboxes and third-party MATLAB dependencies for the first use case.
  - Minimum baseline dependencies:
    - MATLAB itself
    - matching `matlabengine`
    - a valid MATLAB license that permits actual startup
  - First-use-case dependencies to be enforced at block level:
    - required FOOOF-related MATLAB code
    - any toolbox requirements declared by the bundled block
- [x] Decide whether bundled `.m` files will live inside this repo, a companion repo, or an operator-managed external directory.
  - Bundled production assets live in this repo.
  - External paths remain optional extension points only.
- [x] Define the minimum runtime contract for a MATLAB-backed block:
  - input data type:
    - file-based EEG inputs and structured scalar config inputs
  - expected output files:
    - deterministic artifact files written to task/derivatives locations
  - output metadata to track:
    - MATLAB entrypoint name
    - MATLAB runtime/version when available
    - engine/package detection result
    - output file paths
    - execution status and timing
  - failure behavior:
    - fail clearly with actionable runtime, compatibility, or licensing errors
- [x] Define the minimum operational contract for serve workers:
  - preflight requirements:
    - validated interpreter
    - importable engine package
    - successful MATLAB startup check
    - required scripts/toolboxes present
  - interpreter/environment ownership:
    - CLI and worker must use the same MATLAB-capable environment
  - timeout behavior:
    - startup timeout and execution timeout are explicit
  - cancellation behavior:
    - task-level execution must support teardown and non-reuse of broken sessions
  - recovery after MATLAB startup or execution failure:
    - worker records the failure clearly and does not silently reuse a poisoned MATLAB session

Deliverable:
- Approved runtime contract and scope statement for v1, including explicit non-MATLAB behavior.

### Phase 2. MATLAB Runtime Foundation
- [x] Implement `matlab_runtime` service with:
  - engine import detection
  - startup/shutdown
  - path registration
  - function/script invocation
  - shared wrapper helpers for Python-callable MATLAB functions
  - exception normalization
  - structured result model
  - Notes:
    - Implemented in `src/autoclean/utils/matlab_runtime.py`
    - Runtime remains import-safe when MATLAB is unavailable
- [x] Add a typed error hierarchy for:
  - engine not installed
  - matlabengine package unavailable
  - matlabengine package install/build failed because MATLAB is missing
  - incompatible environment
  - startup timeout
  - missing script/function
  - MATLAB execution failure
- [x] Add logging hooks compatible with AutoClean’s existing logging style.
- [x] Ensure engine lifecycle is safe for repeated task execution in serve mode.
  - Notes:
    - Serve queue processing already executes each file in a fresh `autocleaneeg-pipeline process ...` subprocess.
    - `api/tasks.py` now runs MATLAB preflight in that same target runtime before launching a MATLAB-backed Python taskfile.
    - MATLAB-backed route failures now stop before full task execution, and the failed preflight process is not reused for the next queued job.
- [x] Decide whether engine reuse is per process, per task, or per step.
  - v1 implementation direction remains per-task lifecycle ownership with explicit teardown.
- [x] Ensure core imports remain safe when `matlabengine` is not installed.
- [x] Implement explicit timeout and teardown handling suitable for worker execution.
  - Notes:
    - Startup now uses async engine launch with an explicit timeout path.
    - Newly created engines are always torn down after helper execution.
- [x] Define how a failed or wedged MATLAB session is isolated from subsequent jobs.
  - Notes:
    - v1 runtime does not keep a global shared engine.
    - Helper APIs reject `keep_engine=True` when the caller does not supply its own engine, preventing leaked hidden sessions.
    - Per-call teardown remains the current isolation boundary until worker-specific integration lands.

Deliverable:
- Reusable MATLAB runtime module with unit-test coverage via mocks.

### Phase 3. Config Schema and Task Integration
- [x] Add schema descriptors for MATLAB-backed step configuration in [schema.py](/Users/sueo8x/Documents/Github/autoclean_pipeline/src/autoclean/configkit/schema.py).
  - Notes:
    - Added validated `apply_matlab` and `run_matlab` step schemas with standard MATLAB config keys.
- [x] Implement a `MatlabExecutionMixin` that tasks can call directly.
  - Notes:
    - Implemented in `src/autoclean/mixins/utils/matlab.py`.
    - Added `execute_matlab_step(config_key, ...)` so tasks can run one validated MATLAB config block directly from `self.settings`.
- [x] Add first-class helper functions for thin wrappers, for example:
  - `call_matlab_function(...)`
  - `run_matlab_file(...)`
  - optional convenience wrapper utilities if they improve ergonomics without hiding execution behavior
  - Notes:
    - Implemented in `src/autoclean/functions/matlab.py` and re-exported through `src/autoclean/functions/__init__.py`.
    - Added `execute_matlab_config(...)` as the config-driven thin wrapper over the shared runtime.
- [x] Define standard config keys for:
  - script/function name
  - MATLAB search paths
  - input argument mapping
  - output artifact paths
  - timeout and startup behavior
  - toolbox requirements
  - Notes:
    - Standardized keys now include `kind`, `entrypoint`, `args`, `paths`, `startup_options`, `startup_timeout_seconds`, `license_file`, `nargout`, `toolbox_requirements`, and `outputs`.
- [x] Ensure task config validation fails early when malformed MATLAB config is provided.
  - Notes:
    - Added focused schema tests for valid configs and invalid timeout values.
- [x] Decide whether MATLAB blocks should be discoverable using the existing bundled block registry model.
  - Notes:
    - v1 task integration is config-key based (`apply_matlab` / `run_matlab`).
    - Bundled block registry work remains in Phase 4, not as a prerequisite for MATLAB task configs.
- [x] Ensure a Python task file can use the wrapper API while still inheriting AutoClean logging, provenance, and artifact conventions.
  - Notes:
    - Tasks can now call `execute_matlab_step(...)`, `call_matlab_function(...)`, or `run_matlab_file(...)` through `MatlabExecutionMixin`.
- [x] Ensure non-MATLAB tasks and non-MATLAB routes do not pay an import-time or startup-time penalty from the MATLAB integration.
- [x] Keep loose Python wrappers as a thin API over the shared runtime, not as a second orchestration path with separate lifecycle behavior.

Deliverable:
- MATLAB-backed step config plus Python wrapper helpers that both feel native to existing AutoClean tasks.

### Phase 4. First Bundled MATLAB Block
- [x] Implement a concrete proof-of-value block, likely a MATLAB-backed FOOOF analysis block.
- [x] Refactor the example MATLAB workflow from `temp/` into a production-ready callable unit:
  - prefer function-oriented `.m` entrypoints over monolithic scripts
  - parameterize input/output paths
  - avoid hard-coded external machine paths
  - return machine-readable status
- [x] Add block manifest and README following existing block conventions.
- [x] Define how the block writes outputs into task derivatives or route artifact folders.
- [x] Ensure the block records enough metadata to support downstream route review.
- [x] Verify the same underlying MATLAB functionality can be reached both from:
  - a production pipeline block
  - a thin Python wrapper inside a task/helper file
  - Notes:
    - Added bundled block under `src/autoclean/blocks/analysis/matlab_fooof/`.
    - Added production MATLAB entrypoint `autoclean_eeglab_fooof.m`.
    - Added task-usable proxy mixin in `src/autoclean/mixins/analysis/matlab_fooof.py`.
    - The block writes subject-scoped artifacts under `derivatives/matlab/fooof/{subject}/`.

Deliverable:
- One end-to-end MATLAB-backed bundled block that runs inside the current task system.

### Phase 4a. v1 Scope Guardrails
- [x] Keep v1 production scope to:
  - MATLAB runtime adapter
  - doctor/preflight checks
  - one bundled route-safe MATLAB block
  - one thin wrapper API over the shared runtime
- [x] Do not expand v1 into multiple unrelated bundled MATLAB blocks before the first route-safe path is stable.
- [x] Treat broader loose-wrapper ergonomics and additional MATLAB block families as post-v1 expansion unless they are required to land the first production use case.

Deliverable:
- A constrained v1 scope that can be implemented and validated without architectural sprawl.

### Phase 5. Serve and Route Automation Integration
- [x] Verify MATLAB-backed tasks work under the existing worker execution model used by serve.
- [x] Ensure route-triggered runs expose MATLAB startup and execution failures in queue/service logs.
- [x] Confirm route outputs are written into predictable route-specific folders.
- [x] Decide whether route specs need explicit MATLAB requirements, for example:
  - route labels indicating MATLAB dependency
  - route validation warnings when MATLAB is unavailable
  - worker admission checks before dispatch
  - Notes:
    - Serve config parsing now flags Python taskfile routes with `requires_matlab=True`.
    - Non-strict parsing emits explicit warnings for MATLAB-backed route taskfiles.
- [x] Add preflight checks so a route can fail fast before spending time on upstream preprocessing if the MATLAB runtime is unavailable.
- [x] Validate coexistence with automation modes and multi-route execution.
- [x] Ensure routes that do not use MATLAB remain unaffected when MATLAB is absent.
- [x] Ensure serve workers reject or clearly warn on MATLAB-backed work when they are not running from the validated MATLAB-capable interpreter.
- [x] Ensure a failed MATLAB run cannot leave the worker in an unsafe reused state for the next job.
  - Notes:
    - `api/tasks.process_file` runs MATLAB doctor preflight in the target runtime before launching MATLAB-backed Python taskfiles.
    - Queue task results already retain stdout, stderr, and return codes, so MATLAB runtime failures surface in serve logs/results.

Deliverable:
- MATLAB-backed route execution that behaves like any other route from an operator perspective.

### Phase 6. CLI, Operator UX, and Diagnostics
- [x] Add a CLI command such as:
  - `autocleaneeg-pipeline matlab check`
  - `autocleaneeg-pipeline matlab doctor`
  - `autocleaneeg-pipeline matlab test-engine`
  - Notes:
    - `matlab doctor` and `matlab test-engine` are implemented in `src/autoclean/cli.py`.
- [x] Report:
  - detected Python version
  - detected MATLAB engine package version
  - engine startup success/failure
  - MATLAB root/version if discoverable
  - required toolbox availability for a chosen block
  - Notes:
    - Current implementation reports interpreter, engine package, MATLAB root/binary, startup status, and route-environment support.
    - Toolbox-specific reporting remains block-specific future work.
- [ ] Add actionable remediation guidance for common misconfigurations.
- [ ] Add explicit guidance on whether the current install is:
  - a base install with no MATLAB support
  - a MATLAB-capable `.venv`
  - a `uv tool` install that still needs separate MATLAB enablement
- [ ] Add explicit output telling the operator whether the current interpreter is a supported environment for MATLAB-backed routes.
- [ ] Optionally add UI surfacing later in the web/service screens, but do not block backend integration on web UI work.

Deliverable:
- A simple operator path to verify MATLAB readiness before running routes.

### Phase 7. Testing Strategy
- [x] Add unit tests for runtime adapter behavior with mocked `matlab.engine`.
  - Notes:
    - Runtime unit coverage exists in `tests/unit/utils/test_matlab_runtime.py`.
    - Worker/process preflight coverage now exists in `tests/test_serve_cli.py`.
- [x] Add schema validation tests for valid and invalid MATLAB step configs.
  - Notes:
    - Added coverage in `tests/config/test_matlab_schema.py`.
- [ ] Add block-level unit tests for artifact path generation and error propagation.
- [ ] Add an integration test tier that is gated or skipped unless MATLAB is available.
- [ ] Add serve/worker integration tests covering:
  - runtime missing
  - matlabengine not installed
  - matlabengine install/build impossible because MATLAB is not present
  - startup failure
  - successful route execution
  - output artifact discovery
- [ ] Avoid making CI hard-fail on environments without MATLAB unless a dedicated MATLAB-capable job exists.

Deliverable:
- Reliable local coverage without making the base CI matrix unusable.

### Phase 8. Documentation and Rollout
- [ ] Add install guide for `matlabengine` in the project docs.
- [ ] Document compatibility caveats between Python version, MATLAB release, and architecture.
- [ ] Add task config examples and route examples for MATLAB-backed workflows.
- [ ] Document output expectations, provenance behavior, and troubleshooting.
- [ ] Add migration notes if any existing external MATLAB workflow is being replaced.

Deliverable:
- Operator-ready docs for installing and using MATLAB-backed routes.

## Technical Decisions to Make Early

### Engine Reuse Policy
Options:
- Per-step engine startup: simplest, slower, lower state leakage risk.
- Per-task engine reuse: better performance, moderate complexity.
- Per-worker shared engine: highest performance, highest state/isolation risk.

Recommendation:
- Start with per-task engine reuse and explicit teardown.
- Do not use a long-lived per-worker shared engine in v1.

### MATLAB Asset Location
Options:
- Keep `.m` files in this repo under a dedicated folder such as `src/autoclean/matlab/`.
- Store them in a companion repo and reference a configured path.
- Support both, but define one as the primary supported path.

Recommendation:
- Use a repo-local folder for bundled production assets and allow additional search paths for external research code.

### Script Versus Function Entry Points
Options:
- Support raw script execution.
- Require function-based entrypoints for production blocks.

Recommendation:
- Support both in the runtime layer, but require function-based entrypoints for bundled blocks. Scripts are harder to validate, harder to test, and more fragile in automation.
- The pipeline should still retain the ability to execute MATLAB `.m` files directly when needed, even if wrapper-based integrations prefer function entrypoints.

### Environment Ownership
Options:
- Allow mixed interpreter setups and try to discover MATLAB support dynamically.
- Require one explicit MATLAB-capable Python environment for both CLI and worker execution.

Recommendation:
- Require one explicit MATLAB-capable Python environment for all MATLAB-backed route execution in v1.
- Do not support mixed-environment routing semantics in v1.

### Data Exchange Format
Options:
- Pass primitive values only and exchange data through files.
- Convert arrays/tables directly through engine objects.
- Hybrid approach.

Recommendation:
- Use file-based exchange for large EEG artifacts and scalar/dict-like arguments for control parameters. That is the most route-friendly and reproducible pattern.

## Risks and Mitigations

### Risk: MATLAB/Python version mismatch
- Mitigation: CLI doctor command and explicit startup validation before route execution.

### Risk: Engine availability differs across operators or workers
- Mitigation: keep MATLAB support optional and expose route preflight errors clearly.

### Risk: Installing `matlabengine` may fail outright on machines without MATLAB
- Mitigation: document this explicitly, keep it out of the default dependency set, and build doctor/preflight checks around this expected failure mode.

### Risk: Users assume `pip install` or `uv tool install` should automatically provide MATLAB support
- Mitigation: document a two-step installation model and make CLI diagnostics explain whether MATLAB has been enabled in the current environment.

### Risk: MATLAB-backed routes are launched from a different interpreter than the one that has MATLAB configured
- Mitigation: define a strict supported environment model and have doctor/preflight checks validate interpreter ownership before route execution.

### Risk: Long-lived MATLAB state leaks between jobs
- Mitigation: explicit engine lifecycle ownership and per-task isolation by default.

### Risk: A wedged MATLAB session poisons a worker process
- Mitigation: require explicit timeout, teardown, and worker recovery behavior in the runtime and serve integration design.

### Risk: Hard-coded research paths from the current example scripts
- Mitigation: refactor `.m` assets into parameterized functions before production use.

### Risk: CI cannot validate MATLAB features
- Mitigation: mock-first unit tests plus optional integration tests gated by environment availability.

### Risk: Route automation becomes brittle if MATLAB is treated as an unmanaged external sidecar
- Mitigation: make MATLAB execution a typed runtime service inside AutoClean rather than a shell-script convention.

## Acceptance Criteria for v1
- Base AutoClean installation and non-MATLAB workflows work unchanged on machines without MATLAB.
- `pip install autocleaneeg-pipeline` works without MATLAB.
- `uv tool install autocleaneeg-pipeline` works without MATLAB.
- A developer can install MATLAB support into the active AutoClean environment.
- `autocleaneeg-pipeline matlab doctor` or equivalent reports whether the environment is ready.
- `autocleaneeg-pipeline matlab doctor` or equivalent reports whether the current interpreter is a supported environment for MATLAB-backed routes.
- A bundled task or block can call MATLAB via Python engine and produce deterministic artifacts.
- A task author can write a thin Python wrapper around a MATLAB function using shared AutoClean functionality rather than custom engine setup code.
- The pipeline can still execute MATLAB files directly when required.
- The block integrates with existing task config validation.
- The block can run through the current serve/route system without bespoke operator steps.
- MATLAB-backed routes are explicitly supported only from the validated MATLAB-capable environment model defined above.
- Failures appear in normal AutoClean logs and are actionable.
- The feature is documented and optional.

## Open Questions
- Should the thin Python wrapper API be implemented in `src/autoclean/functions/`, a mixin utility module, or both?
- Do you want MATLAB support to be available only for analysis blocks, or also for preprocessing/epoching style steps?
- Should bundled MATLAB code live inside this repo, or should AutoClean point to an external research-code directory/repo?
- Do you want route creation or route sync to warn when a route references a MATLAB-backed task but the local runtime is unavailable?

## Recommended Next Step
Approve the open questions above, then implement Phase 1 and Phase 2 first before touching route UI or converting the example MATLAB scripts into a production block.
