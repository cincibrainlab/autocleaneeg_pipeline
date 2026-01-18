# RFC 002 Reasoning Plan: Automation Idempotency Preparation

## Intent

Document a careful analysis, plan, and discussion for enabling automation runs to reuse a single output folder without triggering automatic backups, while integrating the serve-workspace architecture, test/live runtime separation, configuration governance, and file-watching ingestion strategy (watchfiles/notify).

## Ordered Steps

1. Review current output directory backup behavior and metadata logging in `src/autoclean/utils/file_system.py`, `src/autoclean/core/pipeline.py`, and `src/autoclean/utils/path_resolution.py`.
2. Define the serve workspace model (workspace root, `runtimes/test`, `runtimes/live`, taskfile-montage-version naming, workspace registry).
3. Specify control-plane commands (`workspace`, `list`, `validate`, `deploy`) plus operator-versus-deployed YAML governance.
4. Formalize dual automation-mode inputs (CLI `--automation` flag and YAML key) with clear precedence and audit logging.
5. Design ingestion monitoring with watchfiles + Rust notify, including debounce, readiness, and quarantine rules.
6. Enumerate output/idempotency policies for artifacts, metadata, and audit logging across repeated runs.
7. Draft an APA-style analysis and first-step plan that prioritizes a minimal automation toggle with strong safety gates.



## Rationale

The instruction set is preparatory, so the output focuses on analysis and planning rather than implementation. The reasoning sequence ensures the proposal is grounded in current behavior and highlights the risks to idempotency and data provenance before any code changes are considered.
