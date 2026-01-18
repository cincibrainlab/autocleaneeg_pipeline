# RFC 002 Reasoning Plan: Automation Idempotency Preparation

## Intent

Document a careful analysis, plan, and discussion for enabling automation runs to reuse a single output folder without triggering automatic backups, while integrating the new serve-workspace architecture with test/live runtimes and configuration governance.

## Ordered Steps

1. Review current output directory backup behavior in `src/autoclean/utils/file_system.py` and its orchestration in `src/autoclean/core/pipeline.py`.
2. Map the proposed `autocleaneeg-pipeline serve` workspace layout, including `runtimes/test`, `runtimes/live`, and named task workspaces.
3. Define the serve command family (`workspace`, `list`, `validate`, `deploy`) and its expected governance responsibilities.
4. Specify YAML governance, including operator-edited configs, hidden deployed configs, and validation gates for uptime.
5. Enumerate downstream artifacts and metadata that depend on backup behavior, noting overwrite risk under automation.
6. Draft an APA-style analysis and a first-step plan for a minimal toggle that preserves auditability, uptime, and configuration safety.



## Rationale

The instruction set is preparatory, so the output focuses on analysis and planning rather than implementation. The reasoning sequence ensures the proposal is grounded in current behavior and highlights the risks to idempotency and data provenance before any code changes are considered.
