# RFC 005 Reasoning Plan: Serve Workspace Command

## Intent

Implement the initial `autocleaneeg-pipeline serve workspace` command to create or link a serve workspace, generate `serve-test.yaml`/`serve-live.yaml`, set up runtime folders, and persist the workspace path in the existing setup JSON.

## Ordered Steps

1. Add serve workspace subcommand and minimal inputs (path, mode, package spec).
2. Create the serve workspace structure per RFC 002 (runtimes/test, runtimes/live, automations, YAMLs).
3. Use uv to create runtime environments and install the package spec.
4. Persist the workspace path in setup.json and validate existing workspaces.
5. Run a basic `serve workspace` test using the provided test workspace path.

## Rationale

This focuses on the smallest viable command that aligns with RFC 002 and enables reproducible setup without expanding beyond the initial workspace lifecycle requirements.
