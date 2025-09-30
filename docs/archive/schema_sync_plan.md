# AutoClean Task Schema Sync Strategy (Draft)

This document captures a staged plan for keeping the task registry aligned with
the pipeline’s canonical schema definition.

## Goals

- Single source of truth for task config validation (no duplicated schema
  definitions).
- Versioned reference that external tooling and task authors can consume.
- Automation/checks to prevent drift between the pipeline and the registry.

## Stage 1 – Authoritative Export

1. **Export utility**
   - Add a CLI subcommand under `task` (e.g., `autocleaneeg-pipeline task schema export`) that renders
     the Python schema to JSON (or JSON Schema).
   - Emit `schema_version` and the entire tasks structure layout.
2. **Surface in releases**
   - Bundle the exported artifact with the PyPI package and attach it to GitHub
     releases (e.g., `schema-2025.09.json`).
3. **Documentation**
   - Update pipeline README/docs to point to the export command and describe
     where the schema artifact lives.

## Stage 2 – Registry Consumption

1. **Generator integration**
   - The registry already ships a TypeScript generator
     (`src/utils/pythonGenerator.ts`) that renders the task template for web
     previews/downloads. Update it to ingest the exported schema JSON so
     `schema_version`, required sections, and field defaults are sourced from
     the pipeline rather than hard-coded locally.
   - The schema export remains descriptive: it enumerates fields, intent, and
     types but does not define execution flow. The generator therefore keeps
     using the existing template scaffold for the `Task` subclass and run
     sequence; the schema simply validates and hydrates the `config` block.
   - Fail generation if the schema artifact is missing or the version diverges
     from the locally cached value.
2. **Task validation**
  - Introduce a registry-side validation step (e.g., `python -m jsonschema` or
    invoking the pipeline’s validator) to ensure all checked-in tasks conform to
    the fetched schema prior to publishing.
3. **CI check**
   - In the registry repo, add a CI job that downloads the latest export (or the
     one pinned in the generator) and runs validation across every task file.

## Stage 3 – Continuous Sync

1. **Version bump workflow**
   - Document a release checklist: bump `SCHEMA_VERSION`, run the export, update
     registry generator/tests, link to new artifact.
   - Consider a helper script to propagate `schema_version` into template/seed
     configs automatically.
2. **Notification**
   - Publish release notes announcing schema updates and linking to migration
     guidance for custom tasks.
3. **Optional automation**
   - Provide a public URL or package that hosts the latest schema (e.g., GitHub
     pages or CDN) so downstream projects can fetch it directly.
   - Offer an SDK helper (`autoclean.schema.load()`) for Python clients to
     consume the schema programmatically.

## Stage 4 – Tooling Enhancements (Future)

- **Config validator CLI**: ship a command (`autoclean schema validate config.yaml`)
  that checks arbitrary configs against the canonical schema.
- **Change log generator**: produce a diff between schema versions to highlight
  breaking vs. additive changes for external consumers.
- **Pre-commit hook**: optional hook in the registry (and external projects) to
  validate configs automatically during development.

## Open Questions

- How do we version and store schema artifacts for older releases? (e.g., keep
  per-version files under `docs/schema/`.)
- Does the registry need offline/offline-safe copies of the schema, or can it
  fetch them on demand?
- Should we support multiple schema versions simultaneously (e.g., allow the
  registry to emit tasks pinned to older releases)?
- **Template placement**: task templates currently live in
  `autoclean/templates/` alongside the CLI scaffolding code. Even with schema
  exports, full task files still need that boilerplate (imports, `run()` flow,
  docstrings). Moving templates into `configkit` would mix runtime validation
  assets with CLI scaffolding and complicate existing imports/tests (e.g.,
  `tests/unit/templates/test_custom_task_template.py`). Keeping templates in the
  dedicated `templates/` package maintains a clear separation: configkit owns
  validation logic, while the CLI owns user-facing scaffolding.

## Integration Snapshot (Pipeline ↔ Registry)

- **Pipeline repo**
  - Hosts canonical schema (`configkit`), task templates (`autoclean/templates`),
    built-in tasks, and CLI tooling.
  - Exports/bundles schema JSON artifacts per release.
- **Task registry repo** (`/Volumes/braindata/cbl_github/autocleaneeg-task-registry`)
  - Contains published tasks in `tasks/<category>/…`, an index (`registry.json`),
    and TypeScript tooling (`src/utils/pythonGenerator.ts`) powering the web UI
    and CLI installs.
  - Already generates Python tasks from configs and now should validate against
    the exported schema while still relying on the shared template scaffold.
- **Sync workflow**
  1. Pipeline bumps `SCHEMA_VERSION`, exports JSON, publishes package/release.
  2. Registry fetches the new JSON, updates generator/tests, and republishes
     tasks tied to that version.
  3. CLI users run `task schema export` (for verification) or install tasks from
     the registry knowing both ends agree on the schema contract.

## Stage 1 recap

- **Export command**: Added `autocleaneeg-pipeline task schema export`, which renders
  the canonical task schema as JSON (either to stdout or a specified file).
- **Bundled artifact**: The JSON (`schema-2025.09.json`) ships inside the package at
  `autoclean/configkit/schema_exports/…`, ensuring both the CLI and registry
  can consume the same reference without additional tooling.
- **Documentation**: CLI help/README now describe the command and artifact location.

## Stage 2 updates + user stories

- **Registry generator** now imports the schema JSON, enforcing the version and
  rejecting unknown config sections before rendering templates.
- **User stories**:
  - *Template author*: run `autocleaneeg-pipeline task schema export -o schema.json`
    to view allowed fields and defaults while editing tasks.
  - *Registry maintainer*: after pulling the latest artifact, regenerate tasks;
    the generator surfaces mismatched fields with clear errors.
  - *QA team*: integrate JSON validation in CI to guarantee submitted tasks match the
    pipeline schema.

### Testing suggestions

1. Run `autocleaneeg-pipeline task schema export -o schema.json` and compare the
   output to the bundled artifact.
2. In the task registry repo, modify a sample task config to include an invalid
   section and trigger the generator—you should see a failure referencing the
   disallowed key.
3. Automate regression checks via `jsonschema` or by invoking the pipeline
   validator with exported configs.
