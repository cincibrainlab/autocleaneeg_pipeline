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
   - Modify the registry’s generator scripts to read the exported schema instead
     of hard-coding the shape/`schema_version`.
   - Fail generation if the schema artifact cannot be found or the version
     changes unexpectedly.
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

_Draft prepared for review before implementation._
