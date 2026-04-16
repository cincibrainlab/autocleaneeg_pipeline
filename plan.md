# Production Open-Source Cleanup Plan

## Goal

Turn this repository into a production-grade open-source repo by doing one
intentional cleanup pass across repo hygiene, packaging, documentation,
automation, and release policy.

This plan is review-only. It does not assume implementation has started.

Maintainer decisions captured during cleanup:

- do not add non-standard root docs when the content belongs in `README.md`,
  `CONTRIBUTING.md`, or `docs/INDEX.md`
- use GitHub issues as the repository issue-reporting path
- keep governance additions minimal and conventional
- fold dependency, legacy, release, and repo-structure guidance into standard
  docs instead of adding standalone root policy files
- do not add separate `SECURITY.md` or support-policy files for this repo;
  issue reporting and maintainer boundaries should stay in standard repo docs

## Review Summary

The repo already has substantial implementation and test depth, but it still has
several production-readiness gaps:

- release and contributor surfaces are inconsistent with the real codebase
- CI is effectively disabled for core validation
- docs are split across canonical docs, historical plans, generated HTML, and
  older contributor guides
- tracked/generated artifacts and local-run artifacts are not cleanly separated
- package metadata and version support signals conflict with each other
- active vs experimental vs legacy code paths are not clearly curated
- open-source governance files and templates are incomplete or missing

## Phases

### Phase 1: Freeze Scope And Define Canonical Policies

- [x] Decide and document the repo support policy:
  supported Python versions, supported OS targets, optional dependencies,
  MATLAB support expectations, GUI support expectations, and whether the web UI
  is part of the default install or an optional surface, using standard docs
  instead of a separate root policy file.
- [x] Decide the canonical user entrypoints and contributor entrypoints:
  `autocleaneeg-pipeline`, `autocleaneeg-serve`, TUI, API, and `web/`.
- [x] Decide the lifecycle policy for code/doc states:
  `active`, `experimental`, `pending approval`, `deprecated`, `legacy`,
  `archived`.
- [x] Decide what must be shipped in git versus generated at release time:
  built frontend assets, docs build output, coverage output, Quarto `_site`
  output, screenshots, and sample fixture data.
- [x] Decide the minimum bar for “production open-source ready”:
  required CI checks, required docs, required repo metadata, and release steps.

### Phase 2: Repository Hygiene And Artifact Cleanup

- [x] Remove tracked local-run artifacts and generated noise from the repo where
  they are not intentionally versioned:
  `__pycache__/`, `docs/__pycache__/`, `htmlcov/`, `.serve-run.pid`, and any
  stray compiled files.
- [x] Audit `.gitignore` so it reflects deliberate policy instead of organic
  accumulation.
- [x] Remove redundant or stale ignore rules and ensure real source paths are
  not accidentally treated like temporary output.
- [x] Review whether root `package.json` and `package-lock.json` should be kept,
  replaced, or removed if `web/` is the only real frontend package.
- [x] Verify `.gitattributes` remains minimal and consistent with the intended
  cross-platform line-ending policy.
- [x] Add a simple “generated artifacts policy” note to contributor docs so
  future cleanup debt does not immediately return.

### Phase 3: Package, Build, And Metadata Consistency

- [x] Reconcile all Python version signals:
  `pyproject.toml`, classifiers, Black target versions, mypy target version,
  docs, tests, and block manifests.
- [x] Reconcile install guidance across all docs:
  `uv tool install`, editable dev install, optional extras, and GUI/MATLAB
  extras.
- [x] Audit whether all public console scripts that are documented are actually
  declared in packaging metadata.
- [x] Audit dependency strategy for open-source distribution:
  identify which dependencies are core, optional, platform-sensitive, heavy,
  or likely to fail in CI/user installs.
- [x] Review whether pinned dependencies are intentionally pinned for
  reproducibility or unintentionally over-constrained for public consumers.
- [x] Verify sdist/wheel contents so only intended runtime files are packaged.
- [x] Confirm the published README and PyPI metadata describe the repo the same
  way the code actually behaves today.

### Phase 4: Documentation Consolidation And Canonicalization

- [x] Pick one canonical contributor guide and deprecate or rewrite the others.
  Current drift exists between root docs and published Sphinx development docs.
- [x] Pick one canonical quick-start path for end users and remove conflicting
  command examples.
- [x] Replace outdated command examples such as `autoclean ...` if the actual
  supported CLI is `autocleaneeg-pipeline ...`.
- [x] Reconcile documentation around setup flow, workspace model, Serve mode,
  and task discovery so the current architecture is explained once, not in
  overlapping guides.
- [x] Audit the docs tree for generated/static HTML pages that should be kept
  versus pages that should become normal Sphinx sources or be archived.
- [x] Create a docs information architecture note:
  root README for first contact, `docs/` for published docs, `plans/` for
  active engineering plans, `archive/` only for retired material.
- [x] Add or update a “repo map” section in docs so outside contributors can
  understand the codebase surfaces quickly, without creating a separate
  standalone root document.
- [x] Normalize changelog ownership:
  decide whether `docs/CHANGELOG.md` or `docs/development/changelog.rst` is the
  canonical release history and archive or cross-link the other.

### Phase 5: Archive And Curate Non-Canonical Material

- [x] Review `plans/` and classify every top-level standalone plan as either
  active, archive, or superseded.
- [x] Move obviously historical or one-off cleanup notes into the existing
  archive structure instead of leaving them mixed with active plans.
- [x] Review `docs/archive/` and ensure only genuinely retired docs live there.
- [x] Review `src/autoclean/tasks/pending_approval/` and define the policy:
  promote, rename as experimental, move to archive/examples, or exclude from
  normal user-facing discovery if they are not production supported.
- [x] Review legacy surfaces such as `legacy_app`, deprecated CLI aliases, and
  old workflow notes, and decide whether to keep, document as deprecated, or
  remove in a later cleanup phase.
- [x] Review example scripts and ensure only maintained examples remain in the
  top-level `examples/` directory.
- [x] Review tracked built frontend assets in `src/autoclean/api/static/` and
  decide whether they are release artifacts, checked-in runtime assets, or
  generated output that should be rebuilt during release.

### Phase 6: Quality Gates, CI, And Validation

- [x] Re-enable or replace the disabled CI workflow with a smaller, trustworthy
  baseline pipeline before expanding scope.
- [x] Define the required PR checks for open-source contribution:
  format, lint, unit tests, selected integration tests, docs build, and
  packaging sanity checks.
- [x] Align Make targets, contributor docs, and CI commands so they all invoke
  the same toolchain.
- [x] Remove stale references to tools no longer used, such as `flake8` or
  `pip install -e ".[dev]"`, if those are no longer real workflows.
- [x] Add a validation step for frontend changes if the web UI is a supported
  public surface.
- [x] Add a minimal packaging/release smoke test:
  install from wheel/sdist and verify the documented CLI entrypoints work.
- [x] Decide how to handle heavier tests:
  optional/manual, nightly, or separate workflow.
- [x] Audit the disabled real-data workflow and either formalize it as a
  separate maintainer-only workflow or archive it if it is not part of the
  public maintenance contract.

### Phase 7: Open-Source Governance And Repository UX

- [x] Add missing governance/community files as needed:
  `CODE_OF_CONDUCT.md`, `CITATION.cff`, issue templates, PR
  template, and Dependabot configuration if they are part of the intended
  maintenance model.
- [x] Ensure `LICENSE.md`, README badges, project URLs, and release links are
  consistent and current.
- [x] Add a short support/boundary note in standard repo docs explaining what
  maintainers will and will not support: platforms, data types, experimental
  blocks, and enterprise/regulatory claims.
- [x] Add a clear disclosure for optional external systems:
  MATLAB, Redis/RQ, Cloudflare tunnel, GUI dependencies, and any model-backed
  features.
- [x] Review compliance/security language in docs so claims are precise and do
  not overstate guarantees.
- [x] Decide whether author and organization metadata need refresh before public
  promotion.

### Phase 8: Final Production Readiness Pass

- [x] Run the agreed validation matrix after cleanup:
  install, lint, tests, docs build, frontend build, package build, and a basic
  CLI smoke test.
- [x] Confirm the repo root presents a clean open-source experience:
  README, install path, docs path, contributing path, release path, and issue
  path all obvious within a minute of opening the repository.
- [x] Confirm there is a single canonical answer to each of these:
  how to install, how to run, how to contribute, how to build docs, how to run
  tests, and what is experimental.

## Recommended Execution Order

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7
8. Phase 8

## Notes For Implementation

- Prefer moving or archiving over deleting when material still has historical
  value.
- Keep cleanup PRs narrow enough that generated-file churn does not hide policy
  changes.
- Separate “repo hygiene and docs canonicalization” from “behavior-changing
  refactors” unless the latter are required to make packaging or CI truthful.
