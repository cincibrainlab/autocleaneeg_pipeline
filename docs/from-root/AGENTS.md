# Repository Guidelines

## Project Structure & Module Organization
Core EEG-cleaning logic lives in `src/autoclean/`, grouped by processing phase (e.g., `configkit`, `tasks`, `tools`). Reusable CLI helpers and maintenance scripts reside in `scripts/`. Configuration templates ship in `configs/`, alongside exportable schemas under `src/autoclean/configkit/schema_exports/`. Tests sit in `tests/` split into `unit/`, `integration/`, `performance/`, with shared fixtures in `tests/fixtures/` for signal stubs and metadata. Longer-form docs land in `docs/`, while runnable usage demonstrations stay in `examples/` to mirror the README workflows.

## Build, Test, and Development Commands
`make install-dev` provisions the editable dev environment with uv-managed tooling; use `make install` when you only need runtime dependencies. Run `make format` for Black/isort auto-formatting and `make lint` to execute Ruff. `make check` chains format, lint, and type validation. Execute `make test` or `pytest tests/unit -v` for unit coverage, `make test-integration` for end-to-end validation, and `make test-cov` when you need HTML and terminal coverage reports. Before opening a PR, run `make ci-check` to mirror the CI matrix locally.

## Coding Style & Naming Conventions
Python code must stay Black-compliant with an 88-character limit and isort’s Black profile. Ruff warnings should be fixed rather than ignored. Provide explicit type hints—our mypy configuration blocks untyped definitions. Prefer dataclasses or Pydantic models over ad-hoc dicts, and keep business logic inside `src/autoclean/...` instead of CLI entry points. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.

## Testing Guidelines
Pytest powers all suites. Name files `test_<feature>.py` and individual tests `test_<behavior>`. Reuse fixtures in `tests/fixtures/` to avoid duplicating EEG sample data. New features should land unit coverage plus integration scenarios when they touch IO pipelines. Maintain coverage at or above the default `--cov=autoclean` threshold and regenerate reports with `make test-cov` ahead of reviews.

## Commit & Pull Request Guidelines
Commits follow the concise present-tense prefixes already in history (`feat:`, `fix:`, `refactor:`). Group related edits and keep subjects under 60 characters. PRs must link tracking issues, summarize user-facing impact, list validation commands, and attach before/after diagnostics (plots, metrics) when workflows change. Request review from module owners and wait for green `make ci-check` parity before merging.
