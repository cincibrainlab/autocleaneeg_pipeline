# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `src/autoclean/`, organized by processing stage (e.g., `configkit`, `tasks`, `tools`). Reusable CLI utilities sit in `scripts/`. Configuration templates and built-in schemas live under `configs/` and `src/autoclean/configkit/schema_exports/`. Test assets and cases reside in `tests/` with `unit/`, `integration/`, `performance/`, and shared fixtures in `tests/fixtures/`. Documentation sources stay in `docs/`, while runnable usage examples are collected in `examples/`.

## Build, Test, and Development Commands
Create a local dev environment with `make install-dev`, or run `make install` for an editable pip install. Use `make format`, `make lint`, and `make check` for formatting, linting, and combined quality gates via the uv-managed toolchain. Run unit tests with `make test` or `pytest tests/unit -v`, integration tests via `make test-integration`, and `make test-cov` when you need coverage reports.

## Coding Style & Naming Conventions
Python code follows Black with an 88-character line limit and isort’s Black profile. Ruff enforces lint rules; address warnings rather than suppressing them. Type hints are expected (the mypy config disallows untyped defs), and prefer explicit dataclass or Pydantic models over ad-hoc dicts. Modules and functions use `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`. Keep CLI entry points small and delegate logic into `src/autoclean/...` modules.

## Testing Guidelines
Pytest drives testing; new features need targeted tests beneath `tests/unit/` and scenario coverage in `tests/integration/` when IO or pipelines are involved. Name test files `test_<feature>.py` and functions `test_<behavior>`. Reuse fixtures from `tests/fixtures/` or add new ones there. Aim to keep coverage above the default `--cov=autoclean` threshold, and regenerate reports with `make test-cov` before merges.

## Commit & Pull Request Guidelines
Commits follow conventional prefixes seen in history (`feat:`, `fix:`, `refactor:`). Write present-tense summaries under 60 characters and group related changes. PRs should link tracking issues, outline changes, note testing commands run, and include before/after artifacts (e.g., QA plots) when behavior changes. Request reviews from domain owners for modules you touch and wait for CI parity (`make ci-check`) before merging.
