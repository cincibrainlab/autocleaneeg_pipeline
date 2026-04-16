# Contributing to AutoClean EEG Pipeline

This repository uses `uv`, `make`, and pytest-based validation as the canonical
contributor workflow. If a doc conflicts with this file, treat this file as the
source of truth.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/cincibrainlab/autoclean_pipeline.git
   cd autoclean_pipeline
   ```

2. **Install the package as an editable uv tool:**
   ```bash
   uv tool install -e --upgrade . --force
   ```

3. **Install contributor tooling:**
   ```bash
   make install-dev
   python3 scripts/uv_tools.py run pre-commit install
   ```

## Development Workflow with Makefile

We use a Makefile to standardize development workflows and ensure consistency across all contributors. **Why use Make?** Instead of remembering complex command combinations and tool invocations, developers can use simple, consistent commands like `make format` or `make check`. This approach:

- **Eliminates dependency conflicts**: Uses `uv tool` for isolated tool management
- **Standardizes workflows**: All developers run the same commands
- **Simplifies CI replication**: Run `make ci-check` to test locally before pushing
- **Reduces errors**: No need to remember tool paths, flags, or correct parameter order

### Quick Commands

```bash
make help          # Show all available commands
make check         # Run all code quality checks (format + lint)
make format        # Auto-format code with black and isort
make lint          # Run linting (ruff)
make test          # Run unit tests
make test-cov      # Run tests with coverage report
make ci-check      # Run CI-equivalent checks locally
make clean         # Clean temporary files and caches
```

### Code Style

We use automated tools for code quality. Run before committing:

```bash
make format        # Auto-format code
make lint          # Check code quality
# Or use make check to run both
```

Alternatively, you can run tools directly:

```bash
black src tests scripts
isort src tests scripts
ruff check src tests scripts
mypy src/autoclean
```

Configuration is in `pyproject.toml`.

## Testing

Run the test suite:

```bash
make test          # Run unit tests
make test-cov      # Run tests with coverage
make test-all      # Run all tests (unit + integration)
```

Or run pytest directly:

```bash
pytest tests/unit -v
pytest tests/integration -v --tb=short
```

## Documentation

Build documentation locally:

```bash
make docs-setup    # Install docs dependencies
make docs-build    # Build documentation
make docs-serve    # Serve documentation locally at http://localhost:8000
```

Or build directly:

```bash
make -C docs html
```

Documentation navigation:

- main docs tree: [docs/INDEX.md](docs/INDEX.md)

Docs publishing:

- [`.github/workflows/docs.yml`](.github/workflows/docs.yml) is the canonical
  GitHub Pages deployment path
- pushes to `main` publish `docs/_build/html`
- pull requests should validate docs in CI, not by updating a separate
  `gh-pages` branch

Frontend workflow:

- the Serve frontend lives under `web/`
- run frontend build and test commands from that directory via [web/package.json](web/package.json)
- frontend changes should pass `cd web && npm test` and `cd web && npm run build`

Validation policy:

- required CI checks cover formatting, linting, unit tests, docs build, package
  smoke checks, and frontend validation
- heavier or environment-specific testing is maintainer-driven and should not be
  assumed to run on every public PR

Supported public entrypoints:

- `autocleaneeg-pipeline`
- `autocleaneeg-serve`
- `autocleaneeg-tui`

Serve command model:

- use `autocleaneeg-serve` for the normal daemon lifecycle: foreground start,
  `up`, `down`, `restart`, `status`, and `share`
- use `autocleaneeg-pipeline serve ...` for workspace selection, route
  management, validation, deployment, queue inspection, dispatcher control, and
  lower-level API/TUI/worker commands

Compatibility notes:

- prefer the public entrypoints above in new docs, examples, and contributor work
- treat older or compatibility-only paths in-tree as implementation details unless a maintainer explicitly documents them for public use

## Submitting Changes

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure linting and tests pass:
   ```bash
   make ci-check    # Run all checks (equivalent to CI)
   ```
5. Submit a pull request with a clear description

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/cincibrainlab/autoclean_pipeline/issues). Include:
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or tracebacks

## License

Contributions are licensed under the MIT License. See `LICENSE.md` for details.
