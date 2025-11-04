# Contributing to AutoClean EEG Pipeline

We welcome contributions! This guide will help you get started.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/cincibrainlab/autoclean_pipeline.git
   cd autoclean_pipeline
   ```

2. **Install in editable mode using uv:**
   ```bash
   uv tool install -e --upgrade . --force
   ```

3. **Install pre-commit hooks (recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
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
make lint          # Run linting (ruff) and type checking (mypy)
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
black .
isort .
ruff check .
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
pytest -q
pytest --cov=autoclean
```

## Documentation

Build documentation locally:

```bash
make docs-build    # Build documentation
make docs-serve    # Serve documentation locally at http://localhost:8000
```

Or build directly:

```bash
make -C docs html
```

## Submitting Changes

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure linting, type checking, and tests pass:
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
