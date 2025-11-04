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

## Code Style

We use automated tools for code quality. Run before committing:

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
pytest -q
```

With coverage:

```bash
pytest --cov=autoclean
```

## Documentation

Build documentation locally:

```bash
make -C docs html
```

## Submitting Changes

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure linting, type checking, and tests pass
5. Submit a pull request with a clear description

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/cincibrainlab/autoclean_pipeline/issues). Include:
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or tracebacks

## License

Contributions are licensed under the MIT License. See `LICENSE` for details.
