# Contributing to AutoClean EEG Pipeline

Thanks for your interest in improving AutoClean! We welcome contributions of all kinds — bug reports, fixes, features, docs, and examples.

- Project name (PyPI): `autocleaneeg-pipeline`
- CLI command: `autocleaneeg-pipeline`
- Minimum Python: 3.10 (supports < 3.14)

## Getting Started

1) Fork and clone the repository

```bash
git clone https://github.com/cincibrainlab/autoclean_pipeline.git
cd autoclean_pipeline
```

2) Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

3) Install in editable mode

```bash
pip install -e .
# Optional extras
# pip install -e '.[gui]'     # GUI review tool dependencies
# pip install -e '.[docs]'    # Documentation tooling (if you plan to build docs)
```

4) Install pre-commit hooks (recommended)

```bash
pip install pre-commit
pre-commit install
```

## Linting and Type Checking

We use Black, isort, Ruff, and mypy. Before opening a PR, please run:

```bash
black .
isort .
ruff check .
mypy src/autoclean
```

Configuration lives in `pyproject.toml`.

## Running Tests

```bash
pytest -q
# With coverage
pytest --cov=autoclean
```

## Building Documentation

Docs are built with Sphinx and live in `docs/`.

```bash
# Ensure Sphinx deps are available (examples)
pip install sphinx pydata-sphinx-theme numpydoc sphinx-gallery
# Build
make -C docs html
# Open docs
open docs/_build/html/index.html   # Windows: start, Linux: xdg-open
```

Note: The docs requirements may evolve; see `docs/conf.py` for the authoritative list.

## Development Tips

- Run the CLI locally after editable install:
  - `autocleaneeg-pipeline --help`
- Source code lives under `src/autoclean/`
- Keep changes focused and well-scoped; avoid unrelated refactors in the same PR

## Submitting a Pull Request

- Create a feature branch from `main`
- Add or update tests for new behavior
- Update docs and README where relevant
- Ensure linters, mypy, and tests pass locally
- Write a clear, concise PR description with context and motivation

## Reporting Bugs and Requesting Features

- Issues: https://github.com/cincibrainlab/autoclean_pipeline/issues
- Please include steps to reproduce, expected vs. actual behavior, and relevant logs/tracebacks

## License

By contributing, you agree that your contributions will be licensed under the project’s MIT License. See `LICENSE` for details.

