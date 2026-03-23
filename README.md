# AutoCleanEEG Pipeline

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AutoCleanEEG Pipeline is a modular framework for automated EEG preprocessing, review, and operational workflows built on MNE-Python.

It is designed for research settings that need reproducible task-driven processing, quality-control outputs, and a path from single-file validation to larger-scale batch or serve-style operation.

## What It Includes

- Modular EEG preprocessing with task-based configuration
- Support for multiple paradigms including ASSR, Chirp, MMN, and resting state
- BIDS-aligned outputs and quality-control artifacts
- Plugin-based extensibility for formats, montages, and event handling
- Review tooling for inspecting outputs and exclusions
- Serve/TUI/web tooling in this repository for operational workflows

## Requirements

- Python `3.11` to `3.13`
- `uv` for the recommended install workflow: <https://docs.astral.sh/uv/>

## Installation

### Install The CLI

```bash
uv tool install autocleaneeg-pipeline
autocleaneeg-pipeline --help
```

### Upgrade Or Remove

```bash
uv tool upgrade autocleaneeg-pipeline
uv tool uninstall autocleaneeg-pipeline
```

### Install From Source For Development

```bash
git clone https://github.com/cincibrainlab/autoclean_pipeline.git
cd autoclean_pipeline
uv tool install -e --upgrade . --force
autocleaneeg-pipeline --help
```

For contributor workflow, testing, linting, and local docs commands, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start

List available tasks:

```bash
autocleaneeg-pipeline list-tasks
```

Process a file with a built-in task:

```bash
autocleaneeg-pipeline process RestingEyesOpen /path/to/data.raw
```

Review pipeline output:

```bash
autocleaneeg-pipeline review --output /path/to/output
```

## Documentation

- published docs: <https://cincibrainlab.github.io/autoclean_pipeline/>
- docs tree index: [docs/INDEX.md](docs/INDEX.md)

## Development

Common contributor commands:

```bash
make help
make check
make format
make lint
make test
make test-cov
make ci-check
```

Frontend work lives in [web/package.json](web/package.json). Run frontend build and test commands from the `web/` directory.

## Contributing

See:

- [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow
- [BRANCHING.md](BRANCHING.md) for branch and merge conventions

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md).

## Acknowledgments

- Cincinnati Children's Hospital Research Foundation
- Built with [MNE-Python](https://mne.tools/)
