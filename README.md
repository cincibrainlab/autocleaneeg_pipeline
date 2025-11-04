# AutoCleanEEG Pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular framework for automated EEG data processing, built on MNE‑Python.

## Features

- Framework for automated EEG preprocessing with "lego block" modularity
- Support for multiple EEG paradigms (ASSR, Chirp, MMN, Resting State) 
- BIDS-compatible data organization and comprehensive quality control
- Extensible plugin system for file formats, montages, and event processing
- Research-focused workflow: single file testing → parameter tuning → batch processing
- Detailed output: BIDS‑compatible derivatives, single task log file, stage files, exports, and QA visualizations

## Installation (uv)

Use Astral's uv for fast, isolated installs. If you don't have uv yet, see https://docs.astral.sh/uv/

- Install CLI (recommended for users):

```bash
uv tool install autocleaneeg-pipeline
autocleaneeg-pipeline --help
```

- Upgrade or remove:

```bash
uv tool upgrade autocleaneeg-pipeline
uv tool uninstall autocleaneeg-pipeline
```

- Development install from source (editable install):

```bash
git clone https://github.com/cincibrainlab/autocleaneeg_pipeline.git
cd autocleaneeg_pipeline
uv tool install -e --upgrade . --force
autocleaneeg-pipeline --help # Slow on first run!
```

## Documentation

Full documentation is available at [https://docs.autocleaneeg.org](https://docs.autocleaneeg.org)

## Development

For contributors, we provide a Makefile with convenient development commands:

```bash
make help          # Show all available commands
make check         # Run code quality checks
make format        # Auto-format code
make lint          # Run linting and type checking
make test          # Run unit tests
make test-cov      # Run tests with coverage
make ci-check      # Run CI-equivalent checks locally
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- Cincinnati Children's Hospital Research Foundation
- Built with [MNE-Python](https://mne.tools/)
