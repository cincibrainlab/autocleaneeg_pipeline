# AutoCleanEEG Pipeline

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular framework for automated EEG data processing, built on MNE-Python.



## Features

- Framework for automated EEG preprocessing with "lego block" modularity
- Support for multiple EEG paradigms (ASSR, Chirp, MMN, Resting State) 
- BIDS-compatible data organization and comprehensive quality control
- Extensible plugin system for file formats, montages, and event processing
- Research-focused workflow: single file testing → parameter tuning → batch processing
- Detailed output: logs (stored in your AutoClean workspace), stage files, metadata, and quality control visualizations

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

- Development install from source:

```bash
git clone https://github.com/cincibrainlab/autoclean_pipeline.git
cd autoclean_pipeline
uv venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
uv pip install -e .
# Optional extras
# uv pip install -e '.[gui]'   # GUI review tool dependencies
# uv pip install -e '.[docs]'  # Documentation tooling
```

## Quick Start

Process a file using a built-in task:

```bash
autocleaneeg-pipeline process RestingEyesOpen /path/to/data.raw
```

List tasks and show overrides:

```bash
autocleaneeg-pipeline list-tasks --overrides
```

## Theme and Color

AutoClean’s CLI uses Rich with semantic styles and adaptive themes for readable output across light/dark terminals, limited color depth, and colorless logs.

- Flag: `--theme auto|dark|light|hc|mono` (default: `auto`)
  - `mono`: Monochrome (no hues), ideal for logs or unknown backgrounds
  - `hc`: High-contrast, accessible on both dark and light backgrounds
- Env overrides:
  - `AUTOCLEAN_THEME=auto|dark|light|hc|mono`
  - `AUTOCLEAN_COLOR_DEPTH=auto|8|256|truecolor`
  - `NO_COLOR=1` disables color
  - `FORCE_COLOR=1` forces color even in non-TTY (e.g., CI)

Examples:

```bash
autocleaneeg-pipeline --theme light list-tasks
AUTOCLEAN_THEME=hc autocleaneeg-pipeline version
NO_COLOR=1 autocleaneeg-pipeline list-tasks
```

## Documentation

Full documentation is available at [https://cincibrainlab.github.io/autoclean_pipeline/](https://cincibrainlab.github.io/autoclean_pipeline/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Cincinnati Children's Hospital Research Foundation
- Built with [MNE-Python](https://mne.tools/)
