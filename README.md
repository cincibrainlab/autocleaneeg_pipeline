# AutoCleanEEG Pipeline

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://cincibrainlab.github.io/autoclean_pipeline/)

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
- Optional components depend on workflow. Review GUI support, MATLAB-backed
  tasks, and Serve worker deployments may require additional local dependencies
  or services.

## Supported Surfaces

- `autocleaneeg-pipeline`: primary CLI for processing, workspace, task, and Serve commands
- `autocleaneeg-serve`: convenience launcher for the Serve web/API workflow
- `autocleaneeg-tui`: terminal UI entrypoint
- Python API: `from autoclean import Pipeline`
- `web/`: frontend source for the Serve UI

Experimental, legacy, or pending-approval components may exist in-tree, but the
surfaces above are the supported public entrypoints for this repository.

Dependency versions are intentionally conservative in this repository to keep
research workflows reproducible and to reduce breakage across the processing
stack. New documentation and examples should prefer the supported surfaces
above rather than older compatibility paths retained in-tree.

## Installation

### Install The CLI

```bash
uv tool install autocleaneeg-pipeline
autocleaneeg-pipeline --help
autocleaneeg-serve --help
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

``autocleaneeg-pipeline list-tasks``
List the available built-in and installed tasks.

```bash
autocleaneeg-pipeline list-tasks
```

``autocleaneeg-pipeline process RestingEyesOpen /path/to/data.raw``
Process one file with a built-in task.

```bash
autocleaneeg-pipeline process RestingEyesOpen /path/to/data.raw
```

``autocleaneeg-pipeline review --output /path/to/output``
Open the review flow for a completed output directory.

```bash
autocleaneeg-pipeline review --output /path/to/output
```

``autocleaneeg-serve up``
Start the normal Serve web/API workflow.

```bash
autocleaneeg-serve up
```

## Serve Basics

Use the Serve surfaces with this split:

- `autocleaneeg-serve`: normal operator launcher for starting, stopping, and
  checking the Serve daemon
- `autocleaneeg-pipeline serve ...`: lower-level workspace, route, validation,
  queue, dispatcher, and API control surface

``autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace``
Create a new Serve workspace.

```bash
autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace
```

``autocleaneeg-serve up``
Start the normal Serve daemon for that workspace.

```bash
autocleaneeg-serve up
```

``autocleaneeg-pipeline serve route list --path /path/to/serve-workspace``
List the routes configured for a Serve workspace.

```bash
autocleaneeg-pipeline serve route list --path /path/to/serve-workspace
```

``autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode test``
Validate the Serve configuration before deployment.

```bash
autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode test
```

``autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode test``
Deploy the current Serve draft configuration for test mode.

```bash
autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode test
```

``autocleaneeg-serve status``
Check whether the Serve UI and dispatcher are actually operational.

```bash
autocleaneeg-serve status
```

For the full Serve operator workflow, see [docs/serve_ui_workflow.rst](docs/serve_ui_workflow.rst).

## Documentation

- published docs: <https://cincibrainlab.github.io/autoclean_pipeline/>
- docs tree index: [docs/INDEX.md](docs/INDEX.md)
- issue tracker: <https://github.com/cincibrainlab/autoclean_pipeline/issues>

GitHub Pages publishing:

- pushes to `main` publish the Sphinx docs to GitHub Pages through
  [docs.yml](.github/workflows/docs.yml)
- pull requests should only validate the docs build through CI; they do not
  publish Pages output

## Repository Layout

- `src/autoclean/`: package source
- `configs/`: shipped configuration assets
- `web/`: Serve frontend source
- `src/autoclean/api/static/`: tracked runtime frontend bundle used by the API/Serve surface
- `docs/`: published documentation sources
- `plans/`: active engineering plans
- `tests/`: unit and integration coverage
- `examples/`: maintained example and template material

Generated output such as `docs/_build/`, `plans/_site/`, coverage reports, and
cache directories is not canonical source.

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
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md).

## Acknowledgments

- Cincinnati Children's Hospital Research Foundation
- Built with [MNE-Python](https://mne.tools/)
