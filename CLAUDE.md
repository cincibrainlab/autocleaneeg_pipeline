# CLAUDE.md

This file gives Claude Code a current, repo-local orientation for working in
this repository.

## Commit Guidance

- Do not mention Claude in commit messages or descriptions.
- Prefer conventional prefixes when appropriate: `feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`.

## Project Overview

AutoCleanEEG Pipeline is a modular EEG preprocessing and review framework built
on MNE-Python. The repository includes:

- the main pipeline CLI: `autocleaneeg-pipeline`
- the Serve launcher: `autocleaneeg-serve`
- the TUI entrypoint: `autocleaneeg-tui`
- a Python API centered on `autoclean.Pipeline`
- frontend source in `web/` and tracked runtime assets in
  `src/autoclean/api/static/`

Treat the README and published docs as the canonical user-facing references:

- [`README.md`](README.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`docs/INDEX.md`](docs/INDEX.md)
- [`docs/command_reference.rst`](docs/command_reference.rst)
- [`docs/serve_command_reference.rst`](docs/serve_command_reference.rst)

## Repository Layout

- `src/autoclean/`: package source
- `src/autoclean/core/`: pipeline and task base classes
- `src/autoclean/mixins/`: processing mixins
- `src/autoclean/plugins/`: plugin and registry surfaces
- `src/autoclean/tasks/`: built-in tasks
- `src/autoclean/api/`: API and Serve-facing runtime code
- `configs/`: shipped configuration assets
- `docs/`: Sphinx documentation sources
- `examples/`: maintained example material
- `tests/`: unit and integration coverage
- `web/`: frontend source for the Serve UI

## Development Commands

Recommended setup:

```bash
make dev-setup
autocleaneeg-pipeline --help
autocleaneeg-serve --help
```

Common checks:

```bash
make check
make check-fix
make test
make test-cov
make test-all
make ci-check
make docs-build
```

Source install path used by this repo:

```bash
uv tool install -e --upgrade . --force
```

## CLI Notes

Use these command surfaces:

- `autocleaneeg-pipeline` for processing, workspace, task, config, review, and
  Serve control commands
- `autocleaneeg-serve` for the normal Serve daemon lifecycle
- `autocleaneeg-tui` for the terminal UI

Common examples:

```bash
autocleaneeg-pipeline list-tasks
autocleaneeg-pipeline process RestingEyesOpen /path/to/file.raw
autocleaneeg-pipeline review --output /path/to/output

autocleaneeg-pipeline serve workspace --mode new --path /path/to/serve-workspace
autocleaneeg-serve up
autocleaneeg-pipeline serve route list --path /path/to/serve-workspace
autocleaneeg-pipeline serve validate --path /path/to/serve-workspace --mode test
autocleaneeg-pipeline serve deploy --path /path/to/serve-workspace --mode test
autocleaneeg-serve status
```

For full command coverage, use the command reference docs instead of extending
this file with duplicate command inventories.

## Architecture Notes

- Task behavior is composed from mixins under `src/autoclean/mixins/`.
- Built-in task implementations live under `src/autoclean/tasks/`.
- Serve runtime code is split across the CLI, API, workspace, and frontend
  surfaces rather than one single module.
- Compatibility aliases and older surfaces may still exist in-tree, but new
  docs and examples should prefer the supported public entrypoints listed
  above.

## Workspace Notes

The active Serve and operator workflow in this repo assumes a workspace under
the user's environment, commonly `~/Documents/Autoclean-EEG`. Use the CLI to
inspect or set it rather than assuming an older fixed path:

```bash
autocleaneeg-pipeline workspace show
autocleaneeg-pipeline workspace set /path/to/workspace
autocleaneeg-pipeline serve workspace status
```
