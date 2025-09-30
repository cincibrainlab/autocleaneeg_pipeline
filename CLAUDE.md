# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit Guidelines
- DO NOT add anything about claude in git commit messages or descriptions
- Use conventional commit format when possible (feat:, fix:, docs:, test:, refactor:)

## Project Overview
AutoClean EEG is a sophisticated modular framework for automated EEG data processing built on MNE-Python. It supports multiple EEG paradigms (ASSR, Chirp, MMN, Resting State) with BIDS-compatible data organization and enterprise-grade audit logging.

**Version 2.0.0+ introduces Python-based task files with embedded configuration, replacing YAML pipelines.**

## Core Architecture

### Dynamic Mixin System
The codebase uses an innovative mixin discovery system that automatically finds and combines all "*Mixin" classes:
- **Auto-discovery**: Scans `src/autoclean/mixins/` subdirectories for mixin classes
- **External blocks**: Also discovers plugin blocks from `~/.autoclean/blocks/`, `./blocks/`, and task-registry
- **Dynamic combination**: Creates a single CombinedAutocleanMixins class via multiple inheritance
- **MRO conflict detection**: Advanced error handling for method resolution order issues
- **Collision warnings**: Detects method name conflicts between mixins

**Plugin Blocks (v2.4.0+)**: Single-file Python modules that extend Task with custom methods. Drop `*_plugin.py` files in `~/.autoclean/blocks/` to add functionality without modifying pipeline code. See `PLUGIN_BLOCKS_PLAN.md` for architecture details.

### Key Components
1. **Pipeline** (`src/autoclean/core/pipeline.py`) - Central orchestrator managing workflow
2. **Task** (`src/autoclean/core/task.py`) - Base class inheriting from all discovered mixins
3. **Mixins** (`src/autoclean/mixins/`) - Processing components organized by functionality:
   - `signal_processing/` - Filtering, ICA, epoching, artifacts
   - `viz/` - Reports, plots, topography
   - `analysis/` - Connectivity, source localization, wavelets
   - `utils/` - BIDS handling, file operations, validation

### Plugin Architecture
Auto-registered extensibility system:
- **EEG Plugins** (`src/autoclean/plugins/eeg_plugins/`) - Format + montage handlers (e.g., CNT_GSN129)
- **Event Processors** (`src/autoclean/plugins/event_processors/`) - Paradigm-specific event handling
- **Format Plugins** (`src/autoclean/plugins/formats/`) - EEG file format support (EGI, CNT, EDF)

### Task Implementation Pattern
```python
# Python task file with embedded configuration (v2.0.0+)
config = {
    "schema_version": "2025.09",
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {"l_freq": 1, "h_freq": 100, "notch_freqs": [60, 120]}
    }
}

class CustomTask(Task):  # Inherits all mixins automatically
    def run(self):
        self.import_raw()           # From base

        # Basic preprocessing steps (explicit for transparency)
        self.resample_data()        # From mixins
        self.filter_data()          # From mixins
        self.drop_outer_layer()     # From mixins
        self.assign_eog_channels()  # From mixins
        self.trim_edges()           # From mixins
        self.crop_duration()        # From mixins

        # Channel cleaning and rereferencing
        self.clean_bad_channels()   # From mixins
        self.rereference_data()     # From mixins

        # Advanced processing
        self.run_ica()             # From mixins
        self.create_regular_epochs() # From mixins
```

## Development Commands

### Code Quality & Testing
```bash
# Quick quality checks (recommended before commits)
make check                      # Run all checks (format, lint, type)
make check-fix                  # Auto-fix formatting and linting issues
make fix-all                    # Fix all possible issues automatically

# Testing
make test                       # Run unit tests
make test-cov                   # Run tests with coverage report
make test-all                   # Run all tests (unit + integration)
make ci-check                   # Run CI-equivalent checks locally

# Run specific tests
pytest tests/unit/test_pipeline.py -v                           # Specific file
pytest tests/unit/test_pipeline.py::TestPipeline::test_init -v  # Specific method
pytest tests/unit/ -k "pattern" -v                              # Pattern matching
pytest tests/integration/ --benchmark-only                       # Performance tests
```

### Installation & Setup
```bash
# Development setup
make dev-setup                  # Complete dev environment setup
pip install -e .                # Install package in editable mode
pip install -e ".[gui]"         # Install with GUI dependencies

# Standalone CLI tool
uv tool install autocleaneeg-pipeline    # Install from PyPI
make install-uv-tool                      # Install from source
```

### CLI Usage
```bash
# Core commands
autocleaneeg-pipeline process RestingEyesOpen /path/to/data.raw
autocleaneeg-pipeline list-tasks --overrides
autocleaneeg-pipeline review --output results/

# Task schema management
autocleaneeg-pipeline task schema export -o schema.json
autocleaneeg-pipeline task schema export --bundle

# Audit log export
autocleaneeg-pipeline export-access-log --output audit.jsonl
autocleaneeg-pipeline export-access-log --format csv --output audit.csv
autocleaneeg-pipeline export-access-log --verify-only
```

## Key File Locations
- **Core**: `src/autoclean/core/` - Pipeline and Task base classes
- **Mixins**: `src/autoclean/mixins/` - Processing components (signal_processing/, viz/, analysis/, utils/)
- **Plugins**: `src/autoclean/plugins/` - Auto-registered extensions
- **Tasks**: `src/autoclean/tasks/` - Built-in paradigm implementations
- **Database**: `src/autoclean/database/` - SQLite with audit logging
- **GUI**: `src/autoclean/tools/` - Review GUI and task manager
- **Workspace**: `~/.autoclean/` or OS-specific user directory
- **Custom Tasks**: `workspace/tasks/` - User Python task files

## Audit Trail & Compliance
The system maintains tamper-proof audit logging with cryptographic integrity:
- **Hash chain verification**: Each log entry includes hash of previous entry
- **User context tracking**: Username, hostname, PID, timestamp for all operations
- **Task file tracking**: SHA256 hash and full source code captured for reproducibility
- **Database protection**: SQL triggers prevent modification of completed runs
- **Export formats**: JSONL, CSV, human-readable reports

## API Migration (v1.x → v2.0.0+)
```python
# OLD (v1.x) - YAML configuration
pipeline = Pipeline(
    autoclean_dir="/path/to/output",
    autoclean_config="config.yaml"
)

# NEW (v2.0.0+) - Python task files
pipeline = Pipeline(output_dir="/path/to/output")
pipeline.add_task("my_custom_task.py")
pipeline.process_file("/path/to/data.raw", task="MyTask")
```

## Research Workflow
1. **Setup**: Interactive workspace wizard creates directory structure
2. **Development**: Drop Python task files into workspace/tasks/
3. **Testing**: Process single files to validate parameters
4. **Production**: Batch processing for full datasets
5. **Review**: GUI tools for quality inspection and BIDS derivatives

## Common Patterns

### Creating Custom Mixins
```python
# Add to src/autoclean/mixins/custom/my_mixin.py
class MyCustomMixin:
    def my_processing_step(self):
        # Will be available to all Task classes
        pass
```

### Handling Mixin Conflicts
When mixins have conflicting method names, the last mixin in MRO wins. Use explicit method calls:
```python
def run(self):
    # Explicitly call specific mixin's version
    FilteringMixin.apply_filter(self, ...)
```

### Export Counter System
Processing stages automatically numbered in BIDS derivatives:
- `01_import/`, `02_resample/`, `03_filter/`, etc.
- Replaces legacy `stage_files` approach

## Development Requirements
- **Python**: 3.10-3.13 (requires-python = ">=3.10,<3.14")
- **Core deps**: MNE>=1.10.1, PyTorch==2.8.0, NumPy>=1.20.0
- **GUI deps**: PyQt6, mne-qt-browser, textual
- **Build**: hatchling backend, UV package manager
- **Style**: Black (88 char), isort, ruff
- **Testing**: pytest with >85% coverage target

## Current Status
- **Version**: 2.3.0 (see pyproject.toml)
- **PyPI**: `autocleaneeg-pipeline`
- **Documentation**: https://cincibrainlab.github.io/autoclean_pipeline/
- **CI/CD**: GitHub Actions with automated testing