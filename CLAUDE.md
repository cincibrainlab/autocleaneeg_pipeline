# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commit Guidelines
- DO NOT add anything about claude in git commit messages or descriptions
- Use conventional commit format when possible (feat:, fix:, docs:, test:, refactor:)

## Project Overview
AutoClean EEG is a modular framework for automated EEG data processing built on MNE-Python. It supports multiple EEG paradigms (ASSR, Chirp, MMN, Resting State) with BIDS-compatible data organization and database-backed processing tracking.

**Version 2.0.0 introduces major API changes and simplified workflow.**

## Core Architecture
- **Modular Design**: "Lego Block" approach for task composition
- **Dynamic Mixins**: Automatically discover and combine all "*Mixin" classes
- **Plugin System**: Auto-registration for EEG formats, montages, and event processors
- **Python Task Files**: Embedded configuration in Python files (v2.0.0)
- **No YAML Pipeline Configs**: YAML-based pipeline configs are removed; use Python task modules

### Key Components
1. **Pipeline** (`src/autoclean/core/pipeline.py`) - Main orchestrator handling configuration, file processing, and result management
2. **Task** (`src/autoclean/core/task.py`) - Abstract base class for all EEG processing tasks
3. **Mixins** (`src/autoclean/mixins/`) - Reusable processing components dynamically combined into Task classes

### Mixin System
- **Dynamic Discovery**: Automatically finds and combines all "*Mixin" classes
- **Signal Processing**: Artifacts, ICA, filtering, epoching, channel management
- **Visualization**: Reports, ICA plots, PSD topography  
- **Utils**: BIDS handling, file operations
- **MRO Conflict Detection**: Sophisticated error handling for inheritance conflicts

### Plugin Architecture
- **EEG Plugins** (`src/autoclean/plugins/eeg_plugins/`): Handle specific file format + montage combinations
- **Event Processors** (`src/autoclean/plugins/event_processors/`): Task-specific event annotation processing
- **Format Plugins** (`src/autoclean/plugins/formats/`): Support for new EEG file formats
- **Auto-registration**: Plugins automatically discovered at runtime

### Task Implementation Pattern (v2.0.0)
```python
# Python task file with embedded configuration
class NewTask(Task):  # Inherits all mixins automatically
    def __init__(self, config): 
        super().__init__(config)
    
    def run(self):
        self.import_raw()           # From base
        self.run_basic_steps()      # From mixins
        self.run_ica()             # From mixins
        self.create_regular_epochs() # From mixins

# Embedded configuration (replaces YAML)
config = {
    "eeg_system": {"montage": "GSN-HydroCel-129", "reference": "average"},
    "signal_processing": {"filter": {"highpass": 0.1, "lowpass": 50.0}},
    "output": {"save_stages": ["raw", "epochs"]}
}
```

### Pipeline Usage (v2.0.0)
```python
from autoclean import Pipeline

# Simple initialization (no YAML required)
pipeline = Pipeline(output_dir="/path/to/output")

# Add custom Python task files
pipeline.add_task("my_custom_task.py")

# Process files
pipeline.process_file("/path/to/data.raw", task="MyTask")
```

## Research Workflow & Usage

### Typical Research Workflow (v2.0.0)
1. **Setup Phase**: Interactive workspace setup wizard, drop Python task files into workspace
2. **Testing Phase**: Process single files to validate task quality and parameter tuning  
3. **Production Phase**: Use batch processing methods for full datasets
4. **Quality Review**: Examine results via review GUI and derivatives folder

### Task Design Philosophy (v2.0.0)
- **"Drop and Go" Approach**: Copy Python task files to workspace for instant availability
- **Embedded Configuration**: Task settings included directly in Python files
- **Simplified Workflow**: No separate YAML pipeline files to manage
- **Export Counter System**: Automatic stage numbering (01_, 02_, 03_) replacing stage_files
- **Easy Extension**: Custom mixins added by creating classes in mixins subfolders

### Workspace Management (v2.0.0)
- **Automatic Setup**: Interactive wizard creates workspace structure on first run
- **Task Discovery**: Automatically scans workspace/tasks/ folder for Python files
- **No JSON Tracking**: Pure filesystem-based task management
- **Cross-platform**: Uses platformdirs for proper OS-specific locations

### Common Challenges
- **Quality Failures**: Too many channels/epochs dropped (most common flagging reason)
- **New Dataset Support**: Special events/montages often require code changes
- **Complex Cases**: Pediatric HBCD data with atypical event handling requirements
- **API Migration**: v2.0.0 breaking changes require updating existing scripts

## Development Commands

### Code Quality & Testing
```bash
# Quick quality checks (recommended)
make check                      # Run all checks (format, lint, type)
make check-fix                  # Auto-fix formatting and linting issues
make format                     # Format code (black + isort)
make lint                       # Run linting (ruff + mypy)

# Testing
make test                       # Run unit tests
make test-cov                   # Run tests with coverage
make test-all                   # Run all tests (unit + integration)
make ci-check                   # Run CI-equivalent checks locally

# Single test execution
pytest tests/unit/test_python_task_files.py -v              # Run specific file
pytest tests/unit/test_synthetic_data.py::TestClass::test_method  # Run specific test
pytest tests/unit/ -k "test_pattern" -v                     # Pattern matching
pytest tests/unit/test_file.py -v -s --tb=short            # With debug output
```

### Installation & Setup
```bash
# Development setup (recommended)
make dev-setup                  # Complete dev environment setup
make install-dev                # Install development tools via uv
pip install -e .                # Install package in editable mode
pip install -e ".[gui]"         # Install with GUI dependencies

# Use as standalone CLI tool (via uv)
uv tool install autocleaneeg-pipeline              # From PyPI
uv tool run autocleaneeg-pipeline --help           # Run CLI
make install-uv-tool                               # Install from source
make uninstall-uv-tool                             # Uninstall tool
```

### CLI Usage
```bash
# Core commands
autocleaneeg-pipeline process RestingEyesOpen /path/to/data.raw    # Process file
autocleaneeg-pipeline list-tasks --overrides                       # Show tasks
autocleaneeg-pipeline review --output results/                     # Review GUI
autocleaneeg-pipeline export-access-log --output audit.jsonl       # Export audit log

# Theme and color control
autocleaneeg-pipeline --theme mono list-tasks                      # Monochrome output
AUTOCLEAN_THEME=hc autocleaneeg-pipeline version                   # High contrast
NO_COLOR=1 autocleaneeg-pipeline list-tasks                        # Disable colors
```

## Key File Locations
- **Core Logic**: `src/autoclean/core/` (Pipeline + Task base classes)
- **Processing Steps**: `src/autoclean/mixins/signal_processing/`
- **Built-in Tasks**: `src/autoclean/tasks/`
- **User Workspace**: `~/.autoclean/` or OS-specific user directory (v2.0.0)
- **Custom Tasks**: `workspace/tasks/` (Python files with embedded config)
- **Configuration**: Python task files with embedded `config` dicts (YAML pipeline configs removed)
- **Deployment**: `docker-compose.yml`, `autoclean.sh` (Linux), `profile.ps1` (Windows)

## Development Requirements
- Python 3.10-3.13 compatible
- MNE-Python ecosystem + scientific computing stack
- Entry point: `autocleaneeg-pipeline` CLI command
- Code style: Black (88 char), isort, ruff linting
- Type hints required (mypy strict mode currently disabled)
- Build backend: hatchling
- Test coverage target: >85%

## API Migration (v1.x → v2.0.0)
```python
# OLD (v1.4.1)
pipeline = Pipeline(
    autoclean_dir="/path/to/output",
    autoclean_config="config.yaml"
)

# NEW (v2.0.0)
pipeline = Pipeline(
    output_dir="/path/to/output"
)
```

## Audit Trail & Compliance Features

### Database Access Logging
AutoClean maintains a tamper-proof audit trail of all database operations for compliance and security monitoring. All database access is automatically logged to a write-only table with cryptographic integrity verification.

#### Features:
- **Tamper-Proof**: Database triggers prevent modification or deletion of audit records
- **Hash Chain Integrity**: Each log entry includes cryptographic hash of previous entry
- **User Context Tracking**: Captures username, hostname, PID, and timestamp for each operation
- **Comprehensive Coverage**: All database operations (create, read, update) are logged
- **Space Optimized**: Efficient storage format minimizes database overhead

#### Access Log Export
Export audit logs for compliance reporting and external analysis:

```bash
# Export all access logs to JSONL format
autocleaneeg-pipeline export-access-log --output audit-trail.jsonl

# Export with date filtering
autocleaneeg-pipeline export-access-log --start-date 2025-01-01 --end-date 2025-01-31 --output monthly-audit.jsonl

# Export specific operations only
autocleaneeg-pipeline export-access-log --operation "store" --output store-operations.jsonl

# Export to CSV for analysis
autocleaneeg-pipeline export-access-log --format csv --output audit-data.csv

# Human-readable report
autocleaneeg-pipeline export-access-log --format human --output audit-report.txt

# Verify integrity only (no export)
autocleaneeg-pipeline export-access-log --verify-only
```

#### Export Formats:
- **JSONL**: JSON Lines format with metadata header (default)
- **CSV**: Tabular format for spreadsheet analysis
- **Human**: Formatted text report for manual review

#### Security Features:
- **Integrity Verification**: Each export includes hash chain verification
- **Tamper Detection**: Identifies any attempts to modify audit records
- **Chain Validation**: Cryptographic verification of log sequence
- **Export Metadata**: Includes database path, entry count, and integrity status

#### Task File Tracking
For enhanced reproducibility and compliance, the system captures:
- **Source Code Hash**: SHA256 hash of task file used for each run
- **Full File Content**: Complete source code stored in database
- **File Metadata**: Path, size, line count, and capture timestamp
- **Version Tracking**: Links each run to specific task implementation

### Database Protection
- **Status-Based Locking**: Completed runs cannot be modified (prevents result tampering)
- **Automatic Backups**: Database backups created for significant operations
- **Trigger Protection**: SQL triggers prevent unauthorized data modification
- **Audit Trail**: All changes logged with user context and timestamps

## Current Status
- **Version**: 2.2.6 (Latest release - see pyproject.toml)
- **Production Ready**: Yes (85%+ test coverage, dependency locked)
- **PyPI Package**: `autocleaneeg-pipeline`
- **Python Support**: 3.10, 3.11, 3.12
- **Documentation**: https://cincibrainlab.github.io/autoclean_pipeline/

## Single Test Execution
```bash
# Run specific test file
pytest tests/unit/test_pipeline.py -v

# Run specific test method
pytest tests/unit/test_pipeline.py::TestPipeline::test_initialization -v

# Run tests matching pattern
pytest tests/unit/ -k "test_pipeline" -v

# Run with debugging output
pytest tests/unit/test_pipeline.py -v -s --tb=short
```

## Dependency Management
- **Production dependencies**: Specified in `pyproject.toml` dependencies section
- **GUI dependencies**: PyQt5, mne-qt-browser, PyMuPDF, pyjsonviewer, textual
- **Key dependencies**: MNE>=1.7.0, NumPy>=1.20.0, PyTorch>=1.9.0
- **Python version**: 3.10 to 3.12 supported (requires-python = ">=3.10,<3.14")

## Project Structure Overview
```
src/autoclean/
├── core/              # Pipeline orchestrator + Task base class
├── mixins/            # Dynamically combined processing components
│   ├── signal_processing/  # Filtering, ICA, epoching, artifacts
│   ├── visualization/      # Reports, plots, topography
│   ├── utils/             # BIDS handling, file operations
│   └── connectivity/      # Network analysis components
├── plugins/           # Auto-registered extensions
│   ├── eeg_plugins/       # Format + montage handlers
│   ├── event_processors/  # Task-specific event handling
│   └── formats/          # EEG file format support
├── tasks/            # Built-in EEG processing tasks
├── tools/            # GUI and CLI utilities
├── calc/             # Calculation modules (source, connectivity)
└── database/         # SQLite tracking with audit logging
```

## Common Git Operations
```bash
# Check current branch and status
git status

# View recent commits with diff from main branch
git log --oneline -10
git diff main...HEAD

# Push to remote (creates PR-ready branch)
git push -u origin feature/your-branch-name

# Create PR using GitHub CLI
gh pr create --title "Your PR title" --body "PR description"
```
