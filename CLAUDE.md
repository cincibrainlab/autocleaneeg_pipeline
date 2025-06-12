# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Commit Guidelines
- DO NOT add anything about claude in git commit messages or descriptions

## Project Overview
AutoClean EEG is a modular framework for automated EEG data processing built on MNE-Python. It supports multiple EEG paradigms (ASSR, Chirp, MMN, Resting State) with BIDS-compatible data organization and database-backed processing tracking.

## Core Architecture
- **Modular Design**: "Lego Block" approach for task composition
- **Dynamic Mixins**: Automatically discover and combine all "*Mixin" classes
- **Plugin System**: Auto-registration for EEG formats, montages, and event processors
- **YAML Configuration**: Hierarchical task-specific processing parameters

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

### Task Implementation Pattern
```python
class NewTask(Task):  # Inherits all mixins automatically
    def __init__(self, config): 
        self.required_stages = ["post_import", "post_clean_raw"]
        super().__init__(config)
    
    def run(self):
        self.import_raw()           # From base
        self.run_basic_steps()      # From mixins
        self.run_ica()             # From mixins
        self.create_regular_epochs() # From mixins
```

## Research Workflow & Usage

### Typical Research Workflow
1. **Setup Phase**: Create Python scripts with `Pipeline()` object, modify YAML configs for new tasks
2. **Testing Phase**: Process single files to validate task quality and parameter tuning
3. **Production Phase**: Use batch processing methods for full datasets
4. **Quality Review**: Examine results via review GUI datascroll and derivatives folder

### Task Design Philosophy
- **"Lego Block" Approach**: Users call `self.function()` in task scripts for simple workflows
- **High Customization**: Extensive parameters and function choices available
- **Manual Task Selection**: Users need domain knowledge to choose appropriate tasks
- **Easy Extension**: Custom mixins added by creating classes in mixins subfolders

### Common Challenges
- **Quality Failures**: Too many channels/epochs dropped (most common flagging reason)
- **New Dataset Support**: Special events/montages often require code changes
- **Complex Cases**: Pediatric HBCD data with atypical event handling requirements

## Development Commands

### Tool Installation (uv tool - Recommended)
```bash
# Install all development tools in isolated environments
python scripts/install_dev_tools.py

# Or install directly with uv
python scripts/uv_tools.py install

# List installed tools
python scripts/uv_tools.py list

# Upgrade all tools
python scripts/uv_tools.py upgrade
```

### Code Quality (uv tool)
```bash
# Run all quality checks (uses uv tool automatically)
python scripts/check_code_quality.py
python scripts/check_code_quality.py --fix    # Auto-fix issues

# Individual tools via uv
python scripts/uv_tools.py run black src/autoclean/
python scripts/uv_tools.py run isort src/autoclean/
python scripts/uv_tools.py run ruff check src/autoclean/
python scripts/uv_tools.py run mypy src/autoclean/

# Makefile commands (uses uv tool)
make format                    # Auto-format code
make lint                      # Run linting
make check                     # Run all checks
```

### Code Quality (Fallback - Direct Commands)
```bash
# If uv is not available, use direct commands
python scripts/check_code_quality.py --no-uv

# Or direct tool usage
black src/autoclean/
isort src/autoclean/
ruff check src/autoclean/
mypy src/autoclean/
```

### Testing
```bash
# Testing with coverage
pytest --cov=autoclean

# Run specific test suites
pytest tests/unit/                    # Unit tests only
pytest tests/integration/            # Integration tests only  
pytest tests/unit/ -k "test_pipeline" # Specific test patterns
```

### Build and Installation
```bash
# Development installation
pip install -e .

# With GUI dependencies
pip install -e ".[gui]"

# Build package
python -m build
```

### Using AutoClean as a uv tool (Recommended for Users)
```bash
# Install AutoClean as a standalone uv tool
uv tool install .                    # From source (development)
uv tool install autocleaneeg         # From PyPI (when published)

# Use AutoClean CLI (isolated environment, no conflicts!)
uv tool run autoclean --help
uv tool run autoclean process --task RestingEyesOpen --file data.raw --output results/
uv tool run autoclean list-tasks
uv tool run autoclean review --output results/

# Manage AutoClean tool
uv tool list                         # Show installed tools
uv tool upgrade autocleaneeg         # Upgrade AutoClean
uv tool uninstall autocleaneeg       # Remove AutoClean

# Makefile shortcuts
make install-uv-tool                 # Install AutoClean as uv tool
make uninstall-uv-tool               # Uninstall AutoClean uv tool
```

### Docker Development
```bash
# Build and run pipeline
docker-compose up autoclean

# Run review GUI
docker-compose up review

# Shell access
docker-compose run autoclean bash
```

## Key File Locations
- **Core Logic**: `src/autoclean/core/` (Pipeline + Task base classes)
- **Processing Steps**: `src/autoclean/mixins/signal_processing/`
- **Task Implementations**: `src/autoclean/tasks/`
- **Configuration**: `configs/autoclean_config.yaml`
- **Deployment**: `docker-compose.yml`, `autoclean.sh` (Linux), `profile.ps1` (Windows)

## CI/CD Pipeline
- **Matrix testing**: Python 3.10-3.12 across Ubuntu/macOS/Windows
- **Code quality**: black, isort, ruff, mypy 
- **Security**: bandit, pip-audit
- **Testing**: pytest with coverage reporting
- **Performance benchmarking**: `.github/workflows/benchmark.yml`
- **Synthetic EEG data generation** for realistic testing
- Fast CI execution targeting <15 minute runs

## Development Notes
- Python 3.10+ required, <3.13
- MNE-Python ecosystem + scientific computing stack
- Entry point: `autoclean` CLI command
- Extensive type hints required (mypy strict mode)
- Black formatting with 88 character line length
- pytest with coverage reporting
- Use hatchling as build backend