# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoClean EEG is a modular framework for automated EEG data processing built on MNE-Python. It supports multiple EEG paradigms (ASSR, Chirp, MMN, Resting State) with BIDS-compatible data organization and database-backed processing tracking.

## Development Commands

### Code Quality
```bash
# Format code
black src/autoclean/
isort src/autoclean/

# Type checking
mypy src/autoclean/

# Linting
ruff check src/autoclean/

# Testing with coverage
pytest --cov=autoclean
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

### Documentation
```bash
# Build docs (from docs/ directory)
cd docs
make html

# Clean docs
make clean
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

## User Workflow & Usage Patterns

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

## Architecture Overview

### Core Components

1. **Pipeline** (`src/autoclean/core/pipeline.py`): Main orchestrator handling configuration, file processing, and result management
2. **Task** (`src/autoclean/core/task.py`): Abstract base class for all EEG processing tasks
3. **Mixins** (`src/autoclean/mixins/`): Reusable processing components dynamically combined into Task classes

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

### Configuration System
- **YAML-based**: Task-specific processing parameters in `configs/`
- **Hierarchical**: Tasks → Settings → Individual steps (resample, filter, ICA, etc.)
- **Stage Files**: Controls which intermediate files are saved
- **Multiple Configs**: Standard, HBCD, mouse-specific variants

### File Processing Pipeline
1. Import → Basic Steps → ICA → Epoching → Reports (minutes per run)
2. **Performance Bottlenecks**: ICA decomposition and RANSAC channel cleaning
3. **Quality Control**: Automatic flagging, comprehensive derivatives with visualizations
4. **Output Structure**: Full BIDS structure, logs, stage files, metadata JSON, QC reports
5. **Resumability**: Currently restart-only (resume capability was initial design goal)

### File Format Support
- **Primary**: .raw/.set formats (majority of datasets)
- **Required**: .edf/.mff support
- **Extensible**: Plugin system for new formats, montages, and event types

## Key File Locations

- **Core Logic**: `src/autoclean/core/` (Pipeline + Task base classes)
- **Processing Steps**: `src/autoclean/mixins/signal_processing/`
- **Task Implementations**: `src/autoclean/tasks/`
- **Configuration**: `configs/autoclean_config.yaml`
- **Deployment**: `docker-compose.yml`, `autoclean.sh` (Linux), `profile.ps1` (Windows)

## Testing & CI Notes

- Current testing: `.github/workflows/dev_test.py` for manual testing of tasks/formats
- No formal unit/integration tests yet - planned development area
- Test data: De-identified real EEG data available (files 150MB+), need CI-friendly approach
- Target: Mix of synthetic data for CI + real data for comprehensive testing
- PyQt5 dependency only needed for review GUI (not CI testing)
- Review GUI primarily runs via Docker due to PyQt5 platform issues

## Platform Support

- Target: Linux, Mac, Windows native support
- Primary deployment: Dockerized CLI for platform-agnostic runs
- Python versions: 3.10-3.12
- No GPU dependencies for core pipeline
- Heavy scientific dependencies (MNE, PyTorch) but standard CI handling

## Release Process

- GitHub releases (manual currently)
- Future goal: PyPI publishing
- CI should enforce: all checks pass before PR merge
- Quality gates: Standard professional linting/type checking

## Development Notes

- Use hatchling as build backend
- Python 3.10+ required, <3.13
- MNE-Python ecosystem + scientific computing stack
- Entry point: `autoclean` CLI command
- Extensive type hints required (mypy strict mode)
- Black formatting with 88 character line length
- pytest with coverage reporting

## Guidelines

- Do not add anything about claude in git commit messages or descriptions
```