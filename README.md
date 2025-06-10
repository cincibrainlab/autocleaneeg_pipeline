# AutoClean EEG

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
- Detailed output: logs, stage files, metadata, and quality control visualizations

## Installation

```bash
pip install autoclean-eeg
```

For development installation:

```bash
git clone https://github.com/cincibrainlab/autoclean_pipeline.git
cd autoclean-eeg
uv tool install -e --upgrade ".[dev]"
```

## Quick Start

AutoClean EEG offers two approaches for building custom EEG processing workflows:

### Option 1: Python Task Files (Recommended for New Users)

Create simple Python files that combine configuration and processing logic:

```python
# my_task.py
from typing import Any, Dict
from autoclean.core.task import Task

# Embedded configuration
config = {
    'resample_step': {'enabled': True, 'value': 250},
    'filtering': {'enabled': True, 'value': {'l_freq': 1, 'h_freq': 100}},
    'ICA': {'enabled': True, 'value': {'method': 'picard'}},
    'epoch_settings': {'enabled': True, 'value': {'tmin': -1, 'tmax': 1}}
}

class MyRestingTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self) -> None:
        self.import_raw()
        self.run_basic_steps(export=True)
        self.run_ica(export=True)
        self.create_regular_epochs(export=True)
```

```python
# Use your custom task
from autoclean import Pipeline

pipeline = Pipeline(autoclean_dir="/path/to/output")
pipeline.add_task("my_task.py")
pipeline.process_file("/path/to/data.raw", task="MyRestingTask")
```

### Option 2: Traditional YAML Configuration

For complex workflows or when you prefer separate config files:

```python
from autoclean import Pipeline

# Initialize pipeline with YAML configuration
pipeline = Pipeline(
    autoclean_dir="/path/to/output",
    autoclean_config="configs/autoclean_config.yaml"
)

# Process using built-in tasks
pipeline.process_file(
    file_path="/path/to/test_data.raw", 
    task="rest_eyesopen"
)
```

### Typical Research Workflow

1. **Test single file** to validate task and tune parameters
2. **Review results** in output directories and adjust as needed  
3. **Process full dataset** using batch processing

```python
# Batch processing (works with both approaches)
pipeline.process_directory(
    directory="/path/to/dataset",
    task="MyRestingTask",  # or built-in task name
    pattern="*.raw"
)
```

**Key Benefits of Python Task Files:**
- **Simpler**: No separate YAML files to manage
- **Self-contained**: Configuration and logic in one file
- **Flexible**: Optional `export=True` parameters control file outputs
- **Intuitive**: Pandas-like API with sensible defaults

## Documentation

Full documentation is available at [https://cincibrainlab.github.io/autoclean_pipeline/](https://cincibrainlab.github.io/autoclean_pipeline/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{autoclean_eeg,
  author = {Gammoh, Gavin, Pedapati, Ernest, and Grace Westerkamp},
  title = {AutoClean EEG: Automated EEG Processing Pipeline},
  year = {2024},
  publisher = {GitHub},
  url = {[https://github.com/yourusername/autoclean-eeg](https://github.com/cincibrainlab/autoclean_pipeline/)}
}
```

## Acknowledgments

- Cincinnati Children's Hospital Medical Center
- Built with [MNE-Python](https://mne.tools/)

## Docker Usage

For platform-agnostic execution and GUI components (especially the review interface), the pipeline can be run in a containerized environment:

### Windows PowerShell

```powershell
# Add the autoclean command to your PowerShell profile
Copy-Item profile.ps1 $PROFILE
# or add to existing profile
. "C:\path\to\autoclean.ps1"

# Run the pipeline
autoclean -DataPath "C:\Data\raw" -Task "RestingEyesOpen" -ConfigPath "C:\configs\autoclean_config.yaml"

# View help
autoclean -Help
```

### Linux/WSL/Mac

```bash
# Add the autoclean command to your system
# Create directory if it doesn't exist
mkdir -p ~/.local/bin

# Copy script to this directory
cp autoclean.sh ~/.local/bin/autoclean

# Make it executable
chmod +x ~/.local/bin/autoclean

# Most modern Linux distributions automatically include ~/.local/bin in PATH
# If it's not in your PATH, add it to your shell configuration:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc


# Run the pipeline
autoclean -DataPath "/path/to/data" -Task "RestingEyesOpen" -ConfigPath "/path/to/config.yaml"

# View help
autoclean --help
```

### Command Parameters

- `-DataPath`: Directory containing raw EEG data or path to single data file
- `-Task`: Processing task type (RestingEyesOpen, ASSR, ChirpDefault, etc.)
- `-ConfigPath`: Path to configuration YAML file
- `-OutputPath`: (Optional) Output directory, defaults to "./output"

### Requirements

- Docker Desktop installed and running
- Docker Compose v2 or higher
- PowerShell 5.1+ (Windows) or Bash (Linux/WSL)

### Notes

- Windows paths are automatically converted to the appropriate format in WSL
- The script handles mounting volumes and setting environment variables
- Output directories are created automatically if they don't exist
- All paths can be absolute or relative
