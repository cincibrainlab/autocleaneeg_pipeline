# Installation Guide

This guide will help you install the AutoClean EEG Pipeline and its dependencies.

## Prerequisites

Before installing AutoClean Pipeline, ensure you have the following:

- **Python**: Version 3.8 or higher (3.10 recommended)
- **Operating System**: 
  - Linux (recommended for production use)
  - macOS (fully supported)
  - Windows (supported with some limitations)
- **Disk Space**: At least 2GB of free space for installation and dependencies
- **RAM**: Minimum 8GB, 16GB+ recommended for processing large datasets

## Installation Methods

### Method 1: Using pip (Recommended)

The simplest way to install AutoClean Pipeline is via pip:

```bash
pip install autoclean-pipeline
```

To install with optional dependencies for visualization:

```bash
pip install "autoclean-pipeline[viz]"
```

For development tools and testing dependencies:

```bash
pip install "autoclean-pipeline[dev]"
```

### Method 2: From Source

For the latest development version or to contribute:

```bash
# Clone the repository
git clone https://github.com/ernie-ecs/autoclean_pipeline.git
cd autoclean_pipeline

# Install in development mode
pip install -e .
```

### Method 3: Using Docker

A Docker image is available for containerized usage:

```bash
# Pull the image
docker pull ernie-ecs/autoclean-pipeline:latest

# Run a container
docker run -it --rm \
  -v /path/to/your/data:/data \
  -v /path/to/output:/output \
  ernie-ecs/autoclean-pipeline:latest
```

## Installing Dependencies

### MNE-Python

AutoClean Pipeline relies heavily on MNE-Python. While it's installed automatically as a dependency, you may want to install it separately to ensure proper configuration:

```bash
pip install mne
```

### PyQt5 (for GUI)

For the review interface, PyQt5 is required:

```bash
pip install PyQt5
```

### CUDA Support (Optional)

For GPU acceleration with compatible operations:

```bash
pip install cupy
```

## Verifying Installation

To verify that AutoClean Pipeline was installed correctly:

```python
import autoclean
print(autoclean.__version__)

# Test pipeline initialization
from autoclean.core.pipeline import Pipeline
pipeline = Pipeline()
print(pipeline.list_tasks())  # Should print available tasks
```

## Troubleshooting

### Common Installation Issues

1. **Missing Compiler**: Some dependencies require a C compiler. Install appropriate development tools:
   - Ubuntu/Debian: `sudo apt install build-essential`
   - macOS: `xcode-select --install`
   - Windows: Install Visual C++ Build Tools

2. **PyQt5 Issues**: If you encounter problems with PyQt5:
   - Linux: `sudo apt install python3-pyqt5`
   - macOS: `brew install pyqt5`

3. **Permission Errors**: Use a virtual environment or `pip install --user` to avoid permission issues.

## Next Steps

Once installation is complete, proceed to the [Quick Start Guide](quick-start.md) to begin processing your EEG data.

For development setup, see the [Contributing Guide](../contributing.md).
