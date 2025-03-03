# Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with the AutoClean Pipeline.

## Table of Contents
- [Installation Issues](#installation-issues)
  - [macOS Installation Issues](#macos-installation-issues)
  - [Linux Installation Issues](#linux-installation-issues)
  - [Windows Installation Issues](#windows-installation-issues)
- [Configuration Problems](#configuration-problems)
- [Data Processing Errors](#data-processing-errors)
- [Performance Issues](#performance-issues)
- [Common Error Messages](#common-error-messages)
- [Debug Tools](#debug-tools)
- [Getting Help](#getting-help)

## Installation Issues

### macOS Installation Issues

#### Missing libev dependency

**Issue**: When installing dependencies, you may encounter a build error with `bjoern` package related to missing `ev.h` file:
```
fatal error: 'ev.h' file not found
```

**Solution**: Install the required `libev` library using Homebrew:
```bash
brew install libev
```

After installing libev, retry your package installation command.

#### Missing pycairo and pkg-config dependencies

**Issue**: When installing dependencies, you may encounter a build error with `pycairo` package:
```
Dependency lookup for cairo with method 'pkgconfig' failed:
Pkg-config for machine host machine not found
```

**Solution**: Install the required `py3cairo` package using Homebrew:
```bash
brew install py3cairo pkg-config
```

After installing py3cairo, retry your package installation command.

#### PyQt5 Installation Issues

**Issue**: Problems installing or running PyQt5 on macOS:
```
ModuleNotFoundError: No module named 'PyQt5.sip'
```

**Solution**: Install PyQt5 using Homebrew instead of pip:
```bash
brew install pyqt5
```

Or reinstall with specific versions:
```bash
pip uninstall PyQt5 PyQt5-sip
pip install PyQt5==5.15.6 PyQt5-sip==12.9.1
```

### Linux Installation Issues

#### Missing System Libraries

**Issue**: Missing required system libraries for MNE and other dependencies:
```
error: lMNE Not Found
```

**Solution**: Install required development libraries:

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libffi-dev libssl-dev \
    libxml2-dev libxslt1-dev zlib1g-dev libhdf5-dev libpng-dev libjpeg-dev \
    libfreetype6-dev libopenblas-dev libgl1-mesa-glx
```

**CentOS/RHEL**:
```bash
sudo yum install -y gcc gcc-c++ python3-devel libffi-devel openssl-devel \
    libxml2-devel libxslt-devel zlib-devel hdf5-devel libpng-devel \
    libjpeg-devel freetype-devel openblas-devel mesa-libGL
```

#### CUDA Issues

**Issue**: CUDA-related errors when using GPU acceleration:
```
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
```

**Solution**: Ensure NVIDIA drivers and CUDA toolkit are properly installed:
```bash
# Check NVIDIA driver installation
nvidia-smi

# Install CUDA if needed (Ubuntu example)
sudo apt install nvidia-cuda-toolkit
```

### Windows Installation Issues

#### Visual C++ Build Tools

**Issue**: Missing compiler when installing packages with native extensions:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution**: Install Visual C++ Build Tools:
1. Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select "C++ build tools" during installation
3. Reinstall the package: `pip install autoclean-pipeline`

#### Path Length Limitations

**Issue**: Errors related to long file paths during installation:
```
WindowsError: [Error 206] The filename or extension is too long
```

**Solution**: Enable long path support in Windows 10/11:
1. Run regedit as administrator
2. Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart your computer

## Configuration Problems

### Invalid Configuration File

**Issue**: Pipeline fails to start with configuration validation errors:
```
ValidationError: 'invalid_value' is not one of [...] at /pipeline/eeg_system
```

**Solution**:
1. Check the error message for the specific validation problem
2. Verify allowed values in the [Configuration Guide](configuration-files.md)
3. Use the configuration validation utility:
```python
from autoclean.utils.config import validate_config
errors = validate_config("path/to/config.yaml")
for error in errors:
    print(error)
```

### Missing Configuration File

**Issue**: Pipeline can't find the specified configuration file:
```
FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'
```

**Solution**:
1. Check if the file exists at the specified path
2. Use absolute paths instead of relative paths
3. Create a default configuration if needed:
```python
from autoclean.utils.config import create_default_config
create_default_config("config.yaml")
```

### Environment Variable Issues

**Issue**: Configuration with environment variables not working:
```
KeyError: 'DATA_DIR'
```

**Solution**:
1. Ensure environment variables are set before running the pipeline
2. Use default values in your configuration: `${DATA_DIR:/default/path}`
3. Set environment variables in your script:
```python
import os
os.environ['DATA_DIR'] = '/path/to/data'
```

## Data Processing Errors

### Input File Format Issues

**Issue**: Pipeline fails to load EEG data:
```
ValueError: Raw file format not supported
```

**Solution**:
1. Check if the file format is supported (FIF, EDF, BDF, CNT, SET, etc.)
2. Convert the file to a supported format using MNE:
```python
import mne
raw = mne.io.read_raw_edf('file.edf')
raw.save('file.fif', overwrite=True)
```

### Memory Errors

**Issue**: Pipeline crashes with memory errors during processing:
```
MemoryError: Unable to allocate array with shape (n, m)
```

**Solution**:
1. Reduce the memory usage by adjusting configuration:
```yaml
pipeline:
  memory_optimization: true
  max_memory_usage: "8G"
```
2. Process smaller chunks of data:
```python
pipeline.process_file(
    file_path="large_file.raw",
    task="rest_eyesopen",
    chunk_size="5m"  # Process in 5-minute chunks
)
```

### Bad Channel Detection Issues

**Issue**: Too many or too few channels marked as bad:
```
Warning: 25% of channels marked as bad, exceeding threshold
```

**Solution**:
1. Adjust bad channel detection parameters:
```yaml
pipeline:
  artifacts:
    bad_channel_detection:
      method: "correlation"  # Try different methods
      threshold: 0.8  # Adjust threshold (higher = fewer bad channels)
```
2. Pre-mark known bad channels to exclude them from automatic detection:
```yaml
pipeline:
  bad_channels: ["Fp1", "F8"]  # Pre-mark these channels as bad
```

### ICA Issues

**Issue**: ICA decomposition fails or produces poor results:
```
Warning: ICA did not converge
```

**Solution**:
1. Adjust ICA parameters:
```yaml
pipeline:
  ica:
    method: "extended-infomax"  # Try different ICA method
    n_components: 30  # Adjust number of components
    max_iter: 1000  # Increase maximum iterations
    random_state: 42  # Set fixed random seed for reproducibility
```
2. Apply stronger filtering before ICA:
```yaml
pipeline:
  filter:
    highpass: 1.0  # Increase highpass filter cutoff
```

## Performance Issues

### Slow Processing

**Issue**: Pipeline processing is taking too long:

**Solution**:
1. Enable parallel processing:
```yaml
pipeline:
  parallel_processing: true
  max_workers: 4  # Adjust based on CPU cores
```
2. Reduce processing steps:
```yaml
step_functions:
  time_frequency_analysis:
    enable: false  # Disable expensive steps
```
3. Optimize file I/O:
```yaml
pipeline:
  use_memory_mapping: true
  disk_io_optimization: true
```

### GPU Acceleration Issues

**Issue**: GPU acceleration not working:

**Solution**:
1. Verify CUDA installation:
```python
import torch
print(torch.cuda.is_available())  # Should print True
```
2. Enable GPU acceleration in config:
```yaml
pipeline:
  use_gpu: true
  gpu_device: 0  # Specify GPU device index
```

## Common Error Messages

### "Cannot import name x from y"

**Issue**: Missing or incompatible dependencies:
```
ImportError: cannot import name 'create_info' from 'mne'
```

**Solution**:
1. Check installed package versions:
```bash
pip list | grep mne
```
2. Reinstall with specific versions:
```bash
pip uninstall mne
pip install mne==1.0.3
```

### "Raw data not loaded"

**Issue**: Operations attempted before loading data:
```
RuntimeError: Raw data not loaded. Call load_raw() first.
```

**Solution**:
Ensure proper pipeline execution order:
```python
from autoclean.step_functions.io import load_raw

# Load data first
load_raw(pipeline)

# Then proceed with other operations
preprocessing.apply_filter(pipeline)
```

### "Directory x does not exist"

**Issue**: Missing directories for outputs:
```
FileNotFoundError: Directory 'output/clean' does not exist
```

**Solution**:
1. Create directories manually:
```python
import os
os.makedirs("output/clean", exist_ok=True)
```
2. Allow pipeline to create directories:
```yaml
pipeline:
  create_dirs: true
```

## Debug Tools

### Enabling Debug Mode

Enable detailed debugging output:

```python
from autoclean.core.pipeline import Pipeline

pipeline = Pipeline(debug=True)
```

Or in configuration:
```yaml
pipeline:
  debug: true
  log_level: "DEBUG"
```

### Log Files

Examine log files for detailed error information:

```bash
cat ~/.autoclean/logs/autoclean.log
```

Or programmatically:
```python
with open(pipeline.log_file, 'r') as f:
    log_contents = f.read()
    print(log_contents)
```

### Interactive Debugging

Launch interactive debugger for troubleshooting:

```python
pipeline = Pipeline()
pipeline.debug_mode = True
pipeline.interactive_debug()
```

### Validation Tools

Validate your pipeline setup:

```python
# Check configuration
pipeline.validate_config()

# Check input file
pipeline.validate_file("eeg_data.raw")

# Check task configuration
pipeline.validate_task("rest_eyesopen")
```

## Getting Help

If you've tried the troubleshooting steps and still have issues:

1. **Check Documentation**: Review the [API Reference](../api-reference/pipeline.md)
2. **Search Issues**: Check [GitHub Issues](https://github.com/ernie-ecs/autoclean_pipeline/issues)
3. **Ask for Help**: Post a question with:
   - Error message and traceback
   - Configuration file (remove sensitive data)
   - Python and package version information
   - Code sample reproducing the issue
   - Operating system and environment details
