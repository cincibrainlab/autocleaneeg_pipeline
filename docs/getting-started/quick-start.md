# Quick Start Guide

This guide will help you get up and running with AutoClean Pipeline quickly. Follow these steps to process your first EEG dataset.

## Basic Usage Example

Here's a minimal example to process a single EEG file:

```python
from autoclean.core.pipeline import Pipeline

# Initialize the pipeline
pipeline = Pipeline(
    autoclean_dir="./autoclean_output"  # Where to store outputs
)

# Process a single file with the resting eyes open task
pipeline.process_file(
    file_path="path/to/your/eeg_file.raw",
    task="rest_eyesopen"
)

# Results will be available in the autoclean_output directory
```

## Step-by-Step Tutorial

### 1. Set Up Your Environment

If you haven't already installed AutoClean Pipeline, see the [Installation Guide](installation.md).

```bash
# Create a directory for your project
mkdir my_eeg_project
cd my_eeg_project

# Create a basic configuration file
touch autoclean_config.yaml
```

### 2. Create a Basic Configuration File

Edit `autoclean_config.yaml` with your preferred text editor and add:

```yaml
# Basic pipeline configuration
pipeline:
  # General settings
  eeg_system: "biosemi128"  # EEG system type
  sampling_rate: 1000       # Target sampling rate in Hz
  line_noise: 60            # Line noise frequency (50/60Hz)

  # Processing settings
  reference: "average"      # Reference type
  filter:
    highpass: 1.0           # Highpass filter in Hz
    lowpass: 40.0           # Lowpass filter in Hz
    notch: true             # Enable notch filter for line noise

# Task-specific settings will be loaded from the task definition
```

### 3. Process a Single File

Create a simple Python script `process_eeg.py`:

```python
from autoclean.core.pipeline import Pipeline

# List available tasks
pipeline = Pipeline(autoclean_config="autoclean_config.yaml")
print("Available tasks:", pipeline.list_tasks())

# Process a single file
pipeline.process_file(
    file_path="path/to/your_eeg_file.raw",
    task="rest_eyesopen"  # Choose a task from available tasks
)

# Output the location of the results
print(f"Processing complete. Results are in {pipeline.clean_dir}")
```

Run the script:

```bash
python process_eeg.py
```

### 4. Process Multiple Files

To process all EEG files in a directory:

```python
from autoclean.core.pipeline import Pipeline

pipeline = Pipeline(autoclean_config="autoclean_config.yaml")

# Process all .raw files in a directory
pipeline.process_directory(
    directory="path/to/data_directory",
    task="rest_eyesopen",
    pattern="*.raw",       # File pattern to match
    recursive=True         # Include subdirectories
)
```

### 5. Using Asynchronous Processing

For faster processing of multiple files:

```python
import asyncio
from autoclean.core.pipeline import Pipeline

async def main():
    pipeline = Pipeline(
        autoclean_config="autoclean_config.yaml",
        use_async=True
    )
    
    # Process multiple files concurrently
    await pipeline.process_directory_async(
        directory="path/to/data_directory",
        task="rest_eyesopen",
        max_concurrent=4  # Number of files to process simultaneously
    )
    
    print("All files processed!")

# Run the async function
asyncio.run(main())
```

### 6. View Results

After processing, you can examine the results:

```python
from autoclean.core.pipeline import Pipeline

pipeline = Pipeline(autoclean_dir="./autoclean_output")

# Launch the review interface
pipeline.start_autoclean_review()
```

## Common Workflows

### Resting-state EEG Processing

```python
pipeline.process_file(
    file_path="path/to/resting_eeg.raw",
    task="rest_eyesopen"
)
```

### Event-related Potential (ERP) Analysis

```python
pipeline.process_file(
    file_path="path/to/erp_data.raw",
    task="hbcd_mmn_v3"  # For mismatch negativity paradigm
)
```

### Frequency Response Analysis

```python
pipeline.process_file(
    file_path="path/to/assr_data.raw",
    task="assr_default"  # For auditory steady-state response
)
```

## Next Steps

- Learn about [detailed configuration options](configuration.md)
- Explore the [Pipeline Overview](../user-guide/pipeline-overview.md) for advanced features
- Check out [Troubleshooting](../user-guide/troubleshooting.md) if you encounter issues
- Understand how to create [custom tasks](../user-guide/tasks.md) for your specific needs
