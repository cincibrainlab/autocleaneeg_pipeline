# Configuration Guide

AutoClean Pipeline uses YAML configuration files to customize processing parameters. This guide explains the available configuration options and how to use them effectively.

## Configuration File Structure

The configuration file has several main sections:

```yaml
# Top-level pipeline settings
pipeline:
  # General pipeline parameters
  
task_settings:
  # Task-specific settings
  
step_functions:
  # Customization of individual processing steps
  
plugins:
  # Plugin configurations
```

## Basic Configuration Example

Here's a simple configuration file example:

```yaml
pipeline:
  eeg_system: "biosemi128"
  sampling_rate: 1000
  line_noise: 60
  reference: "average"
  filter:
    highpass: 1.0
    lowpass: 40.0
    notch: true
```

## Core Pipeline Settings

### General Settings

```yaml
pipeline:
  # EEG system configuration
  eeg_system: "biosemi128"  # Supports: biosemi64, biosemi128, egihydrocel128, egihydrocel256, etc.
  channel_types:            # Optional channel type mapping
    "EXG1": "eog"
    "EXG2": "eog"
    "EXG3": "ecg"
  
  # Preprocessing settings
  sampling_rate: 1000       # Target sampling rate in Hz
  line_noise: 60            # Line noise frequency (50/60Hz)
  reference: "average"      # Reference type: "average", "mastoid", "laplacian", or electrode name
  
  # File handling
  file_naming:
    prefix: "sub-"          # Prefix for output files
    suffix: "_proc"         # Suffix for output files
  
  # Processing behavior
  parallel_processing: true  # Enable multiprocessing
  max_workers: 4            # Maximum number of worker processes
  memory_limit: "8G"        # Memory limit per process
```

### Filter Settings

```yaml
pipeline:
  filter:
    highpass: 1.0           # Highpass filter cutoff in Hz
    lowpass: 40.0           # Lowpass filter cutoff in Hz
    notch: true             # Enable notch filter for line noise
    notch_width: 2.0        # Width of notch filter in Hz
    filter_order: 4         # Filter order (higher = steeper rolloff)
    filter_method: "fir"    # "fir" or "iir"
    zerophase: true         # Apply filter forward and backward
```

### Artifact Detection Settings

```yaml
pipeline:
  artifacts:
    bad_channel_detection:
      method: "autoreject"   # Method: "autoreject", "pyprep", "correlation"
      threshold: 0.7         # Correlation threshold (if using "correlation" method)
    
    muscle_detection:
      enable: true           # Enable muscle artifact detection
      threshold: 10          # Threshold for detection
    
    eye_detection:
      blink_detection: true  # Enable blink detection
      saccade_detection: true # Enable saccade detection
```

### ICA Settings

```yaml
pipeline:
  ica:
    enable: true             # Enable ICA processing
    method: "fastica"        # ICA method: "fastica", "infomax", "extended-infomax"
    components: 25           # Number of components to estimate (or "auto")
    random_state: 42         # Random seed for reproducibility
    
    # Automatic component rejection
    reject:
      enable: true           # Automatically reject components
      method: "correlation"  # Method: "correlation", "EOGcorr", "ICLabel"
      threshold: 0.8         # Rejection threshold
```

## Task-Specific Settings

Define settings for specific processing tasks:

```yaml
task_settings:
  rest_eyesopen:
    # Resting state specific settings
    epoch_length: 2.0        # Length of epochs in seconds
    overlap: 0.5             # Overlap between epochs
    min_segment_duration: 30 # Minimum recording duration in seconds
    
  assr_default:
    # ASSR paradigm settings
    trigger_channels:
      marker: "STI 014"      # Trigger channel name
    event_ids:
      "40Hz": 1              # Event ID for 40Hz stimuli
      "20Hz": 2              # Event ID for 20Hz stimuli
    tmin: -0.2               # Time before event in seconds
    tmax: 0.8                # Time after event in seconds
```

## Step Function Customization

Configure individual processing steps:

```yaml
step_functions:
  filtering:
    enable: true             # Enable this step
    parameters:              # Parameters specific to this step
      method: "custom"       # Override default method
  
  channel_interpolation:
    enable: true
    parameters:
      interpolation_type: "spline"
  
  epochs:
    enable: true
    parameters:
      baseline: [-0.2, 0]    # Baseline correction period
      reject_by_annotation: true
```

## Plugin System Configuration

If using plugins to extend pipeline functionality:

```yaml
plugins:
  # Example custom plugin configuration
  eeg_quality_metrics:
    enable: true
    threshold: 0.8
    methods: ["SNR", "alpha_power"]
  
  # Custom export plugin
  export_to_eeglab:
    enable: true
    include_intermediates: false
```

## Environment Variables in Configuration

You can use environment variables in your configuration file:

```yaml
pipeline:
  # Use environment variable with default value
  eeg_system: "${EEG_SYSTEM:biosemi128}"
  
  # Path configurations using environment variables
  paths:
    data_dir: "${DATA_DIR:/data}"
    output_dir: "${OUTPUT_DIR:/output}"
```

## Loading Configuration Files

### From Python

```python
from autoclean.core.pipeline import Pipeline

# Load configuration from file
pipeline = Pipeline(autoclean_config="path/to/config.yaml")

# OR specify configuration directly
from autoclean.utils.config import load_config
config = load_config("path/to/config.yaml")

# Override specific settings
config["pipeline"]["sampling_rate"] = 500
pipeline = Pipeline(config=config)
```

### Merging Multiple Configurations

You can combine multiple configuration files:

```python
from autoclean.utils.config import load_config, merge_configs

# Load base configuration
base_config = load_config("base_config.yaml")

# Load task-specific configuration
task_config = load_config("task_config.yaml")

# Merge configurations (task_config overrides base_config)
merged_config = merge_configs(base_config, task_config)

# Initialize pipeline with merged configuration
pipeline = Pipeline(config=merged_config)
```

## Configuration Validation

AutoClean Pipeline validates your configuration against a schema. If the configuration is invalid, a detailed error message will be provided.

Common validation errors:

- Missing required fields
- Type mismatches (e.g., string instead of number)
- Values outside of allowed ranges
- Unknown configuration keys

To validate a configuration file without running the pipeline:

```python
from autoclean.utils.config import validate_config

errors = validate_config("path/to/config.yaml")
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"- {error}")
```

## Next Steps

- See [Configuration Examples](../examples/configuration-examples.md) for specific use cases
- Learn about [Custom Tasks](../user-guide/tasks.md) for extending the pipeline
- Explore [Advanced Configuration](../user-guide/configuration-files.md) for more options
