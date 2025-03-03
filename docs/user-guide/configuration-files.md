# Configuration Files

Configuration files are a central part of the AutoClean Pipeline, allowing you to customize processing parameters without modifying code. This guide explains how to effectively use and create configuration files.

## Configuration File Format

AutoClean Pipeline uses YAML (Yet Another Markup Language) for configuration files due to its readability and hierarchical structure. YAML files can represent complex nested configurations in a human-readable format.

### Basic YAML Syntax

```yaml
# This is a comment in YAML

# Key-value pairs (scalar values)
name: "AutoClean Pipeline"
version: 2.1.0
enable_feature: true
sampling_rate: 1000

# Nested objects (mappings)
filter:
  highpass: 1.0
  lowpass: 40.0
  notch: true

# Lists (sequences)
processing_steps:
  - "filter"
  - "ica"
  - "epoching"

# Lists of objects
channels:
  - name: "Fz"
    type: "eeg"
  - name: "Cz"
    type: "eeg"
  - name: "ECG"
    type: "ecg"
```

## Configuration File Structure

The AutoClean Pipeline configuration file is organized into several main sections:

```yaml
# General pipeline configuration
pipeline:
  # General settings
  # ...

# Task-specific settings
task_settings:
  task_name:
    # Task-specific parameters
    # ...

# Step function customization
step_functions:
  # Step function parameters
  # ...

# Plugin configurations
plugins:
  # Plugin parameters
  # ...
```

## Loading Configuration Files

Configuration files are loaded when initializing the Pipeline object:

```python
from autoclean.core.pipeline import Pipeline

# Initialize pipeline with config file
pipeline = Pipeline(autoclean_config="path/to/config.yaml")
```

## Configuration Inheritance and Merging

The AutoClean Pipeline supports configuration inheritance and merging from multiple sources:

1. **Default Configuration**: Built-in defaults for all settings
2. **Global Configuration**: System-wide configuration (if present)
3. **User Configuration**: User-specific configuration file
4. **Project Configuration**: Project-specific configuration file
5. **Task Configuration**: Task-specific configuration
6. **Runtime Configuration**: Parameters specified when initializing the pipeline

Later sources override earlier ones, allowing for flexible configuration layering.

### Configuration Loading Order

1. Load built-in defaults
2. Load from `$AUTOCLEAN_GLOBAL_CONFIG` (if set)
3. Load from user's home directory (`~/.autoclean/config.yaml`)
4. Load from project directory
5. Load from specified `autoclean_config` parameter
6. Apply runtime parameters

## Common Configuration Sections

### General Pipeline Settings

```yaml
pipeline:
  # EEG system info
  eeg_system: "biosemi128"
  montage: "standard_1020"
  channel_types:
    "EXG1": "eog"
    "EXG2": "eog"
    "EXG3": "ecg"
  
  # General processing settings
  sampling_rate: 1000
  line_noise: 60
  reference: "average"
  
  # Output settings
  output_format: "fif"
  save_intermediate: false
```

### Filter Settings

```yaml
pipeline:
  filter:
    # Frequency filters
    highpass: 1.0
    lowpass: 40.0
    
    # Notch filter for line noise
    notch: true
    notch_freqs: [60, 120]
    
    # Filter parameters
    filter_method: "fir"
    filter_order: 4
    filter_transition_bandwidth: "auto"
```

### ICA Settings

```yaml
pipeline:
  ica:
    # ICA parameters
    enable: true
    method: "fastica"
    n_components: 20
    random_state: 42
    
    # Component rejection
    auto_reject: true
    rejection_criteria:
      correlation_threshold: 0.8
      z_threshold: 2.5
```

### Artifact Detection

```yaml
pipeline:
  artifacts:
    # Bad channel detection
    bad_channel_detection:
      method: "correlation"
      threshold: 0.7
    
    # Artifact detection
    detect_muscle: true
    detect_blinks: true
    detect_heartbeat: true
```

### Task-Specific Settings

```yaml
task_settings:
  # Resting state task
  rest_eyesopen:
    epoch_length: 2.0
    overlap: 0.5
    min_segment_duration: 30
    
  # Event-related task
  mmn:
    event_id:
      standard: 1
      deviant: 2
    tmin: -0.2
    tmax: 0.5
    baseline: [-0.2, 0]
```

## Environment Variables in Configuration

You can use environment variables in your configuration files for flexibility:

```yaml
pipeline:
  # Use environment variable with default
  eeg_system: "${EEG_SYSTEM:biosemi128}"
  
  # Paths using environment variables
  data_dir: "${DATA_DIR:/data}"
  output_dir: "${OUTPUT_DIR:/output}"
```

Variables are specified as `${VARIABLE_NAME:default_value}`. If the environment variable is not set, the default value is used.

## Configuration Schema Validation

AutoClean Pipeline validates all configuration files against a schema to ensure correctness. The schema defines:

- Required fields
- Field types and allowed values
- Default values
- Field dependencies

### Schema Example

```python
PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "eeg_system": {
            "type": "string",
            "enum": ["biosemi128", "biosemi64", "egihydrocel128", "egihydrocel256"]
        },
        "sampling_rate": {
            "type": "number",
            "minimum": 100,
            "maximum": 5000
        },
        "line_noise": {
            "type": "number",
            "enum": [50, 60]
        }
    },
    "required": ["eeg_system"]
}
```

### Validation Errors

If your configuration file has errors, the pipeline will provide detailed error messages:

```
ValidationError: 'biosemi129' is not one of ['biosemi128', 'biosemi64', 'egihydrocel128', 'egihydrocel256'] at /pipeline/eeg_system
```

## Configuration Profiles

You can create multiple configuration profiles for different scenarios:

```yaml
# Base configuration that applies to all profiles
pipeline:
  eeg_system: "biosemi128"
  
# Profile-specific configurations
profiles:
  development:
    pipeline:
      debug: true
      save_intermediate: true
  
  production:
    pipeline:
      debug: false
      save_intermediate: false
      parallel_processing: true
```

To use a specific profile:

```python
pipeline = Pipeline(
    autoclean_config="config.yaml",
    profile="production"
)
```

## Parameterized Configurations

You can create parameterized configurations with placeholders:

```yaml
# Template configuration
pipeline:
  eeg_system: "{{EEG_SYSTEM}}"
  sampling_rate: {{SAMPLING_RATE}}
  
task_settings:
  {{TASK_NAME}}:
    # Task settings
```

And fill them at runtime:

```python
from autoclean.utils.config import load_template_config

config = load_template_config(
    "template_config.yaml",
    params={
        "EEG_SYSTEM": "biosemi128",
        "SAMPLING_RATE": 1000,
        "TASK_NAME": "rest_eyesopen"
    }
)
```

## Best Practices

### Organizing Configuration Files

1. **Split configurations by purpose**:
   - `base_config.yaml`: Common settings
   - `task_config.yaml`: Task-specific settings
   - `system_config.yaml`: Hardware-specific settings

2. **Use meaningful names and hierarchical structure**:
   ```yaml
   pipeline:
     preprocessing:
       filtering:
         highpass: 1.0
   ```

3. **Add comments for clarity**:
   ```yaml
   pipeline:
     # Sampling rate in Hz
     sampling_rate: 1000
   ```

### Version Control

1. **Include example configurations in source control**:
   - `config.example.yaml`
   - `dev_config.example.yaml`

2. **Exclude actual configurations with sensitive data**:
   - Add actual config files to `.gitignore`

3. **Document configuration changes**:
   - Include configuration changes in changelog

## Common Pitfalls

1. **Indentation Errors**: YAML is indentation-sensitive
2. **Type Mismatches**: Using strings when numbers are required
3. **Missing Required Fields**: Not specifying mandatory parameters
4. **Spelling Errors**: Misspelling configuration keys

## Advanced Configuration Techniques

### Dynamic Configurations

For complex scenarios, you can generate configurations programmatically:

```python
import yaml
from pathlib import Path

# Generate configuration dynamically
config = {
    "pipeline": {
        "eeg_system": "biosemi128",
        "sampling_rate": 1000
    },
    "task_settings": {
        "custom_task": {
            "parameters": [1, 2, 3, 4]
        }
    }
}

# Save to file
with open(Path("generated_config.yaml"), "w") as f:
    yaml.dump(config, f, default_flow_style=False)
```

### Configuration Utilities

AutoClean provides utilities for working with configurations:

```python
from autoclean.utils.config import (
    load_config,
    merge_configs,
    validate_config,
    find_config_file
)

# Load multiple configs and merge
base_config = load_config("base_config.yaml")
task_config = load_config("task_config.yaml")
merged_config = merge_configs(base_config, task_config)

# Validate configuration
errors = validate_config(merged_config)
for error in errors:
    print(f"Error: {error}")

# Find configuration file in standard locations
config_path = find_config_file("my_config.yaml")
```

## Next Steps

- See [Configuration Examples](../examples/configuration-examples.md) for real-world examples
- Learn about [Task Configuration](tasks.md) for task-specific settings
- Explore the [API Reference](../api-reference/utils.md#config) for configuration utilities
