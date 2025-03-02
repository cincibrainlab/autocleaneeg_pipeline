# EEG Pipeline Architecture

## Overview

The EEG pipeline is designed to provide a flexible and comprehensive framework for processing EEG data. It supports various EEG paradigms and ensures that the processing steps are modular, configurable, and can be integrated into automated agents for regulated EEG processing.

## Core Components

### Pipeline Class

The `Pipeline` class is the primary interface for the EEG processing workflow. It manages the entire process, including configuration loading, directory setup, task execution, and results management.

#### Initialization

```python
from autoclean.core.pipeline import Pipeline

pipeline = Pipeline(
    autoclean_dir="path/to/output",     # Root directory for all processing outputs
    autoclean_config="path/to/config",   # Path to YAML configuration file
    use_async=False                      # Optional: Enable async processing (default: False)
)
```

#### Task Registry

The pipeline supports multiple tasks, each corresponding to a specific EEG processing paradigm. The available tasks are registered in the `TASK_REGISTRY`.

```python
TASK_REGISTRY = {
    'rest_eyesopen': RestingEyesOpen,
    'assr_default': AssrDefault,
    'chirp_default': ChirpDefault,
    'mouse_xdat_resting': MouseXdatResting,
    'hbcd_mmn_v3': HBCD_MMN,
}
```

## Processing Methods

### Single File Processing

The pipeline can process individual EEG files using the `process_file` method.

```python
pipeline.process_file(
    file_path="path/to/file.raw",    # Path to EEG data file
    task="task_name",                # Name of processing task
    run_id=None                      # Optional: Unique identifier for the run
)
```

### Directory Processing

The pipeline can process all files in a directory using the `process_directory` method.

```python
pipeline.process_directory(
    directory="path/to/dir",         # Directory containing EEG files
    task="task_name",                # Name of processing task
    pattern="*.raw",                 # Optional: File pattern to match (default: "*.raw")
    recursive=False                  # Optional: Search subdirectories (default: False)
)
```

### Asynchronous Directory Processing

The pipeline supports asynchronous processing of directories using the `process_directory_async` method.

```python
await pipeline.process_directory_async(
    directory="path/to/dir",         # Directory containing EEG files
    task="task_name",                # Name of processing task
    pattern="*.raw",                 # Optional: File pattern to match (default: "*.raw")
    recursive=False,                 # Optional: Search subdirectories (default: False)
    max_concurrent=5                 # Optional: Max concurrent processes (default: 5)
)
```

## Utility Methods

### List Available Tasks

The pipeline can list all available tasks using the `list_tasks` method.

```python
pipeline.list_tasks()                # Returns list of configured tasks
```

### List Stage Files

The pipeline can list all configured stage file types using the `list_stage_files` method.

```python
pipeline.list_stage_files()          # Returns list of configured stage file types
```

### Start Review GUI

The pipeline can launch the review interface using the `start_autoclean_review` method.

```python
pipeline.start_autoclean_review()    # Launches the autoclean review interface
```

## Validation Methods

### Validate Task Configuration

The pipeline can validate the configuration of a specific task using the `validate_task` method.

```python
pipeline.validate_task(task="task_name")
```

### Validate Input File

The pipeline can validate the input EEG file using the `validate_file` method.

```python
pipeline.validate_file(file_path="path/to/file")
```

### Validate Base Configuration

The pipeline can validate the base configuration using the `validate_base_config` method.

```python
pipeline.validate_base_config(run_id="run_id")
```

## Run Configuration

The pipeline generates a run configuration dictionary containing:

```python
run_dict = {
    "run_id": str,                    # Unique run identifier
    "task": str,                      # Task name
    "eeg_system": str,                # EEG system type
    "config_file": Path,              # Config file path
    "stage_files": dict,              # Stage file configuration
    "unprocessed_file": Path,         # Input file path
    "autoclean_dir": Path,           # Root output directory
    "bids_dir": Path,                # BIDS directory
    "metadata_dir": Path,            # Metadata directory
    "clean_dir": Path,               # Cleaned data directory
    "debug_dir": Path,               # Debug output directory
    "stage_dir": Path,               # Stage file directory
}
```

## Database Integration

The pipeline maintains a database to track:
- Run IDs and status
- Processing metadata
- Error information
- File locations

Each run generates:
- JSON metadata file: `{input_file_stem}_autoclean_metadata.json`
- PDF report file: `{input_file_stem}_autoclean_report.pdf`

## Error Handling

The pipeline implements comprehensive error handling:
- Validates all inputs before processing
- Tracks errors in database
- Generates error reports
- Continues processing remaining files in batch operations
- Provides detailed error messages and logging

## Directory Structure

The pipeline automatically creates and manages:

```
autoclean_dir/
├── bids/           # BIDS-formatted data
├── metadata/       # Processing metadata
├── clean/          # Cleaned output data
├── debug/          # Debug information
├── stage/          # Intermediate files
└── script/         # Processing scripts
```

## Key Attributes

- `pipeline.raw`: Current state of MNE Raw object
- `pipeline.ica`: ICA object for artifact removal
- `pipeline.ica2`: Secondary ICA object (if used)
- `pipeline.flags`: Dictionary storing processing flags/markers
- `pipeline.bids_path`: BIDS-formatted path object
- `pipeline.derivatives_path`: Path to derivatives directory
- `pipeline.config`: Configuration dictionary
- `pipeline.metadata`: Processing metadata dictionary

## Directory Structure

### Input/Output Paths

- `raw_path`: Original data location
- `derivatives_path`: Processed data output
- `metadata_path`: Pipeline metadata storage
- `debug_path`: Debug information and logs
- `temp_path`: Temporary processing files

### BIDS Organization

```
derivatives/
├── sub-<participant>/
│   └── ses-<session>/
│       └── eeg/
│           ├── sub-<participant>_ses-<session>_task-<task>_desc-<desc>_eeg.set
│           └── sub-<participant>_ses-<session>_task-<task>_desc-<desc>_eeg.fdt
```

## Processing Steps

### Data Import

- Supports multiple formats (EGI, BrainVision, EEGLab)
- Handles channel type assignment
- Sets up montage information
- Validates data structure

### Preprocessing

- Resampling
- Filtering
- Bad channel detection
- Channel interpolation
- Re-referencing
- ICA-based artifact removal

### Epoching

- Event-based epoching
- Fixed-length epoching
- Epoch rejection based on:
  - Amplitude thresholds
  - Global field power (GFP)
  - Artifact detection
  - Manual marking

### Quality Control

- Automated QC metrics
- Visual inspection tools
- Report generation
- Metadata tracking

## Metadata Management

### Database Structure

- Run ID tracking
- Step-by-step metadata
- Processing parameters
- Quality metrics
- Error logging

### Key Metadata Fields

```python
metadata = {
    "run_id": str,
    "task": str,
    "subject": str,
    "session": str,
    "processing_steps": List[str],
    "quality_metrics": Dict,
    "artifacts_removed": Dict,
    "channel_info": Dict,
    "epoch_info": Dict
}
```

## Configuration Options

### Task Settings

```yaml
task_name:
  enabled: true
  settings:
    resample_step:
      enabled: true
      value: 250
    reference_step:
      enabled: true
      value: "average"
    trim_step:
      enabled: true
      value: 1
```

### Processing Parameters

```yaml
processing:
  filtering:
    l_freq: 1.0
    h_freq: 80.0
  ica:
    n_components: 20
    method: "fastica"
  artifact_detection:
    amplitude_threshold: 100
    gfp_threshold: 3.0
```

## Methods and Functions

### Core Methods

- `pipeline.run()`: Execute full processing pipeline
- `pipeline.get_derivative_path()`: Get BIDS derivative path
- `pipeline.save_metadata()`: Update metadata database
- `pipeline.generate_report()`: Create processing report
- `pipeline.validate_data()`: Check data integrity

### Utility Methods

- `pipeline.get_config()`: Retrieve configuration
- `pipeline.set_montage()`: Set EEG montage
- `pipeline.mark_bad_channels()`: Mark channels as bad
- `pipeline.plot_data()`: Visualization utilities
- `pipeline.export_data()`: Export processed data

## Error Handling

### Common Errors

1. Data Import Errors
   - File format issues
   - Missing channels
   - Incorrect montage

2. Processing Errors
   - Invalid parameters
   - Insufficient data quality
   - Resource constraints

3. Export Errors
   - File permission issues
   - Disk space limitations
   - BIDS validation failures

### Error Recovery

- Automatic state saving
- Processing checkpoints
- Error logging and reporting
- Fallback options

## Integration Points

### MNE Integration

- Raw object handling
- Preprocessing functions
- Visualization tools
- Data structure compatibility

### PyLossless Integration

- Artifact detection
- ICA processing
- Quality metrics
- Report generation

### BIDS Integration

- Path handling
- Metadata organization
- File naming
- Directory structure

## Best Practices

### Performance Optimization

1. Memory Management
   - Use memory-mapped files
   - Implement garbage collection
   - Monitor resource usage

2. Processing Speed
   - Parallel processing where possible
   - Efficient data loading
   - Optimized algorithms

### Data Quality

1. Validation Steps
   - Input data checks
   - Processing parameter validation
   - Output quality assessment

2. Quality Metrics
   - Signal-to-noise ratio
   - Channel correlations
   - Artifact detection rates
   - Data continuity

### Code Organization

1. Modular Structure
   - Separate core functions
   - Clear dependencies
   - Consistent naming

2. Documentation
   - Inline comments
   - Function docstrings
   - Usage examples

## Debugging Tools

### Built-in Debugging

- Verbose logging
- Step-by-step tracking
- Error tracing
- State inspection

### External Tools

- MNE visualization
- Quality metric plots
- Processing reports
- Log analysis

## Common Usage Patterns

### Basic Processing

```python
# Initialize pipeline with output directory and config
pipeline = Pipeline(
    autoclean_dir="path/to/output",
    autoclean_config="config.yaml"
)

# Process a single file
pipeline.process_file(
    file_path="path/to/eeg.raw",
    task="resting_state"
)
```

### Directory Processing

```python
# Initialize pipeline
pipeline = Pipeline(
    autoclean_dir="path/to/output",
    autoclean_config="config.yaml"
)

# Process all files in a directory
pipeline.process_dir(
    dir_path="path/to/data_dir",
    task="hbcd_mmn",
    file_pattern="*.raw"  # Optional: pattern to match specific files
)
```

### Asynchronous Processing

```python
# Initialize pipeline
pipeline = Pipeline(
    autoclean_dir="path/to/output",
    autoclean_config="config.yaml"
)

# Process files asynchronously
pipeline.process_files_async(
    directory="path/to/data_dir",
    task="resting_eyes_open",
    pattern="*.raw",
    recursive=False,
    max_concurrent=3
)
```

### Advanced Usage

```python
# Initialize with specific config
pipeline = Pipeline(
    autoclean_dir="path/to/output",
    autoclean_config="custom_config.yaml"
)
```

## Troubleshooting Guide

### Common Issues

1. Data Loading
   - Check file permissions
   - Verify file format
   - Validate channel names

2. Processing
   - Monitor memory usage
   - Verify parameter values
   - Check step dependencies

3. Output
   - Ensure disk space
   - Validate BIDS compliance
   - Check file integrity

### Solutions

1. Data Validation
   - Use built-in validators
   - Check data dimensions
   - Verify metadata

2. Error Recovery
   - Load from checkpoints
   - Skip problematic steps
   - Use fallback options

3. Quality Checks
   - Review QC metrics
   - Inspect visualizations
   - Validate outputs

## Pipeline-Task Interaction

### Lifecycle Management

The Pipeline acts as a factory and lifecycle manager for Task objects:
1. Task instantiation through TASK_REGISTRY
2. Configuration injection and validation
3. Execution monitoring and state tracking
4. Resource cleanup and result persistence

### Error Propagation Chain

Pipeline implements a multi-level error handling strategy:
1. Task-level errors (processing failures)
2. Pipeline-level errors (configuration, I/O)
3. System-level errors (resources, permissions)

Each level has specific handling and recovery mechanisms.

### Atomic Operations

The Pipeline ensures atomicity through:
- Database-backed run tracking
- File system operation isolation
- State validation checkpoints
- Transaction-like processing steps

### Resource Management

Pipeline handles resource allocation for tasks:
- Directory structure creation and validation
- File handle management
- Memory allocation monitoring
- Concurrent processing limits

## Pipeline-Task Integration Pattern

### Responsibility Division

- Pipeline: Infrastructure, lifecycle, resources
- Task: Processing logic, data manipulation, analysis

### Communication Flow

1. Pipeline validates environment and resources
2. Task receives validated configuration
3. Task reports status through pipeline_results
4. Pipeline persists results and manages cleanup

### Extension Points

When implementing new tasks or pipeline features:
1. Task extensions: Inherit Task class, implement abstract methods
2. Pipeline extensions: Add to TASK_REGISTRY, extend validation
3. Configuration extensions: Update both Pipeline and Task validation

### Safety Guarantees

The integration provides several safety guarantees:
1. Configuration consistency
2. Resource availability
3. State isolation
4. Error containment
5. Data provenance

## Plugin System

The EEG pipeline includes a plugin system for importing, montage, and event trigger handling. This system allows users to extend the pipeline's functionality without modifying the core code.

### Plugin Types

1. **Format Plugins**: For registering new EEG file formats.
2. **EEG Plugins**: For handling specific combinations of file formats and montages.
3. **Event Processor Plugins**: For processing task-specific event annotations.

### Directory Structure

```
src/autoclean/
├── step_functions/
│   └── io.py                     # Core interfaces (don't modify)
├── plugins/
│   ├── formats/                  # Format registrations
│   │   ├── __init__.py
│   │   └── additional_formats.py # Register new formats here
│   ├── eeg_plugins/              # Unified plugins for format+montage combinations
│   │   ├── __init__.py
│   │   └── eeglab_mea30_plugin.py # Example plugin
│   ├── event_processors/         # Event processor plugins
│   │   ├── __init__.py
│   │   └── p300.py               # Example event processor
```

### Using Plugins

#### Importing EEG Data

```python
from autoclean.step_functions.io import import_eeg

# Configure the import
autoclean_dict = {
    "run_id": "my_analysis_001",
    "unprocessed_file": "/path/to/my/data.set",  # Can be any supported format
    "eeg_system": "MEA30",                       # Montage/system name
    "task": "oddball",                           # Optional task type
    # ...other configuration options...
}

# Import the data - plugins are automatically discovered and used
eeg_data = import_eeg(autoclean_dict)

# eeg_data can be either Raw or Epochs depending on the plugin
```

### Creating a Custom Plugin

#### Step 1: Register Your File Format (if needed)

If you're adding support for a new file format, register it first:

```python
# src/autoclean/plugins/formats/my_formats.py
from autoclean.step_functions.io import register_format

# Register your format - extension without dot, format ID in UPPERCASE
register_format('xyz', 'XYZ_FORMAT')
```

#### Step 2: Create Your Plugin

Create a file in `src/autoclean/plugins/eeg_plugins/` named descriptively to indicate the format+montage combination:

```python
# src/autoclean/plugins/eeg_plugins/xyz_custom64_plugin.py
from autoclean.step_functions.io import BaseEEGPlugin
from autoclean.utils.logging import message

class XYZCustom64Plugin(BaseEEGPlugin):
    """Plugin for XYZ files with Custom-64 montage."""
    
    VERSION = "1.0.0"  # Track your plugin version
    
    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check which format+montage combinations this plugin supports."""
        return format_id == 'XYZ_FORMAT' and montage_name == 'CustomCap-64'
    
    def import_and_configure(self, file_path, autoclean_dict, preload=True):
        """Import data and configure montage in one step."""
        message("info", f"Processing {file_path} with {autoclean_dict['eeg_system']}")
        
        try:
            # 1. Import the data
            # Your import code here...
            
            # 2. Configure the montage
            # Your montage configuration code here...
            
            # 3. Process events if applicable
            # Your event processing code here...
            
            # Return either Raw or Epochs object
            return raw  # or return epochs
            
        except Exception as e:
            raise RuntimeError(f"Failed to process data: {str(e)}")
    
    def process_events(self, raw, autoclean_dict):
        """Process events after import (for Raw data)."""
        # This is called automatically by import_eeg for Raw data
        # Events code here...
        return events, event_id, events_df
    
    def get_metadata(self):
        """Return additional metadata about this plugin."""
        return {
            "plugin_version": self.VERSION,
            # Any other metadata you want to include
        }
```

#### Step 3: Use Your Plugin

Your plugin will be automatically discovered and used when the appropriate format and montage are specified:

```python
autoclean_dict = {
    "unprocessed_file": "my_data.xyz",  # Your custom format
    "eeg_system": "CustomCap-64",       # Your custom montage
    # ...other options...
}

# Your plugin will be automatically selected
eeg_data = import_eeg(autoclean_dict)
```

### Complete Example

Here's a complete example that adds support for XDF files with a BioSemi-64 montage:

```python
# src/autoclean/plugins/formats/bids_formats.py
from autoclean.step_functions.io import register_format
register_format('xdf', 'XDF_FORMAT')

# src/autoclean/plugins/eeg_plugins/xdf_biosemi64_plugin.py
import mne
import pyxdf
from autoclean.step_functions.io import BaseEEGPlugin
from autoclean.utils.logging import message

class XDFBioSemi64Plugin(BaseEEGPlugin):
    """Plugin for XDF files with BioSemi-64 montage."""
    
    VERSION = "1.0.0"
    
    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        return format_id == 'XDF_FORMAT' and montage_name == 'biosemi64'
    
    def import_and_configure(self, file_path, autoclean_dict, preload=True):
        message("info", f"Processing XDF file with BioSemi-64 montage: {file_path}")
        
        try:
            # Load XDF file using pyxdf
            streams, header = pyxdf.load_xdf(file_path)
            
            # Find EEG stream and extract data
            eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
            if not eeg_stream:
                raise RuntimeError("No EEG stream found in XDF file")
                
            # Extract data, create MNE Raw object
            data = eeg_stream['time_series'].T
            sfreq = float(eeg_stream['info']['nominal_srate'][0])
            ch_names = [f"EEG{i+1}" for i in range(data.shape[0])]
            
            # Create Raw object
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # Apply BioSemi-64 montage
            montage = mne.channels.make_standard_montage('biosemi64')
            raw.set_montage(montage)
            
            # Process events from marker stream if present
            marker_stream = next((s for s in streams if s['info']['type'][0] == 'Markers'), None)
            if marker_stream:
                onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0]
                descriptions = [str(m[0]) for m in marker_stream['time_series']]
                raw.set_annotations(mne.Annotations(onsets, np.zeros_like(onsets), descriptions))
            
            return raw
            
        except Exception as e:
            raise RuntimeError(f"Failed to process XDF file: {str(e)}")
```

### Plugin Naming Conventions

Use clear, descriptive names that indicate the format and montage combination:

- `eeglab_mea30_plugin.py` - EEGLAB files with MEA30 montage
- `egi_raw_gsn129_plugin.py` - EGI Raw files with GSN-HydroCel-129 montage
- `brainvision_standard1020_plugin.py` - BrainVision files with standard 10-20 montage

### Returning Raw vs. Epochs

Your plugin can return either a `mne.io.Raw` or `mne.Epochs` object:

```python
def import_and_configure(self, file_path, autoclean_dict, preload=True):
    # ... processing code ...
    
    if autoclean_dict.get("return_epochs", False):
        # Create and return epochs
        events, _ = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, preload=True)
        return epochs
    else:
        # Return continuous data
        return raw
```

### Advanced Plugin Features

#### Task-Specific Processing

Handle different experimental paradigms:

```python
def import_and_configure(self, file_path, autoclean_dict, preload=True):
    # ... basic import code ...
    
    # Apply task-specific processing
    task = autoclean_dict.get("task", None)
    if task == "oddball":
        # Apply oddball-specific processing
        mapping = {"1": "Standard", "2": "Target"}
        raw.annotations.rename(mapping)
    elif task == "mmn":
        # Apply MMN-specific processing
        # ...
        
    return raw
```

#### Multiple Format/Montage Support

A single plugin can support multiple combinations:

```python
@classmethod
def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
    # Support multiple combinations
    supported = [
        ('EEGLAB_SET', 'GSN-HydroCel-64'),
        ('EEGLAB_SET', 'GSN-HydroCel-128'),
        ('EEGLAB_SET', 'GSN-HydroCel-256'),
    ]
    return (format_id, montage_name) in supported
```

#### Custom Configuration Options

Pass additional configuration via `autoclean_dict`:

```python
def import_and_configure(self, file_path, autoclean_dict, preload=True):
    # Get custom options
    reference = autoclean_dict.get("reference", "average")
    filter_settings = autoclean_dict.get("filter", {"l_freq": 0.1, "h_freq": 40})
    
    # ... import code ...
    
    # Apply reference
    if reference == "average":
        raw.set_eeg_reference("average")
    elif reference == "mastoids":
        raw.set_eeg_reference(["M1", "M2"])
    
    # Apply filters
    raw.filter(l_freq=filter_settings["l_freq"], h_freq=filter_settings["h_freq"])
    
    return raw
```

### Troubleshooting

If your plugin isn't working:

1. **Plugin Discovery**:
   - Ensure your plugin is in the correct directory (`plugins/eeg_plugins/`)
   - Check that `__init__.py` files exist in all plugin directories

2. **Format Registration**:
   - Verify your format is registered correctly in a file in `plugins/formats/`
   - Check for typos in format IDs

3. **Debugging Tips**:
   - Add `message("info", "Debug message")` calls in your plugin
   - Check if `supports_format_montage()` is returning `True` as expected
   - Test your plugin with a simple test script before integrating

4. **Common Issues**:
   - Mismatch between registered format ID and what's checked in `supports_format_montage()`
   - Forgetting to handle exceptions properly
   - Using incompatible MNE versions
