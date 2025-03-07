# Pipeline Overview

The AutoClean EEG pipeline is a powerful and flexible framework for processing EEG data. It supports various EEG paradigms and ensures that the processing steps are modular, configurable, and can be integrated into automated agents for regulated EEG processing.

## Key Features

- **Modular Design**: The pipeline is built with a modular architecture, allowing users to customize and extend the processing steps according to their needs.
- **Task-Based Processing**: The pipeline supports multiple tasks, each corresponding to a specific EEG processing paradigm.
- **Comprehensive Error Handling**: The pipeline includes robust error handling mechanisms to ensure smooth processing.
- **BIDS Compatibility**: The pipeline is designed to work with BIDS (Brain Imaging Data Structure) datasets.
- **Extensive Quality Control**: The pipeline provides automated quality control metrics and visual inspection tools.

## Detailed Architecture

For a detailed description of the architecture and features of the EEG pipeline, please refer to the [Architecture](../architecture.md) documentation.

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
