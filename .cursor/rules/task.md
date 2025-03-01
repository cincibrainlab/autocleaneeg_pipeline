# Task Object Documentation

## Overview
The Task class serves as the abstract base class for all EEG processing tasks in the autoclean package. It defines a standardized interface that all specific EEG processing tasks must implement, providing a structured approach to EEG data processing through several key stages: configuration validation, data import, preprocessing, task-specific processing, and results management.

## Core Components

### Base Class Structure
```python
from autoclean.core.task import Task

class MyNewTask(Task):
    def __init__(self, config):
        super().__init__(config)
        self.raw = None
        self.epochs = None
        self.pipeline_results = {}
```

### Required Configuration
```python
config = {
    'run_id': str,              # Unique identifier for processing run
    'unprocessed_file': Path,   # Path to raw EEG data file
    'task': str,                # Name of the task (e.g., "rest_eyesopen")
    'tasks': dict,              # Task-specific settings
    'stage_files': dict,        # Configuration for saving intermediate results
}
```

## Abstract Methods

### Configuration Validation
```python
@abstractmethod
def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate task-specific configuration settings.
    
    Must be implemented by each task to validate its unique requirements.
    Called automatically after common validation in validate_config.
    """
    pass
```

### Data Import
```python
@abstractmethod
def import_data(self, file_path: Path) -> None:
    """Import raw EEG data for processing.
    
    Handle loading EEG data from disk and perform initial transformations.
    Typically sets self.raw with the imported MNE Raw object.
    """
    pass
```

### Preprocessing
```python
@abstractmethod
def preprocess(self) -> None:
    """Run standard EEG preprocessing pipeline.
    
    Implement common preprocessing steps:
    1. Resampling to standard sampling rate
    2. Filtering (high-pass, low-pass, notch)
    3. Bad channel detection and interpolation
    4. Independent Component Analysis (ICA)
    """
    pass
```

### Processing
```python
@abstractmethod
def process(self) -> None:
    """Run task-specific processing steps.
    
    Implement specialized analysis steps:
    1. Epoching continuous data
    2. Artifact rejection
    3. Feature extraction
    4. Statistical analysis
    """
    pass
```

## Implementation Guide

### Creating a New Task
1. Create a new Python file in the `tasks` directory
2. Import the Task base class
3. Implement all required abstract methods
4. Add task-specific attributes and methods
5. Register the task in the Pipeline's TASK_REGISTRY

### Example Implementation
```python
class MyNewTask(Task):
    def __init__(self, config):
        # Initialize task-specific attributes
        self.raw = None
        self.pipeline = None
        self.cleaned_raw = None
        self.epochs = None
        
        # Call parent initialization
        super().__init__(config)

    def _validate_task_config(self, config):
        # Implement task-specific validation
        required_fields = {
            'eeg_system': str,
            'settings': dict,
        }
        # Add validation logic
        return config

    def import_data(self, file_path):
        # Import raw data using standard function
        self.raw = import_eeg(self.config)
        # Save imported data if configured
        save_raw_to_set(self.raw, self.config, "post_import")

    def preprocess(self):
        # Implement standard preprocessing pipeline
        if self.raw is None:
            raise RuntimeError("No data has been imported")
        # Add preprocessing steps
        
    def process(self):
        # Implement task-specific processing
        if self.cleaned_raw is None:
            raise RuntimeError("Run preprocess first")
        # Add processing steps

## Task Registration and Pipeline Execution

### Task Registry
Tasks must be registered in the Pipeline's TASK_REGISTRY to be available for use:
```python
TASK_REGISTRY = {
    'rest_eyesopen': RestingEyesOpen,
    'assr_default': AssrDefault,
    'chirp_default': ChirpDefault,
    'mouse_xdat_resting': MouseXdatResting,
    'hbcd_mmn_v3': HBCD_MMN,
}
```

### Pipeline Selection and Execution
The Pipeline selects and instantiates the appropriate task based on the task name provided:
```python
# Pipeline instantiation
pipeline = Pipeline(
    autoclean_dir="path/to/output",
    autoclean_config="config.yaml"
)

# Pipeline selects and runs task
pipeline.process_file(
    file_path="data.raw",
    task="rest_eyesopen"  # Must match a key in TASK_REGISTRY
)

# For directory processing
pipeline.process_directory(
    directory="data_dir",
    task="rest_eyesopen",
    pattern="*.raw"
)

# For async directory processing
await pipeline.process_directory_async(
    directory="data_dir",
    task="rest_eyesopen",
    pattern="*.raw",
    max_concurrent=3
)
```

The Pipeline handles:
1. Task instantiation from registry
2. Configuration validation
3. Error handling and logging
4. Results management

## Configuration Structure

### Task Settings
```yaml
tasks:
  my_new_task:
    mne_task: "my_paradigm"
    description: "Description of your task"
    lossless_config: "path/to/lossless_config.yaml"
    settings:
      resample_step:
        enabled: true
        value: 250  # Hz
      eog_step:
        enabled: true
        value: ["E1", "E8", "E14", "E21", "E25", "E32"]
      trim_step:
        enabled: true
        value: 2  # seconds to trim from start
```

### Stage Files Configuration
```yaml
stage_files:
  post_import:
    enabled: true
    suffix: "_imported"
  post_prepipeline:
    enabled: true
    suffix: "_preprocessed"
  post_pylossless:
    enabled: true
    suffix: "_lossless"
```

## Key Attributes

### Instance Variables
- `self.raw`: MNE Raw object containing EEG data
- `self.epochs`: MNE Epochs object for epoched data
- `self.pipeline_results`: Dictionary storing processing results
- `self.config`: Validated configuration dictionary
- `self.pipeline`: PyLossless pipeline instance
- `self.cleaned_raw`: Preprocessed EEG data

## Processing Flow

### Standard Processing Sequence
1. Configuration Validation
   - Common validation (base class)
   - Task-specific validation
   
2. Data Import
   - Load raw EEG data
   - Initial data validation
   - Save imported data (if configured)
   
3. Preprocessing
   - Basic preprocessing (resampling, filtering)
   - Bad channel detection
   - BIDS conversion
   - PyLossless pipeline
   - Rejection policy application
   
4. Task-Specific Processing
   - Epoching
   - Feature extraction
   - Analysis
   - Results saving

## Error Handling

### Common Errors
1. Configuration Errors
   - Missing required fields
   - Invalid field types
   - Invalid parameter values

2. Processing Errors
   - Missing data
   - Invalid processing sequence
   - Failed processing steps

### Error Handling Strategy
```python
def process_step(self):
    if self.raw is None:
        raise RuntimeError("No data imported")
    try:
        # Processing logic
        pass
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")
```

## Best Practices

### Implementation Guidelines
1. Validate all inputs thoroughly
2. Document processing steps clearly
3. Save intermediate results
4. Generate quality control visualizations
5. Handle errors gracefully
6. Follow consistent naming conventions

### Code Organization
1. Keep methods focused and single-purpose
2. Use clear variable names
3. Document assumptions and limitations
4. Include example usage in docstrings
5. Add helpful comments for complex operations

## Integration Points

### MNE Integration
- Raw object handling
- Preprocessing functions
- Epoching and analysis tools
- Visualization utilities

### PyLossless Integration
- Pipeline configuration
- Artifact detection
- ICA processing
- Quality metrics

### File System Integration
- Stage file management
- Results storage
- Debug information
- Report generation

## Usage Examples

### Basic Task Implementation
```python
from autoclean.core.task import Task

class SimpleRestingTask(Task):
    def __init__(self, config):
        super().__init__(config)
        
    def _validate_task_config(self, config):
        # Add validation
        return config
        
    def import_data(self, file_path):
        self.raw = import_eeg(self.config)
        
    def preprocess(self):
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        
    def process(self):
        # Add processing steps
        pass
```

## Troubleshooting Guide

### Common Issues
1. Data Import
   - File format compatibility
   - Channel configuration
   - Metadata consistency

2. Preprocessing
   - Parameter validation
   - Resource constraints
   - Processing sequence

3. Task Processing
   - Data quality
   - Analysis parameters
   - Result validation

### Solutions
1. Data Validation
   - Check file formats
   - Verify channel setup
   - Validate metadata

2. Processing Validation
   - Monitor resource usage
   - Verify processing steps
   - Check intermediate results

3. Quality Control
   - Review visualizations
   - Validate metrics
   - Check output consistency

## State Management

### Core State Objects
The Task class maintains three critical state objects:
- `self.raw`: MNE Raw object that maintains the continuous EEG data state
- `self.epochs`: MNE Epochs object for segmented data analysis
- `self.pipeline_results`: Dictionary tracking complete processing history

### State Transitions
Processing follows a strict state transition pattern:
1. Initialization (config validation) → Empty state
2. Import → Raw data state
3. Preprocessing → Modified raw state
4. Processing → Epochs state (if applicable)

Each transition must maintain data provenance and update pipeline_results.

### Thread Safety Considerations
- State objects are not thread-safe by default
- Implementations should avoid shared state modifications
- Use atomic operations when updating pipeline_results

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