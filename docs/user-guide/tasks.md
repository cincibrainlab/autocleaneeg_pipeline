# Tasks

Tasks are a fundamental concept in the AutoClean Pipeline architecture. They define specific EEG processing workflows for different experimental paradigms and research needs.

## What Are Tasks?

A task is a Python class that inherits from the `Task` base class and implements a sequence of processing steps tailored to a specific type of EEG data. Each task encapsulates:

1. **Processing Logic**: The sequence and parameters of steps to run
2. **Configuration Validation**: Rules for validating task-specific settings
3. **Data-specific Logic**: Special handling for particular EEG paradigms

## Built-in Tasks

The AutoClean Pipeline comes with several pre-configured tasks for common EEG paradigms:

### Resting State Tasks

- **`rest_eyesopen`**: For resting-state EEG with eyes open
  - Optimized for alpha rhythm analysis
  - ICA-based artifact rejection
  - Spectral analysis outputs

### Auditory Tasks

- **`assr_default`**: For auditory steady-state response (ASSR) paradigms
  - Event-locked processing
  - Specific filtering for auditory ERPs
  - Time-frequency analysis outputs

- **`chirp_default`**: For auditory chirp paradigms
  - Dynamic frequency response analysis
  - Phase-locking value calculations

### MMN Paradigm

- **`hbcd_mmn_v3`**: For mismatch negativity (MMN) studies
  - Designed for the HBCD protocol
  - Automated oddball detection
  - Deviant-standard difference waveforms

### Animal EEG

- **`mouse_xdat_resting`**: For mouse EEG recordings
  - Specialized for rodent EEG characteristics
  - Handles cross-species adaptations

## Using Tasks

To use a task, specify its name when processing a file:

```python
from autoclean.core.pipeline import Pipeline

pipeline = Pipeline()

# Use the rest_eyesopen task
pipeline.process_file(
    file_path="path/to/eeg.raw",
    task="rest_eyesopen"
)
```

## Listing Available Tasks

To see all available tasks:

```python
pipeline = Pipeline()
available_tasks = pipeline.list_tasks()
print(available_tasks)
```

## Task Configuration

Each task has default parameters that can be overridden in your configuration file:

```yaml
task_settings:
  rest_eyesopen:
    epoch_length: 2.0
    overlap: 0.5
```

## Creating Custom Tasks

You can create custom tasks for your specific research needs by extending the `Task` base class.

### Custom Task Example

```python
from autoclean.core.task import Task
from autoclean.step_functions import filtering, artifact_detection, ica

class MyCustomTask(Task):
    """Custom EEG processing task for my experiment."""
    
    # Define the required configuration schema
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "epoch_length": {"type": "number", "minimum": 0.1},
            "custom_parameter": {"type": "string"}
        },
        "required": ["epoch_length"]
    }
    
    def run(self, pipeline):
        """Execute the processing steps for this task."""
        # Access configuration
        epoch_length = self.config.get("epoch_length", 1.0)
        custom_param = self.config.get("custom_parameter", "default")
        
        # Define processing steps
        steps = [
            # Pre-processing
            filtering.apply_bandpass_filter,
            artifact_detection.detect_bad_channels,
            
            # ICA processing
            ica.fit_ica,
            ica.detect_and_remove_artifacts,
            
            # Custom processing
            self.my_custom_step,
            
            # Reporting
            self.generate_report
        ]
        
        # Run each step
        for step in steps:
            step(pipeline)
    
    def my_custom_step(self, pipeline):
        """Custom processing logic specific to this task."""
        # Implementation of custom processing
        pass
    
    def generate_report(self, pipeline):
        """Generate a task-specific report."""
        # Create visualizations and metrics
        pass
```

### Registering Custom Tasks

To make your custom task available to the pipeline:

```python
from autoclean.core.pipeline import Pipeline, TASK_REGISTRY
from my_module import MyCustomTask

# Register your custom task
TASK_REGISTRY["my_custom_task"] = MyCustomTask

# Now you can use it
pipeline = Pipeline()
pipeline.process_file(
    file_path="path/to/eeg.raw",
    task="my_custom_task"
)
```

## Task Lifecycle

When a task is executed, it goes through the following lifecycle:

1. **Initialization**: Task object is created with configuration
2. **Validation**: Configuration is validated against schema
3. **Execution**: `run()` method is called with pipeline instance
4. **Completion**: Task finishes and returns control to pipeline

## Task Inheritance and Composition

You can create task hierarchies for code reuse:

```python
class BaseERPTask(Task):
    """Base task for all ERP paradigms."""
    
    def common_erp_processing(self, pipeline):
        # Common ERP processing steps
        pass

class MyERPTask(BaseERPTask):
    """Specific ERP task that inherits common functionality."""
    
    def run(self, pipeline):
        # Call parent methods
        self.common_erp_processing(pipeline)
        
        # Add specific processing
        self.specific_processing(pipeline)
```

## Best Practices for Tasks

When working with tasks:

1. **Keep tasks focused**: Each task should handle one type of EEG paradigm
2. **Validate all inputs**: Use the CONFIG_SCHEMA to enforce valid parameters
3. **Document task behavior**: Include detailed docstrings and comments
4. **Provide sensible defaults**: Make tasks work reasonably out-of-the-box
5. **Error handling**: Gracefully handle unexpected conditions
6. **Logging**: Use the pipeline's logging system for visibility

## Task Outputs

Tasks typically produce:

1. **Cleaned EEG data**: The processed EEG recordings
2. **Metadata**: Information about the processing steps
3. **Reports**: PDF or HTML reports with visualizations
4. **Metrics**: Quantitative measures of data quality and results

## Advanced Task Features

### Progress Reporting

Tasks can report progress:

```python
def run(self, pipeline):
    total_steps = 5
    
    # Step 1
    pipeline.logger.info("Starting preprocessing")
    pipeline.update_progress(1, total_steps)
    
    # Step 2
    pipeline.logger.info("Running ICA")
    pipeline.update_progress(2, total_steps)
    
    # Etc.
```

### Conditional Processing

Tasks can adapt based on data characteristics:

```python
def run(self, pipeline):
    # Check data properties
    if pipeline.raw.info['sfreq'] > 1000:
        # High sampling rate processing path
        self.high_res_processing(pipeline)
    else:
        # Standard processing path
        self.standard_processing(pipeline)
```

## Next Steps

- Learn about [Step Functions](step-functions.md) used within tasks
- Explore [Configuration Files](configuration-files.md) for customizing tasks
- See [Pipeline Overview](pipeline-overview.md) for the broader context
