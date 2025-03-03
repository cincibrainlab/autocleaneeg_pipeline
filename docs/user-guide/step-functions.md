# Step Functions

Step functions are the fundamental building blocks of the AutoClean Pipeline. They are modular units of processing logic that perform specific operations on EEG data.

## What Are Step Functions?

Step functions are Python functions that:

1. Take a pipeline object as input
2. Perform a specific processing operation
3. Modify the pipeline's state (typically the `raw` attribute)
4. Return None (operations happen in-place)

Each step function follows a consistent pattern, making them interchangeable and composable within tasks.

## Core Step Functions

The AutoClean Pipeline provides a comprehensive library of step functions organized by category:

### File Operations

Functions for loading and saving EEG data:

- **`load_raw`**: Load raw EEG data from various formats
- **`save_raw`**: Save processed EEG data
- **`export_to_eeglab`**: Export data to EEGLAB format
- **`export_to_bids`**: Export data in BIDS-compliant format

### Preprocessing

Basic signal processing operations:

- **`resample`**: Change the sampling rate
- **`apply_bandpass_filter`**: Apply bandpass filtering
- **`apply_notch_filter`**: Remove power line noise
- **`apply_car`**: Apply common average reference
- **`standardize_channel_names`**: Convert to standard nomenclature

### Artifact Detection

Functions for identifying problematic data:

- **`detect_bad_channels`**: Identify malfunctioning electrodes
- **`detect_muscle_artifacts`**: Find EMG contamination
- **`detect_eye_artifacts`**: Find blinks and saccades
- **`detect_flatline_segments`**: Identify dropout periods
- **`detect_high_amplitude_artifacts`**: Find extreme values

### ICA Processing

Independent Component Analysis for artifact removal:

- **`fit_ica`**: Compute ICA decomposition
- **`detect_artifact_components`**: Identify artifact-related components
- **`apply_ica`**: Apply ICA to remove artifacts
- **`plot_ica_components`**: Visualize ICA components for inspection

### Epoching

Functions for segmenting continuous data:

- **`create_epochs`**: Segment data into epochs
- **`create_overlapping_epochs`**: Create epochs with overlap
- **`reject_bad_epochs`**: Remove epochs with artifacts
- **`extract_events`**: Detect and extract event markers

### Frequency Analysis

Spectral processing functions:

- **`compute_psd`**: Calculate power spectral density
- **`extract_frequency_bands`**: Extract power in specific bands
- **`compute_coherence`**: Calculate coherence between channels
- **`time_frequency_analysis`**: Perform time-frequency decomposition

### Source Analysis

Functions for source localization:

- **`setup_source_space`**: Prepare source space
- **`compute_forward_solution`**: Calculate forward model
- **`compute_inverse_solution`**: Solve inverse problem
- **`extract_source_time_courses`**: Extract activity from ROIs

### Reporting

Functions for generating outputs:

- **`generate_html_report`**: Create interactive HTML report
- **`generate_pdf_report`**: Create printable PDF report
- **`plot_raw_data`**: Visualize raw EEG signal
- **`plot_spectra`**: Visualize frequency content
- **`create_montage_plot`**: Create topographic visualizations

## Using Step Functions

Step functions are typically called within a task's `run` method:

```python
from autoclean.core.task import Task
from autoclean.step_functions import filtering, preprocessing, ica

class MyTask(Task):
    def run(self, pipeline):
        # Call step functions in sequence
        preprocessing.resample(pipeline)
        filtering.apply_bandpass_filter(pipeline)
        ica.fit_ica(pipeline)
        ica.detect_artifact_components(pipeline)
        ica.apply_ica(pipeline)
```

## Step Function Anatomy

A typical step function looks like:

```python
def apply_bandpass_filter(pipeline, l_freq=1.0, h_freq=40.0, method='fir'):
    """Apply a bandpass filter to the raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline object containing the raw data
    l_freq : float
        Lower frequency bound in Hz
    h_freq : float
        Upper frequency bound in Hz
    method : str
        Filter method ('fir' or 'iir')
        
    Returns
    -------
    None
        The pipeline.raw is modified in-place
    """
    # Get configuration or use default parameters
    l_freq = pipeline.config.get('filter', {}).get('highpass', l_freq)
    h_freq = pipeline.config.get('filter', {}).get('lowpass', h_freq)
    method = pipeline.config.get('filter', {}).get('method', method)
    
    # Log the operation
    pipeline.logger.info(f"Applying bandpass filter: {l_freq}-{h_freq} Hz using {method}")
    
    # Perform the operation
    pipeline.raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        verbose=False
    )
    
    # Update metadata
    pipeline.metadata['processing_steps'].append({
        'step': 'bandpass_filter',
        'parameters': {
            'l_freq': l_freq,
            'h_freq': h_freq,
            'method': method
        },
        'timestamp': datetime.now().isoformat()
    })
```

## Step Function Categories

AutoClean organizes step functions into several modules:

- **`autoclean.step_functions.io`**: Input/output operations
- **`autoclean.step_functions.preprocessing`**: Basic signal processing
- **`autoclean.step_functions.artifacts`**: Artifact detection and handling
- **`autoclean.step_functions.ica`**: ICA-related functions
- **`autoclean.step_functions.epochs`**: Epoching operations
- **`autoclean.step_functions.frequency`**: Spectral analysis
- **`autoclean.step_functions.connectivity`**: Connectivity measures
- **`autoclean.step_functions.source`**: Source localization
- **`autoclean.step_functions.reports`**: Visualization and reporting

## Creating Custom Step Functions

You can create your own step functions to extend the pipeline:

```python
def my_custom_processing(pipeline, param1=default1, param2=default2):
    """Description of what this step function does.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline object
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    """
    # Implementation logic here
    
    # Always log what you're doing
    pipeline.logger.info(f"Performing custom processing with {param1}, {param2}")
    
    # Typically modify pipeline.raw
    # pipeline.raw = ...
    
    # Update metadata to track what was done
    pipeline.metadata['processing_steps'].append({
        'step': 'my_custom_processing',
        'parameters': {
            'param1': param1,
            'param2': param2
        }
    })
```

## Best Practices for Step Functions

When working with or creating step functions:

1. **Single responsibility**: Each function should do one thing well
2. **Parameter priority**: Use explicit parameters, falling back to config values
3. **Comprehensive logging**: Log operations and important values
4. **Metadata tracking**: Record all operations in metadata
5. **Error handling**: Use try/except and provide informative errors
6. **Input validation**: Check inputs before processing
7. **Documentation**: Include detailed docstrings

## Step Function Parameterization

Step functions can be parameterized in three ways:

1. **Default parameters**: Defined in the function signature
2. **Configuration values**: Retrieved from the pipeline configuration
3. **Explicit parameters**: Passed when calling the function

The precedence order is: explicit parameters > configuration values > defaults.

## Step Function Hooks

The pipeline supports hooks that run before and after each step function:

```python
# Before hook example
pipeline.register_hook('before_step', lambda pipeline, step_name: 
    pipeline.logger.debug(f"Starting {step_name}"))

# After hook example
pipeline.register_hook('after_step', lambda pipeline, step_name, success: 
    pipeline.logger.debug(f"Completed {step_name}, success={success}"))
```

## Step Function Groups

Related step functions can be grouped together for convenience:

```python
# Group definition
preprocessing_steps = [
    preprocessing.resample,
    filtering.apply_bandpass_filter,
    filtering.apply_notch_filter,
    preprocessing.standardize_channel_names
]

# Usage in task
for step in preprocessing_steps:
    step(pipeline)
```

## Conditional Execution

You can conditionally execute step functions:

```python
# Only run ICA if enabled in config
if pipeline.config.get('ica', {}).get('enable', True):
    ica.fit_ica(pipeline)
    ica.detect_artifact_components(pipeline)
    ica.apply_ica(pipeline)
```

## Next Steps

- Learn about [Tasks](tasks.md) that orchestrate step functions
- Explore [API Reference](../api-reference/step-functions.md) for detailed function documentation
- See [Examples](../examples/step-function-examples.md) for practical use cases
