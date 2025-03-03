# Creating Custom Step Functions

This guide explains how to create your own step functions to extend the AutoClean Pipeline's capabilities. Step functions are the building blocks of the pipeline, each performing a specific operation on EEG data.

## Step Function Basics

A step function is a Python function that:

1. Takes a `Pipeline` object as its first argument
2. Performs some operation on the pipeline's data
3. Returns `None` (modifies the pipeline in-place)
4. Updates the pipeline's metadata to track what was done

## Step Function Template

Here's a basic template for creating a step function:

```python
def my_custom_step(pipeline, param1=default1, param2=default2):
    """Description of what this step function does.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance containing the EEG data
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    
    Returns
    -------
    None
        The pipeline is modified in-place
    """
    # Log the operation
    pipeline.logger.info(f"Running my_custom_step with param1={param1}, param2={param2}")
    
    # Get configuration values (fall back to function parameters if not in config)
    param1 = pipeline.config.get('my_section', {}).get('param1', param1)
    param2 = pipeline.config.get('my_section', {}).get('param2', param2)
    
    # Input validation
    if pipeline.raw is None:
        raise ValueError("Raw data must be loaded before running my_custom_step")
    
    # Perform the operation
    # ... your code here ...
    
    # Update the pipeline's metadata
    pipeline.metadata['processing_steps'].append({
        'step': 'my_custom_step',
        'parameters': {
            'param1': param1,
            'param2': param2
        },
        'timestamp': str(datetime.datetime.now())
    })
```

## Complete Example

Here's a complete example of a custom step function that applies a custom filter to the data:

```python
import numpy as np
from scipy import signal
import datetime

def apply_custom_filter(pipeline, filter_type='butterworth', order=4, cutoff=(1, 40)):
    """Apply a custom filter to the raw EEG data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    filter_type : str, default='butterworth'
        Type of filter to apply ('butterworth', 'chebyshev', 'elliptic')
    order : int, default=4
        Filter order
    cutoff : tuple of float, default=(1, 40)
        Cutoff frequencies in Hz (low, high)
    
    Returns
    -------
    None
        Modifies pipeline.raw in-place
    """
    # Log the operation
    pipeline.logger.info(f"Applying {filter_type} filter with order {order} and cutoff {cutoff}")
    
    # Input validation
    if pipeline.raw is None:
        raise ValueError("Raw data must be loaded before filtering")
    
    if filter_type not in ['butterworth', 'chebyshev', 'elliptic']:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Get data
    data = pipeline.raw.get_data()
    sfreq = pipeline.raw.info['sfreq']
    
    # Normalize cutoff frequencies
    nyquist = sfreq / 2.0
    low, high = cutoff
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # Create filter
    if filter_type == 'butterworth':
        b, a = signal.butter(order, [low_norm, high_norm], btype='bandpass')
    elif filter_type == 'chebyshev':
        b, a = signal.cheby1(order, 0.5, [low_norm, high_norm], btype='bandpass')
    elif filter_type == 'elliptic':
        b, a = signal.ellip(order, 0.5, 40, [low_norm, high_norm], btype='bandpass')
    
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    
    # Update the raw data
    pipeline.raw._data = filtered_data
    
    # Update metadata
    pipeline.metadata['processing_steps'].append({
        'step': 'apply_custom_filter',
        'parameters': {
            'filter_type': filter_type,
            'order': order,
            'cutoff': cutoff
        },
        'timestamp': str(datetime.datetime.now())
    })
    
    # Report success
    pipeline.logger.info(f"Successfully applied {filter_type} filter")
```

## Organizing Step Functions

We recommend organizing step functions by their functionality:

1. Create a Python module for your custom step functions
2. Group related functions together
3. Import and use them in your tasks

Example file structure:

```
my_eeg_project/
├── step_functions/
│   ├── __init__.py
│   ├── custom_filters.py
│   ├── custom_artifacts.py
│   └── custom_analysis.py
└── tasks/
    ├── __init__.py
    └── my_custom_task.py
```

Example `custom_filters.py`:

```python
"""Custom filtering step functions for EEG processing."""

import numpy as np
from scipy import signal
import datetime

def apply_wavelet_filter(pipeline, wavelet='db4', level=5):
    """Apply wavelet-based filtering to the raw EEG data."""
    # Implementation...
    pass

def apply_adaptive_filter(pipeline, reference_channel=None, delay=10, mu=0.01):
    """Apply an adaptive filter using a reference channel."""
    # Implementation...
    pass
```

## Accessing Pipeline State

Your step functions can access various properties of the pipeline:

- `pipeline.raw`: The raw EEG data (MNE Raw object)
- `pipeline.ica`: ICA decomposition (if performed)
- `pipeline.epochs`: Epoched data (if created)
- `pipeline.evoked`: Evoked data (if created)
- `pipeline.config`: Configuration dictionary
- `pipeline.metadata`: Processing metadata
- `pipeline.logger`: Logger for output

## Handling Errors

Always include proper error handling in your step functions:

```python
def my_robust_step(pipeline, param1=default1):
    """A robust step function with error handling."""
    
    # Check preconditions
    if pipeline.raw is None:
        raise ValueError("Raw data must be loaded first")
    
    try:
        # Your processing code
        # ...
        
    except Exception as e:
        # Log the error
        pipeline.logger.error(f"Error in my_robust_step: {str(e)}")
        
        # Update metadata with error information
        pipeline.metadata['errors'].append({
            'step': 'my_robust_step',
            'error': str(e),
            'timestamp': str(datetime.datetime.now())
        })
        
        # Re-raise or handle as appropriate
        raise
```

## Creating Step Function Combinations

You can create higher-level step functions that combine multiple other steps:

```python
def preprocess_standard(pipeline, resample_freq=250, l_freq=1, h_freq=40):
    """Standard preprocessing combining multiple steps."""
    
    from autoclean.step_functions.preprocessing import resample, apply_bandpass_filter
    from autoclean.step_functions.artifacts import detect_bad_channels
    
    # Sequentially apply steps
    resample(pipeline, sfreq=resample_freq)
    apply_bandpass_filter(pipeline, l_freq=l_freq, h_freq=h_freq)
    detect_bad_channels(pipeline)
    
    # Update metadata
    pipeline.metadata['processing_steps'].append({
        'step': 'preprocess_standard',
        'parameters': {
            'resample_freq': resample_freq,
            'l_freq': l_freq,
            'h_freq': h_freq
        },
        'timestamp': str(datetime.datetime.now())
    })
```

## Using Custom Step Functions in Tasks

Once you've created custom step functions, you can use them in tasks:

```python
from autoclean.core.task import Task
from autoclean.step_functions import preprocessing, artifacts
from my_project.step_functions.custom_filters import apply_custom_filter

class MyCustomTask(Task):
    """Custom task using standard and custom step functions."""
    
    def run(self, pipeline):
        """Run the task on the given pipeline."""
        
        # Standard steps
        preprocessing.resample(pipeline)
        preprocessing.apply_bandpass_filter(pipeline)
        
        # Custom step
        apply_custom_filter(pipeline, filter_type='chebyshev', order=6)
        
        # More standard steps
        artifacts.detect_bad_channels(pipeline)
        
        return pipeline
```

## Performance Considerations

When writing step functions, consider these performance tips:

1. **Avoid copying data** unless necessary
2. **Use in-place operations** when possible
3. **Use NumPy's optimized functions** rather than Python loops
4. **Consider adding a `picks` parameter** to allow processing only certain channels
5. **Add progress reporting** for long-running operations

Example with progress reporting:

```python
def long_processing_step(pipeline, iterations=100):
    """A long-running processing step with progress updates."""
    
    pipeline.logger.info(f"Starting long processing with {iterations} iterations")
    
    for i in range(iterations):
        # Do some work
        # ...
        
        # Update progress every 5%
        if i % max(1, iterations // 20) == 0:
            progress_pct = (i / iterations) * 100
            pipeline.progress("long_processing", progress_pct, 
                             f"Completed {i}/{iterations} iterations")
    
    pipeline.logger.info("Completed long processing")
```

## Testing Custom Step Functions

It's important to thoroughly test your custom step functions:

```python
import pytest
import numpy as np
from mne.io import RawArray
from autoclean.core.pipeline import Pipeline

def test_custom_filter():
    """Test the custom filter step function."""
    
    # Create test data
    data = np.random.randn(5, 1000)  # 5 channels, 1000 samples
    info = mne.create_info(ch_names=['C3', 'C4', 'Pz', 'Fz', 'Cz'], 
                          sfreq=100, ch_types='eeg')
    raw = RawArray(data, info)
    
    # Create a pipeline with the test data
    pipeline = Pipeline()
    pipeline.raw = raw
    
    # Apply the custom filter
    from my_project.step_functions.custom_filters import apply_custom_filter
    apply_custom_filter(pipeline, filter_type='butterworth', cutoff=(5, 30))
    
    # Check that data was modified
    assert not np.array_equal(data, pipeline.raw.get_data())
    
    # Further testing as appropriate
    # ...
```

## Tips for Creating Effective Step Functions

1. **Single Responsibility Principle**: Each step function should do one thing well
2. **Clear Documentation**: Clearly document parameters, behavior, and side effects
3. **Configuration Integration**: Design to work with the configuration system
4. **Sensible Defaults**: Provide reasonable default values for parameters
5. **Robust Validation**: Validate inputs and pipeline state before processing
6. **Comprehensive Logging**: Log important information and progress
7. **Metadata Updates**: Always update the pipeline's metadata
8. **Error Handling**: Include appropriate error handling and reporting
9. **Progress Reporting**: Report progress for long-running operations
10. **Reuse Existing Functionality**: Build on MNE and other libraries when possible

## Adding Step Functions to the Registry

To make your step functions easily discoverable, you can add them to the step function registry:

```python
from autoclean.utils.registry import register_step_function

@register_step_function(category='preprocessing')
def my_registered_step(pipeline, param1=default1):
    """This step function will be registered for easy discovery."""
    # Implementation
    pass
```

The registered step can then be found and used programmatically:

```python
from autoclean.utils.registry import get_step_function

# Get the function by name
func = get_step_function('my_registered_step')

# Or get all functions in a category
preprocessing_funcs = get_step_functions(category='preprocessing')
```

## Conclusion

Creating custom step functions is a powerful way to extend the AutoClean Pipeline with your own processing methods. By following the patterns and best practices described in this guide, you can create robust, reusable components that integrate seamlessly with the rest of the pipeline.

For more details, see the [API Reference](../api-reference/step-functions.md) for step functions and the [Pipeline API Reference](../api-reference/pipeline.md) to understand how step functions interact with the pipeline object. 