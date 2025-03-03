# Pipeline API Reference

The `Pipeline` class is the central component of the AutoClean framework, managing the flow of EEG data processing and providing a context for all step functions.

## Class Definition

```python
from autoclean.core.pipeline import Pipeline
```

## Constructor

```python
def __init__(self, config=None, data_path=None, verbose=None):
    """
    Initialize a Pipeline instance.
    
    Parameters
    ----------
    config : dict, str, or Path, optional
        Configuration dictionary or path to configuration file
    data_path : str or Path, optional
        Base directory for data storage
    verbose : bool, optional
        Whether to display verbose output
    """
```

## Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `config` | dict | Pipeline configuration settings |
| `raw` | mne.io.Raw | The loaded raw EEG data |
| `ica` | mne.preprocessing.ICA | ICA object if ICA has been fitted |
| `epochs` | mne.Epochs | Epoched data if epochs have been created |
| `evoked` | mne.Evoked | Evoked data if averaged epochs have been created |
| `psd` | mne.time_frequency.psd_array_welch | Power spectral density if calculated |
| `metadata` | dict | Dictionary tracking pipeline operations and results |
| `logger` | logging.Logger | Logger for capturing process information |

## Methods

### File Processing

```python
def process_file(self, file_path, task, config=None, output_dir=None, overwrite=False):
    """
    Process a file using the specified task.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the EEG file to process
    task : str or Task instance
        Name of registered task or Task instance to use for processing
    config : dict, str, or Path, optional
        Configuration to use for this run
    output_dir : str or Path, optional
        Directory to store outputs
    overwrite : bool, default=False
        Whether to overwrite existing files
    
    Returns
    -------
    dict
        Results metadata from processing
    """
```

```python
def process_directory(self, directory, task, file_pattern='*.raw', 
                     recursive=False, parallel=False, max_workers=None,
                     config=None, output_dir=None, overwrite=False):
    """
    Process all matching files in a directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory containing files to process
    task : str or Task instance
        Name of registered task or Task instance to use for processing
    file_pattern : str, default='*.raw'
        Glob pattern for files to process
    recursive : bool, default=False
        Whether to search directories recursively
    parallel : bool, default=False
        Whether to process files in parallel
    max_workers : int, optional
        Maximum number of parallel workers
    config : dict, str, or Path, optional
        Configuration to use for this run
    output_dir : str or Path, optional
        Directory to store outputs
    overwrite : bool, default=False
        Whether to overwrite existing files
    
    Returns
    -------
    list of dict
        List of results metadata from processing each file
    """
```

### Configuration Management

```python
def load_config(self, config=None):
    """
    Load configuration from a file or dictionary.
    
    Parameters
    ----------
    config : dict, str, or Path, optional
        Configuration dictionary or path to configuration file
        
    Returns
    -------
    dict
        The loaded configuration
    """
```

```python
def merge_config(self, *configs):
    """
    Merge multiple configurations together.
    
    Parameters
    ----------
    *configs : dict
        Configurations to merge
        
    Returns
    -------
    dict
        Merged configuration dictionary
    """
```

```python
def get_config_value(self, key_path, default=None):
    """
    Get a configuration value using a dot-notation path.
    
    Parameters
    ----------
    key_path : str
        Path to configuration value (e.g., 'preprocessing.filter.l_freq')
    default : any, optional
        Default value if key not found
    
    Returns
    -------
    any
        Configuration value or default
    """
```

### File Operations

```python
def get_output_path(self, file_type, suffix=None, extension=None,
                   ensure_dir=True, relative_to_input=True):
    """
    Get a path for an output file.
    
    Parameters
    ----------
    file_type : str
        Type of file ('raw', 'ica', 'epochs', 'evoked', 'report', etc.)
    suffix : str, optional
        Suffix to add to filename
    extension : str, optional
        File extension (without the dot)
    ensure_dir : bool, default=True
        Whether to create the directory if it doesn't exist
    relative_to_input : bool, default=True
        Whether to base the output path on the input file's path
    
    Returns
    -------
    Path
        Output file path
    """
```

```python
def get_results_dir(self, result_type, ensure_dir=True):
    """
    Get a directory for storing results.
    
    Parameters
    ----------
    result_type : str
        Type of results ('figures', 'tables', 'logs', etc.)
    ensure_dir : bool, default=True
        Whether to create the directory if it doesn't exist
    
    Returns
    -------
    Path
        Results directory path
    """
```

### State Management

```python
def save_state(self, file_path=None):
    """
    Save the current pipeline state.
    
    Parameters
    ----------
    file_path : str or Path, optional
        Where to save the state. If None, uses default location
    
    Returns
    -------
    Path
        Path where state was saved
    """
```

```python
def load_state(self, file_path):
    """
    Load a previously saved pipeline state.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the saved state file
    
    Returns
    -------
    bool
        True if state was loaded successfully
    """
```

```python
def reset(self):
    """
    Reset the pipeline to its initial state.
    
    Returns
    -------
    None
    """
```

### Logging and Progress

```python
def log(self, level, message):
    """
    Log a message.
    
    Parameters
    ----------
    level : str
        Log level ('debug', 'info', 'warning', 'error', 'critical')
    message : str
        Message to log
    
    Returns
    -------
    None
    """
```

```python
def progress(self, stage, percentage=None, message=None):
    """
    Update processing progress.
    
    Parameters
    ----------
    stage : str
        Current processing stage
    percentage : float, optional
        Percentage complete (0-100)
    message : str, optional
        Status message
    
    Returns
    -------
    None
    """
```

### Database Operations

```python
def save_to_database(self, db_path=None):
    """
    Save processing results to database.
    
    Parameters
    ----------
    db_path : str or Path, optional
        Path to database file. If None, uses default
    
    Returns
    -------
    str
        ID of the database entry
    """
```

```python
def load_from_database(self, run_id, db_path=None):
    """
    Load processing results from database.
    
    Parameters
    ----------
    run_id : str
        ID of the processing run to load
    db_path : str or Path, optional
        Path to database file. If None, uses default
    
    Returns
    -------
    dict
        The loaded results
    """
```

## Usage Examples

### Basic Processing

```python
from autoclean.core.pipeline import Pipeline

# Initialize a pipeline
pipeline = Pipeline()

# Process a single file
results = pipeline.process_file(
    file_path="subject01_resting.raw",
    task="rest_eyesopen"
)

# Check results
print(f"Processed file with {len(pipeline.raw.ch_names)} channels")
print(f"Sampling rate: {pipeline.raw.info['sfreq']} Hz")
print(f"Duration: {pipeline.raw.times.max()} seconds")
```

### Batch Processing

```python
from autoclean.core.pipeline import Pipeline

# Initialize a pipeline with custom config
pipeline = Pipeline(config="configs/my_study_config.yaml")

# Process all files in a directory
results_list = pipeline.process_directory(
    directory="data/raw_files",
    task="rest_eyesopen",
    file_pattern="*.fif",
    recursive=True,
    parallel=True,
    max_workers=4
)

# Summarize results
print(f"Processed {len(results_list)} files")
success_count = sum(1 for result in results_list if result.get('success', False))
print(f"Successfully processed: {success_count}")
```

### Custom Processing Sequence

```python
from autoclean.core.pipeline import Pipeline
from autoclean.step_functions import preprocessing, artifacts, ica

# Initialize pipeline
pipeline = Pipeline()

# Load data
pipeline.process_file("subject01_task.raw", task=None)  # No task - manual processing

# Apply step functions directly
preprocessing.resample(pipeline, sfreq=250)
preprocessing.apply_bandpass_filter(pipeline, l_freq=1, h_freq=40)
artifacts.detect_bad_channels(pipeline)
ica.fit_ica(pipeline)
ica.detect_artifact_components(pipeline, method="correlation")
ica.apply_ica(pipeline)

# Save results
pipeline.save_state()
```

### Using with Custom Configuration

```python
from autoclean.core.pipeline import Pipeline

# Custom configuration dictionary
config = {
    "preprocessing": {
        "sampling_rate": 250,
        "filter": {
            "l_freq": 0.5,
            "h_freq": 45,
            "method": "fir"
        }
    },
    "ica": {
        "n_components": 20,
        "random_state": 42
    }
}

# Initialize with custom config
pipeline = Pipeline(config=config)

# Process file
pipeline.process_file("subject01.edf", task="rest_eyesopen")
```

## Pipeline Events

The Pipeline class uses an event system to allow hooking into different stages of processing:

```python
from autoclean.core.pipeline import Pipeline

# Define event handlers
def on_file_loaded(pipeline, file_path):
    print(f"File loaded: {file_path}")
    print(f"Channels: {len(pipeline.raw.ch_names)}")

def on_processing_complete(pipeline, results):
    print(f"Processing complete: {results.get('success', False)}")

# Create pipeline
pipeline = Pipeline()

# Register event handlers
pipeline.on("file_loaded", on_file_loaded)
pipeline.on("processing_complete", on_processing_complete)

# Process file - events will be triggered
pipeline.process_file("subject01.raw", task="rest_eyesopen")
```

## Error Handling

```python
from autoclean.core.pipeline import Pipeline

try:
    pipeline = Pipeline()
    pipeline.process_file("nonexistent_file.raw", task="rest_eyesopen")
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Handle the error or log it
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## Extending Pipeline

You can extend the Pipeline class to add custom functionality:

```python
from autoclean.core.pipeline import Pipeline

class CustomPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_data = {}
    
    def custom_method(self, param):
        """Add custom functionality"""
        self.logger.info(f"Running custom method with {param}")
        # Implementation
        return result
    
    def get_output_path(self, file_type, *args, **kwargs):
        """Override method to change output path behavior"""
        if file_type == 'custom':
            # Custom logic for 'custom' file type
            return custom_path
        # Fall back to parent implementation for other file types
        return super().get_output_path(file_type, *args, **kwargs)
```
