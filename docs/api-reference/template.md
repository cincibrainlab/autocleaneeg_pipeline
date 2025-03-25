# autoclean.core.Pipeline

```python
class autoclean.core.Pipeline(task_name, data_path, config_path=None, output_path=None, log_level="INFO")
```

Raw data processing pipeline for EEG data.

## Parameters

**task_name** : `str`
: The name of the task to run. Must be a registered task in the task registry.

**data_path** : `path-like`
: Path to the directory containing the data files to process.

**config_path** : `path-like` | `None` (default)
: Path to a configuration file. If None, the default configuration for the specified task will be used.

**output_path** : `path-like` | `None` (default)
: Path to store output files. If None, a default directory will be created based on the task name.

**log_level** : `str` (default: "INFO")
: The logging level to use. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

## Attributes

**task** : `autoclean.core.Task`
: The task instance associated with this pipeline.

**config** : `dict`
: The configuration dictionary for this pipeline.

**data_path** : `pathlib.Path`
: Absolute path to the data directory.

**output_path** : `pathlib.Path`
: Absolute path to the output directory.

## Methods

### Pipeline.run()

```python
def run(self, subject_id=None, session_id=None, run_id=None)
```

Run the pipeline on the specified data.

#### Parameters

**subject_id** : `str` | `None` (default)
: Subject ID to process. If None, all subjects in the data path will be processed.

**session_id** : `str` | `None` (default)
: Session ID to process. If None, all sessions for the selected subjects will be processed.

**run_id** : `str` | `None` (default)
: Run ID to process. If None, all runs for the selected sessions will be processed.

#### Returns

**results** : `dict`
: Dictionary containing the results of the pipeline run.

### Pipeline.load_config()

```python
def load_config(self, config_path=None)
```

Load configuration from a file.

#### Parameters

**config_path** : `path-like` | `None` (default)
: Path to a configuration file. If None, the default configuration for the task will be used.

#### Returns

**config** : `dict`
: Configuration dictionary.

## Examples

```python
from autoclean.core import Pipeline

# Create a pipeline for the ASSR task
pipeline = Pipeline(
    task_name="assr_default",
    data_path="/path/to/data",
    config_path="/path/to/config.yaml",
    output_path="/path/to/output",
    log_level="INFO"
)

# Run the pipeline for a specific subject
results = pipeline.run(subject_id="sub-01")

# Run the pipeline for all subjects
results = pipeline.run()
```

## Notes

The Pipeline class orchestrates the execution of processing steps defined in the associated Task.
It handles file management, logging, and result collection.

Changed in version 0.2.0: Added support for BIDS-formatted data.
