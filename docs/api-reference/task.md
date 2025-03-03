# Task API Reference

The `Task` class is a central component of the AutoClean Pipeline architecture. It defines the specific processing workflow for an EEG paradigm.

## Task Base Class

```python
from autoclean.core.task import Task
```

### Class Definition

```python
class Task:
    """Base class for all EEG processing tasks.
    
    A task defines a specific processing workflow for a particular EEG paradigm.
    Each task must implement a run() method that contains the processing logic.
    
    Attributes
    ----------
    name : str
        The name of the task
    config : dict
        Configuration parameters for the task
    CONFIG_SCHEMA : dict
        JSON schema for validating task configuration
    """
```

### Constructor

```python
def __init__(self, config=None, name=None):
    """Initialize a task with configuration.
    
    Parameters
    ----------
    config : dict, optional
        Configuration parameters for the task
    name : str, optional
        The name of the task. If not provided, uses class name.
    """
```

### Methods

#### run

```python
def run(self, pipeline):
    """Execute the task on the provided pipeline.
    
    This is the main method that must be implemented by all task subclasses.
    It should execute all the necessary steps to process the EEG data
    according to the specific paradigm.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance containing the EEG data to process
    
    Returns
    -------
    None
        The pipeline is modified in-place
    
    Raises
    ------
    NotImplementedError
        If the subclass does not implement this method
    """
    raise NotImplementedError("Task subclasses must implement run() method")
```

#### validate_config

```python
def validate_config(self):
    """Validate the task configuration against the CONFIG_SCHEMA.
    
    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    
    Raises
    ------
    ValidationError
        If the configuration does not match the schema
    """
```

#### get_config_schema

```python
@classmethod
def get_config_schema(cls):
    """Get the schema for validating task configuration.
    
    Returns
    -------
    dict
        JSON schema for validating task configuration
    """
    return cls.CONFIG_SCHEMA
```

### Constants

#### CONFIG_SCHEMA

Default configuration schema that all tasks must follow:

```python
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        # Base properties that all tasks support
    },
    "additionalProperties": True
}
```

## Creating Custom Tasks

To create a custom task, extend the `Task` base class and implement the `run()` method:

```python
class MyCustomTask(Task):
    """Custom task for specific EEG processing needs."""
    
    # Define the configuration schema for this task
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "epoch_length": {"type": "number", "minimum": 0.1},
            "overlap": {"type": "number", "minimum": 0, "maximum": 0.99},
            "custom_parameter": {"type": "string"}
        },
        "required": ["epoch_length"]
    }
    
    def run(self, pipeline):
        """Execute the processing steps for this task.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance containing the EEG data
        """
        # Get configuration parameters with defaults
        epoch_length = self.config.get("epoch_length", 1.0)
        overlap = self.config.get("overlap", 0.5)
        
        # Log task execution
        pipeline.logger.info(f"Running {self.name} with epoch_length={epoch_length}, overlap={overlap}")
        
        # Execute processing steps
        from autoclean.step_functions import preprocessing, artifacts, ica
        
        # Preprocessing
        preprocessing.resample(pipeline)
        preprocessing.filter_raw(pipeline)
        
        # Artifact detection and rejection
        artifacts.detect_bad_channels(pipeline)
        
        # ICA for artifact removal
        ica.fit_ica(pipeline)
        ica.identify_eog_components(pipeline)
        ica.apply_ica(pipeline)
        
        # Add custom processing
        self._custom_processing(pipeline, epoch_length, overlap)
        
        # Generate reports
        self._generate_reports(pipeline)
    
    def _custom_processing(self, pipeline, epoch_length, overlap):
        """Custom processing specific to this task.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        epoch_length : float
            Length of epochs in seconds
        overlap : float
            Overlap between epochs (0-1)
        """
        # Task-specific processing logic
        pass
    
    def _generate_reports(self, pipeline):
        """Generate task-specific reports.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        """
        # Generate visualizations and metrics
        pass
```

## Task Registration

Tasks must be registered in the pipeline system to be available:

```python
from autoclean.core.pipeline import TASK_REGISTRY
from my_module import MyCustomTask

# Register the task
TASK_REGISTRY["my_custom_task"] = MyCustomTask
```

## Built-in Tasks

### RestingEyesOpen

Task for processing resting state EEG with eyes open.

```python
from autoclean.tasks.resting_eyes_open import RestingEyesOpen

# Configuration
config = {
    "epoch_length": 2.0,
    "overlap": 0.5,
    "min_duration": 60  # Minimum recording duration in seconds
}

# Create task
task = RestingEyesOpen(config=config)
```

### AssrDefault

Task for processing auditory steady-state response (ASSR) paradigms.

```python
from autoclean.tasks.assr_default import AssrDefault

# Configuration
config = {
    "stim_frequencies": [40, 20],  # Stimulation frequencies in Hz
    "tmin": -0.2,                  # Time before event in seconds
    "tmax": 0.8                    # Time after event in seconds
}

# Create task
task = AssrDefault(config=config)
```

### ChirpDefault

Task for processing auditory chirp paradigms.

```python
from autoclean.tasks.chirp_default import ChirpDefault

# Configuration
config = {
    "freq_range": [1, 100],        # Frequency range in Hz
    "tmin": -0.5,                  # Time before event in seconds
    "tmax": 2.0                    # Time after event in seconds
}

# Create task
task = ChirpDefault(config=config)
```

### HBCD_MMN

Task for processing mismatch negativity (MMN) paradigms.

```python
from autoclean.tasks.hbcd_mmn import HBCD_MMN

# Configuration
config = {
    "standard_id": 1,              # Event ID for standard stimuli
    "deviant_id": 2,               # Event ID for deviant stimuli
    "tmin": -0.2,                  # Time before event in seconds
    "tmax": 0.5                    # Time after event in seconds
}

# Create task
task = HBCD_MMN(config=config)
```

## Task Lifecycle

Tasks go through the following lifecycle:

1. **Construction**: The task is instantiated with configuration
2. **Validation**: Configuration is validated against the schema
3. **Execution**: The `run()` method is called with a pipeline instance
4. **Termination**: The task completes and control returns to the pipeline

## Task Best Practices

1. **Validate inputs**: Always validate the configuration using `CONFIG_SCHEMA`
2. **Use defaults**: Provide sensible defaults for optional parameters
3. **Clear structure**: Organize processing into logical steps
4. **Error handling**: Catch and handle errors appropriately
5. **Logging**: Log important information about task execution
6. **Clean implementation**: Keep the `run()` method clean by moving details to helper methods
7. **Documentation**: Document the task's purpose, parameters, and processing logic
