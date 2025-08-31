User-Defined Variables in Task Files
=====================================

AutoClean EEG v2.1+ supports user-defined variables through JSON headers in task files. This feature allows you to define custom parameters at the top of your task files and access them within your processing pipeline functions.

Overview
--------

User-defined variables are parsed from JSON headers placed at the beginning of Python task files. These variables become available in the ``task_context`` dictionary, which can be accessed throughout your task implementation.

**Benefits:**
- Configure task parameters without modifying the main class code
- Share variables between different methods in your task
- Make tasks more flexible and reusable
- Document task-specific configuration in a structured format

JSON Header Format
------------------

JSON headers must be placed at the very beginning of your task file (after optional shebang and encoding declarations) and enclosed in triple quotes:

.. code-block:: python

    """
    {
      "stimulus_duration": 500,
      "trial_count": 120,
      "response_window": 1000,
      "baseline_period": [-200, 0],
      "custom_threshold": 75.0,
      "experiment_name": "CustomExperiment"
    }
    """
    
    from typing import Any, Dict
    from autoclean.core.task import Task
    
    class MyCustomTask(Task):
        # ... task implementation

Alternative Formats
~~~~~~~~~~~~~~~~~~~

JSON headers are also supported in these formats:

1. **Block Comments** (C-style):

.. code-block:: python

    /*
    {
      "param1": "value1",
      "param2": 42
    }
    */

2. **Triple-quoted strings** (if not a docstring):

.. code-block:: python

    '''
    {
      "setting": "custom_value"
    }
    '''

Accessing Variables
-------------------

JSON header variables are automatically parsed and made available in ``self.task_context`` when your task is instantiated:

.. code-block:: python

    class MyTask(Task):
        def run(self):
            # Access variables with default fallbacks
            duration = self.task_context.get("stimulus_duration", 250)
            count = self.task_context.get("trial_count", 100)
            name = self.task_context.get("experiment_name", "Default")
            
            print(f"Processing {name} with {count} trials")
            print(f"Stimulus duration: {duration}ms")
            
            # Use in processing pipeline
            if duration > 400:
                # Apply different filtering for long stimuli
                self.apply_custom_filter(cutoff=30)

Best Practices
--------------

1. **Use Descriptive Names**: Choose clear, descriptive variable names that indicate their purpose.

.. code-block:: json

    {
      "stimulus_duration_ms": 500,
      "baseline_start_ms": -200,
      "artifact_threshold_uv": 75.0
    }

2. **Provide Defaults**: Always use ``.get()`` with sensible default values when accessing variables.

.. code-block:: python

    # Good - provides fallback
    threshold = self.task_context.get("artifact_threshold_uv", 50.0)
    
    # Avoid - may cause KeyError if variable is missing
    threshold = self.task_context["artifact_threshold_uv"]

3. **Document Variables**: Add comments in your JSON to explain complex parameters.

.. code-block:: python

    """
    {
      "stimulus_duration": 500,
      "trial_count": 120,
      "_comments": {
        "stimulus_duration": "Duration in milliseconds for each stimulus presentation",
        "trial_count": "Total number of trials to process"
      }
    }
    """

4. **Validate Critical Parameters**: Add validation for important variables in your ``run()`` method.

.. code-block:: python

    def run(self):
        trial_count = self.task_context.get("trial_count", 100)
        
        if trial_count < 50:
            raise ValueError("trial_count must be at least 50 for reliable results")

Complete Example
----------------

Here's a complete example demonstrating user-defined variables:

.. code-block:: python

    """
    {
      "stimulus_duration": 500,
      "trial_count": 120,
      "response_window": 1000,
      "baseline_period": [-200, 0],
      "custom_threshold": 75.0,
      "experiment_name": "P300Experiment",
      "filtering": {
        "highpass": 0.1,
        "lowpass": 30.0
      }
    }
    """

    from typing import Any, Dict
    from autoclean.core.task import Task

    class CustomP300Task(Task):
        """Custom P300 task with user-defined parameters."""

        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)

        def run(self) -> None:
            """Run processing with user-defined parameters."""
            # Get user-defined variables
            experiment_name = self.task_context.get("experiment_name", "P300")
            trial_count = self.task_context.get("trial_count", 100)
            stimulus_duration = self.task_context.get("stimulus_duration", 300)
            threshold = self.task_context.get("custom_threshold", 50.0)
            
            # Extract nested configuration
            filtering = self.task_context.get("filtering", {})
            highpass = filtering.get("highpass", 0.1)
            lowpass = filtering.get("lowpass", 40.0)
            
            print(f"Processing {experiment_name}")
            print(f"Trials: {trial_count}, Duration: {stimulus_duration}ms")
            print(f"Filter: {highpass}-{lowpass} Hz, Threshold: {threshold}μV")
            
            # Import raw data
            self.import_raw()
            
            # Apply custom filtering based on user variables
            self.apply_bandpass_filter(l_freq=highpass, h_freq=lowpass)
            
            # Use threshold for artifact detection
            self.detect_artifacts(threshold_uv=threshold)
            
            # Continue with standard pipeline
            self.run_basic_steps()
            self.create_regular_epochs()

Schema Validation
-----------------

Future versions may include schema validation for JSON headers. You can prepare for this by structuring your JSON consistently:

.. code-block:: json

    {
      "version": "1.0",
      "task_type": "ERP",
      "parameters": {
        "stimulus_duration": 500,
        "trial_count": 120
      },
      "processing": {
        "filtering": {"highpass": 0.1, "lowpass": 30.0},
        "artifacts": {"threshold": 75.0, "method": "peak_to_peak"}
      }
    }

Troubleshooting
---------------

**Variables Not Available**
  - Ensure JSON is valid using an online JSON validator
  - Check that JSON header is at the very beginning of the file
  - Verify triple quotes are properly closed

**JSON Parse Errors**
  - Common issues: trailing commas, unquoted strings, single quotes instead of double quotes
  - Use proper JSON format (double quotes for strings, no trailing commas)

**Variables Not Updating**
  - Task classes are cached; restart Python/pipeline after changes
  - Ensure you're editing the correct task file
  - Check file permissions if running in containers

Integration with Existing Config
--------------------------------

JSON header variables complement the existing module-level ``config`` dictionary:

.. code-block:: python

    """
    {
      "custom_param": 42
    }
    """

    # Module-level config (existing pattern)  
    config = {
        "eeg_system": {"montage": "GSN-HydroCel-129"},
        "signal_processing": {"filter": {"highpass": 0.1}}
    }

    class MyTask(Task):
        def run(self):
            # Access JSON header variables
            custom_val = self.task_context.get("custom_param", 0)
            
            # Access module config 
            montage = self.settings.get("eeg_system", {}).get("montage", "auto")

This allows you to use both patterns together - JSON headers for experiment-specific parameters and module config for technical EEG processing settings.