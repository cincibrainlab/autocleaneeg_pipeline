User-Defined Variables in Task Configuration
============================================

AutoClean EEG provides a simple and powerful way to define custom variables in your EEG processing tasks through the existing configuration system. This allows you to customize your preprocessing pipeline with experiment-specific parameters without modifying the core codebase.

Overview
--------

Every task file can include a ``config`` dictionary containing both standard EEG processing settings and your own custom variables. The Task base class automatically detects and loads these variables, making them accessible throughout your processing pipeline via ``self.settings``.

.. note::
   This feature uses the existing configuration system that has been part of AutoClean EEG since v2.0.0. No additional setup or dependencies are required.

Basic Usage
-----------

1. **Define Variables in Config Dictionary**
   
   Add your custom variables directly to the ``config`` dictionary in your task file:

   .. code-block:: python

      config = {
          # Standard EEG settings
          "resample_step": {"enabled": True, "value": 250},
          "filtering": {"enabled": True, "value": {"l_freq": 1, "h_freq": 100}},
          
          # Your custom variables
          "stimulus_duration": 500,
          "trial_count": 120,
          "custom_threshold": 75.0,
          "experiment_conditions": ["condition_A", "condition_B", "control"]
      }

2. **Access Variables in Your Task**
   
   Use ``self.settings.get()`` to safely access your variables with optional defaults:

   .. code-block:: python

      class MyTask(Task):
          def run(self):
              # Access your custom variables with defaults
              duration = self.settings.get("stimulus_duration", 250)
              count = self.settings.get("trial_count", 100)
              conditions = self.settings.get("experiment_conditions", ["default"])
              
              # Use in your processing pipeline
              self._configure_epochs(duration=duration, count=count)

Variable Types
--------------

You can define any type of Python variable in your configuration:

**Simple Values**

.. code-block:: python

   config = {
       "stimulus_duration": 500,          # Integer
       "response_threshold": 75.5,        # Float  
       "participant_group": "adults",     # String
       "use_custom_filter": True,         # Boolean
   }

**Collections**

.. code-block:: python

   config = {
       "target_frequencies": [10, 20, 40],                    # List of numbers
       "condition_names": ["baseline", "task", "rest"],       # List of strings
       "channel_mapping": {"Fz": 1, "Cz": 2, "Pz": 3},      # Dictionary
   }

**Nested Structures**

.. code-block:: python

   config = {
       "analysis_parameters": {
           "window_size": 100,
           "overlap": 50,
           "method": "custom_algorithm",
           "advanced_settings": {
               "tolerance": 0.001,
               "iterations": 1000
           }
       }
   }

Complete Example
----------------

Here's a complete task file demonstrating user-defined variables:

.. code-block:: python

   """Custom task with user-defined variables."""
   from autoclean.core.task import Task

   config = {
       # Standard EEG processing settings
       "resample_step": {"enabled": True, "value": 250},
       "filtering": {
           "enabled": True,
           "value": {"l_freq": 1, "h_freq": 100}
       },
       "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
       
       # Custom experiment variables
       "stimulus_duration": 500,
       "trial_count": 120,
       "response_window": 1000,
       "baseline_period": [-200, 0],
       "experiment_conditions": ["condition_A", "condition_B"],
       "custom_analysis": {
           "method": "custom_algorithm",
           "threshold": 2.5,
           "window_size": 50
       }
   }

   class MyExperiment(Task):
       def run(self):
           # Import data
           self.import_raw()
           
           # Access custom variables
           duration = self.settings.get("stimulus_duration", 250)
           conditions = self.settings.get("experiment_conditions", [])
           analysis = self.settings.get("custom_analysis", {})
           
           print(f"Processing {len(conditions)} conditions")
           print(f"Stimulus duration: {duration}ms")
           print(f"Analysis method: {analysis.get('method', 'standard')}")
           
           # Use in processing
           self._apply_custom_processing(duration, conditions, analysis)
           
           # Continue with standard pipeline
           self.resample_data()
           self.filter_data()
           self.run_ica()
           self.create_regular_epochs()
       
       def _apply_custom_processing(self, duration, conditions, analysis):
           """Apply processing using custom parameters."""
           # Your custom processing logic here
           threshold = analysis.get("threshold", 1.0)
           method = analysis.get("method", "standard")
           
           print(f"Applying {method} with threshold {threshold}")
           # ... your custom processing code ...

Best Practices
--------------

**Use Descriptive Names**

Choose clear, descriptive variable names that explain their purpose:

.. code-block:: python

   # Good
   "stimulus_duration_ms": 500,
   "response_window_ms": 1000,
   "artifact_rejection_threshold": 75.0
   
   # Avoid
   "dur": 500,
   "win": 1000, 
   "thresh": 75.0

**Provide Default Values**

Always use ``self.settings.get()`` with sensible defaults:

.. code-block:: python

   # Good - safe with fallback
   duration = self.settings.get("stimulus_duration", 250)
   
   # Risky - will error if variable not defined
   duration = self.settings["stimulus_duration"]

**Group Related Variables**

Use nested dictionaries for related parameters:

.. code-block:: python

   config = {
       "epoch_parameters": {
           "tmin": -0.5,
           "tmax": 1.0,
           "baseline": [-0.2, 0],
           "reject_criteria": {"eeg": 100e-6}
       },
       "analysis_settings": {
           "method": "custom",
           "window_size": 100,
           "overlap": 50
       }
   }

**Document Your Variables**

Add comments explaining what each custom variable does:

.. code-block:: python

   config = {
       # Custom experiment parameters
       "stimulus_duration": 500,        # Duration of each stimulus in ms
       "trial_count": 120,             # Expected number of trials per condition
       "response_window": 1000,        # Time window for response detection in ms
       "baseline_period": [-200, 0],   # Baseline correction window in ms
   }

Integration with Processing Pipeline
------------------------------------

Your custom variables integrate seamlessly with all AutoClean processing steps:

**Custom Filtering**

.. code-block:: python

   def run(self):
       # Standard filtering
       self.filter_data()
       
       # Additional custom filtering based on your variables
       custom_freq = self.settings.get("custom_notch_frequency", 50)
       if custom_freq:
           self._apply_custom_notch_filter(custom_freq)

**Custom Epoching**

.. code-block:: python

   def run(self):
       # Access custom epoch parameters
       epoch_params = self.settings.get("custom_epochs", {})
       tmin = epoch_params.get("tmin", -0.5)
       tmax = epoch_params.get("tmax", 1.0)
       
       # Apply custom epoching
       self._create_custom_epochs(tmin=tmin, tmax=tmax)

**Custom Analysis**

.. code-block:: python

   def run(self):
       # Standard pipeline
       self.import_raw()
       self.run_basic_steps()
       
       # Custom analysis using your parameters
       analysis_method = self.settings.get("analysis_method", "standard")
       if analysis_method == "custom":
           self._run_custom_analysis()

Validation and Error Handling
------------------------------

**Validate Custom Variables**

Add validation for your custom variables to catch errors early:

.. code-block:: python

   def run(self):
       # Validate required custom variables
       required_vars = ["stimulus_duration", "trial_count"]
       for var in required_vars:
           if var not in self.settings:
               raise ValueError(f"Required custom variable '{var}' not found in config")
       
       # Validate ranges
       duration = self.settings.get("stimulus_duration", 0)
       if duration <= 0:
           raise ValueError(f"stimulus_duration must be positive, got {duration}")

**Handle Missing Variables Gracefully**

.. code-block:: python

   def run(self):
       # Use defaults for optional variables
       use_advanced_processing = self.settings.get("use_advanced_processing", False)
       
       if use_advanced_processing:
           self._run_advanced_analysis()
       else:
           self._run_standard_analysis()

Advanced Usage
--------------

**Dynamic Configuration**

You can modify configuration based on other variables:

.. code-block:: python

   def run(self):
       participant_age = self.settings.get("participant_age", 25)
       
       # Adjust processing based on age
       if participant_age < 18:
           filter_settings = self.settings.get("pediatric_filter", {})
       else:
           filter_settings = self.settings.get("adult_filter", {})
       
       self._apply_age_appropriate_filtering(filter_settings)

**Conditional Processing**

.. code-block:: python

   def run(self):
       experiment_type = self.settings.get("experiment_type", "standard")
       
       # Different processing pipelines based on experiment type
       if experiment_type == "resting_state":
           self._process_resting_state()
       elif experiment_type == "task_based":
           self._process_task_based()
       else:
           self._process_standard()

Migration from Other Systems
----------------------------

If you're migrating from other configuration systems, here's how to adapt:

**From JSON Headers → Config Variables**

.. code-block:: python

   # Old approach (JSON in comments) - NOT RECOMMENDED
   """
   {
     "stimulus_duration": 500,
     "trial_count": 120
   }
   """
   
   # New approach (config dictionary) - RECOMMENDED
   config = {
       "stimulus_duration": 500,
       "trial_count": 120
   }

**From External Config Files → Embedded Config**

.. code-block:: python

   # Old approach - separate config file
   # config.yaml:
   # custom_vars:
   #   stimulus_duration: 500
   #   trial_count: 120
   
   # New approach - embedded in task file
   config = {
       "stimulus_duration": 500,
       "trial_count": 120
   }

Summary
-------

User-defined variables in AutoClean EEG provide a powerful way to customize your processing pipeline:

- **Simple**: Add variables directly to the ``config`` dictionary
- **Accessible**: Use ``self.settings.get("var_name", default)`` anywhere in your task
- **Flexible**: Support any Python data type (numbers, strings, lists, dictionaries)
- **Safe**: Built-in support for default values and error handling
- **Integrated**: Works seamlessly with all AutoClean processing steps

This approach leverages the existing configuration system, making it both powerful and familiar to use. No additional setup or dependencies are required - just define your variables and start using them!

For a complete working example, see ``examples/user_defined_variables_example.py`` in the AutoClean EEG repository.