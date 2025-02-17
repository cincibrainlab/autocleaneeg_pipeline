#src/autoclean/tasks/template_task.py
"""Template for implementing new EEG processing tasks.

This template provides a starting point for implementing new EEG processing tasks.
It includes detailed documentation and examples for each required component.

Task Implementation Guide:
1. Copy this template to create your new task file
2. Replace TemplateTask with your task name
3. Implement the required methods
4. Add your task to the Pipeline's TASK_REGISTRY

Configuration Example:
    Your task configuration in autoclean_config.yaml should look like:
    
    ```yaml
    tasks:
      my_new_task:
        mne_task: "my_paradigm"
        description: "Description of your task"
        lossless_config: "path/to/lossless_config.yaml"
    settings:
      resample_step:
        enabled: true
        value: 500
      drop_outerlayer:
        enabled: true
        value: ["E17", "E38", "E43", "E44", "E48", "E49", "E113", "E114", "E119",
                "E120", "E121", "E56", "E63", "E68", "E73", "E81", "E88", "E94",
                "E99", "E107" ]
      eog_step:
        enabled: false
        value: []
      trim_step:
        enabled: true
        value: 4
      crop_step:
        enabled: false
        value:
          start: 0
          end: 60 # null uses full duration
      reference_step:
        enabled: true
        value: ["E129"]
      montage:
        enabled: true
        value: "GSN-HydroCel-124"
      event_id:
        enabled: true
        value: {'DIN2'}
      epoch_settings:
        enabled: true
        value:
          tmin: -0.1
          tmax: 0.5
        remove_baseline: #Remove/Correct baseline
          enabled: true
          window: [null, 0]
        threshold_rejection: #Remove artifact laden epoch based on voltage threshold
          enabled: true
          volt_threshold: 
            eeg: 200e-6

    rejection_policy:
      ch_flags_to_reject: ["noisy", "uncorrelated"]
      ch_cleaning_mode: "interpolate"
      interpolate_bads_kwargs:
        method: "spline"
      ic_flags_to_reject: ["muscle", "heart", "eog", "ch_noise", "line_noise"]
      ic_rejection_threshold: 0.3
      remove_flagged_ics: true 
    ```
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict

# Local imports
from autoclean.core.task import Task
from autoclean.step_functions.io import save_epochs_to_set, step_import, save_raw_to_set
from autoclean.step_functions.continuous import (
    step_clean_bad_channels,
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)
from autoclean.step_functions.epochs import (
    step_create_eventid_epochs,
    step_apply_autoreject,
)
from autoclean.step_functions.reports import (
    step_generate_ica_reports,
    step_plot_ica_full,
    step_plot_raw_vs_cleaned_overlay,
    step_psd_topo_figure,
)


class TemplateTask(Task):
    """Template class for implementing new EEG processing tasks.
    
    This class demonstrates how to implement a new task type in the autoclean package.
    Each task must implement four key methods:
    1. _validate_task_config - Validate task-specific settings
    2. import_data - Load and prepare raw EEG data
    3. preprocess - Apply standard preprocessing steps
    4. process - Implement task-specific analysis
    
    The task should handle a specific EEG paradigm (e.g., resting state, ASSR)
    and implement appropriate processing steps for that paradigm.
    
    Attributes:
        raw (mne.io.Raw): Raw EEG data after import
        pipeline (Any): PyLossless pipeline instance after preprocessing
        cleaned_raw (mne.io.Raw): Preprocessed EEG data
        epochs (mne.Epochs): Epoched data after processing
        
    Example:
        To use this template:
        
        1. Create your configuration:
        ```yaml
        # autoclean_config.yaml
        tasks:
          my_new_task:
            settings:
              resample_step:
                enabled: true
                value: 250
              # ... other settings ...
        ```
        
        2. Initialize and run your task:
        ```python
        >>> from autoclean import Pipeline
        >>> pipeline = Pipeline("output/", "autoclean_config.yaml")
        >>> pipeline.process_file("data.raw", "my_new_task")
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize a new task instance.
        
        Args:
            config: Configuration dictionary containing all settings.
                   See class docstring for configuration example.
        
        Note:
            The parent class handles basic initialization and validation.
            Task-specific setup should be added here if needed.
        """
        # Initialize instance variables
        self.raw = None
        self.pipeline = None
        self.cleaned_raw = None
        self.epochs = None
        
        # Call parent initialization
        super().__init__(config)

    def run(self) -> None:
        """Run the complete processing pipeline for this task.
        
        This method orchestrates the complete processing sequence:
        1. Import raw data
        2. Run preprocessing steps
        3. Apply task-specific processing
        
        The results are automatically saved at each stage according to
        the stage_files configuration.
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If any processing step fails
            
        Note:
            Progress and errors are automatically logged and tracked in
            the database. You can monitor progress through the logging
            messages and final report.
        """
        file_path = Path(self.config['unprocessed_file'])
        self.import_data(file_path)
        self.preprocess()
        self.process()
        
    def import_data(self, file_path: Path) -> None:
        """Import raw EEG data for this task.
        
        This method should handle:
        1. Loading the raw EEG data file
        2. Basic data validation
        3. Any task-specific import preprocessing
        4. Saving the imported data if configured
        
        Args:
            file_path: Path to the EEG data file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            RuntimeError: If import fails
            
        Note:
            The imported data should be stored in self.raw as an MNE Raw object.
            Use save_raw_to_set() to save intermediate results if needed.
        """
        # Import raw data using standard function
        self.raw = step_import(self.config)
        
        # Save imported data if configured
        save_raw_to_set(self.raw, self.config, "post_import")
        
    def preprocess(self) -> None:
        """Run standard preprocessing pipeline.
        
        This method implements the common preprocessing steps:
        1. Basic preprocessing (resampling, filtering)
        2. Bad channel detection
        3. BIDS conversion
        4. PyLossless pipeline
        5. Rejection policy application
        6. Report generation
        
        Each step's results are saved according to the stage_files
        configuration, allowing for quality control and debugging.
        
        Raises:
            RuntimeError: If no data has been imported
            ValueError: If preprocessing parameters are invalid
            RuntimeError: If any preprocessing step fails
            
        Note:
            The preprocessing parameters are read from the task's
            configuration. Modify the config file to adjust parameters.
        """
        if self.raw is None:
            raise RuntimeError("No data has been imported")
            
        # Run preprocessing pipeline and save result
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(self.raw, self.config, "post_prepipeline")
        
        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)
        
        # Run PyLossless pipeline
        self.pipeline, pipeline_raw = step_run_pylossless(self.config)
        save_raw_to_set(pipeline_raw, self.config, "post_pylossless")

        # Clean bad channels
        self.raw = step_clean_bad_channels(self.raw, self.config)
        
        # Apply rejection policy
        self.pipeline, self.cleaned_raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )
        save_raw_to_set(self.cleaned_raw, self.config, "post_rejection_policy")
        
        # Generate visualization reports
        self._generate_reports()

        save_raw_to_set(self.cleaned_raw, self.config, "post_clean_raw")

    def process(self) -> None:
        """Implement task-specific processing steps.
        
        This is where you implement the analysis specific to your task.
        Common operations include:
        1. Epoching the continuous data
        2. Artifact rejection
        3. Feature extraction
        4. Statistical analysis
        
        Override this method to implement your task's unique processing.
        Make sure to document:
        - Required configuration settings
        - Processing steps and their purpose
        - Expected outputs and their format
        - Any assumptions or limitations
        
        Raises:
            RuntimeError: If preprocessing hasn't been completed
            ValueError: If processing parameters are invalid
            RuntimeError: If processing fails
            
        Example:
            ```python
            def process(self):
                if self.cleaned_raw is None:
                    raise RuntimeError("Run preprocess first")
                    
                # Create epochs
                events = mne.make_fixed_length_events(
                    self.cleaned_raw,
                    duration=2.0
                )
                
                self.epochs = mne.Epochs(
                    self.cleaned_raw,
                    events,
                    tmin=0,
                    tmax=2.0,
                    baseline=None,
                    preload=True
                )
                
                # Add your analysis steps here
                
                # Save results
                save_epochs_to_set(
                    self.epochs,
                    self.config,
                    "post_processing"
                )
            ```
        """
        # Implement your task-specific processing here
        pass 
        
    def _generate_reports(self) -> None:
        """Generate quality control visualizations.
        
        Creates standard visualization reports including:
        1. Raw vs cleaned data overlay
        2. ICA components
        3. ICA details
        4. PSD topography
        
        The reports are saved in the debug directory specified
        in the configuration.
        
        Note:
            This is automatically called by preprocess().
            Override this method if you need custom visualizations.
        """
        if self.pipeline is None or self.cleaned_raw is None:
            return
            
        # Plot raw vs cleaned overlay
        step_plot_raw_vs_cleaned_overlay(
            self.pipeline.raw,
            self.cleaned_raw,
            self.pipeline,
            self.config
        )
        
        # Plot ICA components
        step_plot_ica_full(self.pipeline, self.config)
        
        # Generate ICA reports
        step_generate_ica_reports(
            self.pipeline,
            self.cleaned_raw,
            self.config,
            duration=60
        )
        
        # Create PSD topography figure
        step_psd_topo_figure(
            self.pipeline.raw,
            self.cleaned_raw,
            self.pipeline,
            self.config
        )


    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task-specific configuration settings.
        
        This method should check that all required settings for your task
        are present and valid. Common validations include:
        - Required fields exist
        - Field types are correct
        - Values are within valid ranges
        - File paths exist and are accessible
        - Settings are compatible with each other
        
        Args:
            config: Configuration dictionary that has passed common validation.
                   Contains all standard fields plus task-specific settings.
        
        Returns:
            Dict[str, Any]: The validated configuration dictionary.
                           You can add derived settings or defaults.
        
        Raises:
            ValueError: If any required settings are missing or invalid.
            TypeError: If settings are of wrong type.
            
        Example:
            ```python
            def _validate_task_config(self, config):
                # Check required fields
                required_fields = {
                    'eeg_system': str,
                    'settings': dict,
                }
                
                for field, field_type in required_fields.items():
                    if field not in config:
                        raise ValueError(f"Missing required field: {field}")
                    if not isinstance(config[field], field_type):
                        raise TypeError(f"Field {field} must be {field_type}")
                
                # Validate specific settings
                settings = config['settings']
                if 'epoch_length' in settings:
                    if settings['epoch_length'] <= 0:
                        raise ValueError("epoch_length must be positive")
                
                return config
            ```
        """
        # Add your validation logic here
        # This is just an example - customize for your needs
        required_fields = {
            'task': str,
            'eeg_system': str,
            'tasks': dict,
        }
        
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field {field} must be {field_type}")
                
        # Validate stage_files structure
        required_stages = [
            'post_import',
            'post_prepipeline',
            'post_pylossless',
            'post_rejection_policy',
        ]
        
        for stage in required_stages:
            if stage not in config['stage_files']:
                raise ValueError(f"Missing stage in stage_files: {stage}")
            stage_config = config['stage_files'][stage]
            if not isinstance(stage_config, dict):
                raise ValueError(f"Stage {stage} configuration must be a dictionary")
            if 'enabled' not in stage_config:
                raise ValueError(f"Stage {stage} must have 'enabled' field")
            if 'suffix' not in stage_config:
                raise ValueError(f"Stage {stage} must have 'suffix' field")
                
        return config