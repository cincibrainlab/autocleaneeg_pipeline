# src/autoclean/tasks/template_task.py
"""Template for implementing new EEG processing tasks.

This template provides a starting point for implementing new EEG processing tasks.
It includes detailed documentation and examples for each required component.

Task Implementation Guide:
1. Copy this template to create your new task file (e.g., my_paradigm.py)
2. Replace TemplateTask with your task name (e.g., MyParadigmTask)
3. Implement the required methods

Key Components:
1. Task Configuration:
   - Define processing steps in autoclean_config.yaml
   - Each step has enabled/disabled flag and parameters
   - Configure artifact rejection policies
   
2. Required Methods:
   - __init__: Initialize task state
   - run: Main processing pipeline
   - import_data: Load and validate raw data
   - _generate_reports: Create quality control visualizations
   - _validate_task_config: Validate task settings

3. Processing Flow:
   a. Data Import:
      - Load raw EEG data
      - Apply montage
      - Basic validation
   b. Preprocessing:
      - Resampling
      - Filtering
      - Bad channel detection
   c. Artifact Rejection:
      - ICA decomposition
      - Component classification
      - Bad segment detection
   d. Task-Specific Processing:
      - Epoching
      - Baseline correction
      - Additional analyses
   e. Quality Control:
      - Generate reports
      - Save processed data
      - Update processing log

4. Best Practices:
   - Use type hints for all methods
   - Add comprehensive docstrings
   - Include error handling
   - Log processing steps
   - Save intermediate results
   - Generate quality control reports
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

# Third-party imports
import mne
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from autoclean.core.task import Task
from autoclean.step_functions.continuous import (
    step_clean_bad_channels,
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)
from autoclean.step_functions.io import save_raw_to_set, import_eeg
# Import the reporting functions directly from the Task class via mixins
# # Import the reporting functions directly from the Task class via mixins
# from autoclean.step_functions.reports import (
#     step_generate_ica_reports,
    step_plot_ica_full,
    step_plot_raw_vs_cleaned_overlay,
    step_psd_topo_figure,

# )
from autoclean.utils.logging import message
from autoclean.utils.database import manage_database


class TemplateTask(Task):
    """Template class for implementing new EEG processing tasks.

    This class demonstrates how to implement a new task type in the autoclean package.
    Each task must implement these key methods:
    1. __init__ - Initialize task state and validate config
    2. run - Main processing pipeline
    3. import_data - Load and prepare raw EEG data
    4. _generate_reports - Create quality control visualizations
    5. _validate_task_config - Validate task-specific settings

    The task should handle a specific EEG paradigm (e.g., resting state, ASSR, MMN)
    and implement appropriate processing steps for that paradigm.

    Attributes:
        raw (mne.io.Raw): Raw EEG data after import
        pipeline (Any): PyLossless pipeline instance after preprocessing
        cleaned_raw (mne.io.Raw): Preprocessed EEG data
        epochs (mne.Epochs): Epoched data after processing
        config (Dict[str, Any]): Task configuration dictionary

    Example:
        To use this template:

        1. Create your configuration in autoclean_config.yaml:
        ```yaml
        # autoclean_config.yaml
        tasks:
          my_paradigm:
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
        >>> pipeline.process_file("data.raw", "my_paradigm")
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize a new task instance.

        Args:
            config: Configuration dictionary containing all settings.
                   See class docstring for configuration example.

        Note:
            The parent class handles basic initialization and validation.
            Task-specific setup should be added here if needed.
        """
        # Initialize instance variables
        self.raw: Optional[mne.io.Raw] = None
        self.pipeline: Optional[Any] = None
        self.cleaned_raw: Optional[mne.io.Raw] = None
        self.epochs: Optional[mne.Epochs] = None

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

        Processing Steps:
        1. Import and validate raw data
        2. Apply preprocessing pipeline
        3. Create BIDS-compliant paths
        4. Run PyLossless pipeline
        5. Clean bad channels
        6. Apply rejection policy
        7. Generate reports
        8. Save processed data

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If any processing step fails

        Note:
            Progress and errors are automatically logged and tracked in
            the database. You can monitor progress through the logging
            messages and final report.
        """
        # Import raw data
        file_path = Path(self.config.get("unprocessed_file", ""))
        self.import_data(file_path)

        """Run preprocessing steps on the raw data."""
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

        # Save final cleaned data
        save_raw_to_set(self.cleaned_raw, self.config, "post_clean_raw")

    def import_data(self, file_path: Path) -> None:
        """Import raw EEG data for this task.

        This method handles:
        1. Loading the raw EEG data file
        2. Basic data validation
        3. Task-specific import preprocessing
        4. Saving imported data


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
        self.raw = import_eeg(self.config)

        # Save imported data if configured
        save_raw_to_set(self.raw, self.config, "post_import")



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
            This is automatically called by run().
            Override this method if you need custom visualizations.
        """
        if self.pipeline is None or self.cleaned_raw is None:
            return

        # Plot raw vs cleaned overlay using mixin method
        self.plot_raw_vs_cleaned_overlay(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )

        # Plot ICA components using mixin method
        self.plot_ica_full(self.pipeline, self.config)

        # Generate ICA reports using mixin method
        self.plot_ica_components(
            self.pipeline.ica2, self.cleaned_raw, self.config, self.pipeline, duration=60
        
        )

        # Create PSD topography figure using mixin method
        self.psd_topo_figure(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )

    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task-specific configuration settings.

        This method checks that all required settings for your task
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
            ValueError: If any required settings are missing or invalid
            TypeError: If settings are of wrong type

        Example:
            ```python
            def _validate_task_config(self, config):
                required_fields = {
                    'eeg_system': str,
                    'settings': dict,
                }
                for field, field_type in required_fields.items():
                    if field not in config:
                        raise ValueError(f"Missing required field: {field}")
                    if not isinstance(config[field], field_type):
                        raise TypeError(f"Field {field} must be {field_type}")
                return config
            ```
        """
        # Add your validation logic here
        required_fields = {
            "task": str,
            "eeg_system": str,
            "tasks": dict,
        }

        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field {field} must be {field_type}")

        # Validate stage_files structure
        required_stages = [
            "post_import",
            "post_prepipeline",
            "post_pylossless",
            "post_rejection_policy",
        ]

        for stage in required_stages:
            if stage not in config["stage_files"]:
                raise ValueError(f"Missing stage in stage_files: {stage}")
            stage_config = config["stage_files"][stage]
            if not isinstance(stage_config, dict):
                raise ValueError(f"Stage {stage} configuration must be a dictionary")
            if "enabled" not in stage_config:
                raise ValueError(f"Stage {stage} must have 'enabled' field")
            if "suffix" not in stage_config:
                raise ValueError(f"Stage {stage} must have 'suffix' field")

        return config
        
    def resample_data(self, data: Union[mne.io.Raw, mne.Epochs], stage_name: str = "resampled") -> Union[mne.io.Raw, mne.Epochs]:
        """Resample raw or epoched data based on configuration settings.
        
        This method checks the resample_step toggle in the configuration and applies
        resampling if enabled. It works with both Raw and Epochs objects.
        
        Args:
            data: The MNE Raw or Epochs object to resample
            stage_name: Name for saving the resampled data (default: "resampled")
            
        Returns:
            The resampled data object (same type as input)
            
        Raises:
            TypeError: If data is not a Raw or Epochs object
            RuntimeError: If resampling fails
        """
        if not isinstance(data, (mne.io.Raw, mne.Epochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object")
            
        task = self.config.get("task")
        
        # Check if resampling is enabled in the configuration
        resample_enabled = self.config.get("tasks", {}).get(task, {}).get("settings", {}).get(
            "resample_step", {}).get("enabled", False)
            
        if not resample_enabled:
            message("info", "Resampling step is disabled in configuration")
            return data
            
        # Get target sampling frequency
        target_sfreq = self.config.get("tasks", {}).get(task, {}).get("settings", {}).get(
            "resample_step", {}).get("value")
            
        if target_sfreq is None:
            message("warning", "Target sampling frequency not specified, skipping resampling")
            return data
            
        # Check if we need to resample (avoid unnecessary resampling)
        current_sfreq = data.info["sfreq"]
        if abs(current_sfreq - target_sfreq) < 0.01:  # Small threshold to account for floating point errors
            message("info", f"Data already at target frequency ({target_sfreq} Hz), skipping resampling")
            return data
            
        message("header", f"Resampling data from {current_sfreq} Hz to {target_sfreq} Hz...")
        
        try:
            # Resample based on data type
            if isinstance(data, mne.io.Raw):
                resampled_data = data.copy().resample(target_sfreq)
                # Save resampled raw data if it's a Raw object
                save_raw_to_set(resampled_data, self.config, f"post_{stage_name}")
            else:  # Epochs
                resampled_data = data.copy().resample(target_sfreq)
                
            message("info", f"Data successfully resampled to {target_sfreq} Hz")
            
            # Update metadata
            metadata = {
                "resampling": {
                    "creationDateTime": datetime.now().isoformat(),
                    "original_sfreq": current_sfreq,
                    "target_sfreq": target_sfreq,
                    "data_type": "raw" if isinstance(data, mne.io.Raw) else "epochs"
                }
            }
            
            run_id = self.config.get("run_id")
            manage_database(
                operation="update", update_record={"run_id": run_id, "metadata": metadata}
            )
            
            return resampled_data
            
        except Exception as e:
            message("error", f"Error during resampling: {str(e)}")
            raise RuntimeError(f"Failed to resample data: {str(e)}") from e
