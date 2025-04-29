# src/autoclean/tasks/hbcd_VEP.py
# Standard library imports
from typing import Any, Dict

# Local imports
from autoclean.core.task import Task
from autoclean.io.export import save_raw_to_set
from autoclean.step_functions.continuous import (
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
)
from autoclean.utils.logging import message

# Third-party imports



class HBCD_VEP(Task):
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

    def run(self) -> None:
        """Run the complete processing pipeline for this task

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
        # Import and save raw EEG data
        self.import_raw()

        message("header", "Running preprocessing steps")

        # Continue with other preprocessing steps
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_prepipeline",
            flagged=self.flagged,
        )

        # Store a copy of the pre-cleaned raw data for comparison in reports
        self.original_raw = self.raw.copy()

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        # Run PyLossless Pipeline
        self.pipeline, self.raw = self.step_custom_pylossless_pipeline(
            self.config, eog_channel="E25"
        )

        # Add more artifact detection steps
        self.detect_dense_oscillatory_artifacts()

        # Apply PyLossless Rejection Policy for artifact removal and channel interpolation
        self.pipeline, self.raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_rejection_policy",
            flagged=self.flagged,
        )

        self.detect_dense_oscillatory_artifacts()

        # Clean bad channels post ICA
        self.clean_bad_channels(cleaning_method="interpolate")

        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_clean_raw",
            flagged=self.flagged,
        )

        self.create_eventid_epochs()

        self._generate_reports()

    def _generate_reports(self) -> None:
        """Generate quality control visualizations and reports.

        Creates standard visualization reports including:
        1. Raw vs cleaned data overlay
        2. ICA components
        3. ICA details
        4. PSD topography

        The reports are saved in the debug directory specified
        in the configuration.

        Note:
            This is automatically called by run().
        """
        if self.pipeline is None or self.raw is None or self.original_raw is None:
            return

        # Plot raw vs cleaned overlay using mixin method
        self.plot_raw_vs_cleaned_overlay(
            self.original_raw, self.raw, self.pipeline, self.config
        )

        # Plot ICA components using mixin method
        self.plot_ica_full(self.pipeline, self.config)

        # Generate ICA reports using mixin method
        self.generate_ica_reports(self.pipeline, self.config)

        # Create PSD topography figure using mixin method
        self.step_psd_topo_figure(
            self.original_raw, self.raw, self.pipeline, self.config
        )
