# src/autoclean/tasks/hbcd_mmn.py
# Standard library imports
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import mne
import numpy as np

# Local imports
from autoclean.core.task import Task
from autoclean.step_functions.io import save_epochs_to_set, step_import, save_raw_to_set
from autoclean.utils.logging import message
from autoclean.step_functions.reports import (
    step_generate_ica_reports,
    step_plot_ica_full,
    step_psd_topo_figure,
    generate_mmn_erp,
)
from autoclean.step_functions.continuous import (
    step_detect_dense_oscillatory_artifacts,
    step_clean_bad_channels,
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)
from autoclean.step_functions.epochs import (
    step_create_eventid_epochs,
)


class HBCD_MMN(Task):

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

    def _verify_annotations(self, raw, stage_name=""):
        """Verify that annotations are properly synchronized after each processing stage."""
        if not hasattr(self, "_original_annotations"):
            # Store only the event annotations we want to track (DIN2 events)
            mask = np.array(["DIN2" in desc for desc in raw.annotations.description])
            self._original_annotations = mne.Annotations(
                onset=raw.annotations.onset[mask],
                duration=raw.annotations.duration[mask],
                description=raw.annotations.description[mask],
            )
            self._original_sfreq = raw.info["sfreq"]
            self._trim_amount = 0  # Initialize trim amount
            return True

        # Filter current annotations to only include DIN2 events
        mask = np.array(["DIN2" in desc for desc in raw.annotations.description])
        current_onsets = raw.annotations.onset[mask]

        # Update trim amount if this is the post_prepipeline stage
        if stage_name == "post_prepipeline":
            trim_settings = self.config["tasks"][self.config["task"]]["settings"][
                "trim_step"
            ]
            if trim_settings["enabled"]:
                self._trim_amount = trim_settings["value"]
                message(
                    "info", f"Updated trim amount to {self._trim_amount}s from config"
                )

        # Adjust original onsets based on trim
        adjusted_onsets = self._original_annotations.onset - self._trim_amount

        # Only compare events that should still be in the data
        # (after trimming from start and end)
        valid_mask = (adjusted_onsets >= 0) & (adjusted_onsets <= raw.times[-1])
        adjusted_onsets = adjusted_onsets[valid_mask]
        original_desc = self._original_annotations.description[valid_mask]

        # Get matching subset of current events
        current_desc = raw.annotations.description[mask]
        matching_events = []
        for i, desc in enumerate(current_desc):
            if desc in original_desc:
                matching_events.append(i)
        current_onsets = current_onsets[matching_events]

        if len(current_onsets) != len(adjusted_onsets):
            message("warning", f"✗ Number of DIN2 events changed at {stage_name}:")
            message(
                "warning",
                f"Original count (after trim adjustment): {len(adjusted_onsets)}",
            )
            message("warning", f"Current count: {len(current_onsets)}")
            return False

        try:
            np.testing.assert_array_almost_equal(
                current_onsets, adjusted_onsets, decimal=6
            )
            message("info", f"✓ DIN2 event timing verified at {stage_name}")
            return True
        except AssertionError:
            # Find first mismatch for detailed analysis
            mismatch_idx = np.where(np.abs(current_onsets - adjusted_onsets) > 1e-6)[0][
                0
            ]
            message("warning", f"✗ DIN2 event timing changed at {stage_name}:")
            message("warning", f"First mismatch at event {mismatch_idx}:")
            message(
                "warning",
                f"Original onset time (adjusted for trim): {adjusted_onsets[mismatch_idx]:.6f}",
            )
            message(
                "warning", f"Current onset time: {current_onsets[mismatch_idx]:.6f}"
            )
            message(
                "warning",
                f"Difference: {current_onsets[mismatch_idx] - adjusted_onsets[mismatch_idx]:.6f}s",
            )
            message("warning", f"Original sfreq: {self._original_sfreq}")
            message("warning", f"Current sfreq: {raw.info['sfreq']}")
            return False

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
        file_path = Path(self.config["unprocessed_file"])
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

        # Verify initial annotations
        self._verify_annotations(self.raw, "post_import")

        # Save imported data if configured
        save_raw_to_set(self.raw, self.config, "post_import")

    def preprocess(self) -> None:
        """Run standard preprocessing pipeline.

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
        # self._verify_annotations(self.raw, "post_prepipeline")
        save_raw_to_set(self.raw, self.config, "post_prepipeline")

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        # Run PyLossless pipeline
        self.pipeline, pipeline_raw = step_run_pylossless(self.config)
        # self._verify_annotations(pipeline_raw, "post_pylossless")
        save_raw_to_set(pipeline_raw, self.config, "post_pylossless")

        self.raw = step_clean_bad_channels(self.raw, self.config)

        # Apply rejection policy
        self.pipeline, self.cleaned_raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )
        self._verify_annotations(self.cleaned_raw, "post_rejection_policy")

        self.cleaned_raw = step_detect_dense_oscillatory_artifacts(
            self.cleaned_raw,
            window_size_ms=100,
            channel_threshold_uv=50,
            min_channels=75,
            padding_ms=500,
        )
        save_raw_to_set(self.cleaned_raw, self.config, "post_clean_raw")

        # Generate visualization reports
        # self._generate_reports()

    def process(self) -> None:
        """Process the MMN task data."""
        if self.cleaned_raw is None:
            raise RuntimeError("Preprocessing must be completed first")

        """Implement task-specific processing steps."""
        self.epochs = step_create_eventid_epochs(
            self.cleaned_raw, self.pipeline, self.config
        )
        save_epochs_to_set(self.epochs, self.config, "post_epochs")
        if self.epochs is None:
            message("error", "Failed to create epochs")
            return

        # Run MMN analysis
        generate_mmn_erp(self.epochs, self.pipeline, self.config)

        # Save final epochs
        save_epochs_to_set(self.epochs, self.config, "post_comp")

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

        # Plot ICA components
        step_plot_ica_full(self.pipeline, self.config)

        # Generate ICA reports
        step_generate_ica_reports(
            self.pipeline, self.cleaned_raw, self.config, duration=60
        )

        # Create PSD topography figure
        step_psd_topo_figure(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )
