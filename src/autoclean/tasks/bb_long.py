# src/autoclean/tasks/resting_eyes_open.py
"""Task implementation for resting state EEG preprocessing."""

# Standard library imports
from typing import Any, Dict

# Local imports
from autoclean.core.task import Task
from autoclean.io.export import save_raw_to_set
from autoclean.io.import_ import import_eeg
from autoclean.step_functions.continuous import (
    step_clean_bad_channels,
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)

# Import the reporting functions directly from the Task class via mixins
# from autoclean.step_functions.reports import (
#     step_generate_ica_reports,
#     step_plot_ica_full,
#     step_plot_raw_vs_cleaned_overlay,
#     step_psd_topo_figure,
# )


class BB_Long(Task): # pylint: disable=invalid-name
    """Task implementation for resting state EEG preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        self.raw = None
        self.pipeline = None
        self.cleaned_raw = None
        self.epochs = None
        super().__init__(config)

    def import_data(self) -> None:
        """Import raw resting state EEG data."""
        # Import and save raw EEG data
        self.raw = import_eeg(self.config)
        save_raw_to_set(self.raw, self.config, "post_import")

    def run(self) -> None:
        """Run the complete resting state processing pipeline."""
        self.import_data()

        # Run preprocessing steps on the raw data.
        if self.raw is None:
            raise RuntimeError("No data has been imported")

        # Run preprocessing pipeline and save intermediate result
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_prepipeline",
            flagged=self.flagged,
        )

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        # Run PyLossless pipeline and save result
        self.pipeline, self.raw = step_run_pylossless(self.config)
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_pylossless",
            flagged=self.flagged,
        )

        # Clean bad channels
        self.pipeline.raw = step_clean_bad_channels(self.raw, self.config)
        save_raw_to_set(
            raw=self.pipeline.raw,
            autoclean_dict=self.config,
            stage="post_bad_channels",
            flagged=self.flagged,
        )

        # # Use PyLossless Rejection Policy
        self.pipeline, self.cleaned_raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )

        save_raw_to_set(
            raw=self.cleaned_raw,
            autoclean_dict=self.config,
            stage="post_rejection_policy",
            flagged=self.flagged,
        )

        # Generate visualization reports
        self._generate_reports()

    def _generate_reports(self) -> None:
        """Generate all visualization reports."""
        if self.pipeline is None or self.cleaned_raw is None:
            return

        # Plot raw vs cleaned overlay using mixin method
        self.plot_raw_vs_cleaned_overlay(
            self.pipeline.raw, self.cleaned_raw, self.config
        )

        # Plot ICA components using mixin method
        self.plot_ica_full(self.pipeline, self.config)

        # # Generate ICA reports using mixin method
        self.generate_ica_reports(
            self.pipeline,
            self.config,
            duration=60,
        )

        # # Create PSD topography figure using mixin method
        self.step_psd_topo_figure(
            self.pipeline.raw, self.cleaned_raw, self.config
        )

    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resting state specific configuration.

        # Args:
        #     config: Configuration dictionary that has passed common validation

        # Returns:
        #     Validated configuration dictionary

        # Raises:
        #     ValueError: If required fields are missing or invalid
        #"""

        required_fields = {
            "task": str,
            "eeg_system": str,
            "tasks": dict,
        }

        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(config[field], field_type):
                raise ValueError(f"Field {field} must be of type {field_type}")

        # Validate stage_files structure
        required_stages = [
            "post_import",
            "post_prepipeline",
            "post_pylossless",
            "post_bad_channels",
            "post_rejection_policy",
            "post_cleaned_raw",
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
