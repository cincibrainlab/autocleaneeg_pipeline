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
        # Import and save raw EEG data
        self.import_raw()

        # Continue with other preprocessing steps
        self.run_basic_steps()

        # Store a copy of the pre-cleaned raw data for comparison in reports
        self.original_raw = self.raw.copy()

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        self.clean_bad_channels(cleaning_method = 'interpolate', reset_bads = True) #reset_bads needs to be true if running ICA later

        self.rereference_data()

        self.annotate_noisy_epochs()

        self.annotate_uncorrelated_epochs()

        # #Segment rejection
        self.detect_dense_oscillatory_artifacts()

        # #ICA
        self.run_ica()

        self.run_ICLabel()

        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_clean_raw",
            flagged=self.flagged,
        )

        # --- EPOCHING BLOCK START ---
        self.create_regular_epochs() # Using fixed-length epochs

        # Prepare epochs for ICA
        self.prepare_epochs_for_ica()

        # Clean epochs using GFP
        self.gfp_clean_epochs()
        # --- EPOCHING BLOCK END ---

        # Generate visualization reports
        self.generate_reports()

    def generate_reports(self) -> None:
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
        if self.raw is None or self.original_raw is None:
            return

        # Plot raw vs cleaned overlay using mixin method
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)

        # Plot PSD topography using mixin method
        self.step_psd_topo_figure(self.original_raw, self.raw)

        # Plot ICA components using mixin method
        self.plot_ica_full()

        # Generate ICA reports using mixin method
        self.generate_ica_reports()

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
            "post_basic_steps", #filtering, cropping, trimming, etc.
            "post_cleaned_raw",
            "post_comp",
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
