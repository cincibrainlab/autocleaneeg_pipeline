# src/autoclean/tasks/chirp_default.py
"""Task implementation for chirp EEG preprocessing."""

from typing import Any, Dict

from ..core.task import Task
from ..io.export import save_raw_to_set
from ..step_functions.continuous import (
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)

# Import the reporting functions directly from the Task class via mixins
# # Import the reporting functions directly from the Task class via mixins
# from autoclean.step_functions.reports import step_generate_ica_reports, step_plot_ica_full, step_plot_raw_vs_cleaned_overlay, step_psd_topo_figure


class ChirpDefault(Task):
    """Task implementation for chirp EEG preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        self.raw = None
        self.pipeline = None
        self.cleaned_raw = None
        self.epochs = None
        super().__init__(config)

    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate chirp specific configuration.

        # Args:
        #     config: Configuration dictionary that has passed common validation

        # Returns:
        #     Validated configuration dictionary

        # Raises:
        #     ValueError: If required fields are missing or invalid
        #"""
        # # Validate chirp specific fields
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
            "post_rejection_policy",
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

    def run(self) -> None:
        """Run the complete chirp processing pipeline."""
        # Import and save raw EEG data
        self.import_raw()

        # Note: step_pre_pipeline_processing will skip resampling if already done
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_prepipeline",
            flagged=self.flagged,
        )

        # Store a copy of the pre-cleaned raw data for comparison in reports
        self.original_raw = self.raw.copy()

        # Clean bad channels
        self.clean_bad_channels()

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

        # Apply PyLossless Rejection Policy for artifact removal
        self.pipeline, self.raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )

        # Detect and mark dense oscillatory artifacts
        self.detect_dense_oscillatory_artifacts()

        # Detect and mark muscle artifacts in beta frequency range
        self.detect_muscle_beta_focus(self.raw)

        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_artifact_detection",
            flagged=self.flagged,
        )

        self.create_eventid_epochs(reject_by_annotation=True)

        self.prepare_epochs_for_ica()

        self.gfp_clean_epochs()

        self._generate_reports()

    def _generate_reports(self) -> None:
        """Generate all visualization reports."""
        if self.pipeline is None or self.raw is None:
            return

        # Plot raw vs cleaned overlay using mixin method using mixin method
        self.plot_raw_vs_cleaned_overlay(
            self.raw, self.original_raw, self.pipeline, self.config
        )

        # Plot ICA components using mixin method using mixin method
        self.plot_ica_full(self.pipeline, self.config)

        # # Generate ICA reports using mixin method using mixin method (uncomment if needed)
        self.generate_ica_reports(self.pipeline, self.config)

        # Create PSD topography figure using mixin method using mixin method
        self.step_psd_topo_figure(
            self.raw, self.original_raw, self.pipeline, self.config
        )
