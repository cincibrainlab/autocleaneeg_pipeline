# src/autoclean/tasks/chirp_default.py
"""Task implementation for chirp EEG preprocessing."""

from pathlib import Path
from typing import Any, Dict

from autoclean.step_functions.reports import step_generate_ica_reports, step_plot_ica_full, step_plot_raw_vs_cleaned_overlay, step_psd_topo_figure

from ..core.task import Task
from ..step_functions.continuous import (
    step_clean_bad_channels,
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)
from ..step_functions.epochs import (
    step_create_eventid_epochs,
    step_gfp_clean_epochs,
    step_prepare_epochs_for_ica,
)
from ..step_functions.io import save_epochs_to_set, save_raw_to_set, import_eeg


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
        file_path = Path(self.config["unprocessed_file"])
        self.import_data(file_path)
        self.preprocess()
        self.process()
        self._generate_reports()

    def import_data(self, file_path: Path) -> None:
        # Import and save raw EEG data
        self.raw = import_eeg(self.config)
        save_raw_to_set(self.raw, self.config, "post_import")

    def preprocess(self) -> None:
        """Run preprocessing steps on the raw data."""
        if self.raw is None:
            raise RuntimeError("No data has been imported")

        # Run preprocessing pipeline and save intermediate result
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(self.raw, self.config, "post_prepipeline")

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        # # Run PyLossless pipeline and save result
        self.pipeline, pipeline_raw = step_run_pylossless(self.config)
        save_raw_to_set(pipeline_raw, self.config, "post_pylossless")

        # Clean bad channels
        self.pipeline.raw = step_clean_bad_channels(self.raw, self.config)
        save_raw_to_set(self.raw, self.config, "post_bad_channels")

        # Use PyLossless Rejection Policy
        self.pipeline, self.cleaned_raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )
        save_raw_to_set(self.cleaned_raw, self.config, "post_rejection_policy")

    def process(self) -> None:
        if self.cleaned_raw is None:
            raise RuntimeError("Need to run preprocess first")

        # Create event-id epochs
        self.epochs = step_create_eventid_epochs(
            self.cleaned_raw, self.pipeline, self.config
        )
        save_epochs_to_set(self.epochs, self.config, "post_epochs")

        # Prepare epochs for ICA
        self.epochs = step_prepare_epochs_for_ica(
            self.epochs, self.pipeline, self.config
        )

        # Clean epochs
        self.epochs = step_gfp_clean_epochs(self.epochs, self.pipeline, self.config)

        # Save cleaned epochs
        save_epochs_to_set(self.epochs, self.config, "post_comp")

    def _generate_reports(self) -> None:
        """Generate all visualization reports."""
        if self.pipeline is None or self.cleaned_raw is None:
            return

        # Plot raw vs cleaned overlay
        step_plot_raw_vs_cleaned_overlay(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )

        # Plot ICA components
        step_plot_ica_full(self.pipeline, self.config)

        # # # Generate ICA reports
        # step_generate_ica_reports(
        #     self.pipeline, self.cleaned_raw, self.config, duration=60
        # )

        # # Create PSD topography figure
        step_psd_topo_figure(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )
