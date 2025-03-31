# src/autoclean/tasks/resting_eyes_open.py
"""Task implementation for resting state EEG preprocessing."""

# Standard library imports
from pathlib import Path
from typing import Any, Dict

# Local imports
from autoclean.core.task import Task
from autoclean.step_functions.continuous import (
    step_clean_bad_channels,
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)
from autoclean.step_functions.epochs import (
    step_create_regular_epochs,
    step_gfp_clean_epochs,
    step_prepare_epochs_for_ica,
)
from autoclean.io.import_ import import_eeg
from autoclean.io.export import save_epochs_to_set, save_raw_to_set, save_stc_to_file

from autoclean.calc.source import estimate_source_function_raw

import mne

from autoclean.utils.logging import message


class resting_eyesopen_grael4k(Task):
    """Task implementation for resting state EEG preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        self.raw = None
        self.pipeline = None
        self.cleaned_raw = None
        self.epochs = None
        super().__init__(config)

    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # """Validate resting state specific configuration.

        # Args:
        #     config: Configuration dictionary that has passed common validation

        # Returns:
        #     Validated configuration dictionary

        # Raises:
        #     ValueError: If required fields are missing or invalid
        # """
        # # Validate resting state specific fields
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
        """Run the complete resting state processing pipeline."""

        self.import_raw()
        
        message("header", "Running preprocessing steps")
        
        # Continue with other preprocessing steps
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(raw = self.raw, autoclean_dict = self.config, stage = "post_prepipeline", flagged = self.flagged)

        # Store a copy of the pre-cleaned raw data for comparison in reports
        self.original_raw = self.raw.copy()

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        #Run PyLossless Pipeline
        self.pipeline, self.raw = self.step_custom_pylossless_pipeline(self.config, eog_channel="VEOG")

        #Add more artifact detection steps
        self.detect_dense_oscillatory_artifacts()

        # Update pipeline with annotated raw data
        self.pipeline.raw = self.raw

        # Apply PyLossless Rejection Policy for artifact removal and channel interpolation
        self.pipeline, self.raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )   
        self.raw = self.pipeline.raw

        save_raw_to_set(raw = self.raw, autoclean_dict = self.config, stage = "post_rejection_policy", flagged = self.flagged)

        #Clean bad channels post ICA
        self.clean_bad_channels(deviation_thresh=3, cleaning_method="interpolate")

        save_raw_to_set(raw = self.raw, autoclean_dict = self.config, stage = "post_clean_raw", flagged = self.flagged)

        estimate_source_function_raw(self.raw, self.config)

        # Create regular epochs
        self.create_regular_epochs()

        # Prepare epochs for ICA
        self.prepare_epochs_for_ica()

        # Clean epochs using GFP
        self.gfp_clean_epochs()

        # Generate visualization reports
        self._generate_reports()

    # def preprocess(self) -> None:
    #     """Run preprocessing steps on the raw data."""
    #     if self.raw is None:
    #         raise RuntimeError("No data has been imported")

    #     # Run preprocessing pipeline and save intermediate result
    #     self.raw = step_pre_pipeline_processing(self.raw, self.config)
    #     save_raw_to_set(self.raw, self.config, "post_prepipeline")

    #     # Create BIDS-compliant paths and filenames
    #     self.raw, self.config = step_create_bids_path(self.raw, self.config)

    #     # Run PyLossless pipeline and save result
    #     self.pipeline, self.raw = step_run_pylossless(self.config)
    #     save_raw_to_set(self.raw, self.config, "post_pylossless")

    #     # Clean bad channels
    #     #self.pipeline.raw = step_clean_bad_channels(self.raw, self.config)
    #     #save_raw_to_set(self.pipeline.raw, self.config, "post_bad_channels")

    #     # # Use PyLossless Rejection Policy
    #     self.pipeline, self.cleaned_raw = step_run_ll_rejection_policy(
    #         self.pipeline, self.config
    #     )

    #     self.cleaned_raw = step_detect_dense_oscillatory_artifacts(
    #         self.cleaned_raw,
    #         window_size_ms=100,
    #         channel_threshold_uv=50,
    #         min_channels=65,
    #         padding_ms=500,
    #     )
    #     save_raw_to_set(self.cleaned_raw, self.config, "post_rejection_policy")

    #     estimate_source_function_raw(self.cleaned_raw, self.config)

    #     # Generate visualization reports
    #     # self._generate_reports()


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
        self.psd_topo_figure(
            self.original_raw, self.raw, self.pipeline, self.config
        )

