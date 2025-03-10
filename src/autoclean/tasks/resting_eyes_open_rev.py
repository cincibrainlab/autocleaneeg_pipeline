# src/autoclean/tasks/resting_eyes_open_rev.py
"""Revised task implementation for resting state EEG preprocessing using mixins."""

# Standard library imports
from pathlib import Path
from typing import Any, Dict, Union, Optional

# Third-party imports
import mne

# Local imports
from autoclean.core.task import Task
from autoclean.step_functions.continuous import (
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
    step_get_pylossless_pipeline
)
from autoclean.step_functions.io import (
    save_raw_to_set,
    import_eeg
)
# # Import the reporting functions directly from the Task class via mixins
# # # Import the reporting functions directly from the Task class via mixins
# # from autoclean.step_functions.reports import (
# #     step_generate_ica_reports,
#     step_plot_ica_full,
#     step_plot_raw_vs_cleaned_overlay,
#     step_psd_topo_figure,

# )
from autoclean.utils.logging import message


class RestingEyesOpenRev(Task):
    """Revised task implementation for resting state EEG preprocessing.
    
    This class extends the base Task class which now includes functionality from mixins,
    demonstrating a more modular approach to task implementation.
    
    Attributes:
        raw (mne.io.Raw): Raw EEG data that gets progressively cleaned through the pipeline
        pipeline (Any): PyLossless pipeline instance after preprocessing
        epochs (mne.Epochs): Epoched data after processing
        original_raw (mne.io.Raw): Original unprocessed raw data, preserved for comparison
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the resting state task.
        
        Args:
            config: Configuration dictionary containing all settings.
        """
        self.raw: Optional[mne.io.Raw] = None
        self.pipeline: Optional[Any] = None
        self.epochs: Optional[mne.Epochs] = None
        self.original_raw: Optional[mne.io.Raw] = None
        super().__init__(config)

    def run(self) -> None:
        """Execute the complete resting state EEG processing pipeline.
        
        This method orchestrates the complete processing sequence including:
        1. Data import
        2. Preprocessing (resampling, filtering)
        3. Artifact detection and rejection
        4. Epoching
        5. Report generation
        
        Raises:
            RuntimeError: If data hasn't been imported successfully
        """
        # Import and save raw EEG data
        self.raw = import_eeg(self.config)
        
        save_raw_to_set(self.raw, self.config, "post_import")

        # Check if data was imported successfully
        if self.raw is None:
            raise RuntimeError("No data has been imported")
        
        message("header", "\nRunning preprocessing steps")
        
        # Use the resample_data mixin method instead of relying on step_pre_pipeline_processing
        # to handle resampling
        self.resample_data(stage_name="post_resample")
        
        # Continue with other preprocessing steps
        # Note: step_pre_pipeline_processing will skip resampling if already done
        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(self.raw, self.config, "post_prepipeline")

        # Store a copy of the pre-cleaned raw data for comparison in reports
        self.original_raw = self.raw.copy()

        # Clean bad channels (flags bad channels, does not drop them yet)
        #self.clean_bad_channels()

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        ################## RUN PYLOSSLESS CUSTOM STEPS ##################
        self.pipeline = step_get_pylossless_pipeline(self.config)


        self.pipeline.filter()

        #Flag bad channels

        self.pipeline.flag_noisy_channels()

        data_r_ch = self.pipeline.flag_uncorrelated_channels()

        self.pipeline.flag_bridged_channels(data_r_ch)

        # self.pipeline.flag_rank_channel(data_r_ch, message="Flagging the rank channel")

        bads = self.pipeline.flags['ch'].get_flagged()
        self.pipeline.raw.info['bads'] = bads
        self.pipeline.raw.interpolate_bads(reset_bads=True)
        self.pipeline.raw.set_eeg_reference()


        #Flag Bad Epochs 

        self.pipeline.flag_noisy_epochs(message="Flagging Noisy Epochs")

        self.pipeline.flag_uncorrelated_epochs(message="Flagging Uncorrelated epochs")

        self.raw = self.pipeline.raw

        # Detect and mark dense oscillatory artifacts
        self.detect_dense_oscillatory_artifacts()

        # Detect and mark muscle artifacts in beta frequency range
        #self.detect_muscle_beta_focus()

        save_raw_to_set(self.raw, self.config, "post_artifact_detection")

        #self.reject_bad_segments()

        self.pipeline.raw = self.raw

        if self.pipeline.config["ica"] is not None:
            self.pipeline.run_ica("run1", message="Running Initial ICA")

            self.pipeline.run_ica("run2", message="Running Final ICA and ICLabel.")

            self.pipeline.flag_noisy_ics(message="Flagging time periods with noisy IC's.")

        self.raw = self.pipeline.raw

        save_raw_to_set(self.raw, self.config, "post_pylossless")

        ################## END PYLOSSLESS ##################

        # Apply PyLossless Rejection Policy for artifact removal and channel interpolation
        self.pipeline, self.raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )
        save_raw_to_set(self.raw, self.config, "post_rejection_policy")

        self.clean_bad_channels(deviation_thresh=3)

        self.raw.interpolate_bads(reset_bads=True)

        save_raw_to_set(self.raw, self.config, "checkpoint")

        # Create regular epochs
        self.create_regular_epochs()

        # Prepare epochs for ICA
        self.prepare_epochs_for_ica()

        # Clean epochs using GFP
        self.gfp_clean_epochs()

        # Generate visualization reports
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
        self.psd_topo_figure(
            self.original_raw, self.raw, self.pipeline, self.config
        )


    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resting state specific configuration.

        Args:
            config: Configuration dictionary that has passed common validation

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate resting state specific fields
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
            "post_artifact_detection",
            "post_epochs",
            "post_comp"
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
