
# src/autoclean/tasks/resting_eyes_open_rev.py
"""Revised task implementation for resting state EEG preprocessing using mixins."""

# Standard library imports
from typing import Any, Dict, Optional

# Third-party imports
import mne

# Local imports
from autoclean.core.task import Task
from autoclean.io.export import save_raw_to_set
from autoclean.step_functions.continuous import (
    step_create_bids_path,
    step_run_ll_rejection_policy,
)

class TestingRest(Task):
    """The template for a task created from the Autoclean Config Wizard.

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
        self.pipeline: Optional[Any] = None
        self.original_raw: Optional[mne.io.Raw] = None

        self.required_stages = [
            "post_import",
            "post_basicsteps",
            "post_pylossless",
            "post_rejection_policy",
            "post_clean_raw",
            "post_epochs",
            "post_comp",
        ]

        super().__init__(config)  # Initialize the base class

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
        self.import_raw()

        # Continue with other preprocessing steps
        self.run_basic_steps()

        # Store a copy of the pre-cleaned raw data for comparison in reports
        self.original_raw = self.raw.copy()

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        # Run PyLossless Pipeline
        self.pipeline, self.raw = self.step_custom_pylossless_pipeline(self.config)

        # Add more artifact detection steps
        self.detect_dense_oscillatory_artifacts()

        # Update pipeline with annotated raw data
        self.pipeline.raw = self.raw

        # Apply PyLossless Rejection Policy for artifact removal and channel interpolation
        self.pipeline, self.raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )
        self.raw = self.pipeline.raw

        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_rejection_policy",
            flagged=self.flagged,
        )

        # Clean bad channels post ICA
        self.clean_bad_channels(
            deviation_thresh=3, cleaning_method="interpolate", reset_bads=True
        )

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
