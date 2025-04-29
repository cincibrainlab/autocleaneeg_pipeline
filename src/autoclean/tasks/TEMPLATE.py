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
from typing import Any, Dict, Optional


# Third-party imports
import mne

# Local imports
from autoclean.core.task import Task
from autoclean.io.export import save_raw_to_set
from autoclean.step_functions.continuous import (
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
)
from autoclean.utils.logging import message


class TemplateTask(Task):
    """Template class for implementing new EEG processing tasks.

    This class demonstrates how to implement a new task type in the autoclean package.
    Each task must implement these key methods:
    1. __init__ - Initialize task state and validate config
    2. run - Main processing pipeline
    3. _generate_reports - Create quality control visualizations

    The task should handle a specific EEG paradigm (e.g., resting state, ASSR, MMN)
    and implement appropriate processing steps for that paradigm.

    Attributes:
        raw (mne.io.Raw): Raw EEG data that gets progressively cleaned through the pipeline
        pipeline (Any): PyLossless pipeline instance after preprocessing
        epochs (mne.Epochs): Epoched data after processing
        config (Dict[str, Any]): Task configuration dictionary containing all settings
        original_raw (mne.io.Raw): Original unprocessed raw data, preserved for comparison

    Example:
        To use this template:

        1. Create your configuration in autoclean_config.yaml:
        ```yaml
        # autoclean_config.yaml
        tasks:
          TemplateTask:
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
        self.epochs: Optional[mne.Epochs] = None
        self.original_raw: Optional[mne.io.Raw] = None

        # Stages that should be configured in the autoclean_config.yaml file
        self.required_stages = [
            "post_import",
            "post_prepipeline",
            "post_pylossless",
            "post_rejection_policy",
            "post_clean_raw",
            "post_epochs",
            "post_comp",
        ]

        # Call parent initialization with validated config
        super().__init__(config)

    def run(self) -> None:
        """Run the complete processing pipeline for this task.

        This method orchestrates the complete processing sequence:
        1. Import raw data
        2. Run preprocessing steps
        3. Apply task-specific processing
        4. Generate quality control reports

        The results are automatically saved at each stage according to
        the stage_files configuration.

        Processing Steps:
        1. Import and validate raw data
        2. Apply preprocessing pipeline (filtering, resampling, etc.)
        3. Create BIDS-compliant paths
        4. Run PyLossless pipeline for artifact detection
        5. Clean bad channels
        6. Apply rejection policy for artifact removal
        7. Create event-based epochs
        8. Prepare epochs for ICA
        9. Apply GFP-based cleaning to epochs
        10. Generate quality control reports

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If any processing step fails or if data hasn't been imported

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

        # Create regular epochs
        self.create_eventid_epochs()

        # Prepare epochs for ICA
        self.prepare_epochs_for_ica()

        # Clean epochs using GFP
        self.gfp_clean_epochs()

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
