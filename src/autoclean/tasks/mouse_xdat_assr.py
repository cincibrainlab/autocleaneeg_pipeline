# src/autoclean/tasks/mouse_xdat_assr.py
"""Mouse XDAT assr State Task"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict
import os
import sys


# Local imports
from autoclean.core.task import Task
from autoclean.step_functions.continuous import (
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_pylossless,
    step_clean_bad_channels,
)
from autoclean.step_functions.epochs import (
    step_apply_autoreject,
    step_create_eventid_epochs,
)
from autoclean.step_functions.io import save_epochs_to_set, save_raw_to_set, import_eeg
from autoclean.step_functions.reports import (
    step_generate_ica_reports,
    step_plot_ica_full,
    step_plot_raw_vs_cleaned_overlay,
    step_psd_topo_figure,
)
from pyprep.find_noisy_channels import NoisyChannels
from autoclean.utils.database import manage_database
from autoclean.utils.logging import message
import mne

from datetime import datetime

class MouseXdatAssr(Task):

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




    def step_clean_bad_channels_by_correlation(self,
        raw: mne.io.Raw, autoclean_dict: Dict[str, Any]
    ) -> mne.io.Raw:
        """Clean bad channels."""
        message("header", "step_clean_bad_channels")
        # Setup options
        options = {
            "random_state": 1337,
            "ransac": True,
            "channel_wise": False,
            "max_chunk_size": None,
            "corr_thresh": 0.75,
        }

        # check if "eog" is in channel type dictionary
        if (
            "eog" in raw.get_channel_types()
            and not autoclean_dict["tasks"][autoclean_dict["task"]]["settings"]["eog_step"][
                "enabled"
            ]
        ):
            eog_picks = mne.pick_types(raw.info, eog=True)
            eog_ch_names = [raw.ch_names[idx] for idx in eog_picks]
            raw.set_channel_types({ch: "eeg" for ch in eog_ch_names})

        # Run noisy channels detection
        cleaned_raw = NoisyChannels(raw, random_state=options["random_state"])
        #cleaned_raw.find_bad_by_SNR()
        cleaned_raw.find_bad_by_correlation(correlation_threshold=0.35, frac_bad=0.01)
        #cleaned_raw.find_bad_by_deviation(deviation_threshold=4.0)
        bad_channels = cleaned_raw.get_bads(as_dict=True)
        raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_correlation"]])

        # cleaned_raw.find_bad_by_ransac(
        #     n_samples=100,
        #     sample_prop=0.25,
        #     corr_thresh=options["corr_thresh"],
        #     frac_bad=0.35,
        #     corr_window_secs=5.0,
        #     channel_wise=True,
        #     max_chunk_size=None,
        # )

        # Create empty lists for the other bad channel types
        # This ensures the subsequent extend operations won't fail
        bad_channels = cleaned_raw.get_bads(as_dict=True)
        
        # Initialize empty lists for channel types we're not detecting
        if "bad_by_ransac" not in bad_channels:
            bad_channels["bad_by_ransac"] = []
        
        if "bad_by_deviation" not in bad_channels:
            bad_channels["bad_by_deviation"] = []
            
        if "bad_by_SNR" not in bad_channels:
            bad_channels["bad_by_SNR"] = []

        bad_channels = cleaned_raw.get_bads(as_dict=True)
        raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_ransac"]])
        raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_deviation"]])
        raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_correlation"]])
        raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_SNR"]])

        print(raw.info["bads"])

        # Record metadata with options
        metadata = {
            "step_clean_bad_channels": {
                "creationDateTime": datetime.now().isoformat(),
                "method": "NoisyChannels",
                "options": options,
                "bads": raw.info["bads"],
                "channelCount": len(raw.ch_names),
                "durationSec": int(raw.n_times) / raw.info["sfreq"],
                "numberSamples": int(raw.n_times),
            }
        }

        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

        return raw


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


        l_freq = 0.5
        h_freq = 100
        self.raw.filter(l_freq = l_freq, h_freq = h_freq)


        notch_freqs = [60]
        notch_widths = 3
        self.raw.notch_filter(freqs = notch_freqs, notch_widths = notch_widths)


        self.raw = step_pre_pipeline_processing(self.raw, self.config)
        save_raw_to_set(self.raw, self.config, "post_prepipeline")


        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        self.raw = self.step_clean_bad_channels_by_correlation(self.raw, self.config)


        self.raw.interpolate_bads(reset_bads=False)

        self.epochs = step_create_eventid_epochs(
            self.raw, self.pipeline, self.config
        )

        save_epochs_to_set(self.epochs, self.config, "post_epochs")

        # Create analysis directory in stage_dir
        analysis_dir = Path(self.config['stage_dir']) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)        
        # Update config with analysis directory path
        self.config['analysis_dir'] = str(analysis_dir)

        from autoclean.calc.assr_runner import run_complete_analysis
        file_basename = Path(self.config["unprocessed_file"]).stem
        run_complete_analysis(epochs = self.epochs, output_dir = self.config['analysis_dir'], file_basename=file_basename)

        self._generate_reports()

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
        self.raw = import_eeg(self.config)

        # Save imported data if configured
        save_raw_to_set(self.raw, self.config, "post_import")


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

        # Plot raw vs cleaned overlay
        step_plot_raw_vs_cleaned_overlay(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )

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

    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
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
            "post_epochs",
            "post_autoreject",
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