# src/autoclean/tasks/hbcd_mmn.py
# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt

# Third-party imports
import mne
import numpy as np

# Local imports
from autoclean.core.task import Task
from autoclean.io.export import save_epochs_to_set, save_raw_to_set
from autoclean.io.import_ import import_eeg

# Import the reporting functions directly from the Task class via mixins
# from autoclean.step_functions.reports import (
#     step_generate_ica_reports,
#     step_plot_ica_full,
#     step_psd_topo_figure,
#     generate_mmn_erp,
# )
from autoclean.step_functions.continuous import (
    step_clean_bad_channels,
    step_create_bids_path,
    step_pre_pipeline_processing,
    step_run_ll_rejection_policy,
    step_run_pylossless,
)
from autoclean.types.task_models import ImportMetadata, ProcessingMetadata
from autoclean.utils.database import manage_database
from autoclean.utils.logging import message


class P300_Grael4k(Task):
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

    def run(self) -> None:
        """Run the complete processing pipeline for this task.
        This method orchestrates the complete processing sequence:
        1. Import raw data
        2. Run preprocessing steps
        3. Apply task-specific processing
        """

        # ==========================================
        #          Data Import and Validation
        # ==========================================

        # ------------------------------------------
        # Import the raw data
        # Use default or custom import function
        # populate valid self.raw object to continue
        # ------------------------------------------
        file_path = Path(self.config.get("unprocessed_file", ""))
        self.import_data(file_path)

        # Run preprocessing pipeline and save result
        # Resample: 250 Hz
        # Drop outerlayer: False
        # EOG: [31, 32]
        # Trim: False
        # Crop: False
        self.raw = step_pre_pipeline_processing(self.raw, self.config)

        # Create BIDS-compliant paths and filenames
        self.raw, self.config = step_create_bids_path(self.raw, self.config)

        # Run PyLossless pipeline
        self.pipeline, pipeline_raw = step_run_pylossless(self.config)

        # Clean Channels
        self.raw = step_clean_bad_channels(self.raw, self.config)

        # Apply rejection policy
        self.pipeline, self.cleaned_raw = step_run_ll_rejection_policy(
            self.pipeline, self.config
        )

        # self.cleaned_raw.set_channel_types({'A1': 'eeg', 'A2': 'eeg'})
        self.cleaned_raw.pick_types(eeg=True)

        save_raw_to_set(self.cleaned_raw, self.config, "post_rejection_policy")

        # Generate visualization reports
        self._generate_reports()

        # self.epochs = step_create_eventid_epochs_p300(
        #     self.cleaned_raw, self.pipeline, self.config
        # )

        # save_epochs_to_set(self.epochs, self.config, "post_epochs")
        # if self.epochs is None:
        #     message("error", "Failed to create epochs")
        #     return

        # Run MMN analysis
        # generate_mmn_erp(self.epochs, self.pipeline, self.config)

        # # Save final epochs
        # save_epochs_to_set(self.epochs, self.config, "post_comp")

        # self.preprocess()
        # self.process()

    def import_data(self, file_path: Path) -> mne.io.Raw:
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

        # ------------------------------------------
        # Check if the specified unprocessed file exists
        # ------------------------------------------
        if not file_path.is_file():
            error_message = f"Input file not found: {file_path}"
            message("error", error_message)
            raise FileNotFoundError(error_message)

        # ------------------------------------------
        # Import raw data using EEGLAB reader
        # ------------------------------------------
        message("info", "Importing data...")
        unprocessed_file = self.config["unprocessed_file"]
        eeg_system = self.config["eeg_system"]

        try:
            raw = mne.io.read_raw_eeglab(
                input_fname=unprocessed_file, preload=True, verbose=True
            )
            message("success", "Successfully loaded .set file")
        except Exception as e:
            error_message = f"Failed to read .set file: {str(e)}"
            message("error", error_message)
            raise RuntimeError(error_message)

        # ------------------------------------------
        # Set up standard 10-20 montage
        # ------------------------------------------
        try:
            montage = mne.channels.make_standard_montage(eeg_system)
            raw.set_montage(montage, match_case=False)
            raw.set_eeg_reference(ref_channels=[])  # No re-referencing, just keep as is

            # Reclassify A1 and A2 as 'misc' channels (not EEG)
            raw.set_channel_types({"A1": "misc", "A2": "misc"})

            raw.info["description"] = "Data referenced to linked ears (A1 + A2)/2"
            message("success", "Successfully configured standard 10-20 montage")
        except Exception as e:
            error_message = f"Failed to set up standard_1020 montage: {str(e)}"
            message("error", error_message)
            raise RuntimeError(error_message)

        # ------------------------------------------
        # Process P300 task-specific annotations
        # ------------------------------------------
        try:
            mapping = {"13": "Standard", "14": "Target"}
            raw.annotations.rename(mapping)
            message("success", "Successfully renamed annotations")
        except Exception as e:
            error_message = f"Failed to rename annotations: {str(e)}"
            message("error", error_message)
            raise RuntimeError(error_message)

        # ------------------------------------------
        # Assign imported data to self.raw
        # ------------------------------------------
        self.raw = raw

        # ------------------------------------------
        # Save imported data if configured
        # ------------------------------------------
        save_raw_to_set(self.raw, self.config, "post_import")

        # Verify initial annotations
        self._verify_annotations(self.raw, "post_import")

        # ------------------------------------------
        # Update database with import metadata
        # ------------------------------------------
        try:
            metadata = ImportMetadata(
                unprocessedFile=file_path.name,
                eegSystem=self.config["eeg_system"],
                sampleRate=self.raw.info["sfreq"],
                channelCount=len(self.raw.ch_names),
                durationSec=self.raw.n_times / self.raw.info["sfreq"],
                numberSamples=int(self.raw.n_times),
                hasEvents=self.raw.annotations is not None,
            )

            manage_database(
                operation="update",
                update_record={
                    "run_id": self.config["run_id"],
                    "metadata": ProcessingMetadata(import_eeg=metadata).model_dump(),
                },
            )

            message("success", "✓ Raw EEG data imported successfully")
        except Exception as e:
            message("error", f"Failed to update database: {str(e)}")
            raise

        return self.raw

    def step_create_eventid_epochs_p300(
        cleaned_raw: mne.io.Raw, pipeline: Any, autoclean_dict: Dict[str, Any]
    ) -> Optional[mne.Epochs]:
        task = autoclean_dict["task"]
        # Get epoch settings
        epoch_settings = autoclean_dict["tasks"][task]["settings"]["epoch_settings"]
        if not epoch_settings["enabled"]:
            return None

        tmin = epoch_settings["value"]["tmin"]
        tmax = epoch_settings["value"]["tmax"]
        message("info", f"Using tmin: {tmin} and tmax: {tmax}")
        baseline = (
            tuple(epoch_settings["remove_baseline"]["window"])
            if epoch_settings["remove_baseline"]["enabled"]
            else None
        )
        message("info", f"Using baseline: {baseline}")
        events, event_id = mne.events_from_annotations(cleaned_raw)

        # Create epochs from -200ms to 800ms relative to the event onset, with baseline correction
        epochs = mne.Epochs(
            cleaned_raw,
            events,
            event_id,
            tmin=-0.5,
            tmax=0.8,
            baseline=(-0.5, 0),
            preload=True,
        )

        # Average epochs for each condition
        evoked_standard = epochs["Standard"].average()
        evoked_target = epochs["Target"].average()

        # Plot both evoked responses on the same figure
        figs = mne.viz.plot_compare_evokeds(
            {"Standard": evoked_standard, "Target": evoked_target},
            colors={"Standard": "blue", "Target": "red"},
            title="ERP: Standard vs. Target",
        )

        fig = figs[0] if isinstance(figs, list) else figs
        fig.savefig("erp_comparison.png", dpi=300)

        # get event_id settings
        event_types = epoch_settings.get("event_id")
        if event_types is None:
            message(
                "warning", "Event ID is not specified in epoch_settings (set to null)"
            )
            return None

        # Add metadata about the epoching
        metadata = {
            "step_create_eventid_epochs": {
                "creationDateTime": datetime.now().isoformat(),
                "event_types": list(event_types),
                "number_of_events": len(events),
                "number_of_epochs": len(epochs),
                "epoch_duration": epochs.times[-1] - epochs.times[0],
                "samples_per_epoch": len(epochs.times),
                "total_duration": (epochs.times[-1] - epochs.times[0]) * len(epochs),
                "total_samples": len(epochs.times) * len(epochs),
                "channel_count": len(epochs.ch_names),
                "event_counts": {
                    name: sum(events[:, 2] == num) for name, num in event_id.items()
                },
                "tmin": tmin,
                "tmax": tmax,
            }
        }

        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

        return epochs

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

    pass

    # Generate visualization reports
    # self._generate_reports()

    def process(self) -> None:
        pass

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

        # Plot ICA components using mixin method
        self.plot_ica_full(self.pipeline, self.config)

        # Generate ICA reports using mixin method
        self.plot_ica_components(
            self.pipeline.ica2,
            self.cleaned_raw,
            self.config,
            self.pipeline,
            duration=60,
        )

        # Create PSD topography figure using mixin method
        self.psd_topo_figure(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )

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
        pass

    def _create_import_metadata(
        self, unprocessed_file: Path, raw: mne.io.Raw, events: Optional[Any]
    ) -> Dict[str, Any]:
        """Generate metadata specific to the data import step."""
        if not isinstance(raw, mne.io.Raw):
            raise TypeError("raw must be an mne.io.Raw object")
        if not isinstance(unprocessed_file, Path):
            raise TypeError("unprocessed_file must be a pathlib.Path object")

        metadata = {
            "import_eeg": {
                "import_function": "import_eeg",
                "creationDateTime": datetime.now().isoformat(),
                "unprocessedFile": str(unprocessed_file.name),
                "eegSystem": self.config.get("eeg_system", "Unknown"),
                "sampleRate": float(raw.info["sfreq"]),
                "channelCount": int(len(raw.ch_names)),
                "durationSec": float(raw.n_times) / raw.info["sfreq"],
                "numberSamples": int(raw.n_times),
                "hasEvents": bool(events is not None),
            }
        }
        return metadata

    def _validate_import_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata specific to the import step."""
        required_keys = {
            "import_function": str,
            "creationDateTime": str,
            "unprocessedFile": str,
            "eegSystem": str,
            "sampleRate": float,
            "channelCount": int,
            "durationSec": float,
            "numberSamples": int,
            "hasEvents": bool,
        }
        import_eeg = metadata.get("import_eeg")
        if not isinstance(import_eeg, dict):
            message("error", "import_eeg must be a dictionary")
            return False

        for key, expected_type in required_keys.items():
            if key not in import_eeg:
                message("error", f"Missing required key in import_eeg: {key}")
                return False
            if not isinstance(import_eeg[key], expected_type):
                message("error", f"Key {key} must be of type {expected_type.__name__}")
                return False
        return True

    def _update_database_with_import_metadata(
        self, unprocessed_file: Path, raw: mne.io.Raw, events: Optional[Any]
    ) -> None:
        """Update the database with metadata from the import step."""
        # Generate and validate metadata
        metadata = self._create_import_metadata(unprocessed_file, raw, events)
        if not self._validate_import_metadata(metadata):
            raise ValueError("Import metadata validation failed")

        # Prepare update record
        if "run_id" not in self.config:
            raise KeyError("run_id not found in config")
        update_record = {"run_id": self.config["run_id"], "metadata": metadata}

        # Perform database update with error handling
        try:
            manage_database(
                operation="update",
                update_record=update_record,
            )
            message("info", "Database updated successfully with import metadata")
        except Exception as e:
            message("error", f"Failed to update database: {str(e)}")
            raise
