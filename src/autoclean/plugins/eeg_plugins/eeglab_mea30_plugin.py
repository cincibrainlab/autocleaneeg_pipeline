# src/autoclean/plugins/eeg_plugins/eeglab_mea30_plugin.py
"""EEGLAB .set file plugin with MEA30 montage configuration.

This plugin handles the complete import and montage configuration
for EEGLAB .set files with MEA30 electrode system.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message


class EEGLABSetMEA30Plugin(BaseEEGPlugin):
    """Plugin for EEGLAB .set files with MEA30 montage.

    This plugin handles the combination of EEGLAB .set files
    with the MEA30 electrode system in a unified way.
    """

    # Version information for tracking
    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "EEGLAB_SET" and montage_name == "MEA30"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import EEGLAB .set file and configure MEA30 montage."""
        message("info", f"Loading EEGLAB .set file with MEA30 montage: {file_path}")

        try:
            # Step 1: Import the .set file
            raw = mne.io.read_raw_eeglab(
                input_fname=file_path, preload=preload, verbose=True
            )
            message("success", "Successfully loaded .set file")

            # Step 2: Configure the MEA30 montage
            message("info", "Configuring MEA30 channels")

            # MEA30 requires EEG channels only
            raw.pick_types(eeg=True, exclude=[])

            # Check if we have the expected channel count
            if len(raw.ch_names) != 30:
                message(
                    "warning",
                    f"Expected 30 channels for MEA30, found {len(raw.ch_names)}",
                )

            # Add custom channel locations if needed
            # In this example, we're just picking EEG channels
            # In a real implementation, you might set up specific positions

            message("success", "Successfully configured MEA30 montage")

            # Step 3: Extract events from MATLAB structure
            events_df = self._get_matlab_annotations_table(file_path)

            if events_df is not None:
                message("info", f"Found events in EEGLAB file: {len(events_df)} events")

                # Create annotations if we have the expected columns
                if all(col in events_df.columns for col in ["type", "onset"]):
                    # Create simple annotations
                    new_annotations = mne.Annotations(
                        onset=events_df["onset"].values,
                        duration=np.zeros(len(events_df)),  # Point events
                        description=[str(t) for t in events_df["type"].values],
                    )
                    raw.set_annotations(new_annotations)
                    message("success", "Successfully created annotations from events")

                # Add more complex annotations if we have additional info
                if all(
                    col in events_df.columns
                    for col in ["Task", "type", "onset", "Condition"]
                ):
                    subset_events_df = events_df[["Task", "type", "onset", "Condition"]]
                    new_annotations = mne.Annotations(
                        onset=subset_events_df["onset"].values,
                        duration=np.zeros(len(subset_events_df)),
                        description=[
                            f"{row['Task']}/{row['type']}/{row['Condition']}"
                            for _, row in subset_events_df.iterrows()
                        ],
                    )
                    raw.set_annotations(new_annotations)

            # Step 4: Apply task-specific processing if needed
            task = autoclean_dict.get("task", None)
            if task and task == "mea30_oddball":
                message("info", "Applying MEA30 oddball task-specific processing")
                # Example: map numerical markers to meaningful names
                mapping = {"1": "Standard", "2": "Target"}
                raw.annotations.rename(mapping)

            return raw

        except Exception as e:
            raise RuntimeError(
                f"Failed to process EEGLAB file with MEA30 montage: {str(e)}"
            ) from e

    def process_events(self, raw: mne.io.Raw, autoclean_dict: dict) -> tuple:
        """Process events and annotations in the EEG data."""
        message("info", "Processing events from EEGLAB file")
        try:
            # Get events from annotations (already created in import_and_configure)
            events, event_id = mne.events_from_annotations(raw)

            # Create a more detailed events DataFrame
            if events is not None and len(events) > 0:
                events_df = pd.DataFrame(
                    {
                        "time": events[:, 0] / raw.info["sfreq"],
                        "id": events[:, 2],
                        "type": [
                            event_id.get(id, f"Unknown-{id}") for id in events[:, 2]
                        ],
                    }
                )
                return events, event_id, events_df
            else:
                return None, None, None

        except Exception as e:  # pylint: disable=broad-except
            message("warning", f"Failed to process events: {str(e)}")
            return None, None, None

    def _get_matlab_annotations_table(self, file_path: Path) -> pd.DataFrame:
        """Extract events table from MATLAB structure in EEGLAB file."""
        try:
            eeglab_data = sio.loadmat(
                file_path, squeeze_me=True, struct_as_record=False
            )
            full_events = eeglab_data["EEG"].event

            event_list = []
            for event in full_events:
                event_dict = {}
                for field_name in event._fieldnames:
                    event_dict[field_name] = getattr(event, field_name)
                event_list.append(event_dict)

            events_df = pd.DataFrame(event_list)
            return events_df
        except Exception as e:  # pylint: disable=broad-except
            message("warning", f"Could not load events table: {str(e)}")
            return None

    def get_metadata(self) -> dict:
        """Get additional metadata about this plugin."""
        return {
            "plugin_name": self.__class__.__name__,
            "plugin_version": self.VERSION,
            "montage_details": {
                "type": "MEA30",
                "channel_count": 30,
                "manufacturer": "Example Manufacturer",
                "reference": "Common Reference",
            },
        }
