# src/autoclean/plugins/eeg_plugins/eeglab_gsn129_plugin.py
"""EEGLAB .set file plugin with GSN-HydroCel-129 montage configuration.

This plugin handles the complete import and montage configuration
for EEGLAB .set files with GSN-HydroCel-129 electrode system.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message


class EEGLABSetGSN129Plugin(BaseEEGPlugin):
    """Plugin for EEGLAB .set files with GSN-HydroCel-129 montage.

    This plugin handles the specific combination of EEGLAB .set files
    with the GSN-HydroCel-129 electrode system, which requires special
    handling for the 129th electrode.
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "EEGLAB_SET" and montage_name == "GSN-HydroCel-129"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import EEGLAB .set file and configure GSN-HydroCel-129 montage."""
        message(
            "info",
            f"Loading EEGLAB .set file with GSN-HydroCel-129 montage: {file_path}",
        )

        try:
            # Step 1: Import the .set file
            raw = mne.io.read_raw_eeglab(
                input_fname=file_path, preload=preload, verbose=True
            )
            message("success", "Successfully loaded .set file")

            # Step 1.5: Check for reference channel and rename if necessary
            ref_channel_names = ["VREF", "REF", "E129"]
            for ref_name in ref_channel_names:
                if ref_name in raw.ch_names:
                    message(
                        "debug",
                        f"Found reference channel '{ref_name}', renaming to 'Cz'",
                    )
                    # Create mapping for renaming
                    rename_mapping = {ref_name: "Cz"}
                    # Rename the channel
                    raw.rename_channels(rename_mapping)
                    message("success", f"Successfully renamed '{ref_name}' to 'Cz'")
                    # Ensure reference channel is marked as EEG type
                    if "Cz" in raw.ch_names:
                        raw.set_channel_types({"Cz": "eeg"})
                        message("info", "Set Cz channel type to EEG")
                    break

            # Step 2: Configure the GSN-HydroCel-129 montage
            message("info", "Configuring GSN-HydroCel-129 montage")

            # Create montage and set the special 129th electrode name
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

            # Apply the montage
            raw.set_montage(montage, match_case=False)

            # Pick only EEG channels
            raw.pick("eeg")

            message("success", "Successfully configured GSN-HydroCel-129 montage")

            # Step 3: Extract and process events
            events_df = self._get_matlab_annotations_table(file_path)

            if events_df is not None:
                message("info", f"Found events in EEGLAB file: {len(events_df)} events")

                # Create annotations from the events table
                if all(
                    col in events_df.columns
                    for col in ["Task", "type", "onset", "Condition"]
                ):
                    subset_events_df = events_df[["Task", "type", "onset", "Condition"]]
                    new_annotations = mne.Annotations(
                        onset=subset_events_df["onset"].values,
                        duration=np.zeros(len(subset_events_df)),  # Point events
                        description=[
                            f"{row['Task']}/{row['type']}/{row['Condition']}"
                            for _, row in subset_events_df.iterrows()
                        ],
                    )
                    raw.set_annotations(new_annotations)
                    message("success", "Successfully created annotations from events")

            # Step 4: Apply task-specific processing if needed
            task = autoclean_dict.get("task", None)
            if task:
                if task == "p300_grael4k":
                    message("info", "Processing P300 task-specific annotations")
                    mapping = {"13": "Standard", "14": "Target"}
                    raw.annotations.rename(mapping)
                elif task == "hbcd_mmn":
                    message("info", "Processing HBCD MMN task-specific annotations")
                    if events_df is not None and all(
                        col in events_df.columns
                        for col in ["Task", "type", "onset", "Condition"]
                    ):
                        subset_events_df = events_df[
                            ["Task", "type", "onset", "Condition"]
                        ]
                        new_annotations = mne.Annotations(
                            onset=subset_events_df["onset"].values,
                            duration=np.zeros(len(subset_events_df)),
                            description=[
                                f"{row['Task']}/{row['type']}/{row['Condition']}"
                                for _, row in subset_events_df.iterrows()
                            ],
                        )
                        raw.set_annotations(new_annotations)
                        message(
                            "success", "Successfully processed HBCD MMN annotations"
                        )

            return raw

        except Exception as e:
            raise RuntimeError(
                f"Failed to process EEGLAB file with GSN-HydroCel-129 montage: {str(e)}"
            ) from e

    def process_events(self, raw: mne.io.Raw, autoclean_dict: dict) -> tuple:
        """Process events and annotations in the EEG data."""
        message("info", "Processing events from EEGLAB file")
        try:
            # Get events from annotations
            events, event_id = mne.events_from_annotations(raw)

            # Create a more detailed events DataFrame
            if events is not None and len(events) > 0:
                events_df = pd.DataFrame(
                    {
                        "time": events[:, 0] / raw.info["sfreq"],
                        "sample": events[:, 0],
                        "id": events[:, 2],
                        "type": [
                            event_id.get(str(id), f"Unknown-{id}")
                            for id in events[:, 2]
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
                "type": "GSN-HydroCel-129",
                "channel_count": 129,
                "manufacturer": "Electrical Geodesics, Inc. (EGI)",
                "reference": "Common Reference",
                "layout": "Geodesic",
            },
        }
