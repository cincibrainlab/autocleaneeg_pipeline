"""EGI .raw file plugin with GSN-HydroCel-124 montage configuration.

This plugin handles the complete import and montage configuration
for EGI .raw files with GSN-HydroCel-124 electrode system.
"""

from pathlib import Path

import mne
import pandas as pd

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message


class EGIRawGSN124Plugin(BaseEEGPlugin):
    """Plugin for EGI .raw files with GSN-HydroCel-124 montage.

    This plugin handles the specific combination of EGI .raw files
    with the GSN-HydroCel-124 electrode system, which requires special
    handling for the ECG channels (125-128).
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "EGI_RAW" and montage_name == "GSN-HydroCel-124"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import EGI .raw file and configure GSN-HydroCel-124 montage."""
        message(
            "info", f"Loading EGI .raw file with GSN-HydroCel-124 montage: {file_path}"
        )

        try:
            # Step 1: Import the .raw file with all events included
            raw = mne.io.read_raw_egi(
                input_fname=file_path,
                preload=preload,
                events_as_annotations=True,
                exclude=[],  # Explicitly include all events, no exclusions
                verbose=True,
            )

            message("success", "Successfully loaded .raw file with all events")

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

            # Debug: Print annotations to verify events are loaded
            message(
                "debug", f"Annotations after loading: {raw.annotations.description}"
            )

            # Log detected event types
            event_types = set(raw.annotations.description)
            message("info", f"Detected event types: {event_types}")

            # Step 2: Configure the GSN-HydroCel-124 montage
            message("info", "Configuring GSN-HydroCel-124 montage")

            # First, handle ECG channels (125-128) if they exist in the data
            ecg_channels = ["E125", "E126", "E127", "E128"]

            # Set channel types for ECG channels
            ecg_mapping = {ch: "ecg" for ch in ecg_channels if ch in raw.ch_names}
            if ecg_mapping:
                message(
                    "info",
                    f"Setting channel types for ECG channels: {list(ecg_mapping.keys())}",
                )
                raw.set_channel_types(ecg_mapping)
                # Drop ECG channels
                message("info", f"Dropping ECG channels: {list(ecg_mapping.keys())}")
                raw.drop_channels(list(ecg_mapping.keys()))

            # Now pick only EEG channels before applying montage
            raw.pick("eeg")

            # Now apply the standard 128 montage, but only use the first 124 channels
            message(
                "info", "Applying GSN-HydroCel-128 montage (modified for 124 channels)"
            )
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

            # Apply the montage
            raw.set_montage(montage, match_case=False)

            message("success", "Successfully configured GSN-HydroCel-124 montage")

            # Step 3: Apply task-specific processing if needed
            task = autoclean_dict.get("task", None)
            if task:
                if task.lower() == "p300_grael4k":
                    message("info", "Processing P300 task-specific annotations")
                    mapping = {"13": "Standard", "14": "Target"}
                    raw.annotations.rename(mapping)
                elif task.lower() in ["hbcd_mmn", "hbcd_vep"]:
                    message("info", f"Processing {task} task-specific annotations")
                    # Let the dedicated HBCD event processor handle this
                    # HBCD specific processing here
                    message("info", f"Using generic HBCD processor for {task}")

            return raw

        except Exception as e:
            raise RuntimeError(
                f"Failed to process EGI .raw file with GSN-HydroCel-124 montage: {str(e)}"
            ) from e

    def process_events(self, raw: mne.io.Raw, autoclean_dict: dict) -> tuple:
        """Process events and annotations in the EEG data."""
        message("info", "Processing events from EGI .raw file")
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

                # Log event information
                unique_event_types = events_df["type"].unique()
                message(
                    "info",
                    f"Found {len(events)} events of {len(unique_event_types)} unique types: {unique_event_types}",  # pylint: disable=line-too-long
                )

                # Count events by type
                event_counts = events_df["type"].value_counts().to_dict()
                message("info", f"Event counts: {event_counts}")

                return events, event_id, events_df
            else:
                message("warning", "No events found in the raw data")
                return None, None, None

        except Exception as e:  # pylint: disable=broad-except
            message("warning", f"Failed to process events: {str(e)}")
            return None, None, None

    def get_metadata(self) -> dict:
        """Get additional metadata about this plugin."""
        return {
            "plugin_name": self.__class__.__name__,
            "plugin_version": self.VERSION,
            "montage_details": {
                "type": "GSN-HydroCel-124",
                "channel_count": 124,
                "manufacturer": "Electrical Geodesics, Inc. (EGI)",
                "reference": "Common Reference",
                "layout": "Geodesic",
                "file_format": "EGI .raw binary format",
                "notes": "124 EEG channels, excludes E129 (vertex reference), may include 4 ECG channels (E125-E128)",  # pylint: disable=line-too-long
            },
        }
