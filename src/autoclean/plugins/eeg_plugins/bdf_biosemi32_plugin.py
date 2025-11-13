# src/autoclean/plugins/eeg_plugins/bdf_biosemi32_plugin.py
"""BioSemi BDF file plugin with biosemi32 montage configuration.

This plugin handles the complete import and montage configuration
for BioSemi BDF files with the 32-channel BioSemi electrode system.
"""

from pathlib import Path

import mne
import pandas as pd

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message


class BDFBiosemi32Plugin(BaseEEGPlugin):
    """Plugin for BioSemi BDF files with biosemi32 montage.

    This plugin handles the specific combination of BioSemi BDF files
    with the 32-channel BioSemi electrode system. BioSemi systems use
    active electrodes with CMS/DRL referencing during acquisition.
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "BIOSEMI_BDF" and montage_name == "biosemi32"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import BioSemi BDF file and configure biosemi32 montage."""
        message("info", f"Loading BioSemi BDF file with biosemi32 montage: {file_path}")

        try:
            # Step 1: Import the BDF file with auto status channel detection
            # BioSemi BDF files contain a status channel with trigger information
            raw = mne.io.read_raw_bdf(
                input_fname=file_path,
                preload=preload,
                stim_channel="auto",  # Auto-detect status channel for triggers
                exclude=[],  # Include all channels initially
                verbose=True,
            )
            message("success", "Successfully loaded BDF file with status channel")

            # Log channel information
            message(
                "debug",
                f"Loaded {len(raw.ch_names)} channels: {', '.join(raw.ch_names[:10])}...",
            )

            # Step 1.5: Rename channels to match standard biosemi naming
            # BioSemi BDF files often have prefixed channel names (e.g., A1_Fp1)
            # that need to be stripped to match MNE's standard montage names
            rename_mapping = {}
            for ch_name in raw.ch_names:
                # Skip Status channel
                if ch_name == "Status":
                    continue
                # BioSemi channels may have prefix like A1_, B5_, etc.
                if "_" in ch_name:
                    prefix, standard_name = ch_name.split("_", 1)
                    # Fix known case issues (e.g., Afz -> AFz)
                    if standard_name == "Afz":
                        standard_name = "AFz"
                    rename_mapping[ch_name] = standard_name

            if rename_mapping:
                raw.rename_channels(rename_mapping)
                message(
                    "debug",
                    f"Renamed {len(rename_mapping)} channels to standard names",
                )

            # Step 2: Configure the biosemi32 montage
            message("info", "Configuring biosemi32 montage")

            # Apply BioSemi 32-channel standard montage
            montage = mne.channels.make_standard_montage("biosemi32")
            raw.set_montage(montage, match_case=False, on_missing="warn")

            message("success", "Successfully applied biosemi32 montage")

            # Step 3: Exclude external channels that won't have montage positions
            # EXG channels (EOG, mastoids, etc.) don't have positions in standard montage
            exg_to_exclude = [
                ch
                for ch in raw.ch_names
                if ch
                in [
                    "LM",
                    "RM",
                    "LVE",
                    "RVE",
                    "LHE",
                    "RHE",
                    "EXG7",
                    "EXG8",
                    "EXG1",
                    "EXG2",
                    "EXG3",
                    "EXG4",
                    "EXG5",
                    "EXG6",
                ]
            ]

            if exg_to_exclude:
                raw.drop_channels(exg_to_exclude)
                message(
                    "debug",
                    f"Dropped {len(exg_to_exclude)} EXG channels without montage positions",
                )

            # Pick EEG and stimulus channels
            # Keep stimulus channels for event extraction
            raw.pick_types(eeg=True, stim=True, exclude=[])

            message(
                "info",
                f"Selected {len(raw.ch_names)} channels (EEG + status)",
            )

            # Note: BioSemi systems use CMS/DRL active referencing during acquisition.
            # Rereferencing should be done in the pipeline preprocessing steps if needed.
            message(
                "info",
                "BioSemi data retains CMS/DRL referencing from acquisition. "
                "Apply rereferencing in pipeline if needed.",
            )

            return raw

        except Exception as e:
            raise RuntimeError(
                f"Failed to process BioSemi BDF file with biosemi32 montage: {str(e)}"
            ) from e

    def process_events(self, raw: mne.io.Raw) -> tuple:
        """Process events and annotations from BDF status channel.

        BioSemi BDF files encode triggers in a 16-bit status channel,
        with system status information in the upper bits.
        """
        message("info", "Processing events from BDF status channel")
        try:
            # Get events from annotations (MNE auto-extracts from status channel)
            events, event_id = mne.events_from_annotations(raw)

            # Create a detailed events DataFrame
            if events is not None and len(events) > 0:
                events_df = pd.DataFrame(
                    {
                        "time": events[:, 0] / raw.info["sfreq"],
                        "sample": events[:, 0],
                        "id": events[:, 2],
                        "type": [
                            (
                                list(event_id.keys())[list(event_id.values()).index(id)]
                                if id in event_id.values()
                                else f"Unknown-{id}"
                            )
                            for id in events[:, 2]
                        ],
                    }
                )

                # Log event information
                unique_event_types = events_df["type"].unique()
                message(
                    "info",
                    f"Found {len(events)} events of {len(unique_event_types)} unique types: {unique_event_types}",
                )

                # Count events by type
                event_counts = events_df["type"].value_counts().to_dict()
                message("info", f"Event counts: {event_counts}")

                return events, event_id, events_df
            else:
                message("warning", "No events found in the BDF status channel")
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
                "type": "biosemi32",
                "channel_count": 32,
                "manufacturer": "BioSemi",
                "reference": "CMS/DRL active electrodes",
                "layout": "International 10-20 extended",
                "file_format": "BioSemi BDF (24-bit)",
            },
        }
