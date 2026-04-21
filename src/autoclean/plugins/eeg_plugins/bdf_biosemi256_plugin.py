# src/autoclean/plugins/eeg_plugins/bdf_biosemi256_plugin.py
"""BioSemi BDF file plugin with biosemi256 montage configuration.

This plugin handles the complete import and montage configuration
for BioSemi BDF files with the 256-channel BioSemi electrode system.
"""

from pathlib import Path

import mne
import pandas as pd

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.plugins.eeg_plugins._biosemi_bdf_common import import_biosemi_bdf
from autoclean.utils.logging import message


class BDFBiosemi256Plugin(BaseEEGPlugin):
    """Plugin for BioSemi BDF files with biosemi256 montage.

    This plugin handles the specific combination of BioSemi BDF files
    with the 256-channel BioSemi electrode system. BioSemi systems use
    active electrodes with CMS/DRL referencing during acquisition.
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "BIOSEMI_BDF" and montage_name == "biosemi256"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import BioSemi BDF file and configure biosemi256 montage."""
        try:
            return import_biosemi_bdf(
                file_path=file_path,
                autoclean_dict=autoclean_dict,
                preload=preload,
                montage_name="biosemi256",
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to process BioSemi BDF file with biosemi256 montage: {str(e)}"
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
                "type": "biosemi256",
                "channel_count": 256,
                "manufacturer": "BioSemi",
                "reference": "CMS/DRL active electrodes",
                "layout": "High-density extended 10-5",
                "file_format": "BioSemi BDF (24-bit)",
            },
        }
