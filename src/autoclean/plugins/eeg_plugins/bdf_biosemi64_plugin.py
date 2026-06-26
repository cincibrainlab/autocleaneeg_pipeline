# src/autoclean/plugins/eeg_plugins/bdf_biosemi64_plugin.py
"""BioSemi BDF file plugin with biosemi64 montage configuration.

This plugin handles the complete import and montage configuration
for BioSemi BDF files with the 64-channel BioSemi electrode system.
"""

from pathlib import Path

import mne

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.plugins.eeg_plugins._biosemi_bdf_common import (
    import_biosemi_bdf,
    process_biosemi_bdf_events,
)
from autoclean.utils.logging import message


class BDFBiosemi64Plugin(BaseEEGPlugin):
    """Plugin for BioSemi BDF files with biosemi64 montage.

    This plugin handles the specific combination of BioSemi BDF files
    with the 64-channel BioSemi electrode system. BioSemi systems use
    active electrodes with CMS/DRL referencing during acquisition.
    """

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "BIOSEMI_BDF" and montage_name == "biosemi64"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ):
        """Import BioSemi BDF file and configure biosemi64 montage."""
        try:
            return import_biosemi_bdf(
                file_path=file_path,
                autoclean_dict=autoclean_dict,
                preload=preload,
                montage_name="biosemi64",
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to process BioSemi BDF file with biosemi64 montage: {str(e)}"
            ) from e

    def process_events(self, raw: mne.io.Raw) -> tuple:
        """Process events and annotations from BDF status channel."""
        return process_biosemi_bdf_events(raw)

    def get_metadata(self) -> dict:
        """Get additional metadata about this plugin."""
        return {
            "plugin_name": self.__class__.__name__,
            "plugin_version": self.VERSION,
            "montage_details": {
                "type": "biosemi64",
                "channel_count": 64,
                "manufacturer": "BioSemi",
                "reference": "CMS/DRL active electrodes",
                "layout": "International 10-20 extended",
                "file_format": "BioSemi BDF (24-bit)",
            },
        }
