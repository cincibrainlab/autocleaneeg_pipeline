# src/autoclean/plugins/eeg_plugins/edf_standard1020_plugin.py
"""EDF plugin for standard 10-20 montage.

This module implements an EEG plugin for reading EDF files and applying
the standard 10-20 montage. A mixin provides the EDF loading function
while the plugin class integrates with the AutoClean plugin system.
"""

from pathlib import Path

import mne

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message

__all__ = ["EDFStandard1020Plugin"]


class EDFImportMixin:
    """Mixin providing EDF file loading functionality."""

    def _load_edf(self, file_path: Path, preload: bool) -> mne.io.Raw:
        """Load an EDF file using MNE."""
        return mne.io.read_raw_edf(file_path, preload=preload, verbose=True)


class EDFStandard1020Plugin(EDFImportMixin, BaseEEGPlugin):
    """Plugin for EDF files with standard_1020 montage."""

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage."""
        return format_id == "EDF_FORMAT" and montage_name == "standard_1020"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ) -> mne.io.Raw:
        """Import EDF data and apply standard_1020 montage."""
        message("info", f"Loading EDF file with standard_1020 montage: {file_path}")
        raw = self._load_edf(file_path, preload)
        message("success", "Successfully loaded EDF file")

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False)
        raw.pick("eeg")
        return raw
