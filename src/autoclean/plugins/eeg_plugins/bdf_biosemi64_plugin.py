# src/autoclean/plugins/eeg_plugins/bdf_biosemi64_plugin.py
"""BioSemi BDF plugin for biosemi64 montage.

This module provides an EEG plugin for handling BioSemi BDF files
with the biosemi64 electrode montage. It follows the plugin and mixin
pattern used throughout the project, where a mixin supplies the core
loading functionality and the plugin class handles format/montage
configuration.
"""

from pathlib import Path

import mne

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message

__all__ = ["BiosemiBDFPlugin"]


class BDFImportMixin:
    """Mixin providing BDF file loading functionality."""

    def _load_bdf(self, file_path: Path, preload: bool) -> mne.io.Raw:
        """Load a BDF file using MNE."""
        return mne.io.read_raw_bdf(file_path, preload=preload, verbose=True)


class BiosemiBDFPlugin(BDFImportMixin, BaseEEGPlugin):
    """Plugin for BioSemi BDF files with biosemi64 montage."""

    VERSION = "1.0.0"

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage."""
        return format_id == "BIOSEMI_BDF" and montage_name.lower() == "biosemi64"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ) -> mne.io.Raw:
        """Import BDF data and apply biosemi64 montage."""
        message("info", f"Loading BDF file with BioSemi64 montage: {file_path}")
        raw = self._load_bdf(file_path, preload)
        message("success", "Successfully loaded BDF file")

        montage = mne.channels.make_standard_montage("biosemi64")
        raw.set_montage(montage, match_case=False)
        raw.pick("eeg")
        return raw
