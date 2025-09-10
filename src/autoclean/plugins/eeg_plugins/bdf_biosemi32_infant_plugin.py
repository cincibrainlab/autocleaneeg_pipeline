# src/autoclean/plugins/eeg_plugins/bdf_biosemi32_infant_plugin.py
"""BioSemi BDF plugin for infant biosemi32 montage.

This module provides an EEG plugin for handling BioSemi BDF files
with the infant BioSemi32 electrode montage. BioSemi BDF stores
channels as A1–A32, so we remap them to standard names before
applying the montage. Additional non-EEG channels (EXG, GSR, Resp, etc.)
are also typed correctly.
"""

from pathlib import Path

import mne

from autoclean.io.import_ import BaseEEGPlugin
from autoclean.utils.logging import message

__all__ = ["BiosemiBDFInfantPlugin"]


class BDFImportMixin:
    """Mixin providing BDF file loading functionality."""

    def _load_bdf(self, file_path: Path, preload: bool) -> mne.io.Raw:
        """Load a BDF file using MNE."""
        return mne.io.read_raw_bdf(file_path, preload=preload, verbose=True)


class BiosemiBDFInfantPlugin(BDFImportMixin, BaseEEGPlugin):
    """Plugin for BioSemi BDF files with infant biosemi32 montage."""

    VERSION = "1.0.0"

    # Mapping from BioSemi A1–A32 to standard biosemi32 names
    CHANNEL_MAPPING = {
        "A1": "Fp1",
        "A2": "AF3",
        "A3": "F7",
        "A4": "F3",
        "A5": "FC1",
        "A6": "FC5",
        "A7": "T7",
        "A8": "C3",
        "A9": "CP1",
        "A10": "CP5",
        "A11": "P7",
        "A12": "P3",
        "A13": "Pz",
        "A14": "PO3",
        "A15": "O1",
        "A16": "Oz",
        "A17": "O2",
        "A18": "PO4",
        "A19": "P4",
        "A20": "P8",
        "A21": "CP6",
        "A22": "CP2",
        "A23": "C4",
        "A24": "T8",
        "A25": "FC6",
        "A26": "FC2",
        "A27": "F4",
        "A28": "F8",
        "A29": "AF4",
        "A30": "Fp2",
        "A31": "Fz",
        "A32": "Cz",
    }

    # Non-EEG channel types
    NON_EEG_TYPES = {
        "EXG1": "eog",
        "EXG2": "eog",  # eye movements
        "EXG3": "ecg",  # heart
        "EXG4": "emg",
        "EXG5": "emg",  # muscle
        "EXG6": "misc",
        "EXG7": "misc",
        "EXG8": "misc",
        "GSR1": "misc",
        "GSR2": "misc",
        "Erg1": "emg",
        "Erg2": "emg",
        "Resp": "resp",
        "Plet": "misc",
        "Temp": "misc",
    }

    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage."""
        return format_id == "BIOSEMI_BDF" and montage_name.lower() == "biosemi32_infant"

    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ) -> mne.io.Raw:
        """Import BDF data and apply infant biosemi32 montage."""
        message("info", f"Loading BDF file with infant BioSemi32 montage: {file_path}")
        raw = self._load_bdf(file_path, preload)
        message("success", "Successfully loaded BDF file")

        # Rename BioSemi A1–A32 → standard channel names
        raw.rename_channels(self.CHANNEL_MAPPING)

        # Assign non-EEG channel types
        raw.set_channel_types(self.NON_EEG_TYPES, verbose="warning")

        # Apply montage for EEG electrodes only
        montage = mne.channels.make_standard_montage("biosemi32")
        raw.set_montage(montage, match_case=False)

        return raw
