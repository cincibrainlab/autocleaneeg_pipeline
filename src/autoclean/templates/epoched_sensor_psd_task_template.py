from __future__ import annotations

from autoclean.core.task import Task

# =============================================================================
#  Minimal epoched sensor-PSD task
# =============================================================================
# Use this template when you already have an epoched EEG file, such as an
# EEGLAB `.set`, and you only want to compute/export electrode-level PSD.
#
# Typical flow:
# 1. Point the pipeline at an epoched `.set` or `.fif`
# 2. Run `self.import_epochs()`
# 3. Run `self.apply_sensor_psd()`
#
# The PSD step writes frequency-resolved spectra and band summaries to
# `reports/sensor_psd/` for the current run.
# =============================================================================

config = {
    "schema_version": "2025.09",
    "montage": {"enabled": False, "value": None},
    "move_flagged_files": False,
    "resample_step": {"enabled": False, "value": None},
    "filtering": {
        "enabled": False,
        "value": {
            "l_freq": None,
            "h_freq": None,
            "notch_freqs": None,
        },
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "eog_step": {"enabled": False, "value": None},
    "trim_step": {"enabled": False, "value": 0},
    "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
    "reference_step": {"enabled": False, "value": None},
    "ICA": {
        "enabled": False,
        "value": {
            "method": "infomax",
        },
    },
    "component_rejection": {
        "enabled": False,
        "method": "iclabel",
        "value": {
            "ic_flags_to_reject": [],
            "ic_rejection_threshold": 0.3,
        },
    },
    "epoch_settings": {
        "enabled": False,
        "value": {"tmin": None, "tmax": None},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": None},
        "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 0.000125}},
    },
    "apply_sensor_psd": {
        "enabled": True,
        "value": {
            "method": "welch",
            "fmin": 1.0,
            "fmax": 45.0,
            "picks": "eeg",
            "n_jobs": 1,
        },
    },
    "ai_reporting": False,
}


class EpochedSensorPSD(Task):
    """Minimal task for importing epoched data and exporting sensor PSD."""

    def run(self) -> None:
        """Load epochs, compute electrode PSD, and export analysis artifacts."""
        # Use import_epochs() for epoched `.set` / `.fif` inputs.
        self.import_epochs()

        # This writes PSD tables into reports/sensor_psd/ and records metadata.
        self.apply_sensor_psd()
