from __future__ import annotations

from autoclean.core.task import Task

# =============================================================================
#  Resting-state sensor PSD from pre-epoched EEG
# =============================================================================
# This built-in task is intentionally minimal. It expects the input file to
# already be epoched, such as an EEGLAB `.set` exported from a prior cleaning
# run, and then writes electrode-level PSD outputs.
#
# Keep using the same input folder at runtime, for example:
#   autocleaneeg-pipeline process RestingState_SensorPSD --dir /path/to/testing
#
# The montage stays fixed to HydroCel-129 so it matches the existing test setup.
# =============================================================================

config = {
    "schema_version": "2025.09",
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
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


class RestingState_SensorPSD(Task):
    """Compute electrode-level PSD from already epoched resting-state EEG."""

    def run(self) -> None:
        """Load pre-epoched data and export sensor PSD artifacts."""
        self.import_epochs()
        self.apply_sensor_psd()
