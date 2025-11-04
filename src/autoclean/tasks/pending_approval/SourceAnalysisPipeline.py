from autoclean.core.task import Task

# =============================================================================
#  FULL SOURCE ANALYSIS PIPELINE TEST
# =============================================================================
# This task tests the complete source analysis pipeline:
# 1. Source localization (MNE inverse)
# 2. Source PSD (ROI-level power spectral density)
# 3. Source connectivity (ROI-to-ROI connectivity)
#
# Tests:
# - apply_source_localization()
# - apply_source_psd()
# - apply_source_connectivity()
#
# Expected outputs:
# - derivatives/source_localization/*.h5 (STCs)
# - derivatives/source_psd/*.parquet, *.csv, *.png
# - derivatives/source_connectivity/*.parquet, *.npy, *.png
# =============================================================================

config = {
    "schema_version": "2025.09",
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 1,
            "h_freq": 100,
            "notch_freqs": [60, 120],
            "notch_widths": 5,
        },
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "eog_step": {"enabled": False, "value": []},
    "trim_step": {"enabled": False, "value": 0},
    "montage": {"enabled": True, "value": "standard_1020"},
    "reference_step": {"enabled": True, "value": "average"},
    "crop_step": {"enabled": True, "value": {"start": 0, "end": 60}},  # 60s for speed
    "ICA": {"enabled": False, "value": {"method": "infomax"}},
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
        "value": {"tmin": -1, "tmax": 1},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": [None, 0]},
        "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 0.000125}},
    },
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",
            "lambda2": 0.111,
            "pick_ori": "normal",
            "n_jobs": 10,
            "convert_to_eeg": False,  # Don't need .set files for this test
        },
    },
    "apply_source_psd": {
        "enabled": True,
        "value": {
            "segment_duration": 60,  # Use all available data
            "n_jobs": 4,
            "generate_plots": True,
        },
    },
    "apply_source_connectivity": {
        "enabled": True,
        "value": {
            "epoch_length": 4.0,  # 4-second epochs
            "n_epochs": 40,  # Number of epochs for averaging
            "n_jobs": 4,  # Parallel jobs
        },
    },
}


class SourceAnalysisPipeline(Task):
    """Test complete source analysis pipeline.

    Note: Connectivity requires continuous data (self.stc), not epochs.
    Source localization creates self.stc from Raw data, or self.stc_list from Epochs.
    """

    def run(self) -> None:
        # Import and basic preprocessing
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.crop_duration()
        self.rereference_data()

        # Source analysis pipeline (no epochs - connectivity needs continuous data)
        self.apply_source_localization()  # Creates self.stc from Raw
        self.apply_source_psd()  # Uses self.stc
        self.apply_source_connectivity()  # Uses self.stc
