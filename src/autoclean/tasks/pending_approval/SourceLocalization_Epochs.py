from autoclean.core.task import Task

# =============================================================================
#  SOURCE LOCALIZATION WITH EEG CONVERSION (EPOCHED DATA)
# =============================================================================
# This task tests MNE source localization on epoched data with optional
# conversion to 68-channel EEG format (Desikan-Killiany ROIs).
#
# Tests:
# - estimate_source_function_epochs() algorithm
# - convert_stc_list_to_eeg() conversion function
# - BIDS derivatives output structure
#
# Expected outputs in derivatives/source_localization_eeg/:
# - {subject}_dk_regions.set (68 ROI time courses in EEGLAB Epochs format)
# - {subject}_dk_montage.fif (ROI centroid positions)
# - {subject}_region_info.csv (ROI metadata)
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
    "crop_step": {
        "enabled": True,
        "value": {"start": 0, "end": 60},
    },  # Limit to 60s for speed
    "ICA": {"enabled": False, "value": {"method": "infomax"}},
    "component_rejection": {
        "enabled": False,
        "method": "iclabel",
        "value": {
            "ic_flags_to_reject": ["muscle", "heart", "eog", "ch_noise", "line_noise"],
            "ic_rejection_threshold": 0.3,
        },
    },
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": -1, "tmax": 1},  # 2-second epochs
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
            "convert_to_eeg": True,  # *** Enable STC→EEG conversion ***
        },
    },
}


class SourceLocalization_Epochs(Task):
    """Test source localization on epoched EEG data with conversion."""

    def run(self) -> None:
        # Import and basic preprocessing
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.crop_duration()  # Limit duration for faster testing

        # Re-reference to average (montage is already set during import)
        self.rereference_data()

        # Create epochs from continuous data
        self.create_regular_epochs()

        # Apply source localization with conversion
        # This will:
        # 1. Compute forward solution using fsaverage
        # 2. Create inverse operator
        # 3. Apply inverse to Epochs → List of SourceEstimates
        # 4. Convert to 68-channel EEG Epochs (Desikan-Killiany ROIs)
        # 5. Save EEGLAB .set + montage.fif + region_info.csv
        self.apply_source_localization()
