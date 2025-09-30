from autoclean.core.task import Task

# =============================================================================
#  SOURCE LOCALIZATION WITH EEG CONVERSION (RAW DATA)
# =============================================================================
# This task tests MNE source localization on continuous Raw data with optional
# conversion to 68-channel EEG format (Desikan-Killiany ROIs).
#
# Tests:
# - estimate_source_function_raw() algorithm
# - convert_stc_to_eeg() conversion function
# - BIDS derivatives output structure
#
# Expected outputs in derivatives/source_localization_eeg/:
# - {subject}_dk_regions.set (68 ROI time courses in EEGLAB format)
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
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "reference_step": {"enabled": True, "value": "average"},
    "crop_step": {"enabled": True, "value": {"start": 0, "end": 60}},  # Limit to 60s for speed
    "ICA": {"enabled": False, "value": {}},
    "component_rejection": {"enabled": False, "method": "none", "value": {}},
    "epoch_settings": {"enabled": False, "value": {}},
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


class SourceLocalization_Raw(Task):
    """Test source localization on continuous (Raw) EEG data with conversion."""

    def run(self) -> None:
        # Import and basic preprocessing
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.crop_duration()  # Limit duration for faster testing

        # Set montage (required for source localization)
        self.set_montage()

        # Re-reference to average
        self.rereference_data()

        # Apply source localization with conversion
        # This will:
        # 1. Compute forward solution using fsaverage
        # 2. Create inverse operator
        # 3. Apply inverse to Raw data → SourceEstimate (10,242 vertices)
        # 4. Convert to 68-channel EEG (Desikan-Killiany ROIs)
        # 5. Save EEGLAB .set + montage.fif + region_info.csv
        stc = self.apply_source_localization()

        # Verify outputs
        if stc is not None:
            self.message("success", f"Source localization complete: {stc.data.shape}")

            if hasattr(self, "source_eeg") and self.source_eeg is not None:
                self.message("success", f"STC→EEG conversion complete: {self.source_eeg.info['nchan']} ROI channels")
                self.message("info", f"Output file: {self.source_eeg_file}")
            else:
                self.message("warning", "STC→EEG conversion was not performed")
        else:
            self.message("warning", "Source localization returned None")