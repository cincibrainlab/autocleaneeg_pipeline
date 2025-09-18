from __future__ import annotations

from autoclean.core.task import Task

# =============================================================================
#  Resting state EEG preprocessing with wavelet thresholding
# =============================================================================
# This configuration extends the basic resting-state pipeline by inserting a
# wavelet denoising step that mirrors the updated HAPPE-style logic.
# =============================================================================

config = {
    "move_flagged_files": False,
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
    "eog_step": {
        "enabled": False,
        "value": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
    },
    "trim_step": {"enabled": True, "value": 4},
    "crop_step": {"enabled": True, "value": {"start": 0, "end": 60}},
    "wavelet_threshold": {
        "enabled": True,
        "value": {
            "wavelet": "sym4",
            "level": 5,
            "threshold_mode": "soft",
            "is_erp": False,
            "bandpass": [1.0, 30.0],
        },
    },
    "reference_step": {"enabled": True, "value": "average"},
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "ICA": {
        "enabled": True,
        "value": {
            "method": "infomax",
            "n_components": None,
            "fit_params": {"extended": True},
            "temp_highpass_for_ica": 1.0,
        },
    },
    "component_rejection": {
        "enabled": True,
        "method": "icvision",
        "value": {
            "ic_flags_to_reject": ["muscle", "heart", "eog", "ch_noise", "line_noise"],
            "ic_rejection_threshold": 0.3,
        },
    },
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": -1, "tmax": 1},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": [None, 0]},
        "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 0.000125}},
    },
}


class RestingState_BasicWavelet(Task):
    """Resting-state pipeline variant that adds wavelet thresholding."""

    def run(self) -> None:
        """Execute the preprocessing pipeline."""
        # Import raw EEG data
        self.import_raw()

        # Basic preprocessing steps
        self.resample_data()
        self.filter_data()
        self.drop_outer_layer()
        self.assign_eog_channels()
        self.trim_edges()
        self.crop_duration()

        # Preserve a copy for reporting overlays
        if self.raw is not None:
            self.original_raw = self.raw.copy()

        # Wavelet denoising prior to channel cleaning and ICA
        self.apply_wavelet_threshold()

        # Channel cleaning
        self.clean_bad_channels()

        # Re-referencing
        self.rereference_data()

        # Artifact detection
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()

        # ICA processing with optional export
        self.run_ica()
        self.classify_ica_components(method="iclabel")

        # Epoching with export
        self.create_regular_epochs(export=True)
        self.detect_outlier_epochs()

        # Clean epochs using GFP with export
        self.gfp_clean_epochs()

        # Generate visualization reports
        self.generate_reports()

    def resume_after_ica(self, raw, post_ica_path=None):
        """Resume the task tail after manual ICA re-apply."""
        self.raw = raw
        try:
            self.original_raw = raw.copy()
        except Exception:
            self.original_raw = None

        self.create_regular_epochs(export=True)
        self.detect_outlier_epochs()
        self.gfp_clean_epochs()
        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate quality control visualizations and reports."""
        if self.raw is None or self.original_raw is None:
            return

        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        self.step_psd_topo_figure(self.original_raw, self.raw)
        self.generate_fastplot_summary()
