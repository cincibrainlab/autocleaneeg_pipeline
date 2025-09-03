from autoclean.core.task import Task

# =============================================================================
#  Resting state EEG recording EEG PREPROCESSING CONFIGURATION
# =============================================================================
# This configuration controls how your Resting state EEG recording EEG data will be
# automatically cleaned and processed. Each section handles a different aspect
# of the preprocessing pipeline.
#
# 🟢 enabled: True  = Apply this processing step
# 🔴 enabled: False = Skip this processing step
#
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
    "crop_step": {"enabled": False, "value": {"start": 0, "end": 60}},
    "reference_step": {"enabled": True, "value": "average"},
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "ICA": {
        "enabled": True,
        "value": {
            "method": "infomax",
            "n_components": None,
            "fit_params": {"extended": True},
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


class RestingState_Basic(Task):
    """Task implementation for Resting state EEG recording EEG preprocessing."""

    def run(self) -> None:
        # Import raw EEG data
        self.import_raw()

        # Basic preprocessing steps
        self.resample_data()

        self.filter_data()

        self.drop_outer_layer()

        self.assign_eog_channels()

        self.trim_edges()

        self.crop_duration()

        self.original_raw = self.raw.copy()

        # Channel cleaning
        self.clean_bad_channels()

        # Re-referencing
        self.rereference_data()

        # Artifact detection
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()

        # ICA processing with optional export
        self.run_ica()  # Export after ICA
        self.classify_ica_components(method="iclabel")

        # Epoching with export
        # self.create_eventid_epochs() # Using event IDs
        # Epoching with export
        self.create_regular_epochs(export=True)  # Export epochs
        # Detect outlier epochs
        self.detect_outlier_epochs()

        # Clean epochs using GFP with export
        self.gfp_clean_epochs()

        # Generate visualization reports
        self.generate_reports()

    def resume_after_ica(self, raw, post_ica_path=None):
        """
        Resume the task tail after manual ICA re-apply.

        Parameters
        ----------
        raw : mne.io.Raw
            The post-ICA cleaned raw loaded from the manual_ica stage.
        post_ica_path : str | None
            Path to the post_ica_manual file (for context/logging only).
        """
        # Attach post-ICA data to the task
        self.raw = raw
        try:
            self.original_raw = raw.copy()  # for report overlays
        except Exception:
            self.original_raw = None

        # Optional log if you want
        # from autoclean.utils.logging import message
        # message("header", f"Resuming {self.__class__.__name__} from post_ICA: {post_ica_path or '(in‑memory)'}")
        self.create_regular_epochs(export=True)  # Export epochs
        # Detect outlier epochs
        self.detect_outlier_epochs()

        # Clean epochs using GFP with export
        self.gfp_clean_epochs()

        # Reports
        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate quality control visualizations and reports."""
        if self.raw is None or self.original_raw is None:
            return

        # Plot raw vs cleaned overlay using mixin method
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)

        # Plot PSD topography using mixin method
        self.step_psd_topo_figure(self.original_raw, self.raw)

        # Additional report generation can be added here
