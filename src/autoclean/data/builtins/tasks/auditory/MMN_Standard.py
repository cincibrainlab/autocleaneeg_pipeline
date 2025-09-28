"""Mismatch Negativity (standard oddball) built-in task."""

from __future__ import annotations

from autoclean.core.task import Task

config = {
    "schema_version": "2025.09",
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "move_flagged_files": False,
    "resample_step": {"enabled": True, "value": 500},
    "filtering": {
        "enabled": True,
        "value": {"l_freq": 0.1, "h_freq": 40.0, "notch_freqs": [60]},
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "eog_step": {
        "enabled": True,
        "value": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
    },
    "trim_step": {"enabled": True, "value": 2},
    "crop_step": {"enabled": False, "value": {"start": 0, "end": 0}},
    "reference_step": {"enabled": True, "value": "average"},
    "ICA": {
        "enabled": True,
        "value": {
            "method": "fastica",
            "n_components": None,
            "fit_params": {"tol": 0.0001},
            "temp_highpass_for_ica": 1.0,
        },
    },
    "component_rejection": {
        "enabled": True,
        "method": "icvision",
        "value": {
            "ic_flags_to_reject": [
                "muscle",
                "heart",
                "eog",
                "ch_noise",
                "line_noise",
            ],
            "ic_rejection_threshold": 0.28,
        },
    },
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": -0.2, "tmax": 0.6},
        "event_id": {"standard": 1, "deviant": 2},
        "remove_baseline": {"enabled": True, "window": [-0.2, 0.0]},
        "threshold_rejection": {
            "enabled": True,
            "volt_threshold": {"eeg": 0.00018},
        },
    },
    "ai_reporting": False,
}


class MMN_Standard(Task):
    """Task for standard auditory oddball mismatch negativity experiments."""

    def run(self) -> None:
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.drop_outer_layer()
        self.assign_eog_channels()
        self.trim_edges()
        self.crop_duration()

        self.original_raw = self.raw.copy()

        self.clean_bad_channels()
        self.rereference_data()

        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()

        self.run_ica()
        self.classify_ica_components(method="iclabel")

        self.create_eventid_epochs()
        self.detect_outlier_epochs()
        self.gfp_clean_epochs()

        self.generate_reports()

    def generate_reports(self) -> None:
        if self.raw is None or self.original_raw is None:
            return

        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        self.step_psd_topo_figure(self.original_raw, self.raw)
