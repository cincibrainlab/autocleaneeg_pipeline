"""RestingEyesOpen built-in task.

This task provides a baseline resting-state pipeline tuned for eyes-open data.
"""

from __future__ import annotations

from autoclean.core.task import Task

config = {
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "move_flagged_files": False,
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {"l_freq": 1.0, "h_freq": 80.0, "notch_freqs": [60, 120]},
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "eog_step": {
        "enabled": True,
        "value": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
    },
    "trim_step": {"enabled": True, "value": 4},
    "crop_step": {"enabled": False, "value": {"start": 0, "end": 0}},
    "reference_step": {"enabled": True, "value": "average"},
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
            "ic_flags_to_reject": [
                "muscle",
                "heart",
                "eog",
                "ch_noise",
                "line_noise",
            ],
            "ic_rejection_threshold": 0.3,
        },
    },
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": -1.0, "tmax": 1.0},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": [None, 0]},
        "threshold_rejection": {
            "enabled": False,
            "volt_threshold": {"eeg": 0.000125},
        },
    },
    "ai_reporting": False,
}


class RestingEyesOpen(Task):
    """Task implementation for resting-state EEG with eyes open."""

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

        self.create_regular_epochs(export=True)
        self.detect_outlier_epochs()
        self.gfp_clean_epochs()

        self.generate_reports()

    def generate_reports(self) -> None:
        if self.raw is None or self.original_raw is None:
            return

        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        self.step_psd_topo_figure(self.original_raw, self.raw)
