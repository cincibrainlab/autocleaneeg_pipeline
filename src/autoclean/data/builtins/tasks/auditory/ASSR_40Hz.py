"""Auditory Steady-State Response (40 Hz) built-in task."""

from __future__ import annotations

from pathlib import Path

from autoclean.calc.assr_analysis import analyze_assr
from autoclean.core.task import Task

config = {
    "schema_version": "2025.09",
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "move_flagged_files": False,
    "resample_step": {"enabled": True, "value": 500},
    "filtering": {
        "enabled": True,
        "value": {"l_freq": 1.0, "h_freq": 120.0, "notch_freqs": [60, 120]},
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "eog_step": {
        "enabled": True,
        "value": {
            "eog_indices": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
            "eog_drop": True,
        },
    },
    "trim_step": {"enabled": True, "value": 2},
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
            "psd_fmax": 80.0,
        },
    },
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": -0.3, "tmax": 0.7},
        "event_id": {"ASSR_40Hz": 1},
        "remove_baseline": {"enabled": True, "window": [-0.2, 0.0]},
        "threshold_rejection": {
            "enabled": True,
            "volt_threshold": {"eeg": 0.0002},
        },
    },
    "assr_analysis": {
        "enabled": False,
        "value": {"profile": "assr_epochs"},
    },
    "ai_reporting": False,
}


class ASSR_40Hz(Task):
    """Task for auditory steady-state response paradigms at 40 Hz."""

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
        self.run_assr_analysis()

        self.generate_reports()

    def run_assr_analysis(self) -> None:
        """Run optional ASSR analysis on the cleaned epochs."""
        if (
            getattr(self, "settings", None) is not None
            and "assr_analysis" not in self.settings
        ):
            return
        is_enabled, step_config = self._check_step_enabled("assr_analysis")
        if not is_enabled:
            return
        if self.epochs is None:
            raise RuntimeError("ASSR analysis requires cleaned epochs")

        analysis_config = dict((step_config or {}).get("value") or {})
        analysis_profile = analysis_config.pop("profile", None)
        input_file = self.config.get("unprocessed_file")
        file_basename = Path(input_file).stem if input_file else None

        analyze_assr(
            output_dir=self._resolve_report_path("assr"),
            save_results=True,
            epochs=self.epochs,
            file_basename=file_basename,
            analysis_profile=analysis_profile,
            analysis_config=analysis_config,
        )

    def generate_reports(self) -> None:
        if self.raw is None or self.original_raw is None:
            return

        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        self.step_psd_topo_figure(self.original_raw, self.raw)
