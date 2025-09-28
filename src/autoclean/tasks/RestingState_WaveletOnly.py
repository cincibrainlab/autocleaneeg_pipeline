"""Resting-state pipeline focused on wavelet-based denoising.

This exemplar task highlights the wavelet thresholding options introduced
in the pipeline. It performs light preprocessing (resample + band-pass),
applies wavelet denoising with an auto-selected decomposition depth, and
generates comparison reports without running ICA.
"""

from __future__ import annotations

from autoclean.core.task import Task


config = {
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "move_flagged_files": False,
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 1.0,
            "h_freq": 45.0,
            "notch_freqs": [60.0],
            "notch_widths": 4.0,
        },
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "trim_step": {"enabled": True, "value": 2.0},
    "crop_step": {"enabled": False, "value": {"start": 0.0, "end": 0.0}},
    "wavelet_threshold": {
        "enabled": True,
        "value": {
            "wavelet": "sym4",
            "level": "auto",           # Use the deepest safe decomposition level
            "threshold_mode": "soft",
            "threshold_scale": 0.85,     # Slightly less aggressive than default
            "psd_fmax": 35.0,            # Align PSD metrics with the band-pass
            "picks": "eeg",
            "bandpass": [1.0, 30.0],
            "filter_kwargs": None,
            "is_erp": False,
        },
    },
    "reference_step": {"enabled": True, "value": "average"},
    "epoch_settings": {
        "enabled": False,
        "value": {"tmin": -1.0, "tmax": 1.0},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": [None, 0.0]},
        "threshold_rejection": {
            "enabled": False,
            "volt_threshold": {"eeg": 0.000125},
        },
    },
    "ai_reporting": False,
}


class RestingState_WaveletOnly(Task):
    """Minimal resting-state workflow that showcases wavelet denoising."""

    def run(self) -> None:
        """Execute the wavelet-centric preprocessing pipeline."""

        # Load and prepare the data
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.trim_edges()
        self.crop_duration()

        if self.raw is not None:
            self.original_raw = self.raw.copy()

        # Apply wavelet thresholding with the configured options
        self.apply_wavelet_threshold()

        # Re-reference after denoising for consistency
        self.rereference_data()

        # Light-weight reporting so users can verify the effect
        self.generate_reports()

    def generate_reports(self) -> None:
        """Emit quick comparison figures for QA."""

        if self.raw is None or self.original_raw is None:
            return

        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        self.step_psd_topo_figure(self.original_raw, self.raw)
