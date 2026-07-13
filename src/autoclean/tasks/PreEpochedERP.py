"""Pre-epoched ERP built-in task.

Use this task when the input file already contains epochs, event IDs, and labels
from a completed ERP paradigm such as P300, MMN, or ASSR. The task intentionally
does not run raw-data preprocessing steps by default.
"""

from __future__ import annotations

from pathlib import Path

from autoclean.core.task import Task
from autoclean.utils.erp import generate_erp_outputs, validate_erp_input

config = {
    "schema_version": "2025.09",
    "montage": {"enabled": True, "value": "auto"},
    "move_flagged_files": False,
    "resample_step": {"enabled": False, "value": None},
    "filtering": {
        "enabled": False,
        "value": {
            "l_freq": None,
            "h_freq": None,
            "notch_freqs": None,
            "notch_widths": None,
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
            "method": "fastica",
            "n_components": None,
            "fit_params": None,
            "temp_highpass_for_ica": None,
        },
    },
    "component_rejection": {
        "enabled": False,
        "method": "iclabel",
        "value": {
            "ic_flags_to_reject": ["muscle", "heart", "eog", "ch_noise", "line_noise"],
            "ic_rejection_threshold": 0.3,
        },
    },
    "epoch_settings": {
        "enabled": False,
        "value": {"tmin": None, "tmax": None},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": [None, 0]},
        "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 0.000125}},
    },
    "conditionwise_epoch_rejection": {
        "enabled": False,
        "value": {
            "robust_z_threshold": 4.0,
            "minimum_metric_flags": 2,
            "absolute_amplitude_uv": None,
            "max_reject_fraction": 0.10,
            "minimum_epochs": 20,
            "exclude_channel_types": ["eog", "ecg", "misc"],
            "exclude_channels_matching": ["EOG"],
            "mode": "apply",
            "group_by": "event_id",
        },
    },
    "apply_erp_outputs": {
        "enabled": True,
        "value": {
            "required_conditions": [],
            "conditions": [],
            "analysis_window": [-0.2, 0.8],
            "difference_waves": [],
            "save_evokeds": True,
            "save_amplitudes": True,
            "examples": {
                "P300": {"conditions": ["target", "standard"]},
                "MMN": {"conditions": ["deviant", "standard"]},
                "ASSR": {"conditions": ["stimulus"]},
            },
        },
    },
    "ai_reporting": False,
}


class PreEpochedERP(Task):
    """Task template for already-epoched ERP datasets."""

    def run(self) -> None:
        self.import_epochs()

        erp_enabled, erp_settings = self._check_step_enabled("apply_erp_outputs")
        erp_config = (erp_settings or {}).get("value", {})

        validation = validate_erp_input(
            self.epochs,
            required_conditions=erp_config.get("required_conditions"),
            analysis_window=erp_config.get("analysis_window"),
        )
        self._update_metadata("step_validate_pre_epoched_erp", validation)

        self.apply_conditionwise_epoch_rejection()

        if not erp_enabled:
            return

        output_dir = self._erp_output_dir()
        conditions = erp_config.get("conditions") or None
        outputs = generate_erp_outputs(
            self.epochs,
            output_dir,
            conditions=conditions,
            difference_waves=erp_config.get("difference_waves"),
            analysis_window=erp_config.get("analysis_window"),
            save_evokeds=erp_config.get("save_evokeds", True),
            save_amplitudes=erp_config.get("save_amplitudes", True),
        )
        self._update_metadata("step_generate_erp_outputs", outputs)

    def _erp_output_dir(self) -> Path:
        reports_dir = self.config.get("reports_dir")
        if reports_dir:
            return Path(reports_dir) / "erp"

        metadata_dir = self.config.get("metadata_dir")
        if metadata_dir:
            return Path(metadata_dir) / "erp"

        return Path(self.config["stage_dir"]) / "erp"
