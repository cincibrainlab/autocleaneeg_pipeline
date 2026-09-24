#!/usr/bin/env python3
"""Create a deterministic epoched EEGLAB file for Exclude GUI smoke testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

SMOKE_TASK_CONFIG = {
    "schema_version": "2025.09",
    "montage": {"enabled": True, "value": "standard_1020"},
    "move_flagged_files": False,
    "prior_preprocessing_detection": {"enabled": False, "strict": False},
    "resample_step": {"enabled": False, "value": 100},
    "filtering": {
        "enabled": False,
        "value": {"l_freq": 1, "h_freq": 40, "notch_freqs": None},
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "bad_channel_log": {
        "enabled": False,
        "value": {"path": "unused.csv", "action": "mark", "strict": False},
    },
    "bad_channel_detection": {"enabled": False, "value": {}},
    "exclusion_list": {
        "enabled": False,
        "value": {"path": "unused.csv", "mode": "tag", "strict": False},
    },
    "eog_step": {"enabled": False, "value": None},
    "trim_step": {"enabled": False, "value": 0},
    "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
    "wavelet_threshold": {
        "enabled": False,
        "value": {
            "wavelet": "db4",
            "level": "auto",
            "threshold_mode": "soft",
            "is_erp": False,
        },
    },
    "reference_step": {"enabled": False, "value": None},
    "ICA": {
        "enabled": True,
        "value": {
            "method": "fastica",
            "n_components": 2,
            "random_state": 97,
            "max_iter": 200,
            "temp_highpass_for_ica": None,
        },
    },
    "component_rejection": {
        "enabled": True,
        "method": "iclabel",
        "value": {"ic_flags_to_reject": [], "ic_rejection_threshold": 0.9},
    },
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": 0, "tmax": 2},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": [None, 0]},
        "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 0.000125}},
    },
    "apply_autoreject": {
        "enabled": False,
        "value": {"n_interpolate": [1], "consensus": [0.1]},
    },
    "postprocessing_analysis": {"enabled": False, "value": {}},
    "conditionwise_epoch_rejection": {"enabled": False, "value": {}},
    "assr_analysis": {"enabled": False, "value": {}},
    "apply_source_localization": {"enabled": False, "value": {}},
    "apply_source_psd": {"enabled": False, "value": {}},
    "apply_sensor_psd": {"enabled": False, "value": {}},
    "apply_source_connectivity": {"enabled": False, "value": {}},
    "apply_fooof_aperiodic": {"enabled": False, "value": {}},
    "apply_fooof_periodic": {"enabled": False, "value": {}},
    "apply_matlab": {
        "enabled": False,
        "value": {"kind": "function", "entrypoint": "unused"},
    },
    "run_matlab": {
        "enabled": False,
        "value": {"kind": "function", "entrypoint": "unused"},
    },
    "apply_matlab_fooof": {
        "enabled": False,
        "value": {"vhtp_path": "unused", "eeglab_path": "unused"},
    },
    "ai_reporting": False,
}


def create_fixture(output_dir: Path) -> Path:
    """Write a small epoched EEG fixture and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    task_root = output_dir.parent
    output_path = output_dir / "exclude_gui_smoke_epo.set"
    raw_path = task_root / "exclude_gui_smoke_raw.set"

    sfreq = 100.0
    ch_names = ["Fp1", "Fp2", "Cz", "Oz"]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    rng = np.random.default_rng(314)
    samples_per_epoch = 300
    times = np.arange(samples_per_epoch) / sfreq

    data = []
    for epoch_idx in range(12):
        epoch = []
        blink_like_source = 4e-6 * np.exp(-((times - 1.2) ** 2) / 0.08)
        for channel_idx, _channel in enumerate(ch_names):
            frequency = channel_idx + 2
            amplitude = (10 + channel_idx * 2) * 1e-6
            phase = channel_idx * np.pi / 7
            drift = 0.1e-6 * epoch_idx * np.linspace(-1, 1, len(times))
            noise = rng.normal(0, 0.35e-6, len(times))
            mixed_source = (1 - channel_idx * 0.12) * blink_like_source
            epoch.append(
                amplitude * np.sin(2 * np.pi * frequency * times + phase)
                + mixed_source
                + drift
                + noise
            )
        data.append(epoch)

    events = np.column_stack(
        [
            np.arange(12, dtype=int) * len(times),
            np.zeros(12, dtype=int),
            np.ones(12, dtype=int),
        ]
    )
    epochs = mne.EpochsArray(
        np.asarray(data),
        info,
        events=events,
        event_id={"smoke": 1},
        tmin=0.0,
        verbose=False,
    )
    epochs.export(str(output_path), fmt="eeglab", overwrite=True)
    raw = mne.io.RawArray(np.concatenate(data, axis=1), info, verbose=False)
    raw.export(str(raw_path), fmt="eeglab", overwrite=True)
    _write_reprocess_context(task_root, raw_path)
    return output_path


def _write_reprocess_context(task_root: Path, raw_path: Path) -> None:
    """Add the minimal task-root files needed for reprocess smoke testing."""
    reports_dir = task_root / "reports" / "run_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = reports_dir / "exclude_gui_smoke_autoclean_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "unprocessed_file": str(raw_path),
                "metadata": {
                    "import_eeg": {"originalChannelNames": ["Fp1", "Fp2", "Cz", "Oz"]},
                    "step_run_ica": {
                        "ica": {
                            "ica_components": 2,
                            "ica_fit_data_type": "epochs",
                            "ica_kwargs": {
                                "method": "fastica",
                                "n_components": 2,
                                "random_state": 97,
                            },
                        }
                    },
                    "classify_ica_components": {
                        "ica": {
                            "classification_method": "iclabel",
                            "ica_components": 2,
                        }
                    },
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    status_dir = task_root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "ExcludeGuiSmokeTask.py").write_text(
        (
            "from autoclean.core.task import Task\n\n"
            f"config = {SMOKE_TASK_CONFIG!r}\n\n"
            "class ExcludeGuiSmokeTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.create_regular_epochs(export=True)\n"
            "        self.run_ica()\n"
            "        self.classify_ica_components(method='iclabel')\n"
            "        self.apply_ica_component_rejection()\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(create_fixture(args.output_dir))


if __name__ == "__main__":
    main()
