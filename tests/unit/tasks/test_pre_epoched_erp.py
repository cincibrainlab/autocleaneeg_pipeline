from __future__ import annotations

import csv

import mne
import numpy as np
import pytest

from autoclean.configkit.schema import validate_task_module_config
from autoclean.utils.erp import (
    _condition_counts,
    generate_erp_outputs,
    validate_erp_input,
)


def _epochs() -> mne.EpochsArray:
    info = mne.create_info(["Fz", "Cz"], sfreq=100, ch_types="eeg")
    data = np.zeros((4, 2, 101))
    data[2:] = 1e-6
    events = np.array(
        [
            [0, 0, 1],
            [150, 0, 1],
            [300, 0, 2],
            [450, 0, 2],
        ]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"standard": 1, "target": 2},
        tmin=-0.2,
        verbose=False,
    )


def test_pre_epoched_erp_task_config_validates() -> None:
    from autoclean import Pipeline

    assert Pipeline.__name__ == "Pipeline"
    from autoclean.tasks.PreEpochedERP import config

    validated = validate_task_module_config(config)

    assert validated["epoch_settings"]["enabled"] is False
    assert validated["filtering"]["enabled"] is False
    assert validated["ICA"]["enabled"] is False
    assert validated["reference_step"]["enabled"] is False


def test_validate_erp_input_preserves_event_labels_and_counts() -> None:
    result = validate_erp_input(
        _epochs(),
        required_conditions=["standard", "target"],
        analysis_window=[-0.1, 0.5],
    )

    assert result["event_id"] == {"standard": 1, "target": 2}
    assert result["condition_counts"] == {"standard": 2, "target": 2}
    assert result["warnings"] == []


def test_validate_erp_input_rejects_missing_condition() -> None:
    with pytest.raises(ValueError, match="missing required condition"):
        validate_erp_input(_epochs(), required_conditions=["deviant"])


def test_validate_erp_input_warns_when_window_exceeds_epochs() -> None:
    result = validate_erp_input(_epochs(), analysis_window=[-0.3, 0.9])

    assert "outside epoch coverage" in result["warnings"][0]


def test_validate_erp_input_rejects_short_analysis_window() -> None:
    with pytest.raises(ValueError, match="analysis_window must contain start and end"):
        validate_erp_input(_epochs(), analysis_window=[0.1])


def test_generate_erp_outputs_rejects_short_analysis_window(tmp_path) -> None:
    with pytest.raises(ValueError, match="analysis_window must contain start and end"):
        generate_erp_outputs(_epochs(), tmp_path, analysis_window=[0.1])


def test_condition_counts_warns_when_condition_slice_fails(monkeypatch) -> None:
    messages = []

    class BrokenEpochs:
        event_id = {"bad": 1}

        def __getitem__(self, condition):
            raise RuntimeError(f"cannot slice {condition}")

    monkeypatch.setattr(
        "autoclean.utils.erp.message",
        lambda level, text: messages.append((level, text)),
    )

    assert _condition_counts(BrokenEpochs()) == {"bad": 0}
    assert messages == [
        ("warning", "Could not count epochs for condition 'bad': cannot slice bad")
    ]


def test_generate_erp_outputs_clamps_out_of_range_amplitude_window(tmp_path) -> None:

    result = generate_erp_outputs(
        _epochs(),
        tmp_path,
        conditions=["standard"],
        analysis_window=[-0.5, 1.2],
    )

    assert result["amplitude_summary_file"] is not None
    assert (tmp_path / "erp_amplitude_summary.csv").exists()


def test_generate_erp_outputs_writes_counts_evokeds_difference_and_amplitudes(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_if_difference_evokeds_are_reloaded(*args, **kwargs):
        raise AssertionError("difference evokeds should stay in memory")

    monkeypatch.setattr(mne, "read_evokeds", fail_if_difference_evokeds_are_reloaded)

    result = generate_erp_outputs(
        _epochs(),
        tmp_path,
        conditions=["standard", "target"],
        difference_waves=[
            {
                "name": "target_minus_standard",
                "positive": "target",
                "negative": "standard",
            }
        ],
        analysis_window=[-0.1, 0.5],
    )

    counts_file = tmp_path / "erp_condition_counts.csv"
    with counts_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"condition": "standard", "event_code": "1", "n_epochs": "2"},
        {"condition": "target", "event_code": "2", "n_epochs": "2"},
    ]
    assert set(result["evoked_files"]) == {"standard", "target"}
    assert "target_minus_standard" in result["difference_files"]
    assert (tmp_path / "erp_amplitude_summary.csv").exists()
