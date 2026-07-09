from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest

from autoclean.configkit.schema import export_task_schema_layout
from autoclean.core.task import Task


class _ConditionwiseTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)

    def run(self):
        pass


def _make_epochs(include_eog: bool = False) -> mne.EpochsArray:
    sfreq = 100.0
    ch_names = ["Fz", "Cz"] + (["EOG1"] if include_eog else [])
    ch_types = ["eeg", "eeg"] + (["eog"] if include_eog else [])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    data = np.zeros((8, len(ch_names), 50), dtype=float)
    rng = np.random.default_rng(42)
    data[:, :2, :] = rng.normal(0, 1e-6, size=(8, 2, 50))
    data[1, :2, 10:20] = 75e-6
    if include_eog:
        data[:, 2, :] = 200e-6
    events = np.column_stack(
        [np.arange(8), np.zeros(8, dtype=int), [1, 1, 1, 1, 2, 2, 2, 2]]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"standard": 1, "deviant": 2},
        tmin=-0.1,
        baseline=None,
        verbose=False,
    )


@pytest.fixture
def task(tmp_path: Path):
    config = {
        "run_id": "run",
        "unprocessed_file": tmp_path / "subject.set",
        "task": "ExampleTask",
        "reports_dir": tmp_path / "reports",
    }
    t = _ConditionwiseTask(config)
    t.epochs = _make_epochs()
    return t


def test_conditionwise_rejection_apply_mode_drops_indices_and_writes_audit(task):
    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {
                "absolute_amplitude_uv": 50,
                "minimum_metric_flags": 1,
                "max_reject_fraction": 0.5,
                "minimum_epochs": 1,
                "mode": "apply",
            },
        }
    }

    with patch.object(task, "_update_metadata") as update_metadata:
        clean_epochs, audit_df, summary_df = task.apply_conditionwise_epoch_rejection()

    assert len(clean_epochs) == 7
    assert task.epochs is clean_epochs
    rejected = audit_df[audit_df["rejected"]]
    assert rejected["epoch_index"].tolist() == [1]
    assert rejected.iloc[0]["condition"] == "standard"
    assert rejected.iloc[0]["event_code"] == 1
    assert summary_df.loc[summary_df["condition"] == "standard", "rejected_epochs"].item() == 1
    update_metadata.assert_called_once()
    reports_dir = task.config["reports_dir"] / "conditionwise_epoch_rejection"
    assert any(path.name.endswith("_audit.csv") for path in reports_dir.iterdir())
    assert any(path.name.endswith("_summary.csv") for path in reports_dir.iterdir())


def test_conditionwise_rejection_report_only_keeps_epochs(task):
    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {
                "absolute_amplitude_uv": 50,
                "minimum_metric_flags": 1,
                "max_reject_fraction": 0.5,
                "minimum_epochs": 1,
                "mode": "report_only",
            },
        }
    }

    clean_epochs, audit_df, summary_df = task.apply_conditionwise_epoch_rejection()

    assert len(clean_epochs) == 8
    assert audit_df["candidate_reject"].sum() == 1
    assert audit_df["rejected"].sum() == 0
    assert set(summary_df["mode"]) == {"report_only"}


def test_conditionwise_rejection_excludes_eog_channels(tmp_path):
    config = {
        "run_id": "run",
        "unprocessed_file": tmp_path / "subject.set",
        "task": "ExampleTask",
        "reports_dir": tmp_path / "reports",
    }
    task = _ConditionwiseTask(config)
    task.epochs = _make_epochs(include_eog=True)
    task.epochs._data[:, :2, :] = 1e-6
    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {
                "absolute_amplitude_uv": 50,
                "exclude_channel_types": ["eog"],
                "minimum_metric_flags": 1,
                "minimum_epochs": 1,
            },
        }
    }

    clean_epochs, audit_df, _ = task.apply_conditionwise_epoch_rejection()

    assert len(clean_epochs) == len(task.epochs)
    assert audit_df["candidate_reject"].sum() == 0


def test_conditionwise_rejection_caps_per_condition(task):
    task.epochs._data[:4, :2, :] = 80e-6
    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {
                "absolute_amplitude_uv": 50,
                "minimum_metric_flags": 1,
                "max_reject_fraction": 0.25,
                "minimum_epochs": 1,
                "mode": "apply",
            },
        }
    }

    clean_epochs, audit_df, summary_df = task.apply_conditionwise_epoch_rejection()

    standard = summary_df[summary_df["condition"] == "standard"].iloc[0]
    assert standard["candidate_rejected_epochs"] == 4
    assert standard["rejected_epochs"] == 1
    assert "candidate_fraction_exceeds_max_reject_fraction" in standard["warning"]
    assert audit_df[audit_df["condition"] == "standard"]["rejected"].sum() == 1
    assert len(clean_epochs) == 7




def test_conditionwise_rejection_robust_z_flags_zero_mad_outlier(tmp_path):
    config = {
        "run_id": "run",
        "unprocessed_file": tmp_path / "subject.set",
        "task": "ExampleTask",
        "reports_dir": tmp_path / "reports",
    }
    task = _ConditionwiseTask(config)
    task.epochs = _make_epochs()
    task.epochs._data[:] = 1e-6
    task.epochs._data[1, :2, :] = 50e-6
    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {
                "robust_z_threshold": 4.0,
                "minimum_metric_flags": 1,
                "absolute_amplitude_uv": None,
                "max_reject_fraction": 0.5,
                "minimum_epochs": 1,
                "mode": "apply",
            },
        }
    }

    clean_epochs, audit_df, _ = task.apply_conditionwise_epoch_rejection()

    assert audit_df.loc[audit_df["epoch_index"] == 1, "candidate_reject"].item()
    assert audit_df.loc[audit_df["epoch_index"] == 1, "rejected"].item()
    assert len(clean_epochs) == 7


def test_conditionwise_rejection_summary_includes_recording_total(task):
    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {
                "absolute_amplitude_uv": 50,
                "minimum_metric_flags": 1,
                "max_reject_fraction": 0.5,
                "minimum_epochs": 1,
                "mode": "apply",
            },
        }
    }

    _, _, summary_df = task.apply_conditionwise_epoch_rejection()

    total = summary_df[summary_df["condition"] == "__recording_total__"].iloc[0]
    assert total["event_codes"] == "all"
    assert total["original_epochs"] == 8
    assert total["rejected_epochs"] == 1


def test_conditionwise_rejection_rejects_unknown_grouping(task):
    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {"group_by": "metadata"},
        }
    }

    with pytest.raises(ValueError, match="group_by='event_id'"):
        task.apply_conditionwise_epoch_rejection()


def test_conditionwise_rejection_defaults_to_eeg_scoring(task):
    sfreq = 100.0
    info = mne.create_info(
        ch_names=["Fz", "Cz", "STI 014"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg", "stim"],
    )
    data = np.zeros((4, 3, 50), dtype=float)
    data[:, :2, :] = 1e-6
    data[1, 2, :] = 1.0
    events = np.column_stack(
        [np.arange(4), np.zeros(4, dtype=int), [1, 1, 1, 1]]
    )
    task.epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"standard": 1},
        tmin=-0.1,
        baseline=None,
        verbose=False,
    )

    task.settings = {
        "conditionwise_epoch_rejection": {
            "enabled": True,
            "value": {
                "absolute_amplitude_uv": 50,
                "minimum_metric_flags": 1,
                "mode": "report_only",
            },
        }
    }

    with patch.object(task, "_update_metadata") as update_metadata:
        _, audit_df, _ = task.apply_conditionwise_epoch_rejection()

    assert task.conditionwise_epoch_rejection_audit is audit_df
    assert audit_df["candidate_reject"].sum() == 0
    metadata = update_metadata.call_args.args[1]
    assert metadata["scored_channels"] == ["Fz", "Cz"]


def test_conditionwise_epoch_rejection_is_exported_in_schema_layout():
    layout = export_task_schema_layout()

    assert "conditionwise_epoch_rejection" in layout["tasks"]
