from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from autoclean.mixins.base import BaseMixin
from autoclean.mixins.signal_processing.channels import ChannelsMixin


class DummyBadChannelLogTask(ChannelsMixin, BaseMixin):
    def __init__(
        self,
        raw: mne.io.Raw,
        log_settings: dict | None,
        input_file: Path,
        *,
        reference_step: dict | None = None,
    ):
        self.raw = raw
        self.config: Dict[str, Any] = {
            "run_id": "unit-test-run",
            "task": "dummy",
            "unprocessed_file": input_file,
            "tasks": {"dummy": {"settings": {}}},
        }
        if log_settings is not None:
            self.config["tasks"]["dummy"]["settings"]["bad_channel_log"] = log_settings
        if reference_step is not None:
            self.config["tasks"]["dummy"]["settings"]["reference_step"] = reference_step
        self.flagged = False
        self.flagged_reasons: List[str] = []
        self.saved_results: List[Tuple[str, mne.io.Raw]] = []
        self.metadata_log: List[Tuple[str, Dict[str, Any]]] = []
        self.tracked_removals: List[Dict[str, Any]] = []

    def _save_raw_result(self, result_data: mne.io.Raw, stage_name: str) -> None:
        self.saved_results.append((stage_name, result_data))

    def _update_metadata(self, operation: str, metadata_dict: Dict[str, Any]) -> None:
        self.metadata_log.append((operation, metadata_dict))

    def _track_channel_removal(
        self,
        channels: Any,
        reason: str,
        source_step: str,
        action: str = "dropped",
    ) -> None:
        if isinstance(channels, str):
            channels = [channels]
        for channel in channels:
            self.tracked_removals.append(
                {
                    "channel": channel,
                    "reason": reason,
                    "source_step": source_step,
                    "action": action,
                }
            )


@pytest.fixture
def dummy_raw() -> mne.io.Raw:
    ch_names = [f"E{i}" for i in range(1, 9)]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    data = np.random.randn(len(ch_names), 512)
    return mne.io.RawArray(data, info)


def _empty_detect(**kwargs):
    return {
        "correlation": [],
        "deviation": [],
        "ransac": [],
        "combined": [],
    }


def _bad_channel_log_removals(task: DummyBadChannelLogTask) -> List[Dict[str, Any]]:
    return [
        entry for entry in task.tracked_removals if entry["reason"] == "BAD_CHANNEL_LOG"
    ]


def _log_settings(path: Path, *, strict: bool = False) -> dict:
    return {
        "enabled": True,
        "value": {
            "path": str(path),
            "file_column": "file",
            "channels_column": "bad_channels",
            "action": "mark",
            "strict": strict,
        },
    }


def test_bad_channel_log_skips_reference_channel_in_tracking_and_bads(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    """Regression for #295: log + reference overlap must not touch or track ref channel."""
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E2;E3\n", encoding="utf-8")
    task = DummyBadChannelLogTask(
        dummy_raw,
        _log_settings(table),
        input_file,
        reference_step={"enabled": True, "value": ["E2"]},
    )
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        _empty_detect,
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert "E2" not in result.info["bads"]
    assert set(result.info["bads"]) == {"E3"}
    log_removals = _bad_channel_log_removals(task)
    assert {entry["channel"] for entry in log_removals} == {"E3"}
    assert all(entry["action"] == "marked" for entry in log_removals)


def test_bad_channel_log_still_tracks_non_reference_channels(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    """Guard: non-reference bad_channel_log channels keep existing tracking behavior."""
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E1;E3\n", encoding="utf-8")
    task = DummyBadChannelLogTask(
        dummy_raw,
        _log_settings(table),
        input_file,
        reference_step={"enabled": True, "value": ["E2"]},
    )
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        _empty_detect,
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert set(result.info["bads"]) == {"E1", "E3"}
    log_removals = _bad_channel_log_removals(task)
    assert {entry["channel"] for entry in log_removals} == {"E1", "E3"}
    assert all(entry["action"] == "marked" for entry in log_removals)


def _removals_by_reason(
    task: DummyBadChannelLogTask, reason: str
) -> List[Dict[str, Any]]:
    return [entry for entry in task.tracked_removals if entry["reason"] == reason]


@pytest.mark.parametrize(
    ("detect_key", "reason"),
    [
        ("correlation", "UNCORRELATED"),
        ("deviation", "DEVIATION"),
        ("ransac", "RANSAC"),
    ],
)
def test_detector_overlap_skips_reference_channel_in_tracking_and_bads(
    monkeypatch,
    tmp_path: Path,
    dummy_raw: mne.io.Raw,
    detect_key: str,
    reason: str,
) -> None:
    input_file = tmp_path / "subject01.set"
    task = DummyBadChannelLogTask(
        dummy_raw,
        None,
        input_file,
        reference_step={"enabled": True, "value": ["E2"]},
    )

    def fake_detect(**kwargs):
        detected = {
            "correlation": [],
            "deviation": [],
            "ransac": [],
            "combined": ["E2"],
        }
        detected[detect_key] = ["E2"]
        return detected

    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        fake_detect,
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert "E2" not in result.info["bads"]
    assert _removals_by_reason(task, reason) == []


def test_reference_step_string_channel_excludes_overlap(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E2;E3\n", encoding="utf-8")
    task = DummyBadChannelLogTask(
        dummy_raw,
        _log_settings(table),
        input_file,
        reference_step={"enabled": True, "value": "E2"},
    )
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        _empty_detect,
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert set(result.info["bads"]) == {"E3"}
    assert {entry["channel"] for entry in _bad_channel_log_removals(task)} == {"E3"}


def test_average_reference_does_not_exclude_log_channels(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E1\n", encoding="utf-8")
    task = DummyBadChannelLogTask(
        dummy_raw,
        _log_settings(table),
        input_file,
        reference_step={"enabled": True, "value": "average"},
    )
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        _empty_detect,
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert result.info["bads"] == ["E1"]


def test_pre_existing_bad_channel_preserved_when_not_reference(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    dummy_raw.info["bads"] = ["E8"]
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E3\n", encoding="utf-8")
    task = DummyBadChannelLogTask(
        dummy_raw,
        _log_settings(table),
        input_file,
        reference_step={"enabled": True, "value": ["E2"]},
    )
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        _empty_detect,
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert set(result.info["bads"]) == {"E8", "E3"}


def test_reference_only_bad_log_does_not_flag_recording(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E2\n", encoding="utf-8")
    task = DummyBadChannelLogTask(
        dummy_raw,
        _log_settings(table),
        input_file,
        reference_step={"enabled": True, "value": ["E2"]},
    )
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        _empty_detect,
    )

    task.clean_bad_channels(cleaning_method=None)

    assert task.flagged is False


def test_bad_channel_log_marks_channels_before_auto_detection(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E1;E3\n", encoding="utf-8")
    task = DummyBadChannelLogTask(dummy_raw, _log_settings(table), input_file)

    def fake_detect(data, **kwargs):
        assert set(data.info["bads"]) == {"E1", "E3"}
        return {
            "correlation": ["E5"],
            "deviation": [],
            "ransac": [],
            "combined": ["E5"],
        }

    detect_mock = MagicMock(side_effect=fake_detect)
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        detect_mock,
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert detect_mock.called
    assert set(result.info["bads"]) == {"E1", "E3", "E5"}
    assert {entry["channel"] for entry in task.tracked_removals} >= {"E1", "E3"}
    assert any(
        entry["reason"] == "BAD_CHANNEL_LOG"
        and entry["source_step"] == "bad_channel_log"
        for entry in task.tracked_removals
    )
    bad_log_meta = [
        item for item in task.metadata_log if item[0] == "step_bad_channel_log"
    ]
    assert bad_log_meta[-1][1]["applied_channels"] == ["E1", "E3"]


def test_bad_channel_log_disabled_does_not_apply_log(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E1\n", encoding="utf-8")
    settings = _log_settings(table)
    settings["enabled"] = False
    task = DummyBadChannelLogTask(dummy_raw, settings, input_file)

    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        lambda **kwargs: {
            "correlation": [],
            "deviation": [],
            "ransac": [],
            "combined": [],
        },
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert result.info["bads"] == []
    assert not any(item[0] == "step_bad_channel_log" for item in task.metadata_log)


def test_manual_bad_channels_override_skips_bad_channel_log(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E1\n", encoding="utf-8")
    task = DummyBadChannelLogTask(dummy_raw, _log_settings(table), input_file)
    detect_mock = MagicMock()
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        detect_mock,
    )

    result = task.clean_bad_channels(
        cleaning_method=None,
        manual_bad_channels=["E2"],
    )

    detect_mock.assert_not_called()
    assert result.info["bads"] == ["E2"]
    assert {entry["channel"] for entry in task.tracked_removals} == {"E2"}
    assert all(entry["reason"] != "BAD_CHANNEL_LOG" for entry in task.tracked_removals)
    assert not any(item[0] == "step_bad_channel_log" for item in task.metadata_log)


def test_bad_channel_log_warns_on_missing_channels_when_not_strict(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E2;Missing\n", encoding="utf-8")
    task = DummyBadChannelLogTask(dummy_raw, _log_settings(table), input_file)

    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        lambda **kwargs: {
            "correlation": [],
            "deviation": [],
            "ransac": [],
            "combined": [],
        },
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert result.info["bads"] == ["E2"]
    bad_log_meta = [
        item for item in task.metadata_log if item[0] == "step_bad_channel_log"
    ]
    assert bad_log_meta[-1][1]["missing_channels"] == ["Missing"]
    assert "not present" in bad_log_meta[-1][1]["warning"]


def test_bad_channel_log_raises_on_missing_match_when_strict(
    tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nother.set,E2\n", encoding="utf-8")
    task = DummyBadChannelLogTask(
        dummy_raw, _log_settings(table, strict=True), input_file
    )

    with pytest.raises(RuntimeError, match="No bad-channel log row matched"):
        task.clean_bad_channels(cleaning_method=None)


def test_bad_channel_log_rejects_unsupported_actions(
    tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\nsubject01.set,E2\n", encoding="utf-8")
    settings = _log_settings(table)
    settings["value"]["action"] = "drop"
    task = DummyBadChannelLogTask(dummy_raw, settings, input_file)

    with pytest.raises(RuntimeError, match="action='mark' only"):
        task.clean_bad_channels(cleaning_method=None)


def test_bad_channel_log_can_match_subject_and_session_fields(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "ambiguous.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text(
        "file,subject,session,bad_channels\n"
        "ambiguous.set,sub-01,ses-1,E1\n"
        "ambiguous.set,sub-02,ses-1,E4\n",
        encoding="utf-8",
    )
    settings = _log_settings(table)
    settings["value"].update(
        {
            "subject_column": "subject",
            "subject": "sub-02",
            "session_column": "session",
            "session": "ses-1",
        }
    )
    task = DummyBadChannelLogTask(dummy_raw, settings, input_file)

    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        lambda **kwargs: {
            "correlation": [],
            "deviation": [],
            "ransac": [],
            "combined": [],
        },
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert result.info["bads"] == ["E4"]
    bad_log_meta = [
        item for item in task.metadata_log if item[0] == "step_bad_channel_log"
    ]
    assert bad_log_meta[-1][1]["matched_by"] == "field"


def test_bad_channel_log_warns_when_configured_field_value_is_absent(
    monkeypatch, tmp_path: Path, dummy_raw: mne.io.Raw
) -> None:
    input_file = tmp_path / "subject01.set"
    table = tmp_path / "bad_channels.csv"
    table.write_text(
        "file,subject,session,bad_channels\nsubject01.set,sub-01,ses-1,E2\n",
        encoding="utf-8",
    )
    settings = _log_settings(table)
    settings["value"].update(
        {
            "subject_column": "subject",
            "session_column": "session",
        }
    )
    task = DummyBadChannelLogTask(dummy_raw, settings, input_file)
    messages = []

    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.message",
        lambda level, text: messages.append((level, text)),
    )
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        lambda **kwargs: {
            "correlation": [],
            "deviation": [],
            "ransac": [],
            "combined": [],
        },
    )

    result = task.clean_bad_channels(cleaning_method=None)

    assert result.info["bads"] == ["E2"]
    assert (
        "warning",
        "bad_channel_log subject_column configured but no subject value was found",
    ) in messages
    assert (
        "warning",
        "bad_channel_log session_column configured but no session value was found",
    ) in messages
