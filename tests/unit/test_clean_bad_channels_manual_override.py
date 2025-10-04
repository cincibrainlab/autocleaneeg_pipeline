"""Tests for manual override of bad channel cleaning."""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from autoclean.mixins.base import BaseMixin
from autoclean.mixins.signal_processing.channels import ChannelsMixin


class DummyManualOverrideTask(ChannelsMixin, BaseMixin):
    """Minimal task harness to exercise bad channel cleaning."""

    def __init__(self, raw: mne.io.Raw):
        self.raw = raw
        self.config: Dict[str, Any] = {
            "run_id": "unit-test-run",
            "task": "dummy",
            "tasks": {"dummy": {"settings": {}}},
        }
        self.flagged = False
        self.flagged_reasons: List[str] = []
        self.saved_results: List[Tuple[str, mne.io.Raw]] = []
        self.metadata_log: List[Tuple[str, Dict[str, Any]]] = []
        self.tracked_removals: List[Dict[str, Any]] = []

    def _save_raw_result(self, result_data: mne.io.Raw, stage_name: str) -> None:
        # Avoid filesystem writes during tests; just capture call arguments.
        self.saved_results.append((stage_name, result_data))

    def _update_metadata(self, operation: str, metadata_dict: Dict[str, Any]) -> None:
        # Store metadata updates locally for assertions.
        self.metadata_log.append((operation, metadata_dict))

    def _track_channel_removal(
        self,
        channels: Any,
        reason: str,
        source_step: str,
    ) -> None:
        if isinstance(channels, str):
            channels = [channels]
        for channel in channels:
            self.tracked_removals.append(
                {"channel": channel, "reason": reason, "source_step": source_step}
            )


@pytest.fixture
def dummy_raw() -> mne.io.Raw:
    ch_names = [f"E{i}" for i in range(1, 17)]
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types="eeg")
    data = np.random.randn(len(ch_names), 512)
    raw = mne.io.RawArray(data, info)
    return raw


def test_manual_override_skips_detection(monkeypatch, dummy_raw):
    task = DummyManualOverrideTask(dummy_raw)

    detect_mock = MagicMock()
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.channels.detect_bad_channels",
        detect_mock,
    )

    result = task.clean_bad_channels(
        cleaning_method=None,
        manual_bad_channels=["E1", "E5"],
    )

    detect_mock.assert_not_called()
    assert set(result.info["bads"]) == {"E1", "E5"}
    assert set(task.raw.bad_channels) == {"E1", "E5"}

    assert task.tracked_removals == [
        {"channel": "E1", "reason": "MANUAL_OVERRIDE", "source_step": "clean_bad_channels"},
        {"channel": "E5", "reason": "MANUAL_OVERRIDE", "source_step": "clean_bad_channels"},
    ]

    assert task.metadata_log
    operation, metadata = task.metadata_log[-1]
    assert operation == "step_clean_bad_channels"
    assert metadata["method"] == "ManualOverride"
    assert metadata["options"] == {"manual_bad_channels": ["E1", "E5"]}
