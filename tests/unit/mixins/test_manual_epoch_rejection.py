"""Unit tests for ManualEpochRejectionMixin."""

from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest

try:
    from autoclean.core.task import Task

    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not TASK_AVAILABLE, reason="Task module not available"
)


class _EpochTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)

    def run(self):
        pass


@pytest.fixture
def task(tmp_path):
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.set",
        "task": "test",
    }
    task = _EpochTask(config)
    info = mne.create_info(["Fz", "Cz"], sfreq=100.0, ch_types=["eeg", "eeg"])
    data = np.zeros((4, 2, 20))
    events = np.array(
        [
            [0, 0, 1],
            [20, 0, 1],
            [40, 0, 1],
            [60, 0, 1],
        ]
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"stim": 1},
        tmin=0.0,
        verbose=False,
    )
    epochs.drop([1], reason="PREV", verbose=False)
    task.epochs = epochs
    return task


class TestManualEpochRejection:
    def test_maps_requested_epoch_numbers_through_selection(self, task):
        with patch.object(task, "_update_metadata") as mock_update, patch.object(
            task, "_auto_export_if_enabled"
        ):
            result = task.drop_manual_bad_epochs(
                manual_bad_epoch_indices=[2, 3, 99],
                manual_bad_epoch_times=["0.400", "0.600"],
                manual_bad_epoch_events=["1", "1"],
            )

        assert result.selection.tolist() == [0]
        assert task.epochs.selection.tolist() == [0]

        metadata = mock_update.call_args.args[1]
        assert metadata["requested_bad_epoch_indices"] == [2, 3, 99]
        assert metadata["applied_bad_epoch_indices"] == [2, 3]
        assert metadata["skipped_bad_epoch_indices"] == [99]

    def test_noop_when_no_manual_indices_provided(self, task):
        original = task.epochs
        with patch.object(task, "_update_metadata") as mock_update, patch.object(
            task, "_auto_export_if_enabled"
        ):
            result = task.drop_manual_bad_epochs(manual_bad_epoch_indices=[])

        assert result is original
        mock_update.assert_not_called()
