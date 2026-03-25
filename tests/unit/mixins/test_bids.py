"""Unit tests for BIDSMixin."""

from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

try:
    from autoclean.core.task import Task

    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not TASK_AVAILABLE, reason="Task module not available"
)


class _BIDSTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)

    def run(self):
        pass


@pytest.fixture
def task(tmp_path):
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "test",
    }
    t = _BIDSTask(config)
    t.raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
    )
    return t


@pytest.fixture
def task_with_epochs(tmp_path):
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "test",
    }
    t = _BIDSTask(config)
    raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=30.0, sfreq=250.0
    )
    events = mne.make_fixed_length_events(raw, duration=2.0)
    t.epochs = mne.Epochs(
        raw, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False
    )
    return t


# ---------------------------------------------------------------------------
# create_mock_raw_from_epochs
# ---------------------------------------------------------------------------


def _create_mock_raw(task_with_epochs):
    """Helper: call create_mock_raw_from_epochs, patching filename validation."""
    epochs = task_with_epochs.epochs
    # MNE validates that filenames exist; patch the setter to avoid filesystem checks
    with patch("mne.io.base.BaseRaw.filenames", new_callable=lambda: property(
        fget=lambda self: getattr(self, "_filenames", []),
        fset=lambda self, v: setattr(self, "_filenames", v),
    )):
        return task_with_epochs.create_mock_raw_from_epochs(epochs)


# ---------------------------------------------------------------------------
# create_bids_path
# ---------------------------------------------------------------------------


class TestCreateBIDSPath:
    """Tests for the create_bids_path mixin method."""

    def _make_task_with_bids_config(self, tmp_path):
        """Return a task with bids_dir set in config."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
            "bids_dir": bids_dir,
            "eeg_system": "auto",
        }
        t = _BIDSTask(config)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
        )
        return t

    def _make_mock_bids_path(self, tmp_path):
        """Return a minimal mock BIDSPath-like object."""
        from unittest.mock import MagicMock
        bids_path = MagicMock()
        bids_path.basename = "sub-test_task-test_eeg.fif"
        bids_path.subject = "test"
        bids_path.task = "test"
        bids_path.run = None
        bids_path.session = None
        bids_path.datatype = "eeg"
        bids_path.suffix = "eeg"
        bids_path.extension = ".fif"
        bids_path.root = str(tmp_path / "bids")
        derivatives_dir = tmp_path / "bids" / "derivatives" / "sub-test"
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        return bids_path, derivatives_dir

    def test_updates_config_with_bids_path(self, tmp_path):
        """After create_bids_path, config['bids_path'] is set."""
        t = self._make_task_with_bids_config(tmp_path)
        mock_bids_path, mock_derivatives = self._make_mock_bids_path(tmp_path)

        with (
            patch("autoclean.mixins.utils.bids.step_convert_to_bids",
                  return_value=(mock_bids_path, mock_derivatives)),
            patch.object(t, "_update_metadata"),
        ):
            t.create_bids_path()

        assert t.config.get("bids_path") is mock_bids_path

    def test_updates_config_with_derivatives_dir(self, tmp_path):
        """After create_bids_path, config['derivatives_dir'] is set."""
        t = self._make_task_with_bids_config(tmp_path)
        mock_bids_path, mock_derivatives = self._make_mock_bids_path(tmp_path)

        with (
            patch("autoclean.mixins.utils.bids.step_convert_to_bids",
                  return_value=(mock_bids_path, mock_derivatives)),
            patch.object(t, "_update_metadata"),
        ):
            t.create_bids_path()

        assert t.config.get("derivatives_dir") == mock_derivatives

    def test_creates_bids_dir_before_conversion(self, tmp_path):
        """bids_dir is passed to step_convert_to_bids as output_dir."""
        bids_dir = tmp_path / "bids_output"
        bids_dir.mkdir()
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
            "bids_dir": bids_dir,
            "eeg_system": "auto",
        }
        t = _BIDSTask(config)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
        )
        mock_bids_path, mock_derivatives = self._make_mock_bids_path(tmp_path)

        with (
            patch("autoclean.mixins.utils.bids.step_convert_to_bids",
                  return_value=(mock_bids_path, mock_derivatives)) as mock_convert,
            patch.object(t, "_update_metadata"),
        ):
            t.create_bids_path()

        call_kwargs = mock_convert.call_args[1]
        assert str(bids_dir) in call_kwargs.get("output_dir", "")

    def test_raises_when_step_convert_fails(self, tmp_path):
        """If step_convert_to_bids raises, create_bids_path propagates the exception."""
        t = self._make_task_with_bids_config(tmp_path)

        with (
            patch("autoclean.mixins.utils.bids.step_convert_to_bids",
                  side_effect=ValueError("BIDS conversion failed")),
            patch.object(t, "_update_metadata"),
        ):
            with pytest.raises(ValueError, match="BIDS conversion failed"):
                t.create_bids_path()

    def test_uses_task_name_as_mne_task(self, tmp_path):
        """Task name is passed as the 'task' field in BIDS conversion call."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "MyRestingTask",
            "bids_dir": bids_dir,
            "eeg_system": "auto",
        }
        t = _BIDSTask(config)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
        )
        mock_bids_path, mock_derivatives = self._make_mock_bids_path(tmp_path)

        with (
            patch("autoclean.mixins.utils.bids.step_convert_to_bids",
                  return_value=(mock_bids_path, mock_derivatives)) as mock_convert,
            patch.object(t, "_update_metadata"),
        ):
            t.create_bids_path()

        call_kwargs = mock_convert.call_args[1]
        # Task label should be lowercase (Python task path)
        assert call_kwargs.get("task") == "myrestingtask"


class TestCreateMockRawFromEpochs:
    def test_returns_mne_raw_object(self, task_with_epochs):
        result = _create_mock_raw(task_with_epochs)
        assert isinstance(result, mne.io.BaseRaw)

    def test_channel_count_preserved(self, task_with_epochs):
        original_n_channels = len(task_with_epochs.epochs.ch_names)
        result = _create_mock_raw(task_with_epochs)
        assert len(result.ch_names) == original_n_channels

    def test_sampling_rate_preserved(self, task_with_epochs):
        original_sfreq = task_with_epochs.epochs.info["sfreq"]
        result = _create_mock_raw(task_with_epochs)
        assert result.info["sfreq"] == original_sfreq

    def test_total_samples_is_n_epochs_times_epoch_length(self, task_with_epochs):
        epochs = task_with_epochs.epochs
        n_epochs = len(epochs)
        n_times_per_epoch = epochs.get_data().shape[2]

        result = _create_mock_raw(task_with_epochs)
        expected_samples = n_epochs * n_times_per_epoch

        assert result.get_data().shape[1] == expected_samples

    def test_channel_names_match_source_epochs(self, task_with_epochs):
        epochs = task_with_epochs.epochs
        result = _create_mock_raw(task_with_epochs)
        assert result.ch_names == epochs.ch_names

    def test_data_is_continuous_concatenation_of_epochs(self, task_with_epochs):
        """Data in mock raw should be a transposed reshape of epoch data."""
        epochs = task_with_epochs.epochs
        epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = epoch_data.shape
        expected_data = epoch_data.transpose(1, 0, 2).reshape(n_channels, -1)

        result = _create_mock_raw(task_with_epochs)
        np.testing.assert_array_almost_equal(
            result.get_data(), expected_data
        )
