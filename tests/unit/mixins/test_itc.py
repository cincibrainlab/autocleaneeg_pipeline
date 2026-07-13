"""Unit tests for InterTrialCoherenceMixin (mixins/analysis/inter_trial_coherence.py)."""

from unittest.mock import MagicMock, patch

import mne
import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

try:
    from autoclean.core.task import Task

    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available")


class _ITCTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)

    def run(self):
        pass


def _make_epochs(sfreq=250.0, duration=20.0, n_channels=8):
    raw = create_synthetic_raw(
        montage="standard_1020", n_channels=n_channels, duration=duration, sfreq=sfreq
    )
    events = mne.make_fixed_length_events(raw, duration=2.0)
    return mne.Epochs(
        raw, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False
    )


@pytest.fixture
def task(tmp_path):
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "test",
    }
    t = _ITCTask(config)
    t.epochs = _make_epochs()
    return t


# ---------------------------------------------------------------------------
# compute_itc_analysis
# ---------------------------------------------------------------------------


class TestComputeItcAnalysis:
    def test_skips_when_step_disabled(self, tmp_path):
        """compute_itc_analysis returns (None, None, None) when disabled in settings."""
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }

        class _DisabledTask(_ITCTask):
            def __init__(self, c):
                self.settings = {"itc_analysis": {"enabled": False, "value": {}}}
                super(_ITCTask, self).__init__(c)

            def run(self):
                pass

        task = _DisabledTask(config)
        result = task.compute_itc_analysis()
        assert result == (None, None, None)

    def test_raises_value_error_when_no_epochs(self, tmp_path):
        """compute_itc_analysis raises ValueError when self.epochs is None and no arg passed."""
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        task = _ITCTask(config)
        # itc_analysis step must be enabled (default settings = {})
        with patch.object(task, "_check_step_enabled", return_value=(True, {})):
            with pytest.raises(ValueError, match="No epochs available"):
                task.compute_itc_analysis()

    def test_raises_type_error_for_non_epochs_input(self, task):
        """Passing a non-Epochs object raises TypeError."""
        with patch.object(task, "_check_step_enabled", return_value=(True, {})):
            with pytest.raises(
                TypeError, match="epochs must be an MNE BaseEpochs object"
            ):
                task.compute_itc_analysis(epochs="not_epochs")

    def test_returns_power_and_itc_from_mocked_function(self, task):
        """compute_itc_analysis returns the (power, itc, band_results) tuple."""
        mock_power = MagicMock()
        mock_itc = MagicMock()

        with (
            patch.object(task, "_check_step_enabled", return_value=(True, {})),
            patch(
                "autoclean.mixins.analysis.inter_trial_coherence.compute_statistical_learning_itc",
                return_value=(mock_power, mock_itc),
            ),
            patch(
                "autoclean.mixins.analysis.inter_trial_coherence.analyze_itc_bands",
                return_value={"delta": 0.5},
            ),
            patch.object(task, "_save_itc_results"),
            patch.object(task, "_update_itc_metadata"),
        ):
            power, itc, band_results = task.compute_itc_analysis(epochs=task.epochs)

        assert power is mock_power
        assert itc is mock_itc
        assert isinstance(band_results, dict)

    def test_skips_when_no_epochs_and_self_epochs_none(self, tmp_path):
        """When self.epochs is not set and no epochs argument, raises ValueError."""
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        task = _ITCTask(config)
        # Don't set task.epochs — leave it as None

        with patch.object(task, "_check_step_enabled", return_value=(True, {})):
            with pytest.raises(ValueError):
                task.compute_itc_analysis(epochs=None)
