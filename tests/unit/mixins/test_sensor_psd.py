"""Unit tests for SensorPSDMixin (mixins/analysis/sensor_psd.py)."""

from unittest.mock import patch

import mne
import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

try:
    from autoclean.core.task import Task

    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available")


class _PSDTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)

    def run(self):
        pass


def _make_epochs(tmp_path, n_channels=8, duration=10.0, sfreq=250.0):
    """Create short synthetic epochs for PSD computation."""
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
    t = _PSDTask(config)
    # Explicitly enable the step so _check_step_enabled returns True
    t.settings = {"apply_sensor_psd": {"enabled": True, "value": {}}}
    t.epochs = _make_epochs(tmp_path)
    return t


# ---------------------------------------------------------------------------
# apply_sensor_psd
# ---------------------------------------------------------------------------


class TestApplySensorPsd:
    def test_returns_psd_and_band_dataframes(self, task, tmp_path):
        """apply_sensor_psd must return a (psd_df, band_df, artifact_paths) tuple."""
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_sensor_psd_tables", return_value={}),
        ):
            result = task.apply_sensor_psd(epochs=task.epochs)

        assert result is not None
        psd_df, band_df, _ = result
        assert psd_df is not None
        assert band_df is not None

    def test_psd_dataframe_has_channel_and_frequency_columns(self, task, tmp_path):
        """PSD DataFrame must include 'channel' and 'frequency' columns."""
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_sensor_psd_tables", return_value={}),
        ):
            psd_df, _, _ = task.apply_sensor_psd(epochs=task.epochs)

        assert "channel" in psd_df.columns
        assert "frequency" in psd_df.columns

    def test_band_dataframe_has_band_column(self, task):
        """Band-power DataFrame must include a 'band' column."""
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_sensor_psd_tables", return_value={}),
        ):
            _, band_df, _ = task.apply_sensor_psd(epochs=task.epochs)

        assert "band" in band_df.columns

    def test_skips_when_step_disabled(self, tmp_path):
        """apply_sensor_psd returns (None, None, {}) when disabled in settings."""
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }

        class _DisabledTask(_PSDTask):
            def __init__(self, c):
                self.settings = {"apply_sensor_psd": {"enabled": False, "value": {}}}
                super(_PSDTask, self).__init__(c)

            def run(self):
                pass

        task = _DisabledTask(config)
        result = task.apply_sensor_psd()
        assert result == (None, None, {})

    def test_raises_value_error_without_epochs(self, tmp_path):
        """apply_sensor_psd raises ValueError when no epochs are available."""
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        task = _PSDTask(config)
        # Enable step but provide no epochs
        task.settings = {"apply_sensor_psd": {"enabled": True, "value": {}}}
        # task.epochs is None by default

        with pytest.raises(ValueError, match="No epochs available"):
            task.apply_sensor_psd()

    def test_custom_bands_and_time_windows_are_reflected_in_outputs(self, task):
        """Custom bands/windows should be accepted and written into PSD tables."""
        with (
            patch.object(task, "_update_metadata") as update_metadata,
            patch.object(task, "_save_sensor_psd_tables", return_value={}),
        ):
            psd_df, band_df, _ = task.apply_sensor_psd(
                data=task.epochs,
                fmin=1,
                fmax=20,
                freq_bands={"theta": [4, 8], "skip_me": None},
                time_windows={"early": [0, 1.0]},
            )

        assert set(psd_df["time_window"]) == {"early"}
        assert set(band_df["band"]) == {"theta"}
        metadata = update_metadata.call_args.args[1]
        assert metadata["time_windows"] == {"early": [0.0, 1.0]}
        assert metadata["freq_bands"]["skip_me"] is None

    def test_freq_bands_none_skips_band_summary(self, task):
        """freq_bands=None should skip band-power rows instead of using defaults."""
        with (
            patch.object(task, "_update_metadata") as update_metadata,
            patch.object(task, "_save_sensor_psd_tables", return_value={}),
        ):
            _, band_df, _ = task.apply_sensor_psd(
                data=task.epochs,
                fmin=1,
                fmax=20,
                freq_bands=None,
            )

        assert band_df.empty
        metadata = update_metadata.call_args.args[1]
        assert metadata["freq_bands"] == {}

    def test_custom_band_outside_psd_range_raises(self, task):
        """Bands outside the computed PSD range fail with an actionable error."""
        with pytest.raises(ValueError, match="outside PSD range"):
            task.apply_sensor_psd(
                data=task.epochs,
                fmin=1,
                fmax=20,
                freq_bands={"gamma": [30, 45]},
            )

    def test_raises_type_error_for_non_epochs_input(self, task):
        """apply_sensor_psd raises TypeError when passed a non-Epochs object."""
        with pytest.raises(TypeError):
            task.apply_sensor_psd(epochs="not_epochs")
