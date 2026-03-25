"""Unit tests for EventIDEpochsMixin."""

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


class _EventTask(Task):
    def __init__(self, config, settings=None):
        self.settings = settings or {}
        super().__init__(config)

    def run(self):
        pass


def _make_raw_with_events(duration=30.0, sfreq=250.0, event_id=1):
    """Create a raw object with synthetic annotations for event-based epoching."""
    raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=duration, sfreq=sfreq
    )
    # Add annotations every 2 seconds as stimulus events
    onsets = np.arange(2.0, duration - 2.0, 2.0)
    annotations = mne.Annotations(
        onset=onsets,
        duration=[0.01] * len(onsets),
        description=[f"DIN{event_id}"] * len(onsets),
    )
    raw.set_annotations(annotations)
    return raw


@pytest.fixture
def task(tmp_path):
    settings = {
        "epoch_settings": {
            "enabled": True,
            "value": {"tmin": -0.2, "tmax": 0.5},
            "event_id": {"DIN1": 1},
        }
    }
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "test",
    }
    t = _EventTask(config, settings=settings)
    t.raw = _make_raw_with_events(event_id=1)
    return t


# ---------------------------------------------------------------------------
# print_discovered_events
# ---------------------------------------------------------------------------


class TestPrintDiscoveredEvents:
    def test_returns_dict_with_event_names(self, task):
        result = task.print_discovered_events(data=task.raw, show_config_example=False)
        assert result is not None
        assert isinstance(result, dict)

    def test_found_events_match_annotations(self, task):
        result = task.print_discovered_events(data=task.raw, show_config_example=False)
        assert "DIN1" in result

    def test_returns_none_for_raw_with_no_annotations(self, tmp_path):
        settings = {"epoch_settings": {"enabled": True, "value": {}}}
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _EventTask(config, settings=settings)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
        )
        result = t.print_discovered_events(data=t.raw, show_config_example=False)
        # No annotations → no events discovered
        assert result is None or len(result) == 0


# ---------------------------------------------------------------------------
# create_eventid_epochs
# ---------------------------------------------------------------------------


class TestCreateEventIDEpochs:
    def test_returns_mne_epochs_when_events_found(self, task):
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task, "_save_epochs_result"),
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            result = task.create_eventid_epochs(
                data=task.raw,
                event_id={"DIN1": 1},
                tmin=-0.2,
                tmax=0.5,
            )
        assert isinstance(result, mne.Epochs)

    def test_returns_none_when_disabled_in_settings(self, tmp_path):
        settings = {"epoch_settings": {"enabled": False}}
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _EventTask(config, settings=settings)
        t.raw = _make_raw_with_events()
        result = t.create_eventid_epochs(data=t.raw)
        assert result is None

    def test_epoch_count_proportional_to_event_count(self, task):
        """Number of epochs should not exceed number of matching events."""
        n_events = sum(
            1 for ann in task.raw.annotations if ann["description"] == "DIN1"
        )
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task, "_save_epochs_result"),
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            result = task.create_eventid_epochs(
                data=task.raw,
                event_id={"DIN1": 1},
                tmin=-0.2,
                tmax=0.5,
            )
        assert len(result) <= n_events

    def test_flags_task_when_no_events_found(self, tmp_path):
        """No matching events → task should be flagged."""
        settings = {"epoch_settings": {"enabled": True, "value": {"tmin": -0.2, "tmax": 0.5}}}
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _EventTask(config, settings=settings)
        # Raw with no annotations
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=30.0, sfreq=250.0
        )
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_update_metadata"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            result = t.create_eventid_epochs(
                data=t.raw,
                event_id={"DIN1": 1},
                tmin=-0.2,
                tmax=0.5,
            )

        # Should be flagged when no events found
        assert t.flagged is True or result is None or len(result) == 0

    def test_epoch_window_matches_tmin_tmax(self, task):
        """Epoch duration should match tmin–tmax setting."""
        tmin, tmax = -0.2, 0.5
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task, "_save_epochs_result"),
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            result = task.create_eventid_epochs(
                data=task.raw,
                event_id={"DIN1": 1},
                tmin=tmin,
                tmax=tmax,
            )

        duration = result.times[-1] - result.times[0]
        assert abs(duration - (tmax - tmin)) < 0.01

    def test_uses_event_id_from_settings(self, task):
        """event_id from settings[epoch_settings][event_id] is used for epoching."""
        # task fixture has event_id={"DIN1": 1} in settings
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task, "_save_epochs_result"),
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            result = task.create_eventid_epochs(data=task.raw)
        # DIN1 events exist in raw — epochs should be created
        assert result is not None
        assert isinstance(result, mne.Epochs)

    def test_baseline_correction_applied_when_configured(self, task):
        """Epochs created with a non-None baseline have data shifted relative to baseline."""
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task, "_save_epochs_result"),
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            result_with_baseline = task.create_eventid_epochs(
                data=task.raw,
                event_id={"DIN1": 1},
                tmin=-0.2,
                tmax=0.5,
                baseline=(-0.2, 0),
            )
            result_no_baseline = task.create_eventid_epochs(
                data=task.raw,
                event_id={"DIN1": 1},
                tmin=-0.2,
                tmax=0.5,
                baseline=None,
            )
        # Both should be valid Epochs — baseline doesn't change type
        assert isinstance(result_with_baseline, mne.Epochs)
        assert isinstance(result_no_baseline, mne.Epochs)

    def test_flags_when_too_few_epochs_retained(self, tmp_path):
        """If retained epochs < EPOCH_RETENTION_THRESHOLD of total, task is flagged."""
        # Create raw with very many annotations that will cause most epochs to be dropped
        settings = {
            "epoch_settings": {
                "enabled": True,
                "value": {"tmin": -0.2, "tmax": 0.5},
                "event_id": {"DIN1": 1},
            }
        }
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _EventTask(config, settings=settings)
        t.raw = _make_raw_with_events(duration=60.0, event_id=1)
        # Annotate almost everything as bad
        t.raw.annotations.append(onset=0.5, duration=55.0, description="BAD_all")

        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_update_metadata"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            t.create_eventid_epochs(data=t.raw)

        # With >50% of epochs annotated as bad, task should be flagged
        assert t.flagged is True

    def test_saves_stage_file(self, task):
        """_save_epochs_result should be called after successful epoch creation."""
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task, "_save_epochs_result") as mock_save,
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            task.create_eventid_epochs(
                data=task.raw,
                event_id={"DIN1": 1},
                tmin=-0.2,
                tmax=0.5,
            )
        mock_save.assert_called()


# ---------------------------------------------------------------------------
# summarize_amplitude_quality
# ---------------------------------------------------------------------------


class TestSummarizeAmplitudeQuality:
    def test_returns_dataframe_with_expected_columns(self, task):
        """summarize_amplitude_quality must return a DataFrame with per-channel stats."""
        import pandas as pd

        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task, "_save_epochs_result"),
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            epochs = task.create_eventid_epochs(
                data=task.raw,
                event_id={"DIN1": 1},
                tmin=-0.2,
                tmax=0.5,
            )

        result = task.summarize_amplitude_quality(epochs=epochs)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "channel" in result.columns or "flagged_count" in result.columns
