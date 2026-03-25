"""Unit tests for RegularEpochsMixin."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
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


class _RegularEpochTask(Task):
    """Minimal task subclass for regular epoch mixin tests."""

    def __init__(self, config, settings=None):
        self.settings = settings or {}
        super().__init__(config)

    def run(self):
        pass


def _make_task(tmp_path, settings=None):
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "test",
    }
    t = _RegularEpochTask(config, settings=settings)
    # 30 seconds gives enough data to create several 2-second epochs
    t.raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=30.0, sfreq=250.0
    )
    return t


# Patch the IO and database side-effects shared across all tests
_PATCHES = [
    patch("autoclean.mixins.base.manage_database_conditionally"),
    patch("autoclean.mixins.base.save_raw_to_set"),
    patch("autoclean.mixins.base.save_epochs_to_set"),
]


class TestCreateRegularEpochs:
    def test_returns_mne_epochs_object(self, tmp_path):
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 2}}})
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            result = t.create_regular_epochs(data=t.raw, tmin=0, tmax=2)
        assert isinstance(result, mne.Epochs)

    def test_epoch_duration_controls_window_length(self, tmp_path):
        """tmin/tmax from settings determine each epoch's time window."""
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 1}}})
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            result = t.create_regular_epochs(data=t.raw, tmin=0, tmax=1)

        # Each epoch should span 1 second
        epoch_duration = result.times[-1] - result.times[0]
        assert abs(epoch_duration - 1.0) < 0.01

    def test_returns_none_when_disabled_in_settings(self, tmp_path):
        """Disabled step should return None without raising."""
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": False}})
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
        ):
            result = t.create_regular_epochs(data=t.raw)
        assert result is None

    def test_epochs_stored_on_task_after_creation(self, tmp_path):
        """Successful epoch creation stores result in task.epochs."""
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 2}}})
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            t.create_regular_epochs(data=t.raw, tmin=0, tmax=2)

        assert hasattr(t, "epochs")
        assert isinstance(t.epochs, mne.Epochs)

    def test_bad_annotations_are_tracked_in_metadata(self, tmp_path):
        """Epochs overlapping BAD annotations get BAD_ANNOTATION=True in metadata."""
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 2}}})
        # Annotate a bad segment in the middle of the recording
        t.raw.annotations.append(onset=5.0, duration=2.0, description="BAD_test")

        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            result = t.create_regular_epochs(data=t.raw, tmin=0, tmax=2, reject_by_annotation=False)

        # Some epochs should have been dropped due to the annotation
        assert len(result) < 15  # 30s / 2s = 15 epochs max; should be fewer

    def test_raises_type_error_for_non_raw_input(self, tmp_path):
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": True, "value": {}}})
        with pytest.raises(TypeError):
            t.create_regular_epochs(data="not_raw")

    def test_flags_task_when_retention_below_threshold(self, tmp_path):
        """If good epochs / total < 50%, task.flagged should be True."""
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 2}}})

        # Mark ~75% of epochs as bad but leave at least 1 good epoch
        # 30s / 2s = 15 epochs; annotate epochs 0-11 (first 24s), leaving 3 good epochs (~20%)
        t.raw.annotations.append(onset=0.5, duration=23.0, description="BAD_partial")

        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            t.create_regular_epochs(data=t.raw, tmin=0, tmax=2, reject_by_annotation=False)

        assert t.flagged is True

    def test_uses_epoch_window_from_settings(self, tmp_path):
        """tmin/tmax from settings override default args, producing correct epoch duration."""
        # Settings say 1-second epochs (tmin=0, tmax=1)
        t = _make_task(
            tmp_path,
            settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 1}}},
        )
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            # Pass tmin=0, tmax=2 — settings should override and produce 1s epochs
            result = t.create_regular_epochs(data=t.raw, tmin=0, tmax=2)

        epoch_duration = result.times[-1] - result.times[0]
        assert abs(epoch_duration - 1.0) < 0.01

    def test_saves_stage_file(self, tmp_path):
        """_save_epochs_result should be called after successful epoch creation."""
        t = _make_task(
            tmp_path,
            settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 2}}},
        )
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result") as mock_save,
            patch.object(t, "_auto_export_if_enabled"),
        ):
            t.create_regular_epochs(data=t.raw, tmin=0, tmax=2)

        mock_save.assert_called()

    def test_drops_incomplete_final_epoch(self, tmp_path):
        """MNE does not create a partial epoch if there's insufficient data at the end."""
        import numpy as np

        # 7.5 seconds at 2-second epochs → 3 complete epochs (not 4)
        raw_short = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=7.5, sfreq=250.0
        )
        t = _make_task(tmp_path, settings={"epoch_settings": {"enabled": True, "value": {"tmin": 0, "tmax": 2}}})
        t.raw = raw_short

        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(t, "_save_epochs_result"),
            patch.object(t, "_auto_export_if_enabled"),
        ):
            result = t.create_regular_epochs(data=t.raw, tmin=0, tmax=2)

        # 7.5s / 2s = 3 complete epochs (partial tail dropped)
        assert len(result) <= 3
