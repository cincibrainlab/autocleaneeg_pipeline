"""Unit tests for GFPCleanEpochsMixin."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


class _GFPTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)

    def run(self):
        pass


@pytest.fixture
def task_with_epochs(tmp_path):
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "test",
    }
    t = _GFPTask(config)
    raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=60.0, sfreq=250.0
    )
    # Create epochs manually so we don't need the full epoching mixin
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(
        raw, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False
    )
    t.epochs = epochs
    return t


class TestGFPCleanEpochs:
    def test_returns_mne_epochs_object(self, task_with_epochs):
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task_with_epochs, "_save_epochs_result"),
            patch.object(task_with_epochs, "_update_metadata"),
            patch.object(task_with_epochs, "_auto_export_if_enabled"),
        ):
            result = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs
            )
        assert isinstance(result, mne.Epochs)

    def test_result_has_fewer_or_equal_epochs_than_input(self, task_with_epochs):
        """GFP cleaning can only drop epochs, never add them."""
        original_count = len(task_with_epochs.epochs)
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task_with_epochs, "_save_epochs_result"),
            patch.object(task_with_epochs, "_update_metadata"),
            patch.object(task_with_epochs, "_auto_export_if_enabled"),
        ):
            result = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs, gfp_threshold=3.0
            )
        assert len(result) <= original_count

    def test_strict_threshold_drops_more_epochs_than_lenient(self, task_with_epochs):
        """A lower z-score threshold should reject more epochs."""
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task_with_epochs, "_save_epochs_result"),
            patch.object(task_with_epochs, "_update_metadata"),
            patch.object(task_with_epochs, "_auto_export_if_enabled"),
        ):
            result_strict = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs.copy(), gfp_threshold=1.0
            )
            result_lenient = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs.copy(), gfp_threshold=5.0
            )
        assert len(result_strict) <= len(result_lenient)

    def test_injected_noisy_epoch_is_removed(self, task_with_epochs):
        """An epoch with extreme amplitude should be dropped."""
        # Inflate one epoch to extreme values
        task_with_epochs.epochs.load_data()
        task_with_epochs.epochs._data[0, :, :] = 5000e-6  # 5 mV — huge outlier

        original_count = len(task_with_epochs.epochs)
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task_with_epochs, "_save_epochs_result"),
            patch.object(task_with_epochs, "_update_metadata"),
            patch.object(task_with_epochs, "_auto_export_if_enabled"),
        ):
            result = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs, gfp_threshold=2.5
            )
        assert len(result) < original_count

    def test_number_of_epochs_parameter_limits_output(self, task_with_epochs):
        """If number_of_epochs is set, exactly that many are returned (if enough exist)."""
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task_with_epochs, "_save_epochs_result"),
            patch.object(task_with_epochs, "_update_metadata"),
            patch.object(task_with_epochs, "_auto_export_if_enabled"),
        ):
            result = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs,
                gfp_threshold=10.0,  # Keep almost everything
                number_of_epochs=5,
                random_seed=42,
            )
        assert len(result) == 5

    def test_runs_regardless_of_settings(self, task_with_epochs):
        """gfp_clean_epochs has no _check_step_enabled call — always runs."""
        task_with_epochs.settings = {"gfp_clean_epochs": {"enabled": False}}
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task_with_epochs, "_save_epochs_result"),
            patch.object(task_with_epochs, "_update_metadata"),
            patch.object(task_with_epochs, "_auto_export_if_enabled"),
        ):
            result = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs
            )
        assert isinstance(result, mne.Epochs)

    def test_strict_threshold_rejects_many_epochs(self, task_with_epochs):
        """Very strict GFP threshold causes large fraction of epochs to be dropped."""
        original_count = len(task_with_epochs.epochs)
        # Inject noisy epochs to ensure many are above threshold
        task_with_epochs.epochs.load_data()
        # Make roughly half the epochs very noisy
        half = original_count // 2
        task_with_epochs.epochs._data[:half, :, :] = 5000e-6  # 5 mV — huge outlier
        with (
            patch("autoclean.mixins.base.manage_database_conditionally"),
            patch("autoclean.mixins.base.save_raw_to_set"),
            patch("autoclean.mixins.base.save_epochs_to_set"),
            patch.object(task_with_epochs, "_save_epochs_result"),
            patch.object(task_with_epochs, "_update_metadata"),
            patch.object(task_with_epochs, "_auto_export_if_enabled"),
        ):
            result = task_with_epochs.gfp_clean_epochs(
                epochs=task_with_epochs.epochs,
                gfp_threshold=0.5,  # Extremely strict
            )
        assert len(result) < original_count
