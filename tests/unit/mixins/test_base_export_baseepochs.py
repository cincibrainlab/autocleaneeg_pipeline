"""Regression tests for BaseMixin export gating on mne.BaseEpochs.

Prior to the fix, `_auto_export_if_enabled` and `_save_epochs_result` in
`autoclean.mixins.base` gated on the concrete `mne.Epochs` type via
`isinstance(data, mne.Epochs)`. Mixins such as `gfp_clean_epochs` return an
`mne.EpochsArray` (a `BaseEpochs` subclass, but NOT a concrete `mne.Epochs`),
so when `export=True` was set the isinstance check silently failed to match,
`_save_epochs_result` was never invoked, and no `.set` file was written --
no error, no warning, just a silently skipped export.

These tests exercise the real `_auto_export_if_enabled` -> `_save_epochs_result`
code path (unmocked) and only patch `save_epochs_to_set` at its import
location in `autoclean.mixins.base` to avoid real file I/O.
"""

from unittest.mock import patch

import mne
import numpy as np
import pytest

from autoclean.mixins.base import BaseMixin


class _MixinHost(BaseMixin):
    """Minimal host object exposing what BaseMixin needs, nothing more."""

    def __init__(self):
        self.config = {"run_id": "test-run", "task": "test"}
        self.flagged = False


@pytest.fixture
def epochs_array():
    n_channels, n_samples, n_epochs = 4, 256, 10
    data = np.random.randn(n_epochs, n_channels, n_samples)
    info = mne.create_info(n_channels, 256, "eeg")
    return mne.EpochsArray(data, info)


@pytest.fixture
def concrete_epochs():
    raw_data = np.random.randn(4, 256 * 12)
    info = mne.create_info(4, 256, "eeg")
    raw = mne.io.RawArray(raw_data, info)
    events = mne.make_fixed_length_events(raw, duration=1.0)
    return mne.Epochs(
        raw, events, tmin=0, tmax=1.0, baseline=None, preload=True, verbose=False
    )


class TestAutoExportAcceptsBaseEpochs:
    def test_epochs_array_export_calls_save_epochs_to_set(self, epochs_array):
        """Regression: EpochsArray must trigger a real export, not be silently skipped."""
        host = _MixinHost()

        with patch("autoclean.mixins.base.save_epochs_to_set") as mock_save:
            host._auto_export_if_enabled(
                epochs_array, "post_gfp_cleaning", export_enabled=True
            )

        mock_save.assert_called_once()
        _, kwargs = mock_save.call_args
        assert kwargs["epochs"] is epochs_array
        assert kwargs["stage"] == "post_gfp_cleaning"

    def test_concrete_epochs_export_still_calls_save_epochs_to_set(
        self, concrete_epochs
    ):
        """Companion check: concrete mne.Epochs still triggers export as before."""
        host = _MixinHost()

        with patch("autoclean.mixins.base.save_epochs_to_set") as mock_save:
            host._auto_export_if_enabled(
                concrete_epochs, "post_epochs", export_enabled=True
            )

        mock_save.assert_called_once()
        _, kwargs = mock_save.call_args
        assert kwargs["epochs"] is concrete_epochs
        assert kwargs["stage"] == "post_epochs"

    def test_export_disabled_does_not_call_save(self, epochs_array):
        """Sanity check: export_enabled=False must not save anything."""
        host = _MixinHost()

        with patch("autoclean.mixins.base.save_epochs_to_set") as mock_save:
            host._auto_export_if_enabled(
                epochs_array, "post_gfp_cleaning", export_enabled=False
            )

        mock_save.assert_not_called()
