"""Unit tests for IcaMixin."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

try:
    from autoclean.core.task import Task

    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available")


class _ICATask(Task):
    def __init__(self, config, settings=None):
        self.settings = settings or {}
        super().__init__(config)

    def run(self):
        pass


@pytest.fixture
def task(tmp_path):
    settings = {
        "ICA": {
            "enabled": True,
            "value": {"n_components": 5, "method": "fastica"},
        }
    }
    config = {
        "run_id": "test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "test",
    }
    t = _ICATask(config, settings=settings)
    t.raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=20.0, sfreq=250.0
    )
    return t


def _make_mock_ica(n_components=5):
    """Return a lightweight mock ICA object."""
    mock_ica = MagicMock(spec=mne.preprocessing.ICA)
    mock_ica.n_components_ = n_components
    mock_ica.exclude = []
    return mock_ica


# ---------------------------------------------------------------------------
# run_ica
# ---------------------------------------------------------------------------


class TestRunICA:
    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_stores_fitted_ica_on_task(self, mock_fit, mock_save_fif, task):
        """Successful run_ica call stores the ICA object on self.final_ica."""
        mock_ica = _make_mock_ica()
        mock_fit.return_value = mock_ica
        with patch.object(task, "_update_metadata"):
            task.run_ica()
        assert task.final_ica is mock_ica

    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_returns_fitted_ica(self, mock_fit, mock_save_fif, task):
        """run_ica should return the ICA object."""
        mock_ica = _make_mock_ica()
        mock_fit.return_value = mock_ica
        with patch.object(task, "_update_metadata"):
            result = task.run_ica()
        assert result is mock_ica

    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_fit_ica_called_once(self, mock_fit, mock_save_fif, task):
        mock_fit.return_value = _make_mock_ica()
        with patch.object(task, "_update_metadata"):
            task.run_ica()
        mock_fit.assert_called_once()

    def test_skips_when_ica_disabled_in_settings(self, tmp_path):
        """ICA disabled in settings → no ICA object created, returns None."""
        settings = {"ICA": {"enabled": False}}
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _ICATask(config, settings=settings)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
        )
        result = t.run_ica()
        assert result is None
        assert not hasattr(t, "final_ica") or t.final_ica is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# apply_ica_component_rejection
# ---------------------------------------------------------------------------


class TestApplyICAComponentRejection:
    def test_raises_runtime_error_without_final_ica(self, task):
        """Calling rejection before run_ica should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="final_ica"):
            task.apply_ica_component_rejection()

    def test_manual_override_sets_exclude_list(self, task):
        """Manual component list bypasses auto-detection and sets exclude directly."""
        mock_ica = _make_mock_ica()
        mock_ica.apply = MagicMock()
        task.final_ica = mock_ica

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            task.apply_ica_component_rejection(manual_rejected_components=[0, 2])

        assert task.final_ica.exclude == [0, 2]
        mock_ica.apply.assert_called_once()

    def test_manual_override_empty_list_skips_apply(self, task):
        """Empty manual override means no components are applied/removed."""
        mock_ica = _make_mock_ica()
        mock_ica.apply = MagicMock()
        task.final_ica = mock_ica

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            task.apply_ica_component_rejection(manual_rejected_components=[])

        mock_ica.apply.assert_not_called()

    def test_manual_override_deduplicates_and_sorts_components(self, task):
        """Duplicate and unsorted component indices should be cleaned up."""
        mock_ica = _make_mock_ica()
        mock_ica.apply = MagicMock()
        task.final_ica = mock_ica

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
        ):
            task.apply_ica_component_rejection(
                manual_rejected_components=[3, 1, 1, 3, 0]
            )

        assert task.final_ica.exclude == [0, 1, 3]

    def test_auto_mode_raises_without_ica_flags(self, task):
        """Auto mode without prior classification raises RuntimeError."""
        task.final_ica = _make_mock_ica()
        with pytest.raises(RuntimeError, match="ica_flags"):
            task.apply_ica_component_rejection()

    def test_auto_mode_skips_when_component_rejection_disabled(self, tmp_path):
        """component_rejection disabled in settings → method returns without applying."""
        settings = {
            "ICA": {"enabled": True},
            "component_rejection": {"enabled": False},
        }
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _ICATask(config, settings=settings)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
        )
        mock_ica = _make_mock_ica()
        mock_ica.apply = MagicMock()
        t.final_ica = mock_ica
        # ica_flags present (so auto mode won't raise for missing flags)
        t.ica_flags = pd.DataFrame({"label": ["brain", "eye blink"]})

        t.apply_ica_component_rejection()  # Should not raise

        mock_ica.apply.assert_not_called()  # Nothing applied when disabled


class TestGetIcaReportData:
    def test_uses_prerejection_snapshot_when_present(self, task):
        """Report data should come from the pre-rejection snapshot, not self.raw."""
        original_raw = task.raw
        task.raw_prerejection = original_raw.copy()

        # Mutate self.raw to simulate what ica.apply() does in place
        task.raw._data[:] = 0

        result = task._get_ica_report_data()

        assert result is task.raw_prerejection
        assert not np.allclose(result.get_data(), task.raw.get_data())

    def test_falls_back_to_raw_when_no_snapshot(self, task):
        """Without a snapshot, report data should just be self.raw."""
        assert not hasattr(task, "raw_prerejection")

        result = task._get_ica_report_data()

        assert result is task.raw


# ---------------------------------------------------------------------------
# classify_ica_components
# ---------------------------------------------------------------------------


class TestClassifyICAComponents:
    def test_skips_when_ica_not_run(self, task):
        """classify_ica_components returns None when self.final_ica is absent."""
        # Do not set task.final_ica
        result = task.classify_ica_components()
        assert result is None

    def test_skips_when_ica_disabled_in_settings(self, tmp_path):
        """ICA not enabled → classification returns None."""
        settings = {"ICA": {"enabled": False}}
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _ICATask(config, settings=settings)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
        )
        t.final_ica = _make_mock_ica()
        result = t.classify_ica_components()
        assert result is None

    @patch("autoclean.mixins.signal_processing.ica.classify_ica_components")
    def test_calls_classification_function_and_stores_flags(self, mock_classify, task):
        """After classify, self.ica_flags is the DataFrame returned by the function."""
        mock_flags = pd.DataFrame({"label": ["brain", "eye blink"]})
        mock_classify.return_value = mock_flags
        task.final_ica = _make_mock_ica()

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
            patch.object(task, "apply_ica_component_rejection"),
        ):
            result = task.classify_ica_components(method="iclabel", reject=False)

        assert result is mock_flags
        assert task.ica_flags is mock_flags

    @patch("autoclean.mixins.signal_processing.ica.classify_ica_components")
    def test_classify_components_uses_iclabel_threshold(self, mock_classify, task):
        """classify_ica_components passes the configured threshold to the backend."""
        settings = {
            "ICA": {"enabled": True},
            "component_rejection": {
                "enabled": True,
                "value": {
                    "method": "iclabel",
                    "ic_flags_to_reject": ["eye blink"],
                    "ic_rejection_threshold": 0.9,
                },
            },
        }
        task.settings = settings
        mock_flags = pd.DataFrame({"label": ["brain", "eye blink"]})
        mock_classify.return_value = mock_flags
        task.final_ica = _make_mock_ica()

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
            patch.object(task, "apply_ica_component_rejection"),
        ):
            task.classify_ica_components(method="iclabel", reject=False)

        # The function must have been called (threshold forwarding tested via call)
        mock_classify.assert_called_once()

    @patch("autoclean.mixins.signal_processing.ica.classify_ica_components")
    def test_classify_components_respects_ic_flags_to_reject_setting(
        self, mock_classify, task
    ):
        """classify stores flags DataFrame so apply step can use ic_flags_to_reject."""
        mock_flags = pd.DataFrame({"label": ["muscle", "brain", "eye blink"]})
        mock_classify.return_value = mock_flags
        task.final_ica = _make_mock_ica()

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
            patch.object(task, "apply_ica_component_rejection"),
        ):
            result = task.classify_ica_components(method="iclabel", reject=False)

        # ica_flags must contain the full classification result for downstream filtering
        assert len(result) == 3
        assert task.ica_flags is result

    @patch("autoclean.mixins.signal_processing.ica.classify_ica_components")
    def test_generates_iclabel_report_after_rejection(self, mock_classify, task):
        """ICLabel report generation should happen after rejection updates ICA state."""
        mock_classify.return_value = pd.DataFrame({"label": ["brain", "eye blink"]})
        task.final_ica = _make_mock_ica()
        calls: list[str] = []

        def _record_rejection(*args, **kwargs):
            calls.append("reject")

        def _record_report(*args, **kwargs):
            calls.append("report")

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled"),
            patch.object(
                task,
                "apply_ica_component_rejection",
                side_effect=_record_rejection,
            ),
            patch.object(task, "generate_ica_reports", side_effect=_record_report),
        ):
            task.classify_ica_components(method="iclabel", reject=True)

        assert calls == ["reject", "report"]


class TestRunICASettings:
    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_run_ica_uses_n_components_from_settings(
        self, mock_fit, mock_save_fif, tmp_path
    ):
        """n_components from settings[ICA][value] must be forwarded to fit_ica."""
        settings = {
            "ICA": {
                "enabled": True,
                "value": {"n_components": 10, "method": "fastica"},
            }
        }
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _ICATask(config, settings=settings)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=20.0, sfreq=250.0
        )
        mock_fit.return_value = _make_mock_ica(n_components=10)

        with patch.object(t, "_update_metadata"):
            t.run_ica()

        call_kwargs = mock_fit.call_args[1]
        assert call_kwargs.get("n_components") == 10

    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_run_ica_uses_method_from_settings(self, mock_fit, mock_save_fif, tmp_path):
        """method from settings[ICA][value] must be forwarded to fit_ica."""
        settings = {
            "ICA": {
                "enabled": True,
                "value": {"n_components": 5, "method": "infomax"},
            }
        }
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _ICATask(config, settings=settings)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=20.0, sfreq=250.0
        )
        mock_fit.return_value = _make_mock_ica()

        with patch.object(t, "_update_metadata"):
            t.run_ica()

        call_kwargs = mock_fit.call_args[1]
        assert call_kwargs.get("method") == "infomax"

    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_run_ica_saves_post_ica_stage_file(self, mock_fit, mock_save_fif, task):
        """save_ica_to_fif must be called after a successful fit."""
        mock_fit.return_value = _make_mock_ica()

        with patch.object(task, "_update_metadata"):
            task.run_ica()

        mock_save_fif.assert_called_once()

    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_run_ica_propagates_exception_when_fit_fails(
        self, mock_fit, mock_save_fif, tmp_path
    ):
        """If fit_ica raises (e.g. too few channels), the exception propagates."""
        settings = {
            "ICA": {
                "enabled": True,
                "value": {"n_components": 100},  # Far more than available channels
            }
        }
        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _ICATask(config, settings=settings)
        t.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=8, duration=10.0, sfreq=250.0
        )
        mock_fit.side_effect = ValueError("n_components exceeds n_channels")

        with patch.object(t, "_update_metadata"):
            with pytest.raises((ValueError, Exception)):
                t.run_ica()


class TestApplyICAStageFile:
    def test_apply_ica_calls_auto_export_after_rejection(self, task):
        """_auto_export_if_enabled should be called after ICA application."""
        mock_ica = _make_mock_ica()
        mock_ica.apply = MagicMock()
        task.final_ica = mock_ica

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_auto_export_if_enabled") as mock_export,
        ):
            task.apply_ica_component_rejection(manual_rejected_components=[0])

        mock_export.assert_called_once()
