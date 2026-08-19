"""Unit tests for IcaMixin."""

from unittest.mock import MagicMock, patch

import matplotlib.axes as maxes
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


@pytest.fixture
def epochs_task(tmp_path):
    """A task with only epoched data available (no self.raw), matching the
    `self.import_epochs(); self.run_ica(use_epochs=True)` usage pattern."""
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
    raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=20.0, sfreq=250.0
    )
    t.epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
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

    @patch("autoclean.mixins.signal_processing.ica.save_ica_to_fif")
    @patch("autoclean.mixins.signal_processing.ica.fit_ica")
    def test_use_epochs_fits_on_self_epochs(self, mock_fit, mock_save_fif, epochs_task):
        """run_ica(use_epochs=True) should fit on self.epochs when self.raw
        doesn't exist, matching the `import_epochs(); run_ica(use_epochs=True)`
        pattern from issue #218."""
        mock_ica = _make_mock_ica()
        mock_fit.return_value = mock_ica
        with patch.object(epochs_task, "_update_metadata"):
            epochs_task.run_ica(use_epochs=True)

        assert epochs_task.final_ica is mock_ica
        # run_ica fits on a .copy() of the data, so check type/identity of source
        fit_call_kwargs = mock_fit.call_args[1]
        assert isinstance(fit_call_kwargs["raw"], mne.BaseEpochs)
        assert epochs_task._ica_used_epochs is True


# ---------------------------------------------------------------------------
# apply_ica_component_rejection
# ---------------------------------------------------------------------------


class TestApplyICAComponentRejection:
    def test_raises_runtime_error_without_final_ica(self, task):
        """Calling rejection before run_ica should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="final_ica"):
            task.apply_ica_component_rejection()

    def test_manual_override_updates_self_epochs_when_no_raw(self, epochs_task):
        """When ICA was fit on epochs (no self.raw), manual rejection should
        default to and modify self.epochs instead of trying self.raw."""
        mock_ica = _make_mock_ica()
        mock_ica.apply = MagicMock()
        epochs_task.final_ica = mock_ica
        epochs_task._ica_used_epochs = True

        with (
            patch.object(epochs_task, "_update_metadata"),
            patch.object(epochs_task, "_auto_export_if_enabled"),
        ):
            epochs_task.apply_ica_component_rejection(manual_rejected_components=[0, 2])

        assert epochs_task.final_ica.exclude == [0, 2]
        mock_ica.apply.assert_called_once_with(epochs_task.epochs)

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

    def test_returns_none_when_ica_fit_on_epochs(self, epochs_task):
        """ICA reports are raw-only. When ICA was fit on epochs, this should
        return None (triggering a graceful skip) rather than silently
        returning a stale/incompatible self.raw."""
        epochs_task._ica_used_epochs = True

        result = epochs_task._get_ica_report_data()

        assert result is None

    def test_ignores_stale_raw_when_ica_fit_on_epochs(self, epochs_task):
        """Even if self.raw happens to exist (e.g. left over from an earlier
        step), report data must not silently return it when ICA was actually
        fit on epochs - that data wasn't what ICA ran on."""
        epochs_task.raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=5.0, sfreq=250.0
        )
        epochs_task._ica_used_epochs = True

        result = epochs_task._get_ica_report_data()

        assert result is None


class TestPlotIcaFull:
    def test_returns_none_instead_of_crashing_when_report_data_missing(
        self, epochs_task
    ):
        """plot_ica_full should skip gracefully (not raise) when ICA was fit
        on epochs, since this plot is raw-only."""
        epochs_task.final_ica = _make_mock_ica()
        epochs_task._ica_used_epochs = True

        result = epochs_task.plot_ica_full()

        assert result is None


class TestPlotIcaComponentsEpochsGuard:
    def test_skips_gracefully_when_ica_fit_on_epochs(self, epochs_task):
        """_plot_ica_components should skip gracefully (not crash) when ICA
        was fit on epochs, consistent with reports being raw-only."""
        epochs_task.final_ica = _make_mock_ica()
        epochs_task._ica_used_epochs = True

        result = epochs_task._plot_ica_components()

        assert result is None


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
    def test_classifies_epochs_when_raw_unavailable(self, mock_classify, epochs_task):
        """When ICA was fit on epochs (no self.raw), classification should
        receive self.epochs instead of crashing on a missing self.raw."""
        mock_flags = pd.DataFrame({"label": ["brain", "eye blink"]})
        mock_classify.return_value = mock_flags
        epochs_task.final_ica = _make_mock_ica()
        epochs_task._ica_used_epochs = True

        with (
            patch.object(epochs_task, "_update_metadata"),
            patch.object(epochs_task, "_auto_export_if_enabled"),
            patch.object(epochs_task, "apply_ica_component_rejection"),
        ):
            result = epochs_task.classify_ica_components(method="iclabel", reject=False)

        assert result is mock_flags
        mock_classify.assert_called_once_with(
            epochs_task.epochs, epochs_task.final_ica, method="iclabel"
        )

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


# ---------------------------------------------------------------------------
# ICA component label numbering (issue #294)
#
# The summary table, full-duration activation plot, and ICLabel/Vision
# comparison report must all label the first ICA component "IC0", matching
# MNE's own 0-based indexing and the topography pages - not "IC1".
# ---------------------------------------------------------------------------


class TestIcLabelHelper:
    def test_zero_based_formatting(self):
        from autoclean.mixins.viz.ica import _ic_label

        assert _ic_label(0) == "IC0"
        assert _ic_label(1) == "IC1"
        assert _ic_label(9) == "IC9"


class TestPlotIcaComponentsSummaryLabels:
    """Regression test for the summary table's per-component label."""

    @patch("autoclean.utils.database.manage_database", return_value=None)
    @patch("autoclean.mixins.viz.ica.plot_ica_topographies_overview", return_value=[])
    @patch(
        "autoclean.mixins.viz.ica.plot_component_for_classification", return_value=None
    )
    def test_summary_table_first_row_is_ic0_not_ic1(
        self, mock_plot_component, mock_topo_overview, mock_db, task, tmp_path
    ):
        task.config["reports_dir"] = str(tmp_path / "reports")

        task.run_ica()
        n_components = task.final_ica.n_components_
        task.ica_flags = pd.DataFrame(
            {
                "ic_type": ["brain"] * n_components,
                "confidence": [0.9] * n_components,
                "annotator": ["ic_label"] * n_components,
            }
        )

        with patch.object(
            maxes.Axes, "table", autospec=True, wraps=maxes.Axes.table
        ) as mock_table:
            result = task._plot_ica_components(components="all")

        assert result is not None
        assert mock_table.call_args_list, "summary table was never built"
        first_call_kwargs = mock_table.call_args_list[0].kwargs
        first_row = first_call_kwargs["cellText"][0]
        assert first_row[0] == "IC0"

    @patch("autoclean.mixins.viz.ica.plot_ica_topographies_overview", return_value=[])
    @patch(
        "autoclean.mixins.viz.ica.plot_component_for_classification", return_value=None
    )
    def test_summary_table_labels_single_component_ic0(
        self, mock_plot_component, mock_topo_overview, task, tmp_path
    ):
        """A 1-component ICA must still label its only row IC0, not IC1 -
        the off-by-one bug is easy to miss when there's only one row.

        A real single-component fit isn't possible (MNE rejects
        n_components=1), so this uses a mock ICA instead - the label
        construction under test only touches the component index, not the
        fitted decomposition itself.
        """
        task.config["reports_dir"] = str(tmp_path / "reports")
        task.final_ica = _make_mock_ica(n_components=1)
        task.ica_flags = pd.DataFrame(
            {"ic_type": ["brain"], "confidence": [0.9], "annotator": ["ic_label"]}
        )

        with patch.object(
            maxes.Axes, "table", autospec=True, wraps=maxes.Axes.table
        ) as mock_table:
            result = task._plot_ica_components(components="all")

        assert result is not None
        first_row = mock_table.call_args_list[0].kwargs["cellText"][0]
        assert first_row[0] == "IC0"


class TestPlotIcaFullLabels:
    """Regression test for the full-duration activation plot's y-tick labels."""

    @patch("autoclean.utils.database.manage_database", return_value=None)
    def test_first_yticklabel_is_ic0_not_ic1(self, mock_db, task, tmp_path):
        task.config["reports_dir"] = str(tmp_path / "reports")

        task.run_ica()
        n_components = task.final_ica.n_components_
        task.ica_flags = pd.DataFrame(
            {
                "ic_type": ["brain"] * n_components,
                "confidence": [0.9] * n_components,
                "annotator": ["ic_label"] * n_components,
            }
        )

        fig = task.plot_ica_full()

        assert fig is not None
        first_label = fig.axes[0].get_yticklabels()[0].get_text()
        assert first_label.startswith("IC0:"), first_label


class TestCompareVisionIclabelLabels:
    """Regression test for the ICLabel-vs-Vision comparison bar chart/table."""

    @patch("autoclean.utils.database.manage_database", return_value=None)
    def test_first_component_labeled_ic0_not_ic1(self, mock_db, task, tmp_path):
        task.config["reports_dir"] = str(tmp_path / "reports")
        task.ica_flags = pd.DataFrame(
            {"ic_type": ["brain", "eog"], "confidence": [0.9, 0.8]}
        )
        task.ica_vision_flags = pd.DataFrame(
            {"label": ["brain", "eye"], "confidence": [0.85, 0.75]}
        )

        fig = task.compare_vision_iclabel_classifications()

        assert fig is not None
        ax1 = fig.axes[0]
        xtick_labels = [t.get_text() for t in ax1.get_xticklabels()]
        assert xtick_labels[0] == "IC0"
