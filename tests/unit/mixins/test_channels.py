"""Unit tests for ChannelsMixin."""

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


class _ChannelTask(Task):
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
    t = _ChannelTask(config)
    t.raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
    )
    return t


# ---------------------------------------------------------------------------
# drop_channels
# ---------------------------------------------------------------------------


class TestDropChannels:
    def test_drops_specified_channels_from_raw(self, task):
        channels_to_drop = [task.raw.ch_names[0], task.raw.ch_names[1]]
        original_count = len(task.raw.ch_names)

        with patch.object(task, "_update_metadata"), patch.object(
            task, "_save_raw_result"
        ), patch.object(task, "_update_instance_data"), patch.object(
            task, "_track_channel_removal"
        ):
            result = task.drop_channels(data=task.raw, channels=channels_to_drop)

        assert len(result.ch_names) == original_count - 2
        for ch in channels_to_drop:
            assert ch not in result.ch_names

    def test_drop_channels_no_op_with_empty_list_returns_data(self, task):
        """Empty channel list → no channels dropped (config path returns early)."""
        settings = {"drop_outerlayer": {"enabled": True, "value": []}}
        task.settings = settings
        # Passing channels=[] explicitly: config path is not used; empty list passed
        original_count = len(task.raw.ch_names)
        with patch.object(task, "_update_metadata"), patch.object(
            task, "_save_raw_result"
        ), patch.object(task, "_update_instance_data"), patch.object(
            task, "_track_channel_removal"
        ):
            result = task.drop_channels(data=task.raw, channels=[])
        # MNE silently accepts empty drop list
        assert len(result.ch_names) == original_count

    def test_drop_channels_raises_on_unknown_channel(self, task):
        with pytest.raises((RuntimeError, ValueError)):
            task.drop_channels(data=task.raw, channels=["NONEXISTENT_CH"])

    def test_returned_raw_does_not_include_dropped_channels(self, task):
        ch_to_drop = task.raw.ch_names[5]
        with patch.object(task, "_update_metadata"), patch.object(
            task, "_save_raw_result"
        ), patch.object(task, "_update_instance_data"), patch.object(
            task, "_track_channel_removal"
        ):
            result = task.drop_channels(data=task.raw, channels=[ch_to_drop])
        assert ch_to_drop not in result.ch_names


# ---------------------------------------------------------------------------
# set_channel_types
# ---------------------------------------------------------------------------


class TestSetChannelTypes:
    def test_updates_channel_type_in_raw_info(self, task):
        """Mapping a channel to 'eog' should change its type in info."""
        ch_name = task.raw.ch_names[0]
        with patch.object(task, "_update_metadata"), patch.object(
            task, "_save_raw_result"
        ), patch.object(task, "_update_instance_data"), patch.object(
            task, "_track_channel_removal"
        ):
            result = task.set_channel_types(
                data=task.raw, ch_types_dict={ch_name: "eog"}
            )

        ch_idx = result.ch_names.index(ch_name)
        assert result.get_channel_types()[ch_idx] == "eog"

    def test_drop_mode_removes_channels_from_result(self, task):
        """When drop=True, the channels in ch_types_dict are removed."""
        ch_to_remove = task.raw.ch_names[0]
        original_count = len(task.raw.ch_names)

        with patch.object(task, "_update_metadata"), patch.object(
            task, "_save_raw_result"
        ), patch.object(task, "_update_instance_data"), patch.object(
            task, "_track_channel_removal"
        ):
            result = task.set_channel_types(
                data=task.raw,
                ch_types_dict={ch_to_remove: "eeg"},
                drop=True,
            )

        assert ch_to_remove not in result.ch_names
        assert len(result.ch_names) == original_count - 1

    def test_drop_mode_silently_skips_missing_channels(self, task):
        """Channels not present in data should be silently ignored when drop=True."""
        original_count = len(task.raw.ch_names)
        with patch.object(task, "_update_metadata"), patch.object(
            task, "_save_raw_result"
        ), patch.object(task, "_update_instance_data"), patch.object(
            task, "_track_channel_removal"
        ):
            result = task.set_channel_types(
                data=task.raw,
                ch_types_dict={"MISSING_CHANNEL": "eeg"},
                drop=True,
            )
        assert len(result.ch_names) == original_count


# ---------------------------------------------------------------------------
# clean_bad_channels (manual override path)
# ---------------------------------------------------------------------------


class TestCleanBadChannels:
    def test_manual_override_marks_specified_channels_as_bad(self, task):
        """Passing manual_bad_channels skips auto-detection and marks those channels."""
        ch_to_mark = task.raw.ch_names[3]

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
        ):
            result = task.clean_bad_channels(
                data=task.raw,
                manual_bad_channels=[ch_to_mark],
                cleaning_method=None,  # Don't interpolate — just mark
            )

        assert ch_to_mark in result.info["bads"]

    def test_manual_override_with_interpolation_returns_raw_without_bad_channels(
        self, task
    ):
        """cleaning_method='interpolate' should remove channels from bads after interpolation."""
        ch_to_mark = task.raw.ch_names[3]

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
        ):
            result = task.clean_bad_channels(
                data=task.raw,
                manual_bad_channels=[ch_to_mark],
                cleaning_method="interpolate",
            )

        # After interpolation with reset_bads=True (default), bads list should be empty
        assert ch_to_mark not in result.info["bads"]

    def test_raises_type_error_for_non_raw_input(self, task):
        with pytest.raises(TypeError):
            task.clean_bad_channels(data="not_raw")

    def test_no_op_when_no_bad_channels_detected(self, task):
        """Auto-detection on clean synthetic data → no channels marked bad."""
        original_count = len(task.raw.ch_names)
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
            patch(
                "autoclean.mixins.signal_processing.channels.detect_bad_channels",
                return_value={
                    "correlation": [],
                    "deviation": [],
                    "ransac": [],
                    "combined": [],
                },
            ),
        ):
            result = task.clean_bad_channels(data=task.raw, cleaning_method=None)

        assert len(result.ch_names) == original_count
        assert result.info["bads"] == []

    def test_records_interpolated_channels_in_metadata(self, task):
        """After interpolation, _update_metadata is called with bads info."""
        ch_to_mark = task.raw.ch_names[3]
        with (
            patch.object(task, "_update_metadata") as mock_meta,
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
        ):
            task.clean_bad_channels(
                data=task.raw,
                manual_bad_channels=[ch_to_mark],
                cleaning_method="interpolate",
            )
        mock_meta.assert_called()

    def test_runs_regardless_of_settings(self, task):
        """clean_bad_channels has no enabled check — always runs."""
        task.settings = {"clean_bad_channels": {"enabled": False}}
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
        ):
            result = task.clean_bad_channels(
                data=task.raw,
                manual_bad_channels=[task.raw.ch_names[0]],
                cleaning_method=None,
            )
        assert isinstance(result, mne.io.BaseRaw)


# ---------------------------------------------------------------------------
# clean_bad_channels (montage-aware presets)
# ---------------------------------------------------------------------------


class TestCleanBadChannelsPresets:
    """The fixture's synthetic raw has 32 channels -> low_density bin."""

    def test_auto_is_default_and_disables_ransac_for_low_density(self, task):
        detect_mock = MagicMock(
            return_value={
                "correlation": [],
                "deviation": [],
                "ransac": [],
                "combined": [],
            }
        )
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
            patch(
                "autoclean.mixins.signal_processing.channels.detect_bad_channels",
                detect_mock,
            ),
        ):
            task.clean_bad_channels(data=task.raw, cleaning_method=None)

        call_kwargs = detect_mock.call_args.kwargs
        assert call_kwargs["correlation_thresh"] == 0.20
        assert call_kwargs["deviation_thresh"] == 4.0
        assert call_kwargs["ransac_corr_thresh"] == 0.0  # RANSAC disabled

    def test_legacy_preset_reproduces_historical_thresholds(self, task):
        detect_mock = MagicMock(
            return_value={
                "correlation": [],
                "deviation": [],
                "ransac": [],
                "combined": [],
            }
        )
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
            patch(
                "autoclean.mixins.signal_processing.channels.detect_bad_channels",
                detect_mock,
            ),
        ):
            task.clean_bad_channels(data=task.raw, preset="legacy", cleaning_method=None)

        call_kwargs = detect_mock.call_args.kwargs
        assert call_kwargs["correlation_thresh"] == 0.35
        assert call_kwargs["deviation_thresh"] == 2.5
        assert call_kwargs["ransac_corr_thresh"] == 0.65

    def test_explicit_kwarg_overrides_preset(self, task):
        detect_mock = MagicMock(
            return_value={
                "correlation": [],
                "deviation": [],
                "ransac": [],
                "combined": [],
            }
        )
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
            patch(
                "autoclean.mixins.signal_processing.channels.detect_bad_channels",
                detect_mock,
            ),
        ):
            task.clean_bad_channels(
                data=task.raw, correlation_thresh=0.42, cleaning_method=None
            )

        assert detect_mock.call_args.kwargs["correlation_thresh"] == 0.42

    def test_config_preset_used_when_no_explicit_kwarg(self, task):
        task.settings = {
            "bad_channel_detection": {"enabled": True, "value": {"preset": "legacy"}}
        }
        detect_mock = MagicMock(
            return_value={
                "correlation": [],
                "deviation": [],
                "ransac": [],
                "combined": [],
            }
        )
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
            patch(
                "autoclean.mixins.signal_processing.channels.detect_bad_channels",
                detect_mock,
            ),
        ):
            task.clean_bad_channels(data=task.raw, cleaning_method=None)

        assert detect_mock.call_args.kwargs["correlation_thresh"] == 0.35

    def test_metadata_records_preset_and_density_bin(self, task):
        with (
            patch.object(task, "_update_metadata") as mock_meta,
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
            patch(
                "autoclean.mixins.signal_processing.channels.detect_bad_channels",
                return_value={
                    "correlation": [],
                    "deviation": [],
                    "ransac": [],
                    "combined": [],
                },
            ),
        ):
            task.clean_bad_channels(data=task.raw, cleaning_method=None)

        metadata = mock_meta.call_args.args[1]
        assert metadata["preset"] == "auto"
        assert metadata["density_bin"] == "low_density"
        assert metadata["eeg_channel_count"] == 32
        assert "resolved_thresholds" in metadata
        assert metadata["resolved_thresholds"]["ransac_enabled"] is False

    def test_bad_fraction_warning_uses_preset_max_bad_fraction(self, task):
        """4/32 = 12.5%: exceeds low_density's 10% cap but not legacy's 15%."""
        four_bad = task.raw.ch_names[:4]

        def _run(preset):
            task.flagged = False
            task.flagged_reasons = []
            with (
                patch.object(task, "_update_metadata"),
                patch.object(task, "_save_raw_result"),
                patch.object(task, "_update_instance_data"),
                patch.object(task, "_track_channel_removal"),
                patch(
                    "autoclean.mixins.signal_processing.channels.detect_bad_channels",
                    return_value={
                        "correlation": list(four_bad),
                        "deviation": [],
                        "ransac": [],
                        "combined": list(four_bad),
                    },
                ),
            ):
                task.clean_bad_channels(data=task.raw, preset=preset, cleaning_method=None)
            return task.flagged

        assert _run("auto") is True
        assert _run("legacy") is False


# ---------------------------------------------------------------------------
# set_channel_types (additional)
# ---------------------------------------------------------------------------


class TestSetChannelTypesAdditional:
    def test_raises_type_error_for_non_raw_input(self, task):
        """set_channel_types raises TypeError when given a non-Raw object."""
        with pytest.raises(TypeError):
            task.set_channel_types(data="not_raw", ch_types_dict={"Fp1": "eog"})


# ---------------------------------------------------------------------------
# drop_eog_channels
# ---------------------------------------------------------------------------


class TestDropEOGChannels:
    def test_removes_eog_typed_channels(self, task):
        """Channels with type 'eog' are removed from the raw object."""
        ch_name = task.raw.ch_names[0]
        # Directly set channel type to EOG in the raw's info
        task.raw.set_channel_types({ch_name: "eog"})

        original_count = len(task.raw.ch_names)
        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
        ):
            result = task.drop_eog_channels(data=task.raw)

        assert ch_name not in result.ch_names
        assert len(result.ch_names) < original_count

    def test_no_op_when_no_eog_channels_present(self, task):
        """Raw with no EOG-typed channels is returned with the same channel count."""
        # Ensure all channels are EEG (default from create_synthetic_raw)
        original_count = len(task.raw.ch_names)
        original_names = list(task.raw.ch_names)

        with (
            patch.object(task, "_update_metadata"),
            patch.object(task, "_save_raw_result"),
            patch.object(task, "_update_instance_data"),
            patch.object(task, "_track_channel_removal"),
        ):
            result = task.drop_eog_channels(data=task.raw)

        assert len(result.ch_names) == original_count
        assert list(result.ch_names) == original_names
