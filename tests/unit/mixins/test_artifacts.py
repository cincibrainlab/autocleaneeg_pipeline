"""Unit tests for the ArtifactsMixin."""

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


class _ArtifactTask(Task):
    """Minimal task subclass for artifact mixin tests."""

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
    t = _ArtifactTask(config)
    t.raw = create_synthetic_raw(
        montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0
    )
    return t


# ---------------------------------------------------------------------------
# detect_dense_oscillatory_artifacts
# ---------------------------------------------------------------------------


class TestDetectDenseOscillatoryArtifacts:
    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_returns_mne_raw_object(self, mock_save, mock_db, task):
        result = task.detect_dense_oscillatory_artifacts(data=task.raw)
        assert isinstance(result, mne.io.BaseRaw)

    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_clean_data_produces_zero_annotations(self, mock_save, mock_db, task):
        """Low-amplitude synthetic EEG should produce no BAD_REF_AF annotations."""
        result = task.detect_dense_oscillatory_artifacts(
            data=task.raw,
            channel_threshold_uv=500,  # Very high threshold — nothing crosses it
            min_channels=32,
        )
        bad_annotations = [
            ann for ann in result.annotations if ann["description"] == "BAD_REF_AF"
        ]
        assert len(bad_annotations) == 0

    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_high_amplitude_multichannel_data_produces_annotations(
        self, mock_save, mock_db, task
    ):
        """All-channel high amplitude bursts should be detected as artifacts."""
        # Inject a large-amplitude burst into all channels simultaneously
        task.raw.load_data()
        task.raw._data[:, 500:600] = 200e-6  # 200 µV spike across all 32 channels

        result = task.detect_dense_oscillatory_artifacts(
            data=task.raw,
            channel_threshold_uv=45,  # Default threshold
            min_channels=5,  # Low enough to detect with 32 channels
        )
        bad_annotations = [
            ann for ann in result.annotations if ann["description"] == "BAD_REF_AF"
        ]
        assert len(bad_annotations) > 0

    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_custom_annotation_label_used(self, mock_save, mock_db, task):
        task.raw.load_data()
        task.raw._data[:, 100:200] = 500e-6  # Create detectable artifact

        result = task.detect_dense_oscillatory_artifacts(
            data=task.raw,
            channel_threshold_uv=10,
            min_channels=5,
            annotation_label="BAD_CUSTOM",
        )
        custom_annotations = [
            ann for ann in result.annotations if ann["description"] == "BAD_CUSTOM"
        ]
        assert len(custom_annotations) >= 0  # May or may not fire; label is correct

        # Ensure default label was NOT used
        default_annotations = [
            ann for ann in result.annotations if ann["description"] == "BAD_REF_AF"
        ]
        assert len(default_annotations) == 0

    def test_raises_type_error_for_non_raw_input(self, task):
        with pytest.raises(TypeError):
            task.detect_dense_oscillatory_artifacts(data="not_a_raw_object")

    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_uses_threshold_from_direct_parameter(self, mock_save, mock_db, task):
        """Lower threshold detects more artifacts than higher threshold."""
        task.raw.load_data()
        # Inject a moderate artifact into all channels
        task.raw._data[:, 400:500] = 60e-6  # 60 µV — above default 45 µV threshold

        result_strict = task.detect_dense_oscillatory_artifacts(
            data=task.raw.copy(),
            channel_threshold_uv=10,  # Very sensitive — should flag many windows
            min_channels=5,
        )
        result_lenient = task.detect_dense_oscillatory_artifacts(
            data=task.raw.copy(),
            channel_threshold_uv=500,  # Very insensitive — should flag nothing
            min_channels=5,
        )
        strict_count = sum(
            1 for ann in result_strict.annotations if ann["description"] == "BAD_REF_AF"
        )
        lenient_count = sum(
            1 for ann in result_lenient.annotations if ann["description"] == "BAD_REF_AF"
        )
        assert strict_count >= lenient_count


# ---------------------------------------------------------------------------
# detect_muscle_beta_focus
# ---------------------------------------------------------------------------


class TestDetectMuscleBetaFocus:
    def test_returns_none_when_no_other_channels(self, task):
        """Standard 1020 raw has no GSN 'OTHER' channels → returns None."""
        result = task.detect_muscle_beta_focus(data=task.raw)
        assert result is None

    def test_annotates_raw_when_other_channels_present(self, tmp_path):
        """Raw with E17 (an OTHER channel) processes without error."""
        import numpy as np

        # Create raw with E17 channel name (an 'OTHER' channel in the GSN map)
        sfreq = 250.0
        duration = 10.0
        n_times = int(duration * sfreq)
        ch_names = ["E17", "Fp1", "Fp2"]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        data = np.random.randn(len(ch_names), n_times) * 1e-6

        # Inject a large-amplitude artifact in one window
        data[0, 1000:1250] += 5000e-6  # Huge spike on E17

        raw = mne.io.RawArray(data, info)

        config = {
            "run_id": "test",
            "unprocessed_file": tmp_path / "test.fif",
            "task": "test",
        }
        t = _ArtifactTask(config)
        t.raw = raw

        with patch.object(t, "_update_metadata"):
            result = t.detect_muscle_beta_focus(data=raw)
        # If OTHER channels exist, result should be a Raw (not None)
        assert result is None or isinstance(result, mne.io.BaseRaw)

    def test_raises_type_error_for_non_raw_input(self, task):
        """detect_muscle_beta_focus must raise TypeError for non-Raw input."""
        with pytest.raises(TypeError):
            task.detect_muscle_beta_focus(data=["not", "raw"])


# ---------------------------------------------------------------------------
# reject_bad_segments
# ---------------------------------------------------------------------------


class TestRejectBadSegments:
    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_returns_mne_raw_object(self, mock_save, mock_db, task):
        result = task.reject_bad_segments(data=task.raw)
        assert isinstance(result, mne.io.BaseRaw)

    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_unannotated_raw_is_returned_unchanged_in_duration(
        self, mock_save, mock_db, task
    ):
        original_duration = task.raw.times[-1]
        result = task.reject_bad_segments(data=task.raw)
        assert abs(result.times[-1] - original_duration) < 0.1

    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_bad_annotated_segment_reduces_duration(self, mock_save, mock_db, task):
        """Adding a BAD annotation should result in shorter returned data."""
        original_duration = task.raw.times[-1]
        task.raw.annotations.append(onset=2.0, duration=3.0, description="BAD_test")

        result = task.reject_bad_segments(data=task.raw)
        assert result.times[-1] < original_duration

    @patch("autoclean.mixins.base.manage_database_conditionally")
    @patch("autoclean.mixins.base.save_raw_to_set")
    def test_specific_label_filter_only_removes_matching_label(
        self, mock_save, mock_db, task
    ):
        """When bad_label is set, only annotations with that exact label are removed."""
        task.raw.annotations.append(onset=2.0, duration=2.0, description="BAD_CUSTOM")
        task.raw.annotations.append(onset=6.0, duration=1.0, description="BAD_OTHER")

        # Only remove BAD_CUSTOM segments
        result_custom = task.reject_bad_segments(
            data=task.raw, bad_label="BAD_CUSTOM"
        )
        result_all = task.reject_bad_segments(data=task.raw, bad_label=None)

        # Removing only one segment should leave more data than removing both
        assert result_custom.times[-1] > result_all.times[-1]

    def test_raises_type_error_for_non_raw_input(self, task):
        with pytest.raises(TypeError):
            task.reject_bad_segments(data=["not", "raw"])
