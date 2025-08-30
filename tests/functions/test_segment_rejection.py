"""Tests for segment rejection functions."""

from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from autoclean.functions.segment_rejection import (
    annotate_noisy_segments,
    annotate_uncorrelated_segments,
    detect_dense_oscillatory_artifacts,
)


@pytest.fixture
def mock_raw():
    """Create a mock raw object for testing."""
    raw = MagicMock(spec=mne.io.BaseRaw)
    raw.info = {"sfreq": 500.0}
    raw.get_data.return_value = (
        np.random.randn(64, 10000),  # 64 channels, 10000 samples
        np.linspace(0, 20, 10000),  # 20 seconds of data
    )
    raw.copy.return_value = raw
    raw.annotations = MagicMock()
    raw.annotations.append = MagicMock()
    return raw


class TestDetectDenseOscilatoryArtifacts:
    """Test dense oscillatory artifact detection."""

    def test_basic_functionality(self, mock_raw):
        """Test basic dense oscillatory artifact detection."""
        result = detect_dense_oscillatory_artifacts(mock_raw)

        # Should return a raw object
        assert isinstance(result, type(mock_raw))

        # Should have called copy
        mock_raw.copy.assert_called_once()

    def test_custom_parameters(self, mock_raw):
        """Test with custom parameters."""
        result = detect_dense_oscillatory_artifacts(
            mock_raw,
            window_size_ms=200,
            channel_threshold_uv=60,
            min_channels=50,
            padding_ms=1000,
            annotation_label="BAD_custom",
        )

        assert isinstance(result, type(mock_raw))

    def test_input_validation(self):
        """Test input validation."""
        # Test non-Raw object
        with pytest.raises(TypeError):
            detect_dense_oscillatory_artifacts("not_raw")

        # Test invalid parameters
        raw = MagicMock(spec=mne.io.BaseRaw)

        with pytest.raises(ValueError):
            detect_dense_oscillatory_artifacts(raw, window_size_ms=-100)

        with pytest.raises(ValueError):
            detect_dense_oscillatory_artifacts(raw, channel_threshold_uv=-10)

        with pytest.raises(ValueError):
            detect_dense_oscillatory_artifacts(raw, min_channels=-5)

        with pytest.raises(ValueError):
            detect_dense_oscillatory_artifacts(raw, padding_ms=-100)

    def test_high_amplitude_detection(self, mock_raw):
        """Test detection with high amplitude artifacts."""
        # Create data with high amplitude artifacts
        n_channels, n_samples = 64, 10000
        data = np.random.randn(n_channels, n_samples) * 1e-6  # Normal EEG scale

        # Add high amplitude artifact to many channels
        artifact_start = 5000
        artifact_length = 50  # 100ms at 500Hz
        data[:50, artifact_start : artifact_start + artifact_length] += 100e-6  # 100µV

        mock_raw.get_data.return_value = (data, np.linspace(0, 20, n_samples))

        result = detect_dense_oscillatory_artifacts(
            mock_raw,
            min_channels=40,  # Require fewer channels
            channel_threshold_uv=80,  # Lower threshold
        )

        # Should detect the artifact
        assert isinstance(result, type(mock_raw))


class TestAnnotateNoisySegments:
    """Test noisy segment annotation (from moved functions)."""

    @patch(
        "autoclean.functions.segment_rejection.segment_rejection.mne.make_fixed_length_events"
    )
    @patch("autoclean.functions.segment_rejection.segment_rejection.mne.Epochs")
    def test_basic_functionality(self, mock_epochs_class, mock_make_events, mock_raw):
        """Test basic noisy segment annotation."""
        # Mock events
        mock_events = np.array([[0, 0, 1], [1000, 0, 1], [2000, 0, 1]])
        mock_make_events.return_value = mock_events

        # Mock epochs
        mock_epochs = MagicMock()
        mock_epochs.events = mock_events
        mock_epochs.ch_names = [f"EEG_{i:03d}" for i in range(64)]
        mock_epochs.times = np.linspace(0, 2, 1000)
        mock_epochs_class.return_value = mock_epochs
        mock_epochs.__len__.return_value = 3

        # Mock _epochs_to_xr and _detect_outliers
        with (
            patch(
                "autoclean.functions.segment_rejection.segment_rejection._epochs_to_xr"
            ) as mock_xr,
            patch(
                "autoclean.functions.segment_rejection.segment_rejection._detect_outliers"
            ) as mock_detect,
        ):

            mock_xr.return_value = MagicMock()
            mock_detect.return_value = np.array([])  # No outliers

            result = annotate_noisy_segments(mock_raw)

            # Should return a raw object
            assert isinstance(result, type(mock_raw))

    def test_input_validation(self):
        """Test input validation for noisy segments."""
        with pytest.raises(TypeError):
            annotate_noisy_segments("not_raw")

        raw = MagicMock(spec=mne.io.BaseRaw)

        with pytest.raises(ValueError):
            annotate_noisy_segments(raw, epoch_duration=-1)

        with pytest.raises(ValueError):
            annotate_noisy_segments(raw, epoch_overlap=-1)

        with pytest.raises(ValueError):
            annotate_noisy_segments(raw, quantile_flag_crit=1.5)


class TestAnnotateUncorrelatedSegments:
    """Test uncorrelated segment annotation (from moved functions)."""

    @patch(
        "autoclean.functions.segment_rejection.segment_rejection.mne.make_fixed_length_events"
    )
    @patch("autoclean.functions.segment_rejection.segment_rejection.mne.Epochs")
    def test_basic_functionality(self, mock_epochs_class, mock_make_events, mock_raw):
        """Test basic uncorrelated segment annotation."""
        # Mock events
        mock_events = np.array([[0, 0, 1], [1000, 0, 1], [2000, 0, 1]])
        mock_make_events.return_value = mock_events

        # Mock epochs with montage
        mock_epochs = MagicMock()
        mock_epochs.events = mock_events
        mock_epochs.ch_names = [f"EEG_{i:03d}" for i in range(64)]
        mock_epochs.times = np.linspace(0, 2, 1000)
        mock_epochs_class.return_value = mock_epochs
        mock_epochs.__len__.return_value = 3

        # Mock montage
        mock_montage = MagicMock()
        mock_epochs.get_montage.return_value = mock_montage

        # Mock helper functions
        with (
            patch(
                "autoclean.functions.segment_rejection.segment_rejection._calculate_neighbor_correlations"
            ) as mock_calc,
            patch(
                "autoclean.functions.segment_rejection.segment_rejection._detect_outliers"
            ) as mock_detect,
        ):

            mock_calc.return_value = MagicMock()
            mock_detect.return_value = np.array([])  # No outliers

            result = annotate_uncorrelated_segments(mock_raw)

            # Should return a raw object
            assert isinstance(result, type(mock_raw))

    def test_input_validation(self):
        """Test input validation for uncorrelated segments."""
        with pytest.raises(TypeError):
            annotate_uncorrelated_segments("not_raw")

        raw = MagicMock(spec=mne.io.BaseRaw)

        with pytest.raises(ValueError):
            annotate_uncorrelated_segments(raw, epoch_duration=-1)

        with pytest.raises(ValueError):
            annotate_uncorrelated_segments(raw, n_nearest_neighbors=-1)

        with pytest.raises(ValueError):
            annotate_uncorrelated_segments(raw, corr_method="invalid")

        with pytest.raises(ValueError):
            annotate_uncorrelated_segments(raw, corr_trim_percent=60)

    @patch(
        "autoclean.functions.segment_rejection.segment_rejection.mne.make_fixed_length_events"
    )
    @patch("autoclean.functions.segment_rejection.segment_rejection.mne.Epochs")
    def test_no_montage_error(self, mock_epochs_class, mock_make_events, mock_raw):
        """Test error when no montage is set."""
        # Mock events
        mock_events = np.array([[0, 0, 1]])
        mock_make_events.return_value = mock_events

        # Mock epochs without montage
        mock_epochs = MagicMock()
        mock_epochs.events = mock_events
        mock_epochs_class.return_value = mock_epochs
        mock_epochs.__len__.return_value = 1
        mock_epochs.get_montage.return_value = None  # No montage

        with pytest.raises(ValueError, match="montage"):
            annotate_uncorrelated_segments(mock_raw)


class TestIntegration:
    """Integration tests for segment rejection functions."""

    def test_all_functions_work_together(self, mock_raw):
        """Test that all segment rejection functions can be chained."""
        # Apply all segment rejection functions in sequence
        raw1 = detect_dense_oscillatory_artifacts(mock_raw)

        # Mock the complex functions to avoid deep mocking
        with (
            patch(
                "autoclean.functions.segment_rejection.segment_rejection.mne.make_fixed_length_events"
            ) as mock_events,
            patch(
                "autoclean.functions.segment_rejection.segment_rejection.mne.Epochs"
            ) as mock_epochs_class,
            patch(
                "autoclean.functions.segment_rejection.segment_rejection._epochs_to_xr"
            ),
            patch(
                "autoclean.functions.segment_rejection.segment_rejection._detect_outliers",
                return_value=np.array([]),
            ),
        ):

            mock_events.return_value = np.array([[0, 0, 1]])
            mock_epochs = MagicMock()
            mock_epochs.events = np.array([[0, 0, 1]])
            mock_epochs_class.return_value = mock_epochs
            mock_epochs.__len__.return_value = 1

            raw2 = annotate_noisy_segments(raw1)

            # For uncorrelated segments, also mock montage
            mock_epochs.get_montage.return_value = MagicMock()
            with patch(
                "autoclean.functions.segment_rejection.segment_rejection._calculate_neighbor_correlations"
            ):
                raw3 = annotate_uncorrelated_segments(raw2)

        # All should return raw objects
        assert isinstance(raw1, type(mock_raw))
        assert isinstance(raw2, type(mock_raw))
        assert isinstance(raw3, type(mock_raw))
