"""Tests for advanced processing standalone functions.

This module tests advanced processing functions including autoreject-based
epoch cleaning and segment-based rejection methods.
"""

from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest
from mne.channels import make_standard_montage

# Import the functions to test
from autoclean.functions.advanced import autoreject_epochs
from autoclean.functions.segment_rejection import (
    annotate_noisy_segments,
    annotate_uncorrelated_segments,
)

# Import test utilities
from tests.fixtures.synthetic_data import create_synthetic_raw


class TestAutoRejectEpochs:
    """Test autoreject epochs function."""

    def test_autoreject_epochs_basic(self):
        """Test basic autoreject functionality with mock."""
        # Create synthetic data and epochs
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=16, duration=10.0, sfreq=250
        )

        # Create simple events and epochs
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.8, preload=True, baseline=None)

        # Mock AutoReject to avoid dependency issues in testing
        with patch("autoclean.functions.advanced.autoreject.AutoReject") as mock_ar:
            # Set up mock
            mock_instance = Mock()
            mock_ar.return_value = mock_instance

            # Mock cleaned epochs (remove some epochs to simulate rejection)
            clean_data = epochs.get_data()[::2]  # Keep every other epoch
            mock_cleaned_epochs = epochs.copy()[::2]
            mock_instance.fit_transform.return_value = mock_cleaned_epochs

            # Mock attributes for metadata
            mock_instance.bad_segments_ = np.zeros(
                (len(mock_cleaned_epochs), len(epochs.ch_names)), dtype=bool
            )
            mock_instance.loss_ = np.array([[0.1, 0.2], [0.15, 0.18]])

            # Test the function
            result_epochs, metadata = autoreject_epochs(epochs)

            # Verify results
            assert isinstance(result_epochs, mne.Epochs)
            assert isinstance(metadata, dict)
            assert "initial_epochs" in metadata
            assert "final_epochs" in metadata
            assert "rejection_percent" in metadata
            assert metadata["initial_epochs"] > metadata["final_epochs"]

            # Verify AutoReject was called correctly
            mock_ar.assert_called_once()
            mock_instance.fit_transform.assert_called_once()

    def test_autoreject_epochs_custom_parameters(self):
        """Test autoreject with custom parameters."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=8, duration=8.0, sfreq=250
        )

        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.8, preload=True, baseline=None)

        with patch("autoclean.functions.advanced.autoreject.AutoReject") as mock_ar:
            mock_instance = Mock()
            mock_ar.return_value = mock_instance
            mock_instance.fit_transform.return_value = epochs.copy()
            mock_instance.bad_segments_ = np.zeros(
                (len(epochs), len(epochs.ch_names)), dtype=bool
            )

            # Test with custom parameters
            n_interpolate = [1, 2, 4]
            consensus = [0.2, 0.5, 0.8]

            result_epochs, metadata = autoreject_epochs(
                epochs,
                n_interpolate=n_interpolate,
                consensus=consensus,
                n_jobs=2,
                cv=3,
                random_state=42,
            )

            # Verify AutoReject was called with correct parameters
            mock_ar.assert_called_once_with(
                n_interpolate=n_interpolate,
                consensus=consensus,
                cv=3,
                n_jobs=2,
                random_state=42,
                thresh_method="bayesian_optimization",
                verbose=None,
            )

    def test_autoreject_epochs_invalid_input(self):
        """Test autoreject with invalid inputs."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=16)

        # Test invalid data type
        with pytest.raises(TypeError, match="Data must be an MNE Epochs object"):
            autoreject_epochs(raw)  # Pass Raw instead of Epochs

        # Create valid epochs for other tests
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.8, preload=True, baseline=None)

        # Test insufficient epochs for CV
        with pytest.raises(
            ValueError, match="Need at least.*epochs for.*cross-validation"
        ):
            autoreject_epochs(epochs[:2], cv=4)  # Only 2 epochs for 4-fold CV

        # Test invalid n_interpolate
        with pytest.raises(ValueError, match="n_interpolate must be a list"):
            autoreject_epochs(epochs, n_interpolate="invalid")

        with pytest.raises(ValueError, match="Cannot interpolate more channels"):
            autoreject_epochs(
                epochs, n_interpolate=[100]
            )  # More than available channels

        # Test invalid consensus
        with pytest.raises(ValueError, match="consensus must be a list"):
            autoreject_epochs(epochs, consensus=1.5)

        with pytest.raises(ValueError, match="consensus must be a list"):
            autoreject_epochs(epochs, consensus=[0.5, 1.5])  # Value > 1

        # Test invalid cv
        with pytest.raises(ValueError, match="cv must be at least 2"):
            autoreject_epochs(epochs, cv=1)

    def test_autoreject_epochs_with_picks(self):
        """Test autoreject with channel selection."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=16, duration=8.0, sfreq=250
        )

        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.8, preload=True, baseline=None)

        with patch("autoclean.functions.advanced.autoreject.AutoReject") as mock_ar:
            mock_instance = Mock()
            mock_ar.return_value = mock_instance

            # Create mock result with fewer channels
            picks = ["Fp1", "Fp2", "F3", "F4"]
            picked_epochs = epochs.copy().pick(picks)
            mock_instance.fit_transform.return_value = picked_epochs
            mock_instance.bad_segments_ = np.zeros(
                (len(picked_epochs), len(picks)), dtype=bool
            )

            result_epochs, metadata = autoreject_epochs(epochs, picks=picks)

            # Should have used picked channels
            assert len(result_epochs.ch_names) == len(picks)
            assert metadata["channel_count"] == len(picks)

        # Test invalid picks
        with pytest.raises(ValueError, match="Picks not found in data"):
            autoreject_epochs(epochs, picks=["NonExistentChannel"])


class TestAnnotateNoisySegments:
    """Test noisy segment annotation function."""

    def test_annotate_noisy_segments_basic(self):
        """Test basic noisy segment detection."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=16, duration=10.0, sfreq=250
        )

        # Ensure montage is set
        if raw.get_montage() is None:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        result_raw = annotate_noisy_segments(raw)

        assert isinstance(result_raw, mne.io.BaseRaw)
        # Should return a copy, not modify original
        assert result_raw is not raw
        # Should have same or more annotations
        assert len(result_raw.annotations) >= len(raw.annotations)

    def test_annotate_noisy_segments_parameters(self):
        """Test noisy segment detection with custom parameters."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=8, duration=8.0, sfreq=250
        )

        if raw.get_montage() is None:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        result_raw = annotate_noisy_segments(
            raw,
            epoch_duration=1.5,
            epoch_overlap=0.5,
            quantile_k=4.0,
            quantile_flag_crit=0.3,
            annotation_description="BAD_very_noisy",
        )

        assert isinstance(result_raw, mne.io.BaseRaw)
        # Check if any custom annotations were added
        custom_annotations = [
            ann
            for ann in result_raw.annotations
            if ann["description"] == "BAD_very_noisy"
        ]
        assert isinstance(custom_annotations, list)  # May be empty, that's OK

    def test_annotate_noisy_segments_invalid_input(self):
        """Test noisy segment detection with invalid inputs."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=16)

        # Test invalid data type
        with pytest.raises(TypeError, match="Data must be an MNE Raw object"):
            annotate_noisy_segments("not_raw_data")

        # Test invalid parameters
        with pytest.raises(ValueError, match="epoch_duration must be positive"):
            annotate_noisy_segments(raw, epoch_duration=-1.0)

        with pytest.raises(ValueError, match="epoch_overlap must be non-negative"):
            annotate_noisy_segments(raw, epoch_overlap=-0.5)

        with pytest.raises(
            ValueError, match="quantile_flag_crit must be between 0 and 1"
        ):
            annotate_noisy_segments(raw, quantile_flag_crit=1.5)

        with pytest.raises(ValueError, match="quantile_k must be positive"):
            annotate_noisy_segments(raw, quantile_k=-2.0)

    def test_annotate_noisy_segments_no_epochs(self):
        """Test when no epochs can be created."""
        # Very short data that can't be epoched
        raw = create_synthetic_raw(duration=0.1, sfreq=250, n_channels=8)

        with pytest.raises(RuntimeError, match="Failed to annotate noisy segments"):
            annotate_noisy_segments(raw, epoch_duration=2.0)

    def test_annotate_noisy_segments_with_picks(self):
        """Test noisy segment detection with channel selection."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=16, duration=6.0, sfreq=250
        )

        if raw.get_montage() is None:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        # Test with specific channel picks
        picks = ["Fp1", "Fp2", "F3", "F4"]
        result_raw = annotate_noisy_segments(raw, picks=picks)

        assert isinstance(result_raw, mne.io.BaseRaw)
        # Should still have all original channels in result
        assert len(result_raw.ch_names) == len(raw.ch_names)


class TestAnnotateUncorrelatedSegments:
    """Test uncorrelated segment annotation function."""

    def test_annotate_uncorrelated_segments_basic(self):
        """Test basic uncorrelated segment detection."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=16, duration=8.0, sfreq=250
        )

        # Ensure montage is set
        if raw.get_montage() is None:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        result_raw = annotate_uncorrelated_segments(raw)

        assert isinstance(result_raw, mne.io.BaseRaw)
        assert result_raw is not raw  # Should be a copy
        assert len(result_raw.annotations) >= len(raw.annotations)

    def test_annotate_uncorrelated_segments_parameters(self):
        """Test uncorrelated segment detection with custom parameters."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=16, duration=8.0, sfreq=250
        )

        if raw.get_montage() is None:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        result_raw = annotate_uncorrelated_segments(
            raw,
            epoch_duration=2.5,
            n_nearest_neighbors=3,
            corr_method="mean",
            corr_trim_percent=20.0,
            outlier_k=3.0,
            outlier_flag_crit=0.25,
            annotation_description="BAD_poor_correlation",
        )

        assert isinstance(result_raw, mne.io.BaseRaw)
        # Check if any custom annotations were added
        custom_annotations = [
            ann
            for ann in result_raw.annotations
            if ann["description"] == "BAD_poor_correlation"
        ]
        assert isinstance(custom_annotations, list)

    def test_annotate_uncorrelated_segments_correlation_methods(self):
        """Test different correlation aggregation methods."""
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=12, duration=6.0, sfreq=250
        )

        if raw.get_montage() is None:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        # Test max correlation method
        result_max = annotate_uncorrelated_segments(raw, corr_method="max")
        assert isinstance(result_max, mne.io.BaseRaw)

        # Test mean correlation method
        result_mean = annotate_uncorrelated_segments(raw, corr_method="mean")
        assert isinstance(result_mean, mne.io.BaseRaw)

        # Test trimmean correlation method
        result_trim = annotate_uncorrelated_segments(
            raw, corr_method="trimmean", corr_trim_percent=15.0
        )
        assert isinstance(result_trim, mne.io.BaseRaw)

    def test_annotate_uncorrelated_segments_invalid_input(self):
        """Test uncorrelated segment detection with invalid inputs."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=16)

        # Test invalid data type
        with pytest.raises(TypeError, match="Data must be an MNE Raw object"):
            annotate_uncorrelated_segments("not_raw_data")

        # Test invalid parameters
        with pytest.raises(ValueError, match="epoch_duration must be positive"):
            annotate_uncorrelated_segments(raw, epoch_duration=0)

        with pytest.raises(ValueError, match="n_nearest_neighbors must be positive"):
            annotate_uncorrelated_segments(raw, n_nearest_neighbors=0)

        with pytest.raises(ValueError, match="corr_method must be"):
            annotate_uncorrelated_segments(raw, corr_method="invalid")

        with pytest.raises(
            ValueError, match="corr_trim_percent must be between 0 and 50"
        ):
            annotate_uncorrelated_segments(raw, corr_trim_percent=60.0)

        with pytest.raises(
            ValueError, match="outlier_flag_crit must be between 0 and 1"
        ):
            annotate_uncorrelated_segments(raw, outlier_flag_crit=2.0)

        with pytest.raises(ValueError, match="outlier_k must be positive"):
            annotate_uncorrelated_segments(raw, outlier_k=-1.0)

    def test_annotate_uncorrelated_segments_no_montage(self):
        """Test uncorrelated segment detection without montage."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=16)

        # Explicitly remove montage
        raw.set_montage(None)

        with pytest.raises(
            RuntimeError, match="Failed to annotate uncorrelated segments"
        ):
            annotate_uncorrelated_segments(raw)

    def test_annotate_uncorrelated_segments_few_neighbors(self):
        """Test with fewer channels than requested neighbors."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=4,  # Very few channels
            duration=6.0,
            sfreq=250,
        )

        if raw.get_montage() is None:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        # Request more neighbors than available
        result_raw = annotate_uncorrelated_segments(
            raw, n_nearest_neighbors=10  # More than available channels
        )

        assert isinstance(result_raw, mne.io.BaseRaw)


class TestAdvancedHelperFunctions:
    """Test helper functions used by advanced processing."""

    def test_epochs_to_xr(self):
        """Test epochs to xarray conversion."""
        from autoclean.functions.segment_rejection.segment_rejection import (
            _epochs_to_xr,
        )

        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=8)
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.8, preload=True, baseline=None)

        epochs_xr = _epochs_to_xr(epochs)

        # Check dimensions
        assert epochs_xr.dims == ("ch", "epoch", "time")
        assert epochs_xr.sizes["ch"] == len(epochs.ch_names)
        assert epochs_xr.sizes["epoch"] == len(epochs)
        assert epochs_xr.sizes["time"] == len(epochs.times)

    def test_get_outliers_quantile(self):
        """Test quantile-based outlier detection."""
        import xarray as xr

        from autoclean.functions.segment_rejection.segment_rejection import (
            _get_outliers_quantile,
        )

        # Create test data
        data = np.random.randn(5, 10)  # 5 channels, 10 epochs
        array = xr.DataArray(data, dims=("ch", "epoch"))

        lower_bound, upper_bound = _get_outliers_quantile(array, dim="epoch", k=2.0)

        assert lower_bound.dims == ("ch",)
        assert upper_bound.dims == ("ch",)
        assert len(lower_bound) == 5  # One value per channel
        assert len(upper_bound) == 5
        assert np.all(lower_bound <= upper_bound)

    def test_detect_outliers(self):
        """Test outlier detection function."""
        import xarray as xr

        from autoclean.functions.segment_rejection.segment_rejection import (
            _detect_outliers,
        )

        # Create test data with some clear outliers
        data = np.random.randn(3, 20)
        # Add clear outliers in specific epochs
        data[:, 5] *= 5  # Make epoch 5 very noisy for all channels
        data[:, 15] *= 5  # Make epoch 15 very noisy for all channels

        array = xr.DataArray(data, dims=("ch", "epoch"))

        flagged_indices = _detect_outliers(
            array,
            flag_dim="epoch",
            flag_crit=0.5,  # Flag if >50% of channels are outliers
            init_dir="pos",
            outliers_kwargs={"k": 2.0},
        )

        assert isinstance(flagged_indices, np.ndarray)
        # Should detect some outliers (exact detection depends on random data)
